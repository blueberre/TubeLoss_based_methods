"""
src/training.py
================
Training utilities for the TubeLoss-vs-CQR experiment family.

This version keeps the original/base losses and model classes from the
TubeLoss-vs-CQR codebase:

    - src.losses.tube_loss
    - src.losses.pinball_loss
    - src.models.TubePINet
    - src.models.QuantileNet

On top of those base losses, it adds the training machinery needed by the
additional conformal methods:

    - TCQR / CQR: additive conformal wrappers, evaluated in run_experiment.py
    - TCQR-r / CQR-r: relative-width conformal wrappers, evaluated in run_experiment.py
    - UATCQR-S/P and UACQR-S/P: ensemble / epoch-snapshot training helpers

The important design choice is that every model is still trained with the
original `tube_loss` or `pinball_loss`; the newer methods are wrappers around
those learned intervals, not replacement losses.
"""

from __future__ import annotations

import copy
from typing import Iterable, Literal

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models import TubePINet, QuantileNet
from src.losses import tube_loss, pinball_loss


# ── data loader ───────────────────────────────────────────────────────────────

def make_loader(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    batch_size: int,
    seed: int = 42,
    *,
    shuffle: bool = True,
) -> DataLoader:
    """Create a deterministic PyTorch DataLoader from numpy arrays."""
    g = torch.Generator()
    g.manual_seed(seed)

    dataset = TensorDataset(
        torch.FloatTensor(x_tr),
        torch.FloatTensor(y_tr).reshape(-1),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
        drop_last=False,
    )


def _seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


# ── base tube-loss training ───────────────────────────────────────────────────

def train_tube(
    x_tr:         np.ndarray,
    y_tr:         np.ndarray,
    batch_size:   int,
    delta:        float = 0.0,
    *,
    epochs:       int   = 100,
    lr:           float = 0.001,
    hid:          int   = 32,
    n_layers:     int   = 1,
    weight_decay: float = 1e-3,
    t:            float = 0.90,
    r:            float = 0.50,
    seed:         int   = 42,
    return_snapshots: bool = False,
    snapshot_count: int | None = None,
) -> tuple:
    """
    Train one TubePINet with the original base `tube_loss`.

    When ``return_snapshots=True``, this returns ``(model, loss_history,
    snapshots)``. The snapshots are CPU copies from the same training run, and
    the final snapshot is the same trained base model returned as ``model``.
    This lets UATCQR reuse the already-trained TubeLoss model instead of
    training a separate ensemble baseline.
    """

    _seed(seed)

    in_dim = x_tr.shape[1]
    model  = TubePINet(in_dim=in_dim, hid=hid, n_layers=n_layers)
    opt    = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    loader = make_loader(x_tr, y_tr, batch_size, seed)

    model.train()
    loss_history: list[float] = []

    snapshots: list[TubePINet] = []
    if return_snapshots:
        if snapshot_count is None:
            snapshot_count = epochs
        snapshot_count = max(1, min(int(snapshot_count), int(epochs)))
        snapshot_start_epoch = int(epochs) - snapshot_count
    else:
        snapshot_start_epoch = int(epochs) + 1

    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            opt.zero_grad()
            lo, hi = model(xb)
            loss   = tube_loss(yb, lo, hi, delta, t=t, r=r)

            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Non-finite Tube Loss detected: {loss.item()}")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            epoch_loss += loss.item()
        loss_history.append(epoch_loss / len(loader))

        if return_snapshots and epoch >= snapshot_start_epoch:
            snapshots.append(copy.deepcopy(model).cpu().eval())

    model = model.eval()
    if return_snapshots:
        return model, loss_history, snapshots
    return model, loss_history


# ── base CQR / quantile-regression training ───────────────────────────────────

def train_qr(
    x_tr:         np.ndarray,
    y_tr:         np.ndarray,
    batch_size:   int,
    *,
    epochs:       int   = 100,
    lr:           float = 0.001,
    hid:          int   = 32,
    n_layers:     int   = 1,
    weight_decay: float = 1e-3,
    alpha:        float = 0.10,
    seed:         int   = 42,
    return_snapshots: bool = False,
    snapshot_count: int | None = None,
) -> tuple:
    """
    Train lower/upper QuantileNet models with the original pinball loss.

    When ``return_snapshots=True``, this returns ``(lo_model, hi_model,
    hist_lo, hist_hi, ensemble_snapshots)``. The snapshot ensemble is built from
    the same lower/upper training runs used for the base Pinball/CQR model, and
    the final snapshot pair is the same trained base quantile pair.
    """

    in_dim = x_tr.shape[1]
    nets: list[QuantileNet] = []
    histories: list[list[float]] = []
    snapshots_by_net: list[list[QuantileNet]] = []

    if return_snapshots:
        if snapshot_count is None:
            snapshot_count = epochs
        snapshot_count = max(1, min(int(snapshot_count), int(epochs)))
        snapshot_start_epoch = int(epochs) - snapshot_count
    else:
        snapshot_start_epoch = int(epochs) + 1

    for i, q in enumerate([alpha / 2.0, 1.0 - alpha / 2.0]):
        _seed(seed + i)

        m   = QuantileNet(in_dim=in_dim, hid=hid, n_layers=n_layers)
        opt = torch.optim.Adam(
            m.parameters(), lr=lr, weight_decay=weight_decay
        )
        loader = make_loader(x_tr, y_tr, batch_size, seed)

        m.train()
        loss_history: list[float] = []
        snapshots_this_net: list[QuantileNet] = []

        for epoch in range(epochs):
            epoch_loss = 0.0
            for xb, yb in loader:
                opt.zero_grad()
                loss = pinball_loss(yb, m(xb), q)

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(f"Non-finite pinball loss detected: {loss.item()}")

                loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 2.0)
                opt.step()
                epoch_loss += loss.item()
            loss_history.append(epoch_loss / len(loader))

            if return_snapshots and epoch >= snapshot_start_epoch:
                snapshots_this_net.append(copy.deepcopy(m).cpu().eval())

        nets.append(m.eval())
        histories.append(loss_history)
        if return_snapshots:
            snapshots_by_net.append(snapshots_this_net)

    if return_snapshots:
        n_snapshots = min(len(snapshots_by_net[0]), len(snapshots_by_net[1]))
        ensemble_snapshots = [
            [snapshots_by_net[0][j], snapshots_by_net[1][j]]
            for j in range(n_snapshots)
        ]
        return nets[0], nets[1], histories[0], histories[1], ensemble_snapshots

    return nets[0], nets[1], histories[0], histories[1]


# ── ensemble helpers for UACQR / UATCQR methods ───────────────────────────────

def _bootstrap_sample(
    x_train: np.ndarray,
    y_train: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw a bootstrap sample of size n with replacement."""
    n = len(y_train)
    idx = rng.integers(0, n, size=n)
    return np.asarray(x_train)[idx], np.asarray(y_train)[idx]


def _order_endpoints(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.minimum(lo, hi), np.maximum(lo, hi)


def train_tube_epoch_snapshots(
    x_tr:         np.ndarray,
    y_tr:         np.ndarray,
    batch_size:   int,
    delta:        float = 0.0,
    *,
    B:            int   = 100,
    lr:           float = 0.001,
    hid:          int   = 32,
    n_layers:     int   = 1,
    weight_decay: float = 1e-3,
    t:            float = 0.90,
    r:            float = 0.50,
    seed:         int   = 42,
) -> tuple[list[TubePINet], list[float]]:
    """
    Train one TubePINet and save a CPU snapshot after each epoch.

    These snapshots are the TubeLoss-side ensemble used by UATCQR-S/P.  The
    model is still optimized with the original `tube_loss`.
    """

    _seed(seed)

    model = TubePINet(in_dim=x_tr.shape[1], hid=hid, n_layers=n_layers)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = make_loader(x_tr, y_tr, batch_size, seed)

    ensemble: list[TubePINet] = []
    loss_history: list[float] = []

    for epoch in range(B):
        model.train()
        epoch_loss = 0.0

        for xb, yb in loader:
            opt.zero_grad()
            lo, hi = model(xb)
            loss = tube_loss(yb, lo, hi, delta, t=t, r=r)

            if not torch.isfinite(loss).item():
                raise FloatingPointError(
                    f"Non-finite Tube Loss detected at snapshot epoch {epoch + 1}: {loss.item()}"
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            epoch_loss += loss.item()

        loss_history.append(epoch_loss / len(loader))
        ensemble.append(copy.deepcopy(model).cpu().eval())

    return ensemble, loss_history


def train_qr_epoch_snapshots(
    x_tr:         np.ndarray,
    y_tr:         np.ndarray,
    batch_size:   int,
    *,
    B:            int   = 100,
    lr:           float = 0.001,
    hid:          int   = 32,
    n_layers:     int   = 1,
    weight_decay: float = 1e-3,
    alpha:        float = 0.10,
    seed:         int   = 42,
) -> tuple[list[list[QuantileNet]], list[list[float]]]:
    """
    Train lower/upper quantile models and save one pair of snapshots per epoch.

    These snapshots are the quantile-side ensemble used by UACQR-S/P.  The
    models are still optimized with the original `pinball_loss`.
    """

    quantiles = [alpha / 2.0, 1.0 - alpha / 2.0]
    models: list[QuantileNet] = []
    opts: list[torch.optim.Optimizer] = []

    for i in range(2):
        _seed(seed + i)
        m = QuantileNet(in_dim=x_tr.shape[1], hid=hid, n_layers=n_layers)
        opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=weight_decay)
        models.append(m)
        opts.append(opt)

    loader = make_loader(x_tr, y_tr, batch_size, seed)
    ensemble: list[list[QuantileNet]] = []
    histories: list[list[float]] = [[], []]

    for epoch in range(B):
        for m in models:
            m.train()

        epoch_losses = [0.0, 0.0]

        for xb, yb in loader:
            for i, q in enumerate(quantiles):
                opts[i].zero_grad()
                loss = pinball_loss(yb, models[i](xb), q)

                if not torch.isfinite(loss).item():
                    raise FloatingPointError(
                        f"Non-finite pinball loss detected at snapshot epoch {epoch + 1}: {loss.item()}"
                    )

                loss.backward()
                nn.utils.clip_grad_norm_(models[i].parameters(), 2.0)
                opts[i].step()
                epoch_losses[i] += loss.item()

        for i in range(2):
            histories[i].append(epoch_losses[i] / len(loader))

        ensemble.append([copy.deepcopy(m).cpu().eval() for m in models])

    return ensemble, histories


def train_tube_ensemble(
    x_tr:         np.ndarray,
    y_tr:         np.ndarray,
    batch_size:   int,
    delta:        float = 0.0,
    *,
    B:            int   = 20,
    epochs:       int   = 100,
    lr:           float = 0.001,
    hid:          int   = 32,
    n_layers:     int   = 1,
    weight_decay: float = 1e-3,
    t:            float = 0.90,
    r:            float = 0.50,
    seed:         int   = 42,
    mode:         Literal["epoch_snapshots", "bootstrap", "deep_ensemble"] = "epoch_snapshots",
) -> tuple[list[TubePINet], list]:
    """Build the TubeLoss-side ensemble for UATCQR-S/P."""

    if mode == "epoch_snapshots":
        return train_tube_epoch_snapshots(
            x_tr, y_tr, batch_size, delta,
            B=B, lr=lr, hid=hid, n_layers=n_layers,
            weight_decay=weight_decay, t=t, r=r, seed=seed,
        )

    if mode not in {"bootstrap", "deep_ensemble"}:
        raise ValueError("mode must be 'epoch_snapshots', 'bootstrap', or 'deep_ensemble'")

    rng = np.random.default_rng(seed)
    ensemble: list[TubePINet] = []
    histories: list[list[float]] = []

    for b in range(B):
        if mode == "bootstrap":
            xb, yb = _bootstrap_sample(x_tr, y_tr, rng)
        else:
            xb, yb = x_tr, y_tr

        model_b, hist_b = train_tube(
            xb, yb, batch_size, delta,
            epochs=epochs, lr=lr, hid=hid, n_layers=n_layers,
            weight_decay=weight_decay, t=t, r=r, seed=seed + b,
        )
        ensemble.append(model_b.cpu().eval())
        histories.append(hist_b)

    return ensemble, histories


def train_qr_ensemble(
    x_tr:         np.ndarray,
    y_tr:         np.ndarray,
    batch_size:   int,
    *,
    B:            int   = 20,
    epochs:       int   = 100,
    lr:           float = 0.001,
    hid:          int   = 32,
    n_layers:     int   = 1,
    weight_decay: float = 1e-3,
    alpha:        float = 0.10,
    seed:         int   = 42,
    mode:         Literal["epoch_snapshots", "bootstrap", "deep_ensemble"] = "epoch_snapshots",
) -> tuple[list[list[QuantileNet]], list]:
    """Build the quantile-side ensemble for UACQR-S/P."""

    if mode == "epoch_snapshots":
        return train_qr_epoch_snapshots(
            x_tr, y_tr, batch_size,
            B=B, lr=lr, hid=hid, n_layers=n_layers,
            weight_decay=weight_decay, alpha=alpha, seed=seed,
        )

    if mode not in {"bootstrap", "deep_ensemble"}:
        raise ValueError("mode must be 'epoch_snapshots', 'bootstrap', or 'deep_ensemble'")

    rng = np.random.default_rng(seed)
    ensemble: list[list[QuantileNet]] = []
    histories: list[tuple[list[float], list[float]]] = []

    for b in range(B):
        if mode == "bootstrap":
            xb, yb = _bootstrap_sample(x_tr, y_tr, rng)
        else:
            xb, yb = x_tr, y_tr

        lo_m, hi_m, hist_lo, hist_hi = train_qr(
            xb, yb, batch_size,
            epochs=epochs, lr=lr, hid=hid, n_layers=n_layers,
            weight_decay=weight_decay, alpha=alpha, seed=seed + b,
        )
        ensemble.append([lo_m.cpu().eval(), hi_m.cpu().eval()])
        histories.append((hist_lo, hist_hi))

    return ensemble, histories


# ── ensemble prediction helpers, scaled target space ──────────────────────────
# The experiment runner inverse-transforms these arrays into the original target
# scale before calibration/evaluation.

@torch.no_grad()
def predict_tube_ensemble_scaled(
    ensemble: Iterable[TubePINet],
    x_sc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lo_all: list[np.ndarray] = []
    hi_all: list[np.ndarray] = []
    x_t = torch.FloatTensor(x_sc)

    for model in ensemble:
        model.eval()
        lo, hi = model(x_t)
        lo_np = lo.cpu().numpy().reshape(-1)
        hi_np = hi.cpu().numpy().reshape(-1)
        lo_np, hi_np = _order_endpoints(lo_np, hi_np)
        lo_all.append(lo_np)
        hi_all.append(hi_np)

    return np.stack(lo_all, axis=0), np.stack(hi_all, axis=0)


@torch.no_grad()
def predict_qr_ensemble_scaled(
    ensemble: Iterable[Iterable[QuantileNet]],
    x_sc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lo_all: list[np.ndarray] = []
    hi_all: list[np.ndarray] = []
    x_t = torch.FloatTensor(x_sc)

    for pair in ensemble:
        lo_model, hi_model = pair
        lo_model.eval()
        hi_model.eval()
        lo = lo_model(x_t).cpu().numpy().reshape(-1)
        hi = hi_model(x_t).cpu().numpy().reshape(-1)
        lo, hi = _order_endpoints(lo, hi)
        lo_all.append(lo)
        hi_all.append(hi)

    return np.stack(lo_all, axis=0), np.stack(hi_all, axis=0)
