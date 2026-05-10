"""
Base quantile / CQR model training.
"""

from __future__ import annotations

import numpy as np

from src.training import train_qr


def train_base_quantile_models(
    cfg,
    x_tr_sc: np.ndarray,
    y_tr_sc: np.ndarray,
    batch_size: int,
    *,
    alpha: float,
    run_ensembles: bool,
    reuse_base_ensembles: bool,
    ensemble_B: int,
):
    print("Training base quantile models for CQR ...")
    qr_base_snapshots = None
    if run_ensembles and reuse_base_ensembles:
        m_qr_lo, m_qr_hi, hist_lo, hist_hi, qr_base_snapshots = train_qr(
            x_tr_sc, y_tr_sc, batch_size,
            epochs=cfg.EPOCHS,
            lr=cfg.LR,
            hid=cfg.HID,
            n_layers=cfg.N_LAYERS,
            weight_decay=cfg.WEIGHT_DECAY,
            alpha=alpha,
            seed=cfg.SEED,
            return_snapshots=True,
            snapshot_count=ensemble_B,
        )
    else:
        m_qr_lo, m_qr_hi, hist_lo, hist_hi = train_qr(
            x_tr_sc, y_tr_sc, batch_size,
            epochs=cfg.EPOCHS,
            lr=cfg.LR,
            hid=cfg.HID,
            n_layers=cfg.N_LAYERS,
            weight_decay=cfg.WEIGHT_DECAY,
            alpha=alpha,
            seed=cfg.SEED,
        )
    print(f"  Final lower-q train loss = {hist_lo[-1]:.4f}")
    print(f"  Final upper-q train loss = {hist_hi[-1]:.4f}")
    if run_ensembles and reuse_base_ensembles:
        print(f"  Saved {len(qr_base_snapshots)} quantile snapshot pairs from this same base training run")
    print()

    return m_qr_lo, m_qr_hi, hist_lo, hist_hi, qr_base_snapshots
