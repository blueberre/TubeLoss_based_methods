"""
Single-dataset experiment runner.

This file is intentionally the only place where the method modules are wired
together. Each method family lives in its own source file under
src/experiment/methods/.
"""

from __future__ import annotations

import importlib

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler

from src.experiment.config import apply_cli_overrides
from src.experiment.data import prepare_raw_splits
from src.experiment.utils import cfg_get
from src.experiment.methods.tube_loss import run_tube_delta_sweep, select_optimal_delta
from src.experiment.methods.quantile import train_base_quantile_models
from src.experiment.methods.base_conformal import evaluate_base_and_conformal_methods
from src.experiment.methods.uncertainty_wrappers import evaluate_uncertainty_wrappers
from src.experiment.output import finalize_results


def run_single_dataset(dataset_name: str, args) -> dict:
    cfg = importlib.import_module(f"configs.{dataset_name}")
    apply_cli_overrides(cfg, args)

    torch.manual_seed(cfg.SEED)
    np.random.seed(cfg.SEED)

    data_file, X_tr_raw, X_ca_raw, X_te_raw, y_tr_raw, y_ca_raw, y_te_raw = prepare_raw_splits(dataset_name, cfg)

    # ── scale, fitting only on train ──────────────────────────────────────────────

    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()

    x_tr_sc = scaler_x.fit_transform(X_tr_raw).astype(np.float32)
    x_ca_sc = scaler_x.transform(X_ca_raw).astype(np.float32)
    x_te_sc = scaler_x.transform(X_te_raw).astype(np.float32)

    y_tr_sc = scaler_y.fit_transform(y_tr_raw).ravel().astype(np.float32)
    y_calib = y_ca_raw.ravel().astype(np.float32)
    y_test = y_te_raw.ravel().astype(np.float32)

    batch_size = len(x_tr_sc) if cfg.BATCH_SIZE == "full" else int(cfg.BATCH_SIZE)

    alpha = 1.0 - cfg.T
    epsilon = float(cfg_get(cfg, "EPSILON", 1e-8))
    run_ensembles = bool(cfg_get(cfg, "RUN_ENSEMBLES", True))
    reuse_base_ensembles = bool(cfg_get(cfg, "REUSE_BASE_ENSEMBLES", True))
    ensemble_B = int(cfg_get(cfg, "ENSEMBLE_B", min(int(cfg.EPOCHS), 50)))
    ensemble_mode = str(cfg_get(cfg, "ENSEMBLE_MODE", "epoch_snapshots"))
    ensemble_epochs = int(cfg_get(cfg, "ENSEMBLE_EPOCHS", min(int(cfg.EPOCHS), 100)))
    base_mode_default = "last" if ensemble_mode == "epoch_snapshots" else "mean"
    uacqr_base_mode = str(cfg_get(cfg, "UACQR_BASE_MODE", base_mode_default))
    uatcqr_base_mode = str(cfg_get(cfg, "UATCQR_BASE_MODE", base_mode_default))

    # If we reuse the already-trained base models, the final snapshot must be the
    # baseline for UACQR-S/UATCQR-S. That is exactly base_mode="last".
    if run_ensembles and reuse_base_ensembles:
        uacqr_base_mode = "last"
        uatcqr_base_mode = "last"

    print(f"Train : {x_tr_sc.shape}  |  Calib : {x_ca_sc.shape}  |  Test : {x_te_sc.shape}")
    print(f"BATCH_SIZE = {batch_size}")
    print(f"Epochs={cfg.EPOCHS}  LR={cfg.LR}  HID={cfg.HID}  WD={cfg.WEIGHT_DECAY}")
    print(f"Delta grid: {cfg.DELTAS[0]} to {cfg.DELTAS[-1]}  ({len(cfg.DELTAS)} steps)")
    print(f"Coverage target t = {cfg.T}")
    print(f"Calibration step size = {1 / len(y_calib):.4f}")
    print(f"Run ensembles = {run_ensembles}")
    if run_ensembles:
        print(f"Reuse base models for ensembles = {reuse_base_ensembles}")
        if reuse_base_ensembles:
            print(f"Snapshot ensemble: last {min(ensemble_B, int(cfg.EPOCHS))} epochs from the base training runs")
            print("UACQR/UATCQR base_mode forced to 'last' so the baseline is the trained base model")
        else:
            print(f"Ensemble mode={ensemble_mode}  B={ensemble_B}  member_epochs={ensemble_epochs}")
    print()

    # ── train Tube Loss models across delta grid ──────────────────────────────────

    tube_results = run_tube_delta_sweep(
        cfg,
        x_tr_sc,
        y_tr_sc,
        batch_size,
        x_ca_sc,
        x_te_sc,
        scaler_y,
        y_calib,
        y_test,
        run_ensembles=run_ensembles,
        reuse_base_ensembles=reuse_base_ensembles,
        ensemble_B=ensemble_B,
    )

    deltas, opt_delta, best_i, best, tube_model = select_optimal_delta(cfg, tube_results)

    # ── train base CQR / quantile models ──────────────────────────────────────────

    m_qr_lo, m_qr_hi, hist_lo, hist_hi, qr_base_snapshots = train_base_quantile_models(
        cfg,
        x_tr_sc,
        y_tr_sc,
        batch_size,
        alpha=alpha,
        run_ensembles=run_ensembles,
        reuse_base_ensembles=reuse_base_ensembles,
        ensemble_B=ensemble_B,
    )

    # ── evaluate base + conformal wrappers ────────────────────────────────────────

    method_rows = evaluate_base_and_conformal_methods(
        cfg,
        tube_model=tube_model,
        m_qr_lo=m_qr_lo,
        m_qr_hi=m_qr_hi,
        x_ca_sc=x_ca_sc,
        x_te_sc=x_te_sc,
        scaler_y=scaler_y,
        y_calib=y_calib,
        y_test=y_test,
        epsilon=epsilon,
    )

    # ── UACQR / UATCQR ensemble wrappers ──────────────────────────────────────────

    if run_ensembles:
        method_rows, _ensemble_histories = evaluate_uncertainty_wrappers(
            method_rows,
            cfg,
            x_tr_sc=x_tr_sc,
            y_tr_sc=y_tr_sc,
            batch_size=batch_size,
            opt_delta=opt_delta,
            best=best,
            hist_lo=hist_lo,
            hist_hi=hist_hi,
            qr_base_snapshots=qr_base_snapshots,
            x_ca_sc=x_ca_sc,
            x_te_sc=x_te_sc,
            scaler_y=scaler_y,
            y_calib=y_calib,
            y_test=y_test,
            alpha=alpha,
            reuse_base_ensembles=reuse_base_ensembles,
            ensemble_B=ensemble_B,
            ensemble_mode=ensemble_mode,
            ensemble_epochs=ensemble_epochs,
            uatcqr_base_mode=uatcqr_base_mode,
            uacqr_base_mode=uacqr_base_mode,
            epsilon=epsilon,
        )

    # ── save plots and metrics ────────────────────────────────────────────────────

    return finalize_results(
        dataset_name=dataset_name,
        data_file=data_file,
        cfg=cfg,
        batch_size=batch_size,
        opt_delta=opt_delta,
        best_i=best_i,
        best=best,
        run_ensembles=run_ensembles,
        reuse_base_ensembles=reuse_base_ensembles,
        ensemble_B=ensemble_B,
        ensemble_mode=ensemble_mode,
        ensemble_epochs=ensemble_epochs,
        uacqr_base_mode=uacqr_base_mode,
        uatcqr_base_mode=uatcqr_base_mode,
        epsilon=epsilon,
        y_tr_sc=y_tr_sc,
        y_calib=y_calib,
        y_test=y_test,
        deltas=deltas,
        tube_results=tube_results,
        method_rows=method_rows,
    )
