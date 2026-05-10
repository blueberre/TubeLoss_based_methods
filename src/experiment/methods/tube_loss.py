"""
Tube Loss training sweep and optimal-delta selection.
"""

from __future__ import annotations

import numpy as np

from src.training import train_tube
from src.evaluation import (
    predict_tube,
    nc_scores,
    raw_metrics,
    conformal_quantile,
    conformal_metrics,
)


def run_tube_delta_sweep(
    cfg,
    x_tr_sc: np.ndarray,
    y_tr_sc: np.ndarray,
    batch_size: int,
    x_ca_sc: np.ndarray,
    x_te_sc: np.ndarray,
    scaler_y,
    y_calib: np.ndarray,
    y_test: np.ndarray,
    *,
    run_ensembles: bool,
    reuse_base_ensembles: bool,
    ensemble_B: int,
) -> list[dict]:
    tube_results = []

    print(f"Training {len(cfg.DELTAS)} Tube Loss models ...")
    print()
    print(f"{'delta':>7} | {'Cal PICP':>9} {'Cal MPIW':>9} {'Q_t':>9} | {'TCQR Test PICP':>14} {'TCQR Test MPIW':>14}")
    print("-" * 86)

    for d in cfg.DELTAS:
        tube_snapshots = None
        if run_ensembles and reuse_base_ensembles:
            model, loss_hist, tube_snapshots = train_tube(
                x_tr_sc, y_tr_sc, batch_size, d,
                epochs=cfg.EPOCHS,
                lr=cfg.LR,
                hid=cfg.HID,
                n_layers=cfg.N_LAYERS,
                weight_decay=cfg.WEIGHT_DECAY,
                t=cfg.T,
                r=cfg.R,
                seed=cfg.SEED,
                return_snapshots=True,
                snapshot_count=ensemble_B,
            )
        else:
            model, loss_hist = train_tube(
                x_tr_sc, y_tr_sc, batch_size, d,
                epochs=cfg.EPOCHS,
                lr=cfg.LR,
                hid=cfg.HID,
                n_layers=cfg.N_LAYERS,
                weight_decay=cfg.WEIGHT_DECAY,
                t=cfg.T,
                r=cfg.R,
                seed=cfg.SEED,
            )

        lo_cal, hi_cal = predict_tube(model, x_ca_sc, scaler_y)
        cal_picp, cal_mpiw = raw_metrics(lo_cal, hi_cal, y_calib)
        nc_cal = nc_scores(lo_cal, hi_cal, y_calib)
        Q_t = conformal_quantile(nc_cal, cfg.T)

        lo_te, hi_te = predict_tube(model, x_te_sc, scaler_y)
        raw_test_picp, raw_test_mpiw = raw_metrics(lo_te, hi_te, y_test)
        tcqr_test_picp, tcqr_test_mpiw = conformal_metrics(lo_te, hi_te, Q_t, y_test)

        tube_results.append(dict(
            delta=float(d),
            model=model,
            snapshots=tube_snapshots,
            loss_hist=loss_hist,
            raw_cal_picp=cal_picp,
            raw_cal_mpiw=cal_mpiw,
            raw_test_picp=raw_test_picp,
            raw_test_mpiw=raw_test_mpiw,
            additive_q_hat=float(Q_t),
            additive_test_picp=tcqr_test_picp,
            additive_test_mpiw=tcqr_test_mpiw,
            lo_cal=lo_cal,
            hi_cal=hi_cal,
            lo_te=lo_te,
            hi_te=hi_te,
        ))

        print(f"{d:>7.3f} | {cal_picp:>9.4f} {cal_mpiw:>9.4f} {Q_t:>9.4f} | {tcqr_test_picp:>14.4f} {tcqr_test_mpiw:>14.4f}"
              f"  [train loss: {loss_hist[-1]:.4f}]")

    return tube_results


def select_optimal_delta(cfg, tube_results: list[dict]):
    # Rule: among deltas where raw calibration PICP >= target, choose narrowest raw
    # calibration interval. Fallback: closest raw calibration PICP to target.
    picp_cals = np.array([r["raw_cal_picp"] for r in tube_results])
    mpiw_cals = np.array([r["raw_cal_mpiw"] for r in tube_results])
    deltas = np.array([r["delta"] for r in tube_results])

    eligible = picp_cals >= cfg.T
    if eligible.any():
        cands = np.where(eligible)[0]
        best_i = int(cands[np.argmin(mpiw_cals[cands])])
    else:
        best_i = int(np.argmin(np.abs(picp_cals - cfg.T)))

    opt_delta = float(deltas[best_i])
    best = tube_results[best_i]
    tube_model = best["model"]

    print()
    print("=" * 76)
    print(f"Optimal delta = {opt_delta:.3f}  (raw cal PICP={best['raw_cal_picp']:.4f}  raw cal MPIW={best['raw_cal_mpiw']:.4f})")
    print("=" * 76)
    print()

    return deltas, opt_delta, best_i, best, tube_model
