"""
UATCQR-S/P and UACQR-S/P ensemble wrappers.
"""

from __future__ import annotations

from src.training import train_tube_ensemble, train_qr_ensemble

from src.experiment.prediction import predict_tube_ensemble, predict_qr_ensemble
from src.experiment.metrics import method_record
from src.experiment.conformal_uncertainty import (
    uacqrs_calibrate_apply,
    uacqrp_calibrate_apply,
)


def evaluate_uncertainty_wrappers(
    method_rows: list[dict],
    cfg,
    *,
    x_tr_sc,
    y_tr_sc,
    batch_size: int,
    opt_delta: float,
    best: dict,
    hist_lo,
    hist_hi,
    qr_base_snapshots,
    x_ca_sc,
    x_te_sc,
    scaler_y,
    y_calib,
    y_test,
    alpha: float,
    reuse_base_ensembles: bool,
    ensemble_B: int,
    ensemble_mode: str,
    ensemble_epochs: int,
    uatcqr_base_mode: str,
    uacqr_base_mode: str,
    epsilon: float,
):
    tube_ensemble = None
    qr_ensemble = None
    ensemble_histories = {}

    if reuse_base_ensembles:
        print("Using snapshots from the already-trained base models for UATCQR/UACQR ...")
        tube_ensemble = best.get("snapshots")
        qr_ensemble = qr_base_snapshots

        if not tube_ensemble:
            raise RuntimeError("Tube snapshot ensemble missing. Set REUSE_BASE_ENSEMBLES=False to retrain ensembles.")
        if not qr_ensemble:
            raise RuntimeError("Quantile snapshot ensemble missing. Set REUSE_BASE_ENSEMBLES=False to retrain ensembles.")

        ensemble_histories["tube"] = best.get("loss_hist", [])[-len(tube_ensemble):]
        ensemble_histories["quantile"] = {
            "lower": hist_lo[-len(qr_ensemble):],
            "upper": hist_hi[-len(qr_ensemble):],
        }
        print(f"  Tube snapshot members     : {len(tube_ensemble)}; final member is the selected Tube Loss base model")
        print(f"  Quantile snapshot members : {len(qr_ensemble)}; final pair is the Pinball/CQR base model")
    else:
        print("Training separate TubeLoss ensemble for UATCQR-S/P ...")
        tube_ensemble, tube_ens_hist = train_tube_ensemble(
            x_tr_sc, y_tr_sc, batch_size, opt_delta,
            B=ensemble_B,
            epochs=ensemble_epochs,
            lr=cfg.LR,
            hid=cfg.HID,
            n_layers=cfg.N_LAYERS,
            weight_decay=cfg.WEIGHT_DECAY,
            t=cfg.T,
            r=cfg.R,
            seed=cfg.SEED,
            mode=ensemble_mode,
        )
        ensemble_histories["tube"] = tube_ens_hist

        print("Training separate quantile ensemble for UACQR-S/P ...")
        qr_ensemble, qr_ens_hist = train_qr_ensemble(
            x_tr_sc, y_tr_sc, batch_size,
            B=ensemble_B,
            epochs=ensemble_epochs,
            lr=cfg.LR,
            hid=cfg.HID,
            n_layers=cfg.N_LAYERS,
            weight_decay=cfg.WEIGHT_DECAY,
            alpha=alpha,
            seed=cfg.SEED,
            mode=ensemble_mode,
        )
        ensemble_histories["quantile"] = qr_ens_hist

    # Tube ensemble predictions in original target scale
    tube_lo_ens_cal, tube_hi_ens_cal = predict_tube_ensemble(tube_ensemble, x_ca_sc, scaler_y)
    tube_lo_ens_te, tube_hi_ens_te = predict_tube_ensemble(tube_ensemble, x_te_sc, scaler_y)

    # Quantile ensemble predictions in original target scale
    qr_lo_ens_cal, qr_hi_ens_cal = predict_qr_ensemble(qr_ensemble, x_ca_sc, scaler_y)
    qr_lo_ens_te, qr_hi_ens_te = predict_qr_ensemble(qr_ensemble, x_te_sc, scaler_y)

    # UATCQR-S / UACQR-S
    lo_cal_s, hi_cal_s, t_hat = uacqrs_calibrate_apply(
        tube_lo_ens_cal, tube_hi_ens_cal, y_calib,
        tube_lo_ens_cal, tube_hi_ens_cal,
        coverage=cfg.T,
        base_mode=uatcqr_base_mode,
        epsilon=epsilon,
    )
    lo_te_s, hi_te_s, _ = uacqrs_calibrate_apply(
        tube_lo_ens_cal, tube_hi_ens_cal, y_calib,
        tube_lo_ens_te, tube_hi_ens_te,
        coverage=cfg.T,
        base_mode=uatcqr_base_mode,
        epsilon=epsilon,
    )
    method_rows.append(method_record(
        "Tubeloss", "UATCQR-S",
        lo_cal_s, hi_cal_s, lo_te_s, hi_te_s,
        y_calib, y_test,
        q_hat=t_hat,
    ))

    lo_cal_s, hi_cal_s, t_hat = uacqrs_calibrate_apply(
        qr_lo_ens_cal, qr_hi_ens_cal, y_calib,
        qr_lo_ens_cal, qr_hi_ens_cal,
        coverage=cfg.T,
        base_mode=uacqr_base_mode,
        epsilon=epsilon,
    )
    lo_te_s, hi_te_s, _ = uacqrs_calibrate_apply(
        qr_lo_ens_cal, qr_hi_ens_cal, y_calib,
        qr_lo_ens_te, qr_hi_ens_te,
        coverage=cfg.T,
        base_mode=uacqr_base_mode,
        epsilon=epsilon,
    )
    method_rows.append(method_record(
        "Quantile", "UACQR-S",
        lo_cal_s, hi_cal_s, lo_te_s, hi_te_s,
        y_calib, y_test,
        q_hat=t_hat,
    ))

    # UATCQR-P / UACQR-P
    lo_cal_p, hi_cal_p, t_hat = uacqrp_calibrate_apply(
        tube_lo_ens_cal, tube_hi_ens_cal, y_calib,
        tube_lo_ens_cal, tube_hi_ens_cal,
        coverage=cfg.T,
    )
    lo_te_p, hi_te_p, _ = uacqrp_calibrate_apply(
        tube_lo_ens_cal, tube_hi_ens_cal, y_calib,
        tube_lo_ens_te, tube_hi_ens_te,
        coverage=cfg.T,
    )
    method_rows.append(method_record(
        "Tubeloss", "UATCQR-P",
        lo_cal_p, hi_cal_p, lo_te_p, hi_te_p,
        y_calib, y_test,
        q_hat=t_hat,
    ))

    lo_cal_p, hi_cal_p, t_hat = uacqrp_calibrate_apply(
        qr_lo_ens_cal, qr_hi_ens_cal, y_calib,
        qr_lo_ens_cal, qr_hi_ens_cal,
        coverage=cfg.T,
    )
    lo_te_p, hi_te_p, _ = uacqrp_calibrate_apply(
        qr_lo_ens_cal, qr_hi_ens_cal, y_calib,
        qr_lo_ens_te, qr_hi_ens_te,
        coverage=cfg.T,
    )
    method_rows.append(method_record(
        "Quantile", "UACQR-P",
        lo_cal_p, hi_cal_p, lo_te_p, hi_te_p,
        y_calib, y_test,
        q_hat=t_hat,
    ))

    return method_rows, ensemble_histories
