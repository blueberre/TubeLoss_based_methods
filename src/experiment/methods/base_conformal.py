"""
Raw, additive conformal, and relative-width conformal methods.
"""

from __future__ import annotations

from src.evaluation import predict_tube, nc_scores, conformal_quantile

from src.experiment.prediction import predict_qr_ordered
from src.experiment.metrics import method_record
from src.experiment.conformal_relative import _relative_qhat, _apply_relative


def evaluate_base_and_conformal_methods(
    cfg,
    *,
    tube_model,
    m_qr_lo,
    m_qr_hi,
    x_ca_sc,
    x_te_sc,
    scaler_y,
    y_calib,
    y_test,
    epsilon: float,
) -> list[dict]:
    method_rows: list[dict] = []

    # Raw Tube Loss
    lo_tube_cal, hi_tube_cal = predict_tube(tube_model, x_ca_sc, scaler_y)
    lo_tube_te, hi_tube_te = predict_tube(tube_model, x_te_sc, scaler_y)
    method_rows.append(method_record(
        "Tubeloss", "Tube Loss",
        lo_tube_cal, hi_tube_cal, lo_tube_te, hi_tube_te,
        y_calib, y_test,
    ))

    # Raw Quantile / Pinball Loss
    lo_qr_cal, hi_qr_cal = predict_qr_ordered(m_qr_lo, m_qr_hi, x_ca_sc, scaler_y)
    lo_qr_te, hi_qr_te = predict_qr_ordered(m_qr_lo, m_qr_hi, x_te_sc, scaler_y)
    method_rows.append(method_record(
        "Quantile", "Pinball Loss",
        lo_qr_cal, hi_qr_cal, lo_qr_te, hi_qr_te,
        y_calib, y_test,
    ))

    # TCQR / CQR additive conformal
    q_tube_add = float(conformal_quantile(nc_scores(lo_tube_cal, hi_tube_cal, y_calib), cfg.T))
    method_rows.append(method_record(
        "Tubeloss", "TCQR",
        lo_tube_cal - q_tube_add, hi_tube_cal + q_tube_add,
        lo_tube_te - q_tube_add, hi_tube_te + q_tube_add,
        y_calib, y_test,
        q_hat=q_tube_add,
    ))

    q_qr_add = float(conformal_quantile(nc_scores(lo_qr_cal, hi_qr_cal, y_calib), cfg.T))
    method_rows.append(method_record(
        "Quantile", "CQR",
        lo_qr_cal - q_qr_add, hi_qr_cal + q_qr_add,
        lo_qr_te - q_qr_add, hi_qr_te + q_qr_add,
        y_calib, y_test,
        q_hat=q_qr_add,
    ))

    # TCQR-r / CQR-r relative-width conformal
    q_tube_rel = _relative_qhat(lo_tube_cal, hi_tube_cal, y_calib, cfg.T, epsilon)
    lo_cal_rel, hi_cal_rel = _apply_relative(lo_tube_cal, hi_tube_cal, q_tube_rel, epsilon)
    lo_te_rel, hi_te_rel = _apply_relative(lo_tube_te, hi_tube_te, q_tube_rel, epsilon)
    method_rows.append(method_record(
        "Tubeloss", "TCQR-r",
        lo_cal_rel, hi_cal_rel, lo_te_rel, hi_te_rel,
        y_calib, y_test,
        q_hat=q_tube_rel,
    ))

    q_qr_rel = _relative_qhat(lo_qr_cal, hi_qr_cal, y_calib, cfg.T, epsilon)
    lo_cal_rel, hi_cal_rel = _apply_relative(lo_qr_cal, hi_qr_cal, q_qr_rel, epsilon)
    lo_te_rel, hi_te_rel = _apply_relative(lo_qr_te, hi_qr_te, q_qr_rel, epsilon)
    method_rows.append(method_record(
        "Quantile", "CQR-r",
        lo_cal_rel, hi_cal_rel, lo_te_rel, hi_te_rel,
        y_calib, y_test,
        q_hat=q_qr_rel,
    ))

    return method_rows
