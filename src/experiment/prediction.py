"""
Prediction adapters used by conformal wrappers.
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from src.evaluation import predict_qr
from src.training import predict_tube_ensemble_scaled, predict_qr_ensemble_scaled


def inverse_y_array(arr: np.ndarray, scaler_y: MinMaxScaler) -> np.ndarray:
    """Inverse-transform a possibly 1D/2D array, preserving +/-inf values."""
    arr = np.asarray(arr, dtype=float)
    out = arr.copy()
    flat = out.reshape(-1)
    finite = np.isfinite(flat)
    if finite.any():
        flat[finite] = scaler_y.inverse_transform(flat[finite].reshape(-1, 1)).reshape(-1)
    return out


def predict_tube_ensemble(
    ensemble,
    x_sc: np.ndarray,
    scaler_y: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    lo_s, hi_s = predict_tube_ensemble_scaled(ensemble, x_sc)
    return inverse_y_array(lo_s, scaler_y), inverse_y_array(hi_s, scaler_y)


def predict_qr_ensemble(
    ensemble,
    x_sc: np.ndarray,
    scaler_y: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    lo_s, hi_s = predict_qr_ensemble_scaled(ensemble, x_sc)
    return inverse_y_array(lo_s, scaler_y), inverse_y_array(hi_s, scaler_y)


def predict_qr_ordered(
    m_lo,
    m_hi,
    x_sc: np.ndarray,
    scaler_y: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """Base predict_qr plus defensive endpoint ordering against quantile crossing."""
    lo, hi = predict_qr(m_lo, m_hi, x_sc, scaler_y)
    return np.minimum(lo, hi), np.maximum(lo, hi)
