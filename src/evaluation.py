"""
src/evaluation.py
"""

from __future__ import annotations

import numpy as np
import torch



@torch.no_grad()
def predict_tube(
    model,
    x_sc: np.ndarray,
    scaler_y,
) -> tuple[np.ndarray, np.ndarray]:
    lo_s, hi_s = model(torch.FloatTensor(x_sc))
    lo = scaler_y.inverse_transform(
        lo_s.numpy().reshape(-1, 1)
    ).ravel().astype(np.float32)
    hi = scaler_y.inverse_transform(
        hi_s.numpy().reshape(-1, 1)
    ).ravel().astype(np.float32)
    return lo, hi


@torch.no_grad()
def predict_qr(
    m_lo,
    m_hi,
    x_sc: np.ndarray,
    scaler_y,
) -> tuple[np.ndarray, np.ndarray]:
    lo = scaler_y.inverse_transform(
        m_lo(torch.FloatTensor(x_sc)).numpy().reshape(-1, 1)
    ).ravel().astype(np.float32)
    hi = scaler_y.inverse_transform(
        m_hi(torch.FloatTensor(x_sc)).numpy().reshape(-1, 1)
    ).ravel().astype(np.float32)
    return lo, hi


# ── nonconformity ─────────────────────────────────────────────────────────────

def nc_scores(
    lo: np.ndarray,
    hi: np.ndarray,
    y:  np.ndarray,
) -> np.ndarray:
    return np.maximum(lo - y, y - hi)


# ── metrics ───────────────────────────────────────────────────────────────────

def raw_metrics(
    lo: np.ndarray,
    hi: np.ndarray,
    y:  np.ndarray,
) -> tuple[float, float]:
    picp = float(((y >= lo) & (y <= hi)).mean())
    mpiw = float((hi - lo).mean())
    return picp, mpiw


def conformal_quantile(
    nc_calib: np.ndarray,
    t: float = 0.90,
) -> float:
    n = len(nc_calib)
    k = int(np.ceil((n + 1) * t))
    k = min(max(k, 1), n)
    return float(np.sort(nc_calib)[k - 1])


def conformal_metrics(
    lo_te:  np.ndarray,
    hi_te:  np.ndarray,
    Q_t:    float,
    y_te:   np.ndarray,
) -> tuple[float, float]:
    c_lo = lo_te - Q_t
    c_hi = hi_te + Q_t
    picp = float(((y_te >= c_lo) & (y_te <= c_hi)).mean())
    mpiw = float((c_hi - c_lo).mean())
    return picp, mpiw