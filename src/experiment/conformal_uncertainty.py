"""
Uncertainty-aware conformal wrappers: UACQR-S/P and UATCQR-S/P.
"""

from __future__ import annotations

import numpy as np


def _cutoff_with_upper_bound(scores: np.ndarray, coverage: float, upper_bound=None):
    """Split-conformal ceiling cutoff with optional finite upper bound."""
    scores = np.asarray(scores).reshape(-1)
    n = scores.size
    if n == 0:
        raise ValueError("Calibration scores cannot be empty.")

    k = int(np.ceil((n + 1) * coverage))
    if k > n:
        return np.inf if upper_bound is None else upper_bound
    return np.sort(scores)[k - 1]


def _base_and_scale(
    lo_ens: np.ndarray,
    hi_ens: np.ndarray,
    *,
    base_mode: str,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if base_mode == "last":
        lo_base = lo_ens[-1]
        hi_base = hi_ens[-1]
    elif base_mode == "mean":
        lo_base = lo_ens.mean(axis=0)
        hi_base = hi_ens.mean(axis=0)
    else:
        raise ValueError("base_mode must be 'last' or 'mean'")

    lo_base, hi_base = np.minimum(lo_base, hi_base), np.maximum(lo_base, hi_base)
    g_lo = np.maximum(lo_ens.std(axis=0), epsilon)
    g_hi = np.maximum(hi_ens.std(axis=0), epsilon)
    return lo_base, hi_base, g_lo, g_hi


def uacqrs_calibrate_apply(
    lo_ens_cal: np.ndarray,
    hi_ens_cal: np.ndarray,
    y_calib: np.ndarray,
    lo_ens_eval: np.ndarray,
    hi_ens_eval: np.ndarray,
    *,
    coverage: float,
    base_mode: str,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """UACQR-S / UATCQR-S style calibration and application."""
    y_calib = np.asarray(y_calib).reshape(-1)

    lo_base_cal, hi_base_cal, g_lo_cal, g_hi_cal = _base_and_scale(
        lo_ens_cal, hi_ens_cal, base_mode=base_mode, epsilon=epsilon
    )
    scores = np.maximum(
        (lo_base_cal - y_calib) / g_lo_cal,
        (y_calib - hi_base_cal) / g_hi_cal,
    )
    t_hat = float(_cutoff_with_upper_bound(scores, coverage))

    lo_base_eval, hi_base_eval, g_lo_eval, g_hi_eval = _base_and_scale(
        lo_ens_eval, hi_ens_eval, base_mode=base_mode, epsilon=epsilon
    )
    lo = lo_base_eval - t_hat * g_lo_eval
    hi = hi_base_eval + t_hat * g_hi_eval
    return np.minimum(lo, hi), np.maximum(lo, hi), t_hat


def uacqrp_calibrate_apply(
    lo_ens_cal: np.ndarray,
    hi_ens_cal: np.ndarray,
    y_calib: np.ndarray,
    lo_ens_eval: np.ndarray,
    hi_ens_eval: np.ndarray,
    *,
    coverage: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """UACQR-P / UATCQR-P style calibration and application."""
    y_calib = np.asarray(y_calib).reshape(-1)
    B = lo_ens_cal.shape[0]

    lo_sorted_cal = np.sort(lo_ens_cal, axis=0)
    hi_sorted_cal = np.sort(hi_ens_cal, axis=0)

    t_star = np.full(len(y_calib), B + 1, dtype=int)
    for t_idx in range(1, B + 1):
        lo_t = lo_sorted_cal[B - t_idx, :]
        hi_t = hi_sorted_cal[t_idx - 1, :]
        covered = (y_calib >= lo_t) & (y_calib <= hi_t)
        update_mask = (t_star == B + 1) & covered
        t_star[update_mask] = t_idx

    t_hat = int(_cutoff_with_upper_bound(t_star, coverage, upper_bound=B + 1))

    lo_sorted_eval = np.sort(lo_ens_eval, axis=0)
    hi_sorted_eval = np.sort(hi_ens_eval, axis=0)

    if t_hat <= 0:
        lo = np.full(lo_ens_eval.shape[1], np.inf)
        hi = np.full(lo_ens_eval.shape[1], -np.inf)
    elif t_hat >= B + 1:
        lo = np.full(lo_ens_eval.shape[1], -np.inf)
        hi = np.full(lo_ens_eval.shape[1], np.inf)
    else:
        lo = lo_sorted_eval[B - t_hat, :]
        hi = hi_sorted_eval[t_hat - 1, :]

    return np.minimum(lo, hi), np.maximum(lo, hi), t_hat
