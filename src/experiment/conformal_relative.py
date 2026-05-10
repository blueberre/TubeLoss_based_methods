"""
Relative-width conformal wrappers: TCQR-r and CQR-r.
"""

from __future__ import annotations

import numpy as np

from src.evaluation import conformal_quantile


def _relative_qhat(
    lo_cal: np.ndarray,
    hi_cal: np.ndarray,
    y_cal: np.ndarray,
    coverage: float,
    epsilon: float,
) -> float:
    width = np.maximum(hi_cal - lo_cal, epsilon)
    scores = np.maximum((lo_cal - y_cal) / width, (y_cal - hi_cal) / width)
    return float(conformal_quantile(scores, coverage))


def _apply_relative(
    lo: np.ndarray,
    hi: np.ndarray,
    q_hat: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray]:
    width = np.maximum(hi - lo, epsilon)
    return lo - q_hat * width, hi + q_hat * width
