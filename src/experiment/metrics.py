"""
Shared interval metrics and method-row helpers.
"""

from __future__ import annotations

import numpy as np


def _metrics(lo: np.ndarray, hi: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    y = np.asarray(y).reshape(-1)
    lo = np.asarray(lo).reshape(-1)
    hi = np.asarray(hi).reshape(-1)
    return float(((y >= lo) & (y <= hi)).mean()), float((hi - lo).mean())


def method_record(
    family: str,
    method_name: str,
    cal_lo: np.ndarray,
    cal_hi: np.ndarray,
    te_lo: np.ndarray,
    te_hi: np.ndarray,
    y_calib: np.ndarray,
    y_test: np.ndarray,
    q_hat=None,
) -> dict:
    cal_picp, cal_mpiw = _metrics(cal_lo, cal_hi, y_calib)
    test_picp, test_mpiw = _metrics(te_lo, te_hi, y_test)
    return {
        "family": family,
        "method_name": method_name,
        "q_hat": q_hat,
        "cal_picp": cal_picp,
        "cal_mpiw": cal_mpiw,
        "test_picp": test_picp,
        "test_mpiw": test_mpiw,
    }


def lookup_method(method_rows: list[dict], family: str, method_name: str) -> dict:
    for row in method_rows:
        if row["family"] == family and row["method_name"] == method_name:
            return row
    raise KeyError((family, method_name))
