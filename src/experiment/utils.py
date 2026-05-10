"""
Small shared utilities.
"""

from __future__ import annotations

import numpy as np


def cfg_get(cfg, name: str, default):
    return getattr(cfg, name, default)


def clean_value(x):
    """Make numpy scalars and non-finite values safe for JSON."""
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        x = float(x)
        if np.isnan(x):
            return None
        if np.isposinf(x):
            return "inf"
        if np.isneginf(x):
            return "-inf"
        return x
    if isinstance(x, dict):
        return {k: clean_value(v) for k, v in x.items()}
    if isinstance(x, list):
        return [clean_value(v) for v in x]
    return x
