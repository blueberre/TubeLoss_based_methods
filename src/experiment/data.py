"""
Dataset file loading, synthetic generation, and train/calib/test splitting.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.experiment.utils import cfg_get


def candidate_dataset_files(dataset_name: str, cfg) -> list[Path]:
    """Build a forgiving list of possible benchmark file paths."""
    candidates: list[Path] = []
    configured = getattr(cfg, "DATA_FILE", None)
    if configured:
        p = Path(configured)
        candidates.append(p)
        if not p.is_absolute():
            candidates.append(Path("datasets") / p.name)

    for ext in ["txt", "csv", "data", "dat", "tsv"]:
        candidates.append(Path("datasets") / f"{dataset_name}.{ext}")
        candidates.append(Path("data") / f"{dataset_name}.{ext}")

    out = []
    seen = set()
    for p in candidates:
        key = str(p)
        if key not in seen:
            out.append(p)
            seen.add(key)
    return out


def _read_numeric_table(path: Path, cfg) -> np.ndarray:
    """
    Read numeric benchmark data from .txt/.csv/.data/.tsv.

    Supported optional config fields:
      DATA_DELIMITER = None | "," | "\t" | "whitespace"
      DATA_SKIPROWS = 0
      HAS_HEADER = False
    """
    delimiter = cfg_get(cfg, "DATA_DELIMITER", None)
    skiprows = int(cfg_get(cfg, "DATA_SKIPROWS", 0))
    has_header = bool(cfg_get(cfg, "HAS_HEADER", False))

    if delimiter == "whitespace":
        delimiter = None

    try:
        arr = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        return arr
    except Exception:
        pass

    try:
        import pandas as pd

        if delimiter is None:
            df = pd.read_csv(
                path,
                sep=None,
                engine="python",
                header=0 if has_header else None,
                skiprows=skiprows,
            )
        else:
            df = pd.read_csv(
                path,
                sep=delimiter,
                header=0 if has_header else None,
                skiprows=skiprows,
            )

        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all").dropna(axis=0, how="any")
        arr = df.to_numpy(dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Expected at least 2 numeric columns in {path}")
        return arr
    except Exception as exc:
        raise ValueError(
            f"Could not read numeric benchmark data from {path}. "
            "Check DATA_DELIMITER, DATA_SKIPROWS, HAS_HEADER, and TARGET_COL in the config."
        ) from exc


def load_dataset_file(dataset_name: str, cfg) -> tuple[Path, np.ndarray]:
    for p in candidate_dataset_files(dataset_name, cfg):
        if p.exists():
            return p, _read_numeric_table(p, cfg)

    tried = "\n".join(f"  - {p}" for p in candidate_dataset_files(dataset_name, cfg))
    raise FileNotFoundError(
        f"Dataset file not found for {dataset_name!r}. Tried:\n{tried}\n"
        "Put the benchmark file in datasets/ or set DATA_FILE in configs/<dataset>.py."
    )


def split_features_target(data: np.ndarray, cfg) -> tuple[np.ndarray, np.ndarray]:
    """Split a loaded numeric table into X/y. Default target is the last column."""
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected a 2D array with at least 2 columns, got shape {data.shape}")

    target_col = cfg_get(cfg, "TARGET_COL", -1)
    if isinstance(target_col, str):
        raise ValueError("String TARGET_COL requires pandas column names; use an integer column index here.")

    target_col = int(target_col)
    if target_col < 0:
        target_col = data.shape[1] + target_col
    if not (0 <= target_col < data.shape[1]):
        raise ValueError(f"TARGET_COL={target_col} out of range for data shape {data.shape}")

    y = data[:, target_col].reshape(-1, 1).astype(np.float32)
    X = np.delete(data, target_col, axis=1).astype(np.float32)
    return X, y


def prepare_raw_splits(dataset_name: str, cfg):
    data_file = None

    if cfg.DATA_MODE == "file":
        data_file, data = load_dataset_file(dataset_name, cfg)
        X_raw, y_raw = split_features_target(data, cfg)

        print(f"Dataset : {dataset_name.upper()}  |  file = {data_file}  |  shape = {data.shape}")
        print(f"Features: {X_raw.shape[1]}  |  Target range: [{y_raw.min():.3f}, {y_raw.max():.3f}]")

        X_tr_raw, X_tmp, y_tr_raw, y_tmp = train_test_split(
            X_raw, y_raw, test_size=cfg.SPLIT_TMP, random_state=cfg.SEED
        )
        X_ca_raw, X_te_raw, y_ca_raw, y_te_raw = train_test_split(
            X_tmp, y_tmp, test_size=cfg.SPLIT_TEST, random_state=cfg.SEED
        )

    elif cfg.DATA_MODE == "synthetic":
        # y = sin(x)/x + ε, x ~ U(0,1), ε ~ N(0, SYNTH_NOISE)
        X_raw = np.random.uniform(0, 1, cfg.SYNTH_N).reshape(-1, 1).astype(np.float32)
        epsilon = np.random.normal(0, cfg.SYNTH_NOISE, cfg.SYNTH_N).reshape(-1, 1).astype(np.float32)
        y_raw = np.where(X_raw == 0, 1.0 + epsilon, np.sin(X_raw) / X_raw + epsilon).astype(np.float32)

        print(f"Dataset : SYNTHETIC  |  N = {cfg.SYNTH_N}")
        print(f"y range : [{y_raw.min():.3f}, {y_raw.max():.3f}]")

        X_tr_raw, X_tmp, y_tr_raw, y_tmp = train_test_split(
            X_raw, y_raw,
            train_size=cfg.SPLIT_TRAIN,
            random_state=cfg.SEED,
            shuffle=True,
        )
        X_ca_raw, X_te_raw, y_ca_raw, y_te_raw = train_test_split(
            X_tmp, y_tmp,
            test_size=0.50,
            random_state=cfg.SEED,
            shuffle=True,
        )

    else:
        raise ValueError(f"Unknown DATA_MODE: {cfg.DATA_MODE!r}  (must be 'file' or 'synthetic')")

    return data_file, X_tr_raw, X_ca_raw, X_te_raw, y_tr_raw, y_ca_raw, y_te_raw
