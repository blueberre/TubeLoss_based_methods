"""configs/yacht.py — hyperparameters for the Boston dataset."""
import numpy as np

DATASET      = "yacht"
DATA_FILE    = "datasets/yacht.txt"
DATA_MODE    = "file"          # "file" | "synthetic"

# ── splits ────────────────────────────────────────────────────────────────────
# Fractions used in two successive train_test_splits (60 / 20 / 20).
SPLIT_TMP    = 0.40             # first split:  1-0.40 = 60% train, 40% tmp
SPLIT_TEST   = 0.50             # second split: 50% of tmp → calib, 50% → test

# ── model / optimiser ─────────────────────────────────────────────────────────
EPOCHS       = 100
LR           = 0.001
HID          = 32
N_LAYERS     = 1
WEIGHT_DECAY = 1e-3
BATCH_SIZE   = "full"          # "full" → loader uses len(x_tr); or set an int

# ── tube loss ─────────────────────────────────────────────────────────────────
T            = 0.90            # coverage target
R            = 0.50            # midpoint weight
DELTAS       = [round(x, 3) for x in np.arange(0.000, 0.031, 0.001)]

# ── reproducibility ───────────────────────────────────────────────────────────
SEED         = 42
