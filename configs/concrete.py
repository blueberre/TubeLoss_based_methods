"""configs/concrete.py — hyperparameters for the Concrete dataset."""
import numpy as np

DATASET      = "concrete"
DATA_FILE    = "datasets/concrete.txt"
DATA_MODE    = "file"           # "file" | "synthetic"

# ── splits ────────────────────────────────────────────────────────────────────
SPLIT_TMP    = 0.40             # 60% train, 40% tmp
SPLIT_TEST   = 0.50             # 50% of tmp → calib, 50% → test

# ── model / optimiser — !! FILL IN YOUR TUNED VALUES !! ─────────────────────
EPOCHS       = 50             # TODO: replace with your tuned value
LR           = 0.001            # TODO: replace with your tuned value
HID          = 64               # TODO: replace with your tuned value
N_LAYERS     = 2
WEIGHT_DECAY = 1e-3             # TODO: replace with your tuned value
BATCH_SIZE   = "full"           # "full" or an int

# ── tube loss ─────────────────────────────────────────────────────────────────
T            = 0.90
R            = 0.50
DELTAS       = [round(x, 3) for x in np.arange(-0.010, 0.031, 0.001)]  # TODO: adjust grid if needed

# ── reproducibility ───────────────────────────────────────────────────────────
SEED         = 42

