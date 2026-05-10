"""configs/synthetic.py — hyperparameters for the Synthetic dataset."""

DATASET      = "synthetic"
DATA_FILE    = None             # generated, not loaded from disk
DATA_MODE    = "synthetic"      # "file" | "synthetic"

# ── data generation ───────────────────────────────────────────────────────────
# y = sin(x)/x + eps,  x ~ U(0,1),  eps ~ N(0, 0.8),  N = 1500 total
SYNTH_N      = 1500
SYNTH_NOISE  = 0.8

# ── splits ────────────────────────────────────────────────────────────────────
# Absolute counts (not fractions) — 500 / 500 / 500
SPLIT_TRAIN  = 500
SPLIT_CALIB  = 500              # remainder after train is split 50/50 → calib/test
SPLIT_TEST   = 500

# ── model / optimiser ─────────────────────────────────────────────────────────
EPOCHS       = 250
LR           = 0.001
HID          = 32
N_LAYERS = 1
WEIGHT_DECAY = 1e-4             # note: 1e-4, not 1e-3 like the file datasets
BATCH_SIZE   = "full"           # "full" → 500 (= SPLIT_TRAIN)

# ── tube loss ─────────────────────────────────────────────────────────────────
T            = 0.90
R            = 0.50
DELTAS       = [round(i * 0.001, 3) for i in range(-20, 20)] 

# ── reproducibility ───────────────────────────────────────────────────────────
SEED         = 42
