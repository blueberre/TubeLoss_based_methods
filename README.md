# TubeLoss Based Uncertainty Methods

This repository studies prediction interval methods for regression, with a focus on comparing Tube Loss based intervals against quantile regression and conformalized quantile regression baselines.

The main idea is simple: instead of only learning point predictions or fixed quantiles, the project trains models that directly produce lower and upper prediction bounds. These intervals are then evaluated before and after conformal calibration. The experiments are run on both a synthetic regression problem and several benchmark datasets.

The code is organized so that each method family lives in its own module, while `run_experiment.py` acts as the main runner for the full experiment suite.

## Project motivation

In many regression problems, the quality of a model is not only about how close the prediction is to the target. It is also important to know how uncertain the model is. A prediction interval gives a range of values that is expected to contain the true response with a chosen probability.

For example, a 90 percent prediction interval should contain the true value about 90 percent of the time, while still being as narrow as possible. This creates a natural tradeoff:

- Wide intervals usually give better coverage, but they are less informative.
- Narrow intervals are more useful, but they may miss the true value too often.

This repository compares several ways of learning and calibrating these intervals. The experiments focus on two main metrics:

- PICP, Prediction Interval Coverage Probability: the fraction of targets covered by the interval.
- MPIW, Mean Prediction Interval Width: the average width of the interval.

A good method should reach the target coverage while keeping the interval width reasonably small.

## Methods included

The repository compares Tube Loss based methods with quantile regression based methods.

| Family | Method | Description |
|---|---|---|
| TubeLoss | Tube Loss | Trains a neural network to directly output lower and upper interval bounds. |
| TubeLoss | TCQR | Applies additive conformal calibration to Tube Loss intervals. |
| TubeLoss | TCQR-r | Applies relative-width conformal calibration to Tube Loss intervals. |
| TubeLoss | UATCQR-S | Uses uncertainty-aware calibration based on snapshot-style interval uncertainty. |
| TubeLoss | UATCQR-P | Uses a percentile/rank style uncertainty-aware conformal wrapper. |
| Quantile | Pinball Loss | Trains lower and upper quantile models using pinball loss. |
| Quantile | CQR | Applies standard conformalized quantile regression calibration. |
| Quantile | CQR-r | Applies relative-width conformal calibration to quantile intervals. |
| Quantile | UACQR-S | Uses uncertainty-aware calibration based on quantile model uncertainty. |
| Quantile | UACQR-P | Uses a percentile/rank style uncertainty-aware conformal wrapper. |

The uncertainty-aware methods can reuse snapshots from the already trained base models. This makes the comparison cleaner because the uncertainty methods are built from the same base training process rather than from fully separate retraining runs.

## Repository structure

```text
.
├── configs/
│   ├── boston.py
│   ├── concrete.py
│   ├── energy.py
│   ├── power.py
│   ├── protein.py
│   ├── synthetic.py
│   ├── wine.py
│   └── yacht.py
│
├── datasets/
│   ├── boston.txt
│   ├── concrete.txt
│   ├── energy.txt
│   ├── power.txt
│   ├── protein.txt
│   ├── wine.txt
│   └── yacht.txt
│
├── src/
│   ├── losses.py
│   ├── models.py
│   ├── training.py
│   ├── evaluation.py
│   ├── plotting.py
│   └── experiment/
│       ├── cli.py
│       ├── config.py
│       ├── data.py
│       ├── metrics.py
│       ├── output.py
│       ├── prediction.py
│       ├── runner.py
│       ├── conformal_relative.py
│       ├── conformal_uncertainty.py
│       └── methods/
│           ├── base_conformal.py
│           ├── quantile.py
│           ├── tube_loss.py
│           └── uncertainty_wrappers.py
│
├── results/
├── compiled_results/
├── run_experiment.py
├── compile_results.py
├── show_results.ipynb
└── results.json
```

### Important files

`run_experiment.py`

Main entry point for running experiments. A normal run executes all configured datasets. Internally, each dataset is run in a separate subprocess so that model state, random state, and failures are isolated.

`compile_results.py`

Collects the saved experiment outputs and builds comparison tables, delta summaries, and plots. This is useful after running the full experiment suite.

`src/losses.py`

Contains the core loss functions:

- `tube_loss`
- `pinball_loss`

`src/models.py`

Defines the neural network models:

- `TubePINet`, which outputs lower and upper interval bounds.
- `QuantileNet`, which outputs one quantile at a time.

`src/training.py`

Contains the training utilities for Tube Loss, quantile regression, and ensemble or snapshot-based variants.

`src/experiment/methods/`

Contains the method-level experiment code. This folder is where the actual comparison logic is separated by method family.

`configs/`

Each dataset has its own configuration file. These files define the dataset path, split strategy, model hyperparameters, Tube Loss settings, delta grid, and reproducibility seed.

## Datasets

The current experiment suite includes:

- Synthetic
- Boston
- Concrete
- Energy
- Protein
- Wine
- Power
- Yacht

The synthetic dataset is generated inside the code. The benchmark datasets are loaded from the `datasets/` folder.

By default, benchmark datasets are expected to be numeric files where the target column is the last column. This can be changed in the dataset config file.

Example:

```python
DATA_FILE = "datasets/wine.txt"
DATA_MODE = "file"
TARGET_COL = -1
```

If the file has a header or a different delimiter, the loader supports optional config fields such as:

```python
DATA_DELIMITER = ","
HAS_HEADER = True
DATA_SKIPROWS = 1
TARGET_COL = -1
```

## Installation

Create a virtual environment first.

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install numpy pandas matplotlib scikit-learn torch jupyter
```

The code uses Python type annotations that are best supported in Python 3.10 or later.

## Running the experiments

To run the full experiment suite:

```bash
python run_experiment.py
```

This runs all configured datasets:

```text
synthetic, boston, concrete, energy, protein, wine, power, yacht
```

For a faster smoke test, skip the uncertainty-aware ensemble methods and reduce the number of epochs:

```bash
python run_experiment.py --no-ensembles --epochs 50
```

To stop the full run immediately if one dataset fails:

```bash
python run_experiment.py --strict
```

To change the number of snapshots or ensemble members:

```bash
python run_experiment.py --ensemble-b 25
```

To train separate uncertainty ensembles instead of reusing snapshots from the base models:

```bash
python run_experiment.py --retrain-ensembles
```

## Compiling results

After running the experiments, compile the results with:

```bash
python compile_results.py
```

This reads `results.json` and produces tables and plots in `compiled_results/`.

The script produces outputs such as:

```text
compiled_results/
├── method_rows.csv
├── side_by_side.csv
├── tube_delta_sweep.csv
├── calib_side_by_side_comparison_table.csv
├── calib_side_by_side_comparison_table_flat.csv
├── calib_side_by_side_comparison_table.html
├── calib_side_by_side_printed_tables.txt
├── delta_target_summary.csv
├── delta_target_summary.txt
├── method_summary_calib.csv
└── plots/
    ├── delta_side_by_side/
    ├── delta_sweep/
    └── method_calib/
```

The compiled tables are designed to make Tube Loss and quantile-based methods easy to compare side by side.

## Output format

Each dataset gets its own output folder:

```text
results/<dataset>/
├── metrics.json
└── plots/
    ├── picp_vs_delta.png
    └── mpiw_vs_delta.png
```

A combined `results.json` file is also written at the project root. This file is used by `compile_results.py`.

The saved metrics include:

- dataset name
- configuration values
- train, calibration, and test sizes
- selected optimal delta
- Tube Loss delta sweep results
- method-level calibration and test metrics
- side-by-side comparison rows

## How the experiment works

For each dataset, the runner performs the following steps.

1. Load the dataset configuration.
2. Load or generate the dataset.
3. Split the data into train, calibration, and test sets.
4. Scale features and targets using training data only.
5. Train Tube Loss models across a grid of delta values.
6. Select the Tube Loss delta based on calibration coverage and interval width.
7. Train lower and upper quantile regression models.
8. Evaluate raw intervals.
9. Apply additive conformal calibration.
10. Apply relative-width conformal calibration.
11. Optionally evaluate uncertainty-aware conformal wrappers.
12. Save plots, tables, and JSON metrics.

The key point is that the calibration set is separate from the training set. This separation is important because conformal methods use calibration errors to adjust the interval bounds.

## Delta selection for Tube Loss

Tube Loss uses a delta parameter that controls the width penalty during training. The experiment trains a model for each delta in the configured grid.

The selected delta follows this rule:

1. Find all deltas where raw calibration PICP is at least the target coverage.
2. Among those deltas, choose the one with the smallest raw calibration MPIW.
3. If no delta reaches the target coverage, choose the delta with calibration PICP closest to the target.

This keeps the selection rule simple and tied directly to the coverage-width tradeoff.

## Evaluation metrics

### PICP

PICP measures how often the true target falls inside the predicted interval.

```text
PICP = number of covered targets / total number of targets
```

Higher PICP means better coverage. However, a method can get high PICP by producing very wide intervals, so PICP should not be read alone.

### MPIW

MPIW measures the average width of the prediction intervals.

```text
MPIW = average(upper_bound - lower_bound)
```

Lower MPIW means narrower intervals. A useful method should keep MPIW small while still achieving the desired PICP.

### q_hat and t_hat

For conformal methods, `q_hat` is the calibration correction applied to the interval bounds. For some uncertainty-aware methods, the reported value may be a rank or scale parameter, shown as `t_hat` in the code and output tables.

## Adding a new dataset

To add a new dataset:

1. Place the dataset file in `datasets/`.
2. Add a new config file in `configs/`.
3. Follow the same structure as the existing config files.
4. Add the dataset name to `ALL_DATASETS` in `src/experiment/cli.py` if it should run by default.

Example config:

```python
import numpy as np

DATASET = "my_dataset"
DATA_FILE = "datasets/my_dataset.txt"
DATA_MODE = "file"

SPLIT_TMP = 0.40
SPLIT_TEST = 0.50

EPOCHS = 100
LR = 0.001
HID = 32
N_LAYERS = 1
WEIGHT_DECAY = 1e-3
BATCH_SIZE = "full"

T = 0.90
R = 0.50
DELTAS = [round(x, 3) for x in np.arange(-0.030, 0.031, 0.001)]

SEED = 42
```

## Adding a new method

To add a new method cleanly:

1. Create a new file under `src/experiment/methods/`.
2. Keep the method-specific logic inside that file.
3. Add prediction or calibration helpers only if they are shared across methods.
4. Wire the method into `src/experiment/runner.py`.
5. Add the method row using the shared `method_record` helper.
6. Update `compile_results.py` if the method should appear in the side-by-side summary table.

This keeps the repository easier to read and prevents `run_experiment.py` from becoming a large mixed-purpose script again.

## Reproducibility notes

The configs define a random seed for each run. The runner sets both NumPy and PyTorch seeds before training each dataset.

The code also uses separate subprocesses for multiple datasets. This makes the full run more stable because each dataset starts from a clean process.

Results can still vary slightly depending on the local Python version, PyTorch version, BLAS backend, and machine configuration.

## Interpreting the results

A useful comparison should look at both coverage and width.

A method is usually better when:

- its calibration and test PICP are close to the target coverage, and
- its MPIW is lower than other methods with similar coverage.

For Tube Loss, the delta sweep plots are especially important. They show how the interval behavior changes as the width penalty changes. This makes it easier to see whether Tube Loss is stable across the delta grid or sensitive to small parameter changes.

The compiled side-by-side tables are meant to answer practical questions such as:

- Does Tube Loss reach the desired coverage before conformal calibration?
- How much does conformal calibration widen the intervals?
- Is Tube Loss narrower or wider than CQR at similar coverage?
- Do uncertainty-aware wrappers improve the coverage-width balance?

## Current limitations

This repository is written as an experimental research codebase. It is useful for controlled comparisons, but it is not packaged as a production library.

Some limitations to keep in mind:

- The models are intentionally small neural networks.
- The current runner uses CPU-style PyTorch code and does not include explicit device management.
- Hyperparameters are manually defined in config files.
- The benchmark data loader expects mostly numeric tabular files.
- Results should be interpreted as empirical comparisons, not universal conclusions about one method always being better.

## Suggested workflow

A typical workflow is:

```bash
python run_experiment.py --no-ensembles --epochs 50
python compile_results.py
```

After confirming that the pipeline works, run the full experiment:

```bash
python run_experiment.py
python compile_results.py
```

Then review:

```text
results.json
compiled_results/calib_side_by_side_printed_tables.txt
compiled_results/delta_target_summary.txt
compiled_results/plots/delta_side_by_side/
```
