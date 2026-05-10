"""
CLI helpers for running TubeLoss-vs-CQR experiments.

Top-level runs now execute every dataset by default. The hidden --single/--dataset
path is kept only for run_experiment.py's internal per-dataset subprocess calls.
"""

from __future__ import annotations

import argparse


ALL_DATASETS = ["synthetic", "boston", "concrete", "energy", "protein", "wine", "power", "yacht"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TubeLoss-vs-CQR experiments on all configured datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-ensembles",
        action="store_true",
        help="Skip UATCQR-S/P and UACQR-S/P to make full runs faster.",
    )
    parser.add_argument(
        "--retrain-ensembles",
        action="store_true",
        help=(
            "Train separate UATCQR/UACQR ensembles from scratch. By default, "
            "the script reuses snapshots from the already-trained base TubeLoss "
            "and Quantile models so the uncertainty-aware methods share the same "
            "base model as Tube Loss / Pinball Loss."
        ),
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override cfg.EPOCHS for this run.",
    )
    parser.add_argument(
        "--ensemble-b",
        type=int,
        default=None,
        help="Override ENSEMBLE_B / number of snapshots or members.",
    )
    parser.add_argument(
        "--ensemble-epochs",
        type=int,
        default=None,
        help="Override ENSEMBLE_EPOCHS for bootstrap/deep_ensemble modes.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Stop immediately if one dataset fails.",
    )
    parser.add_argument(
        "--single",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def datasets_from_args(args: argparse.Namespace) -> list[str]:
    # Normal user-facing behavior: always run the full dataset suite.
    # Internal behavior: run_experiment.py calls itself with --single --dataset <name>
    # so each dataset gets its own isolated process without infinite recursion.
    if args.single:
        if not args.dataset:
            raise ValueError("Internal --single mode requires --dataset.")
        return [args.dataset]

    return list(ALL_DATASETS)
