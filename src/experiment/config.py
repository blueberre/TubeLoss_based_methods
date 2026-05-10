"""
Config override helpers.
"""

from __future__ import annotations

import argparse


def apply_cli_overrides(cfg, args: argparse.Namespace) -> None:
    """Apply command-line overrides directly to the imported config module."""
    if args.epochs is not None:
        cfg.EPOCHS = int(args.epochs)
    if args.ensemble_b is not None:
        cfg.ENSEMBLE_B = int(args.ensemble_b)
    if args.ensemble_epochs is not None:
        cfg.ENSEMBLE_EPOCHS = int(args.ensemble_epochs)
    if args.no_ensembles:
        cfg.RUN_ENSEMBLES = False
    if args.retrain_ensembles:
        cfg.REUSE_BASE_ENSEMBLES = False
