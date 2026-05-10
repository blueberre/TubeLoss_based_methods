"""
run_experiment.py
=================
Entry point for TubeLoss-vs-CQR experiments.

The experiment logic has been split into source modules under src/experiment/.
This file now only handles CLI routing and multi-dataset orchestration.
"""

from __future__ import annotations

import os
import subprocess
import sys

from src.experiment.cli import parse_args, datasets_from_args
from src.experiment.runner import run_single_dataset


# Make relative paths resolve from the folder that contains this file.
os.chdir(os.path.dirname(os.path.abspath(__file__)) or ".")


def main() -> None:
    args = parse_args()
    requested_datasets = datasets_from_args(args)

    # Multi-dataset mode runs this same script once per dataset. This avoids sharing
    # model state between datasets and keeps each dataset's failure isolated.
    if not args.single and len(requested_datasets) > 1:
        failures: list[tuple[str, int]] = []
        for ds in requested_datasets:
            print("\n" + "#" * 92)
            print(f"# Running dataset: {ds}")
            print("#" * 92 + "\n")

            cmd = [sys.executable, __file__, "--single", "--dataset", ds]
            if args.no_ensembles:
                cmd.append("--no-ensembles")
            if args.retrain_ensembles:
                cmd.append("--retrain-ensembles")
            if args.epochs is not None:
                cmd += ["--epochs", str(args.epochs)]
            if args.ensemble_b is not None:
                cmd += ["--ensemble-b", str(args.ensemble_b)]
            if args.ensemble_epochs is not None:
                cmd += ["--ensemble-epochs", str(args.ensemble_epochs)]

            rc = subprocess.call(cmd)
            if rc != 0:
                failures.append((ds, rc))
                print(f"\nDataset {ds!r} failed with exit code {rc}.")
                if args.strict:
                    sys.exit(rc)

        if failures:
            print("\nCompleted with failures:")
            for ds, rc in failures:
                print(f"  - {ds}: exit code {rc}")
            sys.exit(1 if args.strict else 0)

        print("\nAll requested datasets completed.")
        sys.exit(0)

    dataset_name = requested_datasets[0]
    run_single_dataset(dataset_name, args)


if __name__ == "__main__":
    main()
