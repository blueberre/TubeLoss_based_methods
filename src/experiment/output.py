"""
Final console output, plot saving, and metrics serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.plotting import plot_picp_vs_delta, plot_mpiw_vs_delta
from src.experiment.metrics import lookup_method
from src.experiment.utils import clean_value


def print_final_method_comparison(method_rows: list[dict]) -> None:
    print()
    print("Final method comparison on test set")
    print(f"{'Family':<10} {'Method':<14} {'Cal PICP':>9} {'Cal MPIW':>10} {'Test PICP':>10} {'Test MPIW':>10} {'q_hat/t_hat':>12}")
    print("-" * 83)
    for row in method_rows:
        q_display = "-" if row["q_hat"] is None else str(clean_value(row["q_hat"]))
        print(f"{row['family']:<10} {row['method_name']:<14} {row['cal_picp']:>9.4f} {row['cal_mpiw']:>10.4f} {row['test_picp']:>10.4f} {row['test_mpiw']:>10.4f} {q_display:>12}")
    print()


def build_side_by_side(method_rows: list[dict], run_ensembles: bool) -> list[dict]:
    # Side-by-side rows used by compile_results.ipynb.
    pairs = [
        ("Tube Loss", "Pinball Loss"),
        ("TCQR", "CQR"),
        ("TCQR-r", "CQR-r"),
    ]
    if run_ensembles:
        pairs += [
            ("UATCQR-S", "UACQR-S"),
            ("UATCQR-P", "UACQR-P"),
        ]

    side_by_side = []
    for tube_method, quant_method in pairs:
        tr = lookup_method(method_rows, "Tubeloss", tube_method)
        qr = lookup_method(method_rows, "Quantile", quant_method)
        side_by_side.append({
            "tube_method": tube_method,
            "tube_cal_picp": tr["cal_picp"],
            "tube_cal_mpiw": tr["cal_mpiw"],
            "tube_test_picp": tr["test_picp"],
            "tube_test_mpiw": tr["test_mpiw"],
            "tube_q_hat": tr["q_hat"],
            "quantile_method": quant_method,
            "quantile_cal_picp": qr["cal_picp"],
            "quantile_cal_mpiw": qr["cal_mpiw"],
            "quantile_test_picp": qr["test_picp"],
            "quantile_test_mpiw": qr["test_mpiw"],
            "quantile_q_hat": qr["q_hat"],
        })

    return side_by_side


def save_plots(
    *,
    dataset_name: str,
    cfg,
    results_dir: Path,
    deltas: np.ndarray,
    tube_results: list[dict],
    method_rows: list[dict],
    opt_delta: float,
    y_calib,
) -> Path:
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plot_picp_vs_delta(
        deltas,
        np.array([r["raw_cal_picp"] for r in tube_results]),
        np.array([r["additive_test_picp"] for r in tube_results]),
        lookup_method(method_rows, "Quantile", "Pinball Loss")["cal_picp"],
        lookup_method(method_rows, "Quantile", "CQR")["test_picp"],
        opt_delta,
        len(y_calib),
        t=cfg.T,
        dataset_name=dataset_name,
        save_path=str(plots_dir / "picp_vs_delta.png"),
    )
    plot_mpiw_vs_delta(
        deltas,
        np.array([r["raw_cal_mpiw"] for r in tube_results]),
        np.array([r["additive_test_mpiw"] for r in tube_results]),
        lookup_method(method_rows, "Quantile", "Pinball Loss")["cal_mpiw"],
        lookup_method(method_rows, "Quantile", "CQR")["test_mpiw"],
        opt_delta,
        dataset_name=dataset_name,
        save_path=str(plots_dir / "mpiw_vs_delta.png"),
    )
    plt.close("all")

    return plots_dir


def build_metrics(
    *,
    dataset_name: str,
    data_file,
    cfg,
    batch_size: int,
    opt_delta: float,
    best_i: int,
    best: dict,
    run_ensembles: bool,
    reuse_base_ensembles: bool,
    ensemble_B: int,
    ensemble_mode: str,
    ensemble_epochs: int,
    uacqr_base_mode,
    uatcqr_base_mode,
    epsilon: float,
    y_tr_sc,
    y_calib,
    y_test,
    tube_results: list[dict],
    method_rows: list[dict],
    side_by_side: list[dict],
) -> dict:
    tube_rows_legacy = [
        {
            "delta": r["delta"],
            "cal_picp": round(r["raw_cal_picp"], 6),
            "cal_mpiw": round(r["raw_cal_mpiw"], 6),
            "Q_t": round(r["additive_q_hat"], 6),
            "conf_picp": round(r["additive_test_picp"], 6),
            "conf_mpiw": round(r["additive_test_mpiw"], 6),
        }
        for r in tube_results
    ]

    cqr_legacy = lookup_method(method_rows, "Quantile", "CQR")

    metrics = {
        "dataset": dataset_name,
        "data_file": str(data_file) if data_file is not None else None,
        "config": {
            "epochs": int(cfg.EPOCHS),
            "lr": float(cfg.LR),
            "hid": int(cfg.HID),
            "n_layers": int(cfg.N_LAYERS),
            "weight_decay": float(cfg.WEIGHT_DECAY),
            "batch_size": int(batch_size),
            "t": float(cfg.T),
            "r": float(cfg.R),
            "seed": int(cfg.SEED),
            "opt_delta": float(opt_delta),
            "run_ensembles": bool(run_ensembles),
            "reuse_base_ensembles": bool(reuse_base_ensembles) if run_ensembles else False,
            "ensemble_B": int(ensemble_B) if run_ensembles else 0,
            "ensemble_mode": ("base_training_snapshots" if (run_ensembles and reuse_base_ensembles) else (ensemble_mode if run_ensembles else None)),
            "ensemble_epochs": int(ensemble_epochs) if run_ensembles else 0,
            "uacqr_base_mode": uacqr_base_mode if run_ensembles else None,
            "uatcqr_base_mode": uatcqr_base_mode if run_ensembles else None,
            "epsilon": float(epsilon),
        },
        "n_train": int(len(y_tr_sc)),
        "n_calib": int(len(y_calib)),
        "n_test": int(len(y_test)),
        "optimal_delta": {
            "delta": float(opt_delta),
            "best_index": int(best_i),
            "selection_rule": "raw cal PICP >= t, then narrowest raw cal MPIW; fallback closest raw cal PICP to t",
            "raw_cal_picp": best["raw_cal_picp"],
            "raw_cal_mpiw": best["raw_cal_mpiw"],
        },
        "tube_delta_sweep": [
            {
                "delta": r["delta"],
                "raw_cal_picp": r["raw_cal_picp"],
                "raw_cal_mpiw": r["raw_cal_mpiw"],
                "raw_test_picp": r["raw_test_picp"],
                "raw_test_mpiw": r["raw_test_mpiw"],
                "additive_q_hat": r["additive_q_hat"],
                "additive_test_picp": r["additive_test_picp"],
                "additive_test_mpiw": r["additive_test_mpiw"],
            }
            for r in tube_results
        ],
        "method_rows": method_rows,
        "side_by_side": side_by_side,
        # Backward-compatible keys for older compile notebooks.
        "tube_rows": tube_rows_legacy,
        "cqr_row": {
            "cal_picp": cqr_legacy["cal_picp"],
            "cal_mpiw": cqr_legacy["cal_mpiw"],
            "Q_t": cqr_legacy["q_hat"],
            "conf_picp": cqr_legacy["test_picp"],
            "conf_mpiw": cqr_legacy["test_mpiw"],
        },
    }

    return clean_value(metrics)


def save_metrics(dataset_name: str, metrics: dict, results_dir: Path) -> Path:
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    results_file = Path("results.json")
    if results_file.exists():
        with open(results_file, "r") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    all_results[dataset_name] = metrics
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)

    return results_file


def finalize_results(
    *,
    dataset_name: str,
    data_file,
    cfg,
    batch_size: int,
    opt_delta: float,
    best_i: int,
    best: dict,
    run_ensembles: bool,
    reuse_base_ensembles: bool,
    ensemble_B: int,
    ensemble_mode: str,
    ensemble_epochs: int,
    uacqr_base_mode,
    uatcqr_base_mode,
    epsilon: float,
    y_tr_sc,
    y_calib,
    y_test,
    deltas: np.ndarray,
    tube_results: list[dict],
    method_rows: list[dict],
) -> dict:
    print_final_method_comparison(method_rows)
    side_by_side = build_side_by_side(method_rows, run_ensembles)

    results_dir = Path("results") / dataset_name
    plots_dir = save_plots(
        dataset_name=dataset_name,
        cfg=cfg,
        results_dir=results_dir,
        deltas=deltas,
        tube_results=tube_results,
        method_rows=method_rows,
        opt_delta=opt_delta,
        y_calib=y_calib,
    )

    metrics = build_metrics(
        dataset_name=dataset_name,
        data_file=data_file,
        cfg=cfg,
        batch_size=batch_size,
        opt_delta=opt_delta,
        best_i=best_i,
        best=best,
        run_ensembles=run_ensembles,
        reuse_base_ensembles=reuse_base_ensembles,
        ensemble_B=ensemble_B,
        ensemble_mode=ensemble_mode,
        ensemble_epochs=ensemble_epochs,
        uacqr_base_mode=uacqr_base_mode,
        uatcqr_base_mode=uatcqr_base_mode,
        epsilon=epsilon,
        y_tr_sc=y_tr_sc,
        y_calib=y_calib,
        y_test=y_test,
        tube_results=tube_results,
        method_rows=method_rows,
        side_by_side=side_by_side,
    )

    results_file = save_metrics(dataset_name, metrics, results_dir)

    print(f"Per-dataset metrics saved → {results_dir / 'metrics.json'}")
    print(f"Combined results saved    → {results_file}")
    print(f"Plots saved               → {plots_dir}")
    print("Done.")

    return metrics
