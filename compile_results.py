"""
Compile TubeLoss-vs-CQR benchmark results into calibration-set comparison tables,
print TubeLoss delta-at-target summaries, and save side-by-side delta plots.

What this script does
---------------------
1. Prints a plain side-by-side calibration table for every dataset:

       Tubeloss: method_name, PICP, MPIW, q_hat
       Quantile: method_name, PICP, MPIW, q_hat

2. For every dataset, prints the TubeLoss delta whose calibration PICP is closest
   to the target coverage, default 0.90. If an exact value exists within the
   tolerance, it prints the exact delta. If the curve crosses the target between
   two delta-grid points, it also prints a linear interpolation estimate.

3. Saves one side-by-side plot per dataset:

       left panel  : delta vs TubeLoss calibration PICP
       right panel : delta vs TubeLoss calibration MPIW

Expected input
--------------
Run experiments first so results.json exists, for example:

    python run_experiment.py --benchmarks
    python compile_results.py

Outputs
-------
compiled_results/
    method_rows.csv
    side_by_side.csv
    tube_delta_sweep.csv
    calib_side_by_side_comparison_table.csv
    calib_side_by_side_comparison_table_flat.csv
    calib_side_by_side_comparison_table.html
    calib_side_by_side_printed_tables.txt
    delta_target_summary.csv
    delta_target_summary.txt
    method_summary_calib.csv
    plots/
        delta_side_by_side/<dataset>_delta_picp_mpiw_side_by_side.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import math
from typing import Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASET_ORDER = ["boston", "concrete", "energy", "protein", "wine", "synthetic"]
METHOD_PAIRS = [
    ("Tube Loss", "Pinball Loss"),
    ("TCQR", "CQR"),
    ("TCQR-r", "CQR-r"),
    ("UATCQR-S", "UACQR-S"),
    ("UATCQR-P", "UACQR-P"),
]


# ── loading / type helpers ──────────────────────────────────────────────────

def to_number(x: Any) -> Any:
    """Convert JSON-safe non-finite strings back to numeric values when possible."""
    if x is None:
        return np.nan
    if isinstance(x, str):
        xl = x.lower()
        if xl in {"inf", "+inf", "infinity", "+infinity"}:
            return np.inf
        if xl in {"-inf", "-infinity"}:
            return -np.inf
        try:
            return float(x)
        except ValueError:
            return x
    return x


def clean_loaded_row(row: dict) -> dict:
    return {k: to_number(v) for k, v in row.items()}


def dataset_sort_key(name: str):
    try:
        return (0, DATASET_ORDER.index(name))
    except ValueError:
        return (1, name)


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns for a normal one-header CSV export."""
    out = df.copy()
    flat_cols = []
    for col in out.columns:
        if isinstance(col, tuple):
            left, right = col
            if left == "Dataset":
                flat_cols.append("dataset")
            else:
                flat_cols.append(f"{left.lower()}_{right}")
        else:
            flat_cols.append(str(col))
    out.columns = flat_cols
    return out


def _safe_float(value: Any, default: float = np.nan) -> float:
    try:
        value = to_number(value)
        if isinstance(value, str):
            return default
        return float(value)
    except Exception:
        return default


# ── calibration side-by-side table construction ─────────────────────────────

def build_calib_side_by_side_from_side_records(result: dict, dataset: str) -> list[dict]:
    """Use the precomputed side_by_side rows saved by run_experiment.py."""
    rows = []
    for row in result.get("side_by_side", []):
        row = clean_loaded_row(row)
        rows.append({
            ("Dataset", "name"): dataset,
            ("Tubeloss", "method_name"): row.get("tube_method"),
            ("Tubeloss", "PICP"): row.get("tube_cal_picp"),
            ("Tubeloss", "MPIW"): row.get("tube_cal_mpiw"),
            ("Tubeloss", "q_hat"): row.get("tube_q_hat"),
            ("Quantile", "method_name"): row.get("quantile_method"),
            ("Quantile", "PICP"): row.get("quantile_cal_picp"),
            ("Quantile", "MPIW"): row.get("quantile_cal_mpiw"),
            ("Quantile", "q_hat"): row.get("quantile_q_hat"),
        })
    return rows


def build_calib_side_by_side_from_method_rows(result: dict, dataset: str) -> list[dict]:
    """
    Fallback for results.json files that have method_rows but do not have
    side_by_side. This still uses calibration metrics.
    """
    method_rows = [clean_loaded_row(r) for r in result.get("method_rows", [])]
    by_key = {(r.get("family"), r.get("method_name")): r for r in method_rows}

    rows = []
    for tube_method, quant_method in METHOD_PAIRS:
        tube = by_key.get(("Tubeloss", tube_method))
        quant = by_key.get(("Quantile", quant_method))
        if tube is None or quant is None:
            continue
        rows.append({
            ("Dataset", "name"): dataset,
            ("Tubeloss", "method_name"): tube_method,
            ("Tubeloss", "PICP"): tube.get("cal_picp"),
            ("Tubeloss", "MPIW"): tube.get("cal_mpiw"),
            ("Tubeloss", "q_hat"): tube.get("q_hat"),
            ("Quantile", "method_name"): quant_method,
            ("Quantile", "PICP"): quant.get("cal_picp"),
            ("Quantile", "MPIW"): quant.get("cal_mpiw"),
            ("Quantile", "q_hat"): quant.get("q_hat"),
        })
    return rows


def build_calib_comparison_table(all_results: dict, round_digits: int = 4) -> pd.DataFrame:
    """
    Build a comparison.py-style MultiIndex table for every dataset in results.json.

    Values are calibration-set values:
      - PICP  = cal_picp / tube_cal_picp / quantile_cal_picp
      - MPIW  = cal_mpiw / tube_cal_mpiw / quantile_cal_mpiw
      - q_hat = conformal correction or uncertainty-aware t_hat
    """
    records = []
    for dataset in sorted(all_results.keys(), key=dataset_sort_key):
        result = all_results[dataset]
        if result.get("side_by_side"):
            records.extend(build_calib_side_by_side_from_side_records(result, dataset))
        else:
            records.extend(build_calib_side_by_side_from_method_rows(result, dataset))

    columns = pd.MultiIndex.from_tuples([
        ("Dataset", "name"),
        ("Tubeloss", "method_name"),
        ("Tubeloss", "PICP"),
        ("Tubeloss", "MPIW"),
        ("Tubeloss", "q_hat"),
        ("Quantile", "method_name"),
        ("Quantile", "PICP"),
        ("Quantile", "MPIW"),
        ("Quantile", "q_hat"),
    ])

    df = pd.DataFrame(records)
    if df.empty:
        return pd.DataFrame(columns=columns)

    df = df.reindex(columns=columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].round(round_digits)
    return df


# ── plain side-by-side printer ───────────────────────────────────────────────

def _is_missing(value) -> bool:
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _fmt_value(value, digits: int = 4) -> str:
    """Format table cells to look like comparison.py output but as plain text."""
    if _is_missing(value):
        return "NaN"
    if isinstance(value, (np.integer, int)) and not isinstance(value, bool):
        return f"{float(value):.{digits}f}"
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return f"{value:.{digits}f}"
    return str(value)


def _center(text: str, width: int) -> str:
    return str(text).center(width)


def _left(text: str, width: int) -> str:
    return str(text).ljust(width)


def _right(text: str, width: int) -> str:
    return str(text).rjust(width)


def format_side_by_side_table_for_dataset(
    calib_table: pd.DataFrame,
    dataset: str,
    *,
    digits: int = 4,
    include_title: bool = True,
) -> str:
    """
    Return a plain-text table for one dataset. This is intentionally a string
    printer, not a DataFrame display.
    """
    dataset_col = ("Dataset", "name")
    if dataset_col in calib_table.columns:
        df = calib_table[calib_table[dataset_col] == dataset].copy()
    else:
        df = calib_table.copy()

    if df.empty:
        return f"Dataset: {dataset}\n  No rows found."

    df = df.reset_index(drop=True)

    specs = [
        (("Tubeloss", "method_name"), "method_name", "left"),
        (("Tubeloss", "PICP"), "PICP", "right"),
        (("Tubeloss", "MPIW"), "MPIW", "right"),
        (("Tubeloss", "q_hat"), "q_hat", "right"),
        (("Quantile", "method_name"), "method_name", "left"),
        (("Quantile", "PICP"), "PICP", "right"),
        (("Quantile", "MPIW"), "MPIW", "right"),
        (("Quantile", "q_hat"), "q_hat", "right"),
    ]

    cell_columns: list[list[str]] = []
    widths: list[int] = []
    for key, header, _align in specs:
        values = [_fmt_value(v, digits=digits) for v in df[key].tolist()]
        width = max(len(header), *(len(v) for v in values))
        if header == "method_name":
            width = max(width, 12)
        widths.append(width)
        cell_columns.append(values)

    idx_width = max(len(str(len(df) - 1)), 1)
    gap = "   "
    inner_gap = "  "

    tube_width = sum(widths[:4]) + len(inner_gap) * 3
    quant_width = sum(widths[4:]) + len(inner_gap) * 3

    lines = []
    if include_title:
        title = f"Dataset: {dataset}"
        lines.append(title)
        lines.append("=" * len(title))

    lines.append(
        " " * (idx_width + len(gap))
        + _center("Tubeloss", tube_width)
        + gap
        + _center("Quantile", quant_width)
    )

    header_parts = []
    for (_, header, align), width in zip(specs, widths):
        header_parts.append(_left(header, width) if align == "left" else _right(header, width))
    lines.append(
        " " * (idx_width + len(gap))
        + inner_gap.join(header_parts[:4])
        + gap
        + inner_gap.join(header_parts[4:])
    )

    for i in range(len(df)):
        row_parts = []
        for col_values, (_, _, align), width in zip(cell_columns, specs, widths):
            value = col_values[i]
            row_parts.append(_left(value, width) if align == "left" else _right(value, width))
        lines.append(
            _right(str(i), idx_width)
            + gap
            + inner_gap.join(row_parts[:4])
            + gap
            + inner_gap.join(row_parts[4:])
        )

    return "\n".join(lines)


# ── delta sweep helpers ─────────────────────────────────────────────────────

def extract_delta_rows(result: dict) -> list[dict]:
    """Normalize either new tube_delta_sweep rows or legacy tube_rows."""
    if result.get("tube_delta_sweep"):
        rows = []
        for row in result["tube_delta_sweep"]:
            row = clean_loaded_row(row)
            rows.append({
                "delta": row.get("delta"),
                "cal_picp": row.get("raw_cal_picp"),
                "cal_mpiw": row.get("raw_cal_mpiw"),
                "conf_picp": row.get("additive_test_picp"),
                "conf_mpiw": row.get("additive_test_mpiw"),
            })
        return rows

    rows = []
    for row in result.get("tube_rows", []):
        row = clean_loaded_row(row)
        rows.append({
            "delta": row.get("delta"),
            "cal_picp": row.get("cal_picp"),
            "cal_mpiw": row.get("cal_mpiw"),
            "conf_picp": row.get("conf_picp"),
            "conf_mpiw": row.get("conf_mpiw"),
        })
    return rows


def delta_frame_for_result(result: dict) -> pd.DataFrame:
    rows = extract_delta_rows(result)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["delta", "cal_picp", "cal_mpiw", "conf_picp", "conf_mpiw"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["delta", "cal_picp", "cal_mpiw"]).sort_values("delta").reset_index(drop=True)
    return df


def _interpolated_crossing_delta(df: pd.DataFrame, target: float) -> float:
    """
    Return a linear interpolation estimate for delta where cal_picp hits target.
    Returns NaN if the curve does not cross the target between grid points.
    """
    if df.empty or len(df) < 2:
        return np.nan

    delta = df["delta"].to_numpy(dtype=float)
    picp = df["cal_picp"].to_numpy(dtype=float)

    crossings = []
    for i in range(len(df) - 1):
        p0, p1 = picp[i], picp[i + 1]
        d0, d1 = delta[i], delta[i + 1]
        if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(d0) and np.isfinite(d1)):
            continue
        if p0 == target:
            crossings.append(d0)
            continue
        if p1 == target:
            crossings.append(d1)
            continue
        if (p0 - target) * (p1 - target) < 0 and p1 != p0:
            crossings.append(d0 + (target - p0) * (d1 - d0) / (p1 - p0))

    if not crossings:
        return np.nan

    # If there are multiple crossings, return the one closest to the closest-grid delta.
    closest_idx = int(np.nanargmin(np.abs(picp - target)))
    closest_delta = delta[closest_idx]
    return float(min(crossings, key=lambda x: abs(x - closest_delta)))


def find_delta_for_tubeloss_picp(
    df: pd.DataFrame,
    *,
    target: float = 0.90,
    tolerance: float = 5e-4,
) -> dict:
    """Find exact/closest/interpolated delta for TubeLoss calibration PICP target."""
    if df.empty:
        return {
            "delta_exact": np.nan,
            "picp_exact": np.nan,
            "delta_closest": np.nan,
            "picp_closest": np.nan,
            "picp_abs_error": np.nan,
            "delta_interpolated": np.nan,
            "used_delta": np.nan,
            "used_kind": "missing",
            "target_picp": target,
            "tolerance": tolerance,
        }

    diffs = (df["cal_picp"] - target).abs()
    closest_idx = int(diffs.idxmin())
    closest = df.loc[closest_idx]

    exact_df = df[diffs <= tolerance]
    if not exact_df.empty:
        # If several points are exact within tolerance, pick the narrowest interval.
        exact_row = exact_df.sort_values(["cal_mpiw", "delta"]).iloc[0]
        delta_exact = float(exact_row["delta"])
        picp_exact = float(exact_row["cal_picp"])
    else:
        delta_exact = np.nan
        picp_exact = np.nan

    delta_interpolated = _interpolated_crossing_delta(df, target)

    if np.isfinite(delta_exact):
        used_delta = delta_exact
        used_kind = "exact"
    elif np.isfinite(delta_interpolated):
        used_delta = float(delta_interpolated)
        used_kind = "interpolated"
    else:
        used_delta = float(closest["delta"])
        used_kind = "closest"

    return {
        "delta_exact": delta_exact,
        "picp_exact": picp_exact,
        "delta_closest": float(closest["delta"]),
        "picp_closest": float(closest["cal_picp"]),
        "picp_abs_error": float(abs(float(closest["cal_picp"]) - target)),
        "delta_interpolated": float(delta_interpolated) if np.isfinite(delta_interpolated) else np.nan,
        "used_delta": used_delta,
        "used_kind": used_kind,
        "target_picp": target,
        "tolerance": tolerance,
    }


def format_delta_summary_for_dataset(dataset: str, summary: dict, digits: int = 4) -> str:
    target = summary["target_picp"]
    tol = summary["tolerance"]
    lines = [f"TubeLoss delta for calibration PICP target — {dataset}"]
    lines.append(f"  target PICP        : {target:.{digits}f}")
    lines.append(f"  reported delta     : {_fmt_value(summary['used_delta'], digits)}  ({summary['used_kind']})")

    if np.isfinite(_safe_float(summary["delta_exact"])):
        lines.append(
            f"  exact grid delta   : {summary['delta_exact']:.{digits}f} "
            f"(PICP={summary['picp_exact']:.{digits}f}, tol=±{tol:g})"
        )
    else:
        lines.append(f"  exact grid delta   : none within ±{tol:g}")

    if np.isfinite(_safe_float(summary["delta_interpolated"])):
        lines.append(f"  interpolated delta : {summary['delta_interpolated']:.{digits}f}")
    else:
        lines.append("  interpolated delta : no crossing found")

    lines.append(
        f"  closest grid delta : {summary['delta_closest']:.{digits}f} "
        f"(PICP={summary['picp_closest']:.{digits}f}, |diff|={summary['picp_abs_error']:.{digits}f})"
    )
    return "\n".join(lines)


def build_delta_target_summaries(
    all_results: dict,
    *,
    target: float = 0.90,
    tolerance: float = 5e-4,
) -> tuple[pd.DataFrame, str]:
    records = []
    text_blocks = []

    for dataset in sorted(all_results.keys(), key=dataset_sort_key):
        result = all_results[dataset]
        cfg_target = _safe_float(result.get("config", {}).get("t"), default=target)
        target_used = float(target)

        df_delta = delta_frame_for_result(result)
        summary = find_delta_for_tubeloss_picp(
            df_delta,
            target=target_used,
            tolerance=tolerance,
        )
        record = {"dataset": dataset, **summary}
        records.append(record)
        text_blocks.append(format_delta_summary_for_dataset(dataset, summary))

    return pd.DataFrame(records), "\n\n".join(text_blocks)


# ── plotting ────────────────────────────────────────────────────────────────

def _method_calibration_lines(result: dict) -> dict[str, float]:
    """Return reference calibration PICP/MPIW for Pinball and CQR if available."""
    refs: dict[str, float] = {}
    for row in result.get("method_rows", []):
        row = clean_loaded_row(row)
        family = row.get("family")
        method = row.get("method_name")
        if family == "Quantile" and method == "Pinball Loss":
            refs["pinball_cal_picp"] = _safe_float(row.get("cal_picp"))
            refs["pinball_cal_mpiw"] = _safe_float(row.get("cal_mpiw"))
        if family == "Quantile" and method == "CQR":
            refs["cqr_cal_picp"] = _safe_float(row.get("cal_picp"))
            refs["cqr_cal_mpiw"] = _safe_float(row.get("cal_mpiw"))

    # Legacy fallback.
    cqr_row = clean_loaded_row(result.get("cqr_row", {}))
    refs.setdefault("cqr_cal_picp", _safe_float(cqr_row.get("cal_picp")))
    refs.setdefault("cqr_cal_mpiw", _safe_float(cqr_row.get("cal_mpiw")))
    return refs


def save_delta_side_by_side_plots(
    all_results: dict,
    plots_dir: Path,
    delta_summary_df: pd.DataFrame,
    *,
    target: float = 0.90,
) -> list[Path]:
    """Save one side-by-side delta-vs-PICP/MPIW plot per dataset."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    summary_by_dataset = {
        row.dataset: row._asdict()
        for row in delta_summary_df.itertuples(index=False)
    } if not delta_summary_df.empty else {}

    for dataset in sorted(all_results.keys(), key=dataset_sort_key):
        result = all_results[dataset]
        df_delta = delta_frame_for_result(result)
        if df_delta.empty:
            continue

        cfg = clean_loaded_row(result.get("config", {}))
        target_used = float(target)
        refs = _method_calibration_lines(result)
        summary = summary_by_dataset.get(dataset, {})
        used_delta = _safe_float(summary.get("used_delta"))
        used_kind = summary.get("used_kind", "target")
        opt_delta = _safe_float(cfg.get("opt_delta", result.get("optimal_delta", {}).get("delta")))

        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharex=False)
        fig.suptitle(f"TubeLoss delta sweep — {dataset.upper()}", fontsize=14, fontweight="bold")

        # Left panel: delta vs calibration PICP.
        ax = axes[0]
        ax.plot(df_delta["delta"], df_delta["cal_picp"], marker="o", linewidth=1.8, label="TubeLoss Cal PICP")
        ax.axhline(target_used, linestyle="--", linewidth=1.5, label=f"Target PICP = {target_used:.3f}")
        if np.isfinite(refs.get("pinball_cal_picp", np.nan)):
            ax.axhline(refs["pinball_cal_picp"], linestyle="-.", linewidth=1.2, label=f"Pinball Cal PICP = {refs['pinball_cal_picp']:.3f}")
        if np.isfinite(refs.get("cqr_cal_picp", np.nan)):
            ax.axhline(refs["cqr_cal_picp"], linestyle=":", linewidth=1.5, label=f"CQR Cal PICP = {refs['cqr_cal_picp']:.3f}")
        if np.isfinite(used_delta):
            ax.axvline(used_delta, linestyle="--", linewidth=1.3, label=f"δ for target = {used_delta:.4f} ({used_kind})")
        if np.isfinite(opt_delta) and (not np.isfinite(used_delta) or abs(opt_delta - used_delta) > 1e-12):
            ax.axvline(opt_delta, linestyle=":", linewidth=1.2, label=f"Optimal δ = {opt_delta:.4f}")
        ax.set_title("δ vs Calibration PICP")
        ax.set_xlabel("δ  (width penalty)")
        ax.set_ylabel("Calibration PICP")
        y_min = max(0.0, float(np.nanmin(df_delta["cal_picp"])) - 0.02)
        y_max = min(1.05, max(1.0, float(np.nanmax(df_delta["cal_picp"])) + 0.02))
        ax.set_ylim(y_min, y_max)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        # Right panel: delta vs calibration MPIW.
        ax = axes[1]
        ax.plot(df_delta["delta"], df_delta["cal_mpiw"], marker="o", linewidth=1.8, label="TubeLoss Cal MPIW")
        if np.isfinite(refs.get("pinball_cal_mpiw", np.nan)):
            ax.axhline(refs["pinball_cal_mpiw"], linestyle="-.", linewidth=1.2, label=f"Pinball Cal MPIW = {refs['pinball_cal_mpiw']:.3f}")
        if np.isfinite(refs.get("cqr_cal_mpiw", np.nan)):
            ax.axhline(refs["cqr_cal_mpiw"], linestyle=":", linewidth=1.5, label=f"CQR Cal MPIW = {refs['cqr_cal_mpiw']:.3f}")
        if np.isfinite(used_delta):
            ax.axvline(used_delta, linestyle="--", linewidth=1.3, label=f"δ for target = {used_delta:.4f} ({used_kind})")
        if np.isfinite(opt_delta) and (not np.isfinite(used_delta) or abs(opt_delta - used_delta) > 1e-12):
            ax.axvline(opt_delta, linestyle=":", linewidth=1.2, label=f"Optimal δ = {opt_delta:.4f}")
        ax.set_title("δ vs Calibration MPIW")
        ax.set_xlabel("δ  (width penalty)")
        ax.set_ylabel("Calibration MPIW")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = plots_dir / f"{dataset}_delta_picp_mpiw_side_by_side.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    return saved


# ── main compile flow ───────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compile TubeLoss-vs-CQR results into calib tables and side-by-side delta plots.")
    parser.add_argument("--results", default="results.json", help="Path to results.json")
    parser.add_argument("--out", default="compiled_results", help="Output directory")
    parser.add_argument("--target", type=float, default=0.90, help="Target TubeLoss calibration PICP, default 0.90")
    parser.add_argument("--tolerance", type=float, default=5e-4, help="Tolerance for exact PICP match on the delta grid")
    parser.add_argument("--digits", type=int, default=4, help="Printed decimal places")
    args = parser.parse_args()

    results_file = Path(args.results)
    output_dir = Path(args.out)
    plots_dir = output_dir / "plots" / "delta_side_by_side"
    output_dir.mkdir(exist_ok=True, parents=True)
    plots_dir.mkdir(exist_ok=True, parents=True)

    if not results_file.exists():
        raise FileNotFoundError(f"{results_file} not found. Run run_experiment.py first.")

    with open(results_file, "r") as f:
        all_results = json.load(f)

    method_records = []
    side_records = []
    sweep_records = []

    for dataset in sorted(all_results.keys(), key=dataset_sort_key):
        result = all_results[dataset]
        for row in result.get("method_rows", []):
            method_records.append({"dataset": dataset, **clean_loaded_row(row)})
        for row in result.get("side_by_side", []):
            side_records.append({"dataset": dataset, **clean_loaded_row(row)})

        df_delta = delta_frame_for_result(result)
        if not df_delta.empty:
            for row in df_delta.to_dict(orient="records"):
                sweep_records.append({"dataset": dataset, **row})

    method_df = pd.DataFrame(method_records)
    side_df = pd.DataFrame(side_records)
    sweep_df = pd.DataFrame(sweep_records)

    method_df.to_csv(output_dir / "method_rows.csv", index=False)
    side_df.to_csv(output_dir / "side_by_side.csv", index=False)
    sweep_df.to_csv(output_dir / "tube_delta_sweep.csv", index=False)

    calib_table = build_calib_comparison_table(all_results)
    calib_table.to_csv(output_dir / "calib_side_by_side_comparison_table.csv", index=False)
    flatten_columns(calib_table).to_csv(output_dir / "calib_side_by_side_comparison_table_flat.csv", index=False)
    calib_table.to_html(output_dir / "calib_side_by_side_comparison_table.html", index=False, border=0)

    # Print plain side-by-side tables and delta target summaries by dataset.
    dataset_col = ("Dataset", "name")
    if dataset_col in calib_table.columns:
        datasets = list(dict.fromkeys(calib_table[dataset_col].tolist()))
    else:
        datasets = sorted(all_results.keys(), key=dataset_sort_key)

    delta_summary_df, delta_summary_text = build_delta_target_summaries(
        all_results,
        target=args.target,
        tolerance=args.tolerance,
    )
    delta_summary_df.to_csv(output_dir / "delta_target_summary.csv", index=False)
    with open(output_dir / "delta_target_summary.txt", "w") as f:
        f.write(delta_summary_text + "\n")

    table_blocks = []
    for dataset in datasets:
        table_blocks.append(format_delta_summary_for_dataset(
            dataset,
            delta_summary_df[delta_summary_df["dataset"] == dataset].iloc[0].to_dict()
            if not delta_summary_df[delta_summary_df["dataset"] == dataset].empty
            else find_delta_for_tubeloss_picp(pd.DataFrame(), target=args.target, tolerance=args.tolerance),
            digits=args.digits,
        ))
        table_blocks.append(format_side_by_side_table_for_dataset(calib_table, dataset, digits=args.digits))

    printed_output = "\n\n".join(table_blocks)
    with open(output_dir / "calib_side_by_side_printed_tables.txt", "w") as f:
        f.write(printed_output + "\n")

    if not method_df.empty:
        method_summary_calib = method_df.pivot_table(
            index=["dataset", "family", "method_name"],
            values=["cal_picp", "cal_mpiw", "q_hat"],
            aggfunc="first",
        ).reset_index()
        method_summary_calib.to_csv(output_dir / "method_summary_calib.csv", index=False)

    saved_plots = save_delta_side_by_side_plots(
        all_results,
        plots_dir,
        delta_summary_df,
        target=args.target,
    )

    print(f"Loaded datasets: {', '.join(sorted(all_results.keys(), key=dataset_sort_key))}")
    print(f"Wrote calibration CSV table to {output_dir / 'calib_side_by_side_comparison_table.csv'}")
    print(f"Wrote printed tables to {output_dir / 'calib_side_by_side_printed_tables.txt'}")
    print(f"Wrote TubeLoss delta target summary to {output_dir / 'delta_target_summary.txt'}")
    print(f"Wrote side-by-side delta plots to {plots_dir}")
    print()
    print(printed_output)
    print()
    print(f"method_rows: {method_df.shape}")
    print(f"side_by_side: {side_df.shape}")
    print(f"delta_sweep: {sweep_df.shape}")
    print(f"calib_table: {calib_table.shape}")
    print(f"side-by-side delta plots: {len(saved_plots)}")
    if saved_plots:
        print("Saved plot files:")
        for path in saved_plots:
            print(f"  {path}")


if __name__ == "__main__":
    main()
