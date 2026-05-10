"""
src/plotting.py
===============
PICP-vs-delta and MPIW-vs-delta plot functions.

The figure code is taken verbatim from the original notebooks.
The only structural change is that every value that was a notebook
global (t, opt_delta, CQR scalars, y_calib length) is now an explicit
argument, so the functions can be called from run_experiment.py
and from compile_results.ipynb without any shared state.

Both functions:
  - return the matplotlib Figure so the caller can embed it in a
    notebook or a subplot grid
  - optionally save a PNG when `save_path` is given

Functions
---------
plot_picp_vs_delta  : Graph 1 — Cal & Conf Test PICP vs δ.
plot_mpiw_vs_delta  : Graph 2 — Cal & Conf Test MPIW vs δ.
"""

from __future__ import annotations

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def _save_fig(fig: plt.Figure, save_path: str) -> None:
    fig.savefig(save_path, dpi=150, bbox_inches="tight")

def plot_picp_vs_delta(
    deltas:        np.ndarray,
    picp_cals:     np.ndarray,
    picp_confs:    np.ndarray,
    qr_cal_picp:   float,
    cqr_conf_picp: float,
    opt_delta:     float,
    n_calib:       int,
    *,
    t:             float = 0.90,
    dataset_name:  str   = "",
    save_path:     str | None = None,
) -> plt.Figure:
    """
    Graph 1 — PICP vs δ (Calibration & Test).

    Identical to the notebook figure; all styling taken verbatim from
    the original notebooks.

    Parameters
    ----------
    deltas        : (n_deltas,) delta grid used during training.
    picp_cals     : (n_deltas,) calibration PICP for each delta.
    picp_confs    : (n_deltas,) conformal test PICP for each delta.
    qr_cal_picp   : scalar  CQR calibration PICP.
    cqr_conf_picp : scalar  CQR conformal test PICP.
    opt_delta     : scalar  the chosen optimal delta.
    n_calib       : int     number of calibration points (for step-size label).
    t             : float   coverage target line (default 0.90).
    dataset_name  : str     appended to suptitle when given.
    save_path     : str     if given, figure is saved as PNG at this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    title_suffix = f"  —  {dataset_name.upper()}" if dataset_name else ""

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        f"Graph 1 — PICP vs δ  (Calibration & Test){title_suffix}",
        fontsize=13, fontweight="bold",
    )

    ax.plot(
        deltas, picp_cals,
        marker="o", lw=1.5, ms=5, color="steelblue",
        label=f"Tube Loss Cal PICP  (step = 1/{n_calib} ≈ {1/n_calib:.4f})",
    )
    ax.plot(
        deltas, picp_confs,
        marker="s", lw=1.5, ms=5, color="black",
        label="Tube Loss Conf Test PICP",
    )
    ax.axhline(t,              color="red",         ls="--", lw=2.0,
               label=f"Target t = {t}")
    ax.axhline(qr_cal_picp,   color="darkorange",  ls="-.", lw=1.8,
               label=f"CQR Cal PICP = {qr_cal_picp:.3f}")
    ax.axhline(cqr_conf_picp, color="saddlebrown", ls=":",  lw=1.8,
               label=f"CQR Conf Test PICP = {cqr_conf_picp:.3f}")
    ax.axvline(opt_delta,     color="green",        ls=":",  lw=1.5,
               label=f"Optimal δ = {opt_delta:.3f}")

    ax.fill_between(
        deltas, t, picp_cals.max() + 0.01,
        color="green", alpha=0.07, label="Over-coverage zone (Cal)",
    )
    ax.fill_between(
        deltas, picp_cals.min() - 0.01, t,
        color="salmon", alpha=0.07, label="Under-coverage zone",
    )

    ax.set_xlabel("δ  (width penalty)", fontsize=11)
    ax.set_ylabel("PICP (original scale)", fontsize=11)
    ax.set_title("Cal & Conf Test PICP vs δ", fontsize=9)
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)
    ax.set_xticks(deltas[::2])
    ax.set_ylim(
        max(0.0, min(picp_cals.min(), picp_confs.min()) - 0.02),
        min(1.0, max(picp_cals.max(), picp_confs.max()) + 0.02),
    )

    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig


def plot_mpiw_vs_delta(
    deltas:        np.ndarray,
    mpiw_cals:     np.ndarray,
    mpiw_confs:    np.ndarray,
    qr_cal_mpiw:   float,
    cqr_conf_mpiw: float,
    opt_delta:     float,
    *,
    dataset_name:  str       = "",
    save_path:     str | None = None,
) -> plt.Figure:
    """
    Graph 2 — MPIW vs δ (Calibration & Test).

    Identical to the notebook figure; all styling taken verbatim from
    the original notebooks.

    Parameters
    ----------
    deltas        : (n_deltas,) delta grid used during training.
    mpiw_cals     : (n_deltas,) calibration MPIW for each delta.
    mpiw_confs    : (n_deltas,) conformal test MPIW for each delta.
    qr_cal_mpiw   : scalar  CQR calibration MPIW.
    cqr_conf_mpiw : scalar  CQR conformal test MPIW.
    opt_delta     : scalar  the chosen optimal delta.
    dataset_name  : str     appended to suptitle when given.
    save_path     : str     if given, figure is saved as PNG at this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    title_suffix = f"  —  {dataset_name.upper()}" if dataset_name else ""

    fig, ax = plt.subplots(figsize=(13, 5))
    fig.suptitle(
        f"Graph 2 — MPIW vs δ  (Calibration & Test){title_suffix}",
        fontsize=13, fontweight="bold",
    )

    ax.plot(
        deltas, mpiw_cals,
        marker="o", lw=1.5, ms=5, color="seagreen",
        label="Tube Loss Cal MPIW",
    )
    ax.plot(
        deltas, mpiw_confs,
        marker="s", lw=1.5, ms=5, color="black",
        label="Tube Loss Conf Test MPIW",
    )
    ax.axhline(qr_cal_mpiw,   color="darkorange",  ls="-.", lw=1.8,
               label=f"CQR Cal MPIW = {qr_cal_mpiw:.3f}")
    ax.axhline(cqr_conf_mpiw, color="saddlebrown", ls=":",  lw=1.8,
               label=f"CQR Conf Test MPIW = {cqr_conf_mpiw:.3f}")
    ax.axvline(opt_delta,     color="green",        ls=":",  lw=1.5,
               label=f"Optimal δ = {opt_delta:.3f}")

    ax.set_xlabel("δ  (width penalty)", fontsize=11)
    ax.set_ylabel("MPIW (original scale)", fontsize=11)
    ax.set_title("Cal & Conf Test MPIW vs δ", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_xticks(deltas[::2])

    plt.tight_layout()

    if save_path:
        _save_fig(fig, save_path)

    return fig
