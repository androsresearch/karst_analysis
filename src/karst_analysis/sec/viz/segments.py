"""Per-segment plots and the interactive breakpoint slider.

Adapted from ``legacy/modules/plots.py``. The interactive slider is
the workhorse of the breakpoint-evaluation notebook.
"""

from __future__ import annotations

from math import ceil
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import IntSlider, interact

from karst_analysis.sec.breakpoints.selection import get_global_metrics


def plot_segments(segments_info: dict, metrics: list[dict], title: str = ""):
    """Plot each segment with its data, fitted line, and metrics in subplots.

    Parameters
    ----------
    segments_info : dict
        Output of :func:`extract_segments`.
    metrics : list of dict
        Output of :func:`calculate_metrics_per_segment`.
    title : str
    """
    segments = segments_info["segments"]
    n_segments = len(segments)
    n_rows = ceil(np.sqrt(n_segments))
    n_cols = ceil(n_segments / n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10), squeeze=False)
    axes = axes.ravel()

    for i, segment in enumerate(segments):
        ax = axes[i]
        x = segment["data_x"]
        y = segment["data_y"]
        y_fit = segment["fitted_model"]["fitted_y"]

        ax.scatter(x, y, label="Data", alpha=0.7)
        ax.plot(x, y_fit, color="red", label="Fit")

        metric = next((m for m in metrics if m["Segment"] == segment["segment"]), None)
        if metric:
            metric_text = "\n".join([
                f"R²: {metric['R^2']:.3f}",
                f"RMS%: {metric['RMS%']:.3f}",
                f"RMS% (min-max): {metric['RMS% (min-max)']:.3f}",
            ])
            ax.text(0.05, 0.95, metric_text, transform=ax.transAxes,
                    fontsize=11, verticalalignment="top",
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.5))

        ax.set_title(f"Segment {segment['segment']}: {len(x)} points")
        ax.set_xlabel("Depth [m]")
        ax.set_ylabel("SEC")
        ax.legend()

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    if title:
        fig.suptitle(title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()


def interactive_segmented_regression(
    x: np.ndarray,
    y: np.ndarray,
    df: Any,
    title: str = "",
    breakpoints: int = 2,
):
    """Interactive slider to explore piecewise fits with 0–10 breakpoints.

    Parameters
    ----------
    x, y : np.ndarray
    df : Any
        DataFrame with ``n_breakpoints`` and ``estimates`` columns.
    title : str
    breakpoints : int
        Initial slider position.
    """
    def extract_estimate(param):
        return param.get("estimate", 0.0) if isinstance(param, dict) else param

    @interact(n_breakpoints=IntSlider(min=0, max=10, step=1, value=breakpoints))
    def update_plot(n_breakpoints: int = 0):
        row = df[df["n_breakpoints"] == n_breakpoints]
        if row.empty:
            print(f"No parameters available for {n_breakpoints} breakpoints.")
            return

        row = row.iloc[0]
        estimates = row["estimates"]

        c = extract_estimate(estimates["const"])
        alpha1 = extract_estimate(estimates["alpha1"])
        betas = [extract_estimate(estimates[f"beta{i}"]) for i in range(1, n_breakpoints + 1)]
        bps = [extract_estimate(estimates[f"breakpoint{i}"]) for i in range(1, n_breakpoints + 1)]

        x_sorted = np.sort(np.array(x))
        y_hat = []
        for xx in x_sorted:
            val = c + alpha1 * xx
            for b, bp in zip(betas, bps):
                if xx > bp:
                    val += b * (xx - bp)
            y_hat.append(val)

        y_pred = []
        for xx in x:
            val = c + alpha1 * xx
            for b, bp in zip(betas, bps):
                if xx > bp:
                    val += b * (xx - bp)
            y_pred.append(val)

        p = 2 + 2 * n_breakpoints
        rss, tss, r2, r2_adj = get_global_metrics(np.array(y), np.array(y_pred), p)

        plt.figure(figsize=(10, 6))
        plt.scatter(x, y, color="blue", alpha=0.6, label="Real Data")
        plt.plot(x_sorted, y_hat, color="darkorange", lw=3, label="Piecewise Fit")

        for i, bp in enumerate(bps, start=1):
            val_bp = c + alpha1 * bp
            for b, bp_j in zip(betas, bps):
                if bp > bp_j:
                    val_bp += b * (bp - bp_j)
            plt.scatter(bp, val_bp, color="limegreen", s=100, edgecolors="k", zorder=5)
            plt.annotate(
                str(i), (bp, val_bp), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=12, fontweight="bold",
                color="black",
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black",
                          facecolor="white", alpha=0.8),
            )

        plt.xlabel("Depth [m]")
        plt.ylabel("SEC")
        plot_title = (
            f"Segmented regression with {n_breakpoints} breakpoint(s): "
            f"({len(x)}) points"
        )
        if title:
            plot_title += f" — {title}"
        plt.title(plot_title)

        plt.text(
            0.05, 0.95,
            f"RSS: {rss:.2f}\nTSS: {tss:.2f}\n$R^2$: {r2:.4f}\n"
            f"$R^2$ Adjusted: {r2_adj:.4f}",
            transform=plt.gca().transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )
        plt.grid(which="major", linestyle="-", linewidth=0.7, alpha=0.8)
        plt.grid(which="minor", linestyle="--", linewidth=0.6, alpha=0.8)
        plt.minorticks_on()
        plt.legend()
        plt.show()
