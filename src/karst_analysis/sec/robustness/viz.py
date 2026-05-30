"""Figures for the robustness analysis.

Two figure types:

    1. Per-well robustness panel: scatter of (N, depth) coloured by
       smoothing, plus horizontal histogram of pooled depths, plus
       bars for clusters ranked by agreement.
    2. BIC curves panel: BIC(N) for each smoothing across all
       priority wells, with the optimal N marked.

Conventions
-----------
* Y-axis is BGL-positive ("Depth below ground level (m)") with 0 at
  the top of the figure (consistent with caliper / convergence panels).
* Savgol shown in green, lowess in purple — both distinguishable from
  the convergence panels' brown/blue scheme.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from karst_analysis.sec.export.api import load_bic_curve
from karst_analysis.sec.robustness.scoring import (
    DEFAULT_DELTA_M, RobustnessResult, compute_robustness,
)


# ──────────────────────────────────────────────────────────────────────
#  Colours used throughout this module
# ──────────────────────────────────────────────────────────────────────
SAVGOL_COLOR: str = "#0f7a4d"    # green
LOWESS_COLOR: str = "#8b3a9e"    # purple
CLUSTER_BAR_COLOR: str = "#ff7f0e"  # orange (matches BP marker colour in v6)


# ──────────────────────────────────────────────────────────────────────
#  Per-well robustness panel
# ──────────────────────────────────────────────────────────────────────
def plot_robustness_panel(
    result: RobustnessResult,
    *,
    well_id: str,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (12, 10),
    show_top_n_clusters: int = 8,
) -> plt.Figure:
    """Render the per-well robustness diagnostic panel.

    Three columns sharing the y-axis:
        Left   — scatter of (N, depth) for every BP, coloured by smoothing.
        Center — horizontal histogram of pooled depths (one bar per
                 0.5 m bin), to visualise where BPs accumulate.
        Right  — horizontal bars for the top-K clusters, with bar width
                 proportional to agreement and the cluster span shown
                 as a vertical extent.

    Parameters
    ----------
    result : RobustnessResult
        From ``compute_robustness``.
    well_id : str
        Used in the title.
    output_path : path-like, optional
        Save the figure as PNG.
    figsize : tuple
    show_top_n_clusters : int
        Number of top-ranked clusters to highlight in the right panel.
    """
    fig, (ax_scatter, ax_hist, ax_bars) = plt.subplots(
        1, 3, figsize=figsize, sharey=True,
        gridspec_kw=dict(width_ratios=(1.4, 1.0, 1.6), wspace=0.05),
    )

    bp = result.bp_records
    clusters = result.clusters

    # Defensive: drop any rows whose depth is NaN/Inf (can happen when
    # piecewise_regression fails to converge for a particular N — its
    # breakpoint depth is then undefined). Without this guard, np.min /
    # np.max on the column would propagate NaN into the axis limits and
    # matplotlib would raise "Axis limits cannot be NaN or Inf".
    bp = bp[np.isfinite(bp["depth_bgl_m"])].copy()
    if bp.empty:
        # Nothing to plot. Build a placeholder figure so callers get a
        # consistent file path back instead of an exception.
        ax_scatter.text(0.5, 0.5, f"No finite breakpoints for {well_id}",
                        ha="center", va="center", transform=ax_scatter.transAxes)
        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out, dpi=170, bbox_inches="tight")
        return fig

    # Y-limits derived from the data (BGL-positive: small numbers = surface)
    y_min = float(bp["depth_bgl_m"].min()) - 1.0
    y_max = float(bp["depth_bgl_m"].max()) + 1.0
    ax_scatter.set_ylim(y_min, y_max)

    # ── Panel 1: scatter (N, depth) ─────────────────────────────────
    for smoothing, color in (("savgol", SAVGOL_COLOR),
                              ("lowess", LOWESS_COLOR)):
        sub = bp[bp["smoothing"] == smoothing]
        if sub.empty:
            continue
        # Slight horizontal jitter for the two methods so they don't sit
        # on top of each other when the same depth appears in both.
        x_jitter = -0.12 if smoothing == "savgol" else 0.12
        ax_scatter.scatter(sub["N"].to_numpy() + x_jitter,
                           sub["depth_bgl_m"].to_numpy(),
                           c=color, s=24, alpha=0.7, edgecolor="none",
                           label=smoothing)
    ax_scatter.set_xlabel("N (number of breakpoints in fit)", fontsize=10)
    ax_scatter.set_ylabel("Depth below ground level (m)", fontsize=10)
    ax_scatter.set_xticks(range(1, 11))
    ax_scatter.tick_params(labelsize=8)
    ax_scatter.grid(True, alpha=0.25, linestyle=":")
    ax_scatter.legend(loc="lower left", fontsize=8, framealpha=0.92,
                      edgecolor="#cccccc")

    # ── Panel 2: pooled-depth histogram ─────────────────────────────
    # 0.5-m bins covering the data range
    bin_edges = np.arange(np.floor(y_min), np.ceil(y_max) + 0.5, 0.5)
    counts_savgol, _ = np.histogram(
        bp[bp["smoothing"] == "savgol"]["depth_bgl_m"].to_numpy(),
        bins=bin_edges,
    )
    counts_lowess, _ = np.histogram(
        bp[bp["smoothing"] == "lowess"]["depth_bgl_m"].to_numpy(),
        bins=bin_edges,
    )
    bin_centres = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bar_h = 0.45  # half-bin height
    ax_hist.barh(bin_centres - 0.12, counts_savgol, height=bar_h * 0.5,
                 color=SAVGOL_COLOR, alpha=0.75, label="savgol")
    ax_hist.barh(bin_centres + 0.12, counts_lowess, height=bar_h * 0.5,
                 color=LOWESS_COLOR, alpha=0.75, label="lowess")
    ax_hist.set_xlabel("Count of BPs in 0.5 m bin", fontsize=10)
    ax_hist.tick_params(labelsize=8)
    ax_hist.tick_params(axis="y", left=False, labelleft=False)
    ax_hist.grid(True, axis="x", alpha=0.25, linestyle=":")
    ax_hist.legend(loc="lower right", fontsize=8, framealpha=0.92,
                   edgecolor="#cccccc")

    # ── Panel 3: top-K cluster bars ─────────────────────────────────
    ax_bars.tick_params(axis="y", left=False, labelleft=False)
    ax_bars.set_xlabel("Agreement (min over methods)", fontsize=10)
    ax_bars.tick_params(labelsize=8)
    ax_bars.grid(True, axis="x", alpha=0.25, linestyle=":")

    if not clusters.empty:
        top = clusters.head(show_top_n_clusters).copy()
        # Pre-compute label-row-height in data units, then PAV-displace
        # so 22.39 m and 23.05 m don't sit on top of each other.
        from karst_analysis.convergence._layout import (
            minimum_displacement_positions,
        )
        y_span = (y_max - y_min)
        axis_h_in = figsize[1] * 0.86  # rough "drawable height in inches"
        pt_per_data = 72 * axis_h_in / y_span
        line_h_data = (8.5 * 1.30) / pt_per_data
        half_h = 0.5 * (line_h_data + 0.7 * line_h_data)
        anchors = top["depth_median"].to_numpy()
        text_y = minimum_displacement_positions(
            anchors, np.full(len(top), half_h),
            y_lo=y_min + 0.4, y_hi=y_max - 0.4,
            pad=0.04 * line_h_data,
        )

        for (_, row), ty in zip(top.iterrows(), text_y):
            agreement = row["agreement"]
            depth_med = row["depth_median"]
            depth_lo = row["depth_min"]
            depth_hi = row["depth_max"]
            ax_bars.fill_betweenx(
                [depth_lo, depth_hi], 0, agreement,
                color=CLUSTER_BAR_COLOR,
                alpha=0.45 if row["wide_flag"] else 0.65,
                edgecolor=CLUSTER_BAR_COLOR, linewidth=0.6,
            )
            ax_bars.plot([0, agreement], [depth_med, depth_med],
                         color="black", lw=1.0, alpha=0.8)
            wide_str = " (wide)" if row["wide_flag"] else ""
            # Connector line if PAV displaced the label
            if abs(ty - depth_med) > 0.3:
                ax_bars.plot([agreement, agreement * 1.05],
                             [depth_med, ty],
                             color="#888888", lw=0.5, alpha=0.7)
            ax_bars.annotate(
                f"{depth_med:.2f} m  P={int(row['persistence'])}/20{wide_str}",
                xy=(agreement * 1.05, ty),
                xytext=(4, 0), textcoords="offset points",
                ha="left", va="center", fontsize=8.5,
                annotation_clip=False,
            )
        n_combos = sum(result.n_max_smoothing.values())
        ax_bars.set_xlim(0, max(top["agreement"].max() * 1.45,
                                 n_combos * 0.55))

    # Title
    n_savgol = result.n_max_smoothing.get("savgol", 0)
    n_lowess = result.n_max_smoothing.get("lowess", 0)
    title = (
        f"Robustness of SEC breakpoints — well {well_id}  "
        f"(δ = {result.delta_m:.1f} m, "
        f"savgol N=1..{n_savgol}, lowess N=1..{n_lowess})"
    )
    fig.suptitle(title, fontsize=11.5, fontweight="bold")
    fig.subplots_adjust(top=0.93, left=0.07, right=0.97, bottom=0.08)

    # BGL-positive: invert y-axis after everything is drawn
    ax_scatter.invert_yaxis()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────────
#  BIC curves panel
# ──────────────────────────────────────────────────────────────────────
def plot_bic_curves(
    well_ids: list[str],
    *,
    campaign: str = "2022_02",
    project_root: Optional[Path] = None,
    output_path: Optional[str | Path] = None,
    figsize: Optional[tuple] = None,
    n_max: Optional[int] = None,
) -> plt.Figure:
    """Plot BIC(N) for every well, both smoothings, on one figure.

    One subplot per well, arranged in a row. The N that minimises BIC
    is highlighted with a star.

    Parameters
    ----------
    well_ids : list[str]
        Subplots are rendered left-to-right in this exact order. To get
        an alphabetical layout, pass ``sorted(well_ids)``.
    campaign : str
    project_root : Path, optional
    output_path : path-like, optional
    figsize : tuple, optional
    n_max : int, optional
        If given, every subplot uses the same fixed x-axis range
        ``[0, n_max]`` (xticks at every integer from 0 to n_max, xlim
        ``(-0.5, n_max + 0.5)``). Subplots whose data is sparser than
        ``n_max`` show empty space on the right — this is a deliberate
        visual cue that those (well, smoothing) combinations did not
        converge at higher N. If ``None`` (legacy default), each
        subplot is auto-scaled by matplotlib to its own data, with
        xticks at integers up to the data max (or 10, whichever is
        larger, for backwards compatibility).
    """
    n_wells = len(well_ids)
    if figsize is None:
        figsize = (3.2 * n_wells, 4.2)

    fig, axes = plt.subplots(1, n_wells, figsize=figsize, sharey=False,
                              gridspec_kw=dict(wspace=0.30))
    if n_wells == 1:
        axes = [axes]

    for ax, well in zip(axes, well_ids):
        max_n_seen = 0
        for smoothing, color in (("savgol", SAVGOL_COLOR),
                                  ("lowess", LOWESS_COLOR)):
            try:
                bic_df = load_bic_curve(
                    well_id=well, campaign=campaign,
                    smoothing=smoothing, project_root=project_root,
                )
            except Exception:
                continue
            if bic_df.empty:
                continue
            ax.plot(bic_df["n_breakpoints"], bic_df["bic"],
                    color=color, marker="o", ms=5, lw=1.2,
                    label=smoothing)
            valid = bic_df.dropna(subset=["bic"])
            if not valid.empty:
                idx_min = valid["bic"].idxmin()
                ax.scatter(valid.loc[idx_min, "n_breakpoints"],
                           valid.loc[idx_min, "bic"],
                           marker="*", s=200, c=color,
                           edgecolor="black", lw=0.8, zorder=5)
                max_n_seen = max(max_n_seen, int(valid["n_breakpoints"].max()))
        ax.set_title(well, fontsize=11, fontweight="bold")
        ax.set_xlabel("N", fontsize=10)
        ax.set_ylabel("BIC", fontsize=10)

        # X-axis range. Fixed when n_max is given (uniform across
        # wells, exposes non-convergence as empty space on the right);
        # otherwise per-subplot dynamic, clamped at >= 10 for backwards
        # compatibility with the v17.2 behaviour.
        if n_max is not None:
            ax.set_xticks(range(0, n_max + 1))
            ax.set_xlim(-0.5, n_max + 0.5)
        else:
            upper = max(max_n_seen, 10)
            ax.set_xticks(range(0, upper + 1))

        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=7.5, loc="best", framealpha=0.92,
                  edgecolor="#cccccc")

    fig.suptitle(
        f"BIC curves — campaign {campaign}  "
        f"(★ = N minimising BIC)",
        fontsize=11.5, fontweight="bold",
    )
    fig.subplots_adjust(top=0.86, left=0.06, right=0.98, bottom=0.13)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────────
#  Sensitivity panel
# ──────────────────────────────────────────────────────────────────────
def plot_delta_sensitivity(
    sensitivity_df: pd.DataFrame,
    *,
    well_id: str,
    output_path: Optional[str | Path] = None,
    figsize: tuple = (8, 6),
    top_n: int = 6,
) -> plt.Figure:
    """Plot the top-N clusters across multiple delta values.

    Each delta gets one row of horizontal bars. Useful to convince the
    reader that the qualitative result doesn't depend on the specific
    delta.

    Parameters
    ----------
    sensitivity_df : pd.DataFrame
        Output of ``compute_robustness_sensitivity``.
    well_id : str
    output_path : path-like, optional
    figsize : tuple
    top_n : int
        How many top clusters per delta to display.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if sensitivity_df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=12)
        return fig

    deltas = sorted(sensitivity_df["delta_m"].unique())
    # Defensive: drop any rows whose depths are NaN (can occur if the
    # underlying piecewise_regression fit failed to converge for some N).
    sensitivity_df = sensitivity_df[
        np.isfinite(sensitivity_df["depth_min"])
        & np.isfinite(sensitivity_df["depth_max"])
    ].copy()
    if sensitivity_df.empty:
        ax.text(0.5, 0.5, f"No finite clusters for {well_id}",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        return fig
    y_min = sensitivity_df["depth_min"].min() - 1.0
    y_max = sensitivity_df["depth_max"].max() + 1.0

    n_deltas = len(deltas)
    bar_width = 0.7 / n_deltas
    cmap = plt.get_cmap("viridis")

    for i, d in enumerate(deltas):
        sub = (sensitivity_df[sensitivity_df["delta_m"] == d]
                 .head(top_n))
        color = cmap(i / max(n_deltas - 1, 1))
        for _, row in sub.iterrows():
            agreement = row["agreement"]
            depth_lo = row["depth_min"]
            depth_hi = row["depth_max"]
            x_centre = i + 0.5
            ax.fill_betweenx(
                [depth_lo, depth_hi],
                x_centre - bar_width * agreement / 10,
                x_centre + bar_width * agreement / 10,
                color=color, alpha=0.7,
                edgecolor=color, linewidth=0.6,
            )

    ax.set_xticks([i + 0.5 for i in range(n_deltas)])
    ax.set_xticklabels([f"δ = {d} m" for d in deltas])
    ax.set_ylabel("Depth below ground level (m)", fontsize=10)
    ax.set_xlabel("Linkage threshold", fontsize=10)
    ax.set_ylim(y_min, y_max)
    ax.invert_yaxis()
    ax.tick_params(labelsize=9)
    ax.grid(True, axis="y", alpha=0.25, linestyle=":")
    ax.set_title(
        f"δ-sensitivity of robustness clusters — {well_id}  "
        f"(top-{top_n} per δ; bar width ∝ agreement)",
        fontsize=10.5, fontweight="bold",
    )
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.10)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170, bbox_inches="tight")
    return fig
