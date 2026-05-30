"""Plot breakpoints from a saved BIC-sweep JSON without re-fitting.

The BIC-sweep JSON contains, for each trial, a table with one row per
``n_breakpoints`` value (1 .. max_bp) including the fitted breakpoint
positions in the ``estimates`` column. This module reads such a JSON
and produces overlay plots — markers at the breakpoint depths over the
smoothed profile — for any chosen N.

Two flavours of plot are supported:

    plot_breakpoints_overlay
        One panel: a single (well, smoothing method) profile with the
        breakpoints for one chosen N.

    plot_breakpoints_compare_methods
        Two panels side-by-side: same well at the same N, with savgol
        on the left and lowess on the right. Useful to decide which
        smoother yields more interpretable breakpoints.

The functions DO NOT re-fit anything. All breakpoint positions come
straight from the JSON. The y-coordinate of each marker is the value
of the smoothed profile at the breakpoint depth (linearly interpolated),
not the raw conductivity.

Convention
----------
The y-axis is "Depth below ground level (m)" with 0 at the top
(``invert_y=True`` default). Use this convention everywhere in
karst_analysis until absolute elevation (m above sea level) becomes
available from differential GPS.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────
#  JSON helpers
# ─────────────────────────────────────────────────────────────────────────
def load_bic_json(json_path: str | Path) -> dict[str, Any]:
    """Load a BIC-sweep JSON saved by notebook 03 / breakpoints_batch."""
    with open(json_path) as f:
        return json.load(f)


def extract_breakpoints_for_n(
    bic_data: dict, n_breakpoints: int, trial: str = "trial_1",
) -> list[float]:
    """Return a list of breakpoint x-positions for a given N.

    Parameters
    ----------
    bic_data : dict
        Output of :func:`load_bic_json`.
    n_breakpoints : int
        The N to extract breakpoints for. Must be in the range used by
        the original sweep.
    trial : str, default "trial_1"

    Returns
    -------
    list of float
        Sorted breakpoint depths. Empty if N=0 or the entry is missing.
    """
    if trial not in bic_data:
        raise KeyError(f"Trial '{trial}' not in JSON. Available: {list(bic_data.keys())}")

    df_dict = bic_data[trial]["df"]
    df = pd.DataFrame(df_dict)

    if "n_breakpoints" not in df.columns:
        raise ValueError("'n_breakpoints' column missing from JSON df.")

    matches = df[df["n_breakpoints"] == n_breakpoints]
    if len(matches) == 0:
        raise ValueError(
            f"N={n_breakpoints} not found in trial '{trial}'. "
            f"Available: {sorted(df['n_breakpoints'].unique().tolist())}"
        )

    estimates = matches.iloc[0]["estimates"]
    if estimates is None or n_breakpoints == 0:
        return []

    if isinstance(estimates, str):
        estimates = json.loads(estimates)

    bps = []
    for i in range(1, n_breakpoints + 1):
        key = f"breakpoint{i}"
        if key in estimates:
            entry = estimates[key]
            if isinstance(entry, dict):
                bps.append(float(entry.get("estimate")))
            else:
                bps.append(float(entry))
    return sorted(bps)


def get_metric_at_n(
    bic_data: dict, n_breakpoints: int, metric: str = "bic",
    trial: str = "trial_1",
) -> Optional[float]:
    """Look up a scalar metric (bic, rss, ...) for a given N.

    Returns None if the metric is missing, NaN, or non-numeric.
    Callers should handle None gracefully (e.g. show 'n/a' in plots).
    """
    df_dict = bic_data[trial]["df"]
    df = pd.DataFrame(df_dict)
    matches = df[df["n_breakpoints"] == n_breakpoints]
    if len(matches) == 0 or metric not in matches.columns:
        return None
    val = matches.iloc[0][metric]
    try:
        v = float(val)
    except (TypeError, ValueError):
        return None
    if math.isnan(v) or math.isinf(v):
        return None
    return v


def _format_bic(bic: Optional[float]) -> str:
    """Render a BIC value for display, NaN/None safe."""
    if bic is None:
        return "BIC: n/a"
    if abs(bic) >= 1e4:
        return f"BIC = {bic:,.1f}"
    return f"BIC = {bic:.1f}"


# ─────────────────────────────────────────────────────────────────────────
#  Marker placement
# ─────────────────────────────────────────────────────────────────────────
def _est(estimates: dict, key: str) -> Optional[float]:
    """Pull a numeric value from an 'estimates' dict.

    Each entry can be either a scalar or a dict like
    ``{"estimate": 4.24, "confidence_interval": [4.18, 4.30], ...}``.
    """
    v = estimates.get(key)
    if v is None:
        return None
    if isinstance(v, dict):
        v = v.get("estimate")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compute_sec_at_breakpoints(
    bic_data: dict,
    n_breakpoints: int,
    *,
    trial: str = "trial_1",
    output_space: str = "linear",
) -> np.ndarray:
    """Return the SEC value AT each breakpoint, computed from the fitted model.

    For each breakpoint x_i, the piecewise-linear model gives a unique
    y value: it is the intersection of the two adjacent linear segments,
    which by construction equals

        y(x_i) = const + alpha1 * x_i + sum_{j: x_j < x_i} beta_j * (x_i - x_j)

    where (const, alpha1, beta_j, x_j) are the fitted parameters stored
    in the JSON.

    This is the **correct** y-coordinate for plotting markers — using
    interpolation onto the smoothed curve gives slightly wrong values
    when the smoothed curve and the piecewise model don't perfectly agree.

    Parameters
    ----------
    bic_data : dict
        Output of :func:`load_bic_json`.
    n_breakpoints : int
    trial : str, default "trial_1"
    output_space : {"linear", "log10"}
        - ``"linear"``: returns 10**y (µS/cm) — appropriate for plotting
          on a linear x-axis labelled "SEC [µS/cm]". This is the default
          because it matches how the figures display the data.
        - ``"log10"``: returns y as fitted (in log10 µS/cm). Useful if
          the consumer wants to plot on a log-scaled x-axis.

    Returns
    -------
    np.ndarray
        SEC value at each breakpoint, sorted by depth ascending. Length
        matches the number of breakpoints requested. Values are NaN if
        the model could not be reconstructed (e.g. trial did not converge).
    """
    if output_space not in ("linear", "log10"):
        raise ValueError(f"output_space must be 'linear' or 'log10'; got '{output_space}'")
    if n_breakpoints == 0:
        return np.array([])
    if trial not in bic_data:
        raise KeyError(f"Trial '{trial}' not in JSON.")

    df = pd.DataFrame(bic_data[trial]["df"])
    match = df[df["n_breakpoints"] == n_breakpoints]
    if not len(match):
        raise ValueError(f"N={n_breakpoints} not in trial '{trial}'.")

    estimates = match.iloc[0]["estimates"]
    if estimates is None:
        return np.full(n_breakpoints, np.nan)
    if isinstance(estimates, str):
        estimates = json.loads(estimates)

    const  = _est(estimates, "const")
    alpha1 = _est(estimates, "alpha1")
    if const is None or alpha1 is None:
        return np.full(n_breakpoints, np.nan)

    bps = []
    for i in range(1, n_breakpoints + 1):
        x = _est(estimates, f"breakpoint{i}")
        if x is not None:
            bps.append(x)
    bps_sorted = sorted(bps)

    betas_in_index_order = []
    for i in range(1, n_breakpoints + 1):
        b = _est(estimates, f"beta{i}")
        x = _est(estimates, f"breakpoint{i}")
        if b is None or x is None:
            return np.full(n_breakpoints, np.nan)
        betas_in_index_order.append((x, b))

    y_vals = []
    for x_target in bps_sorted:
        y = const + alpha1 * x_target
        for (x_other, beta) in betas_in_index_order:
            if x_target > x_other:
                y += beta * (x_target - x_other)
        y_vals.append(y)
    y_arr = np.asarray(y_vals, dtype=float)

    if output_space == "linear":
        return 10.0 ** y_arr
    return y_arr


# ─────────────────────────────────────────────────────────────────────────
#  Internal: draw markers + leader-line labels OUTSIDE the curve
# ─────────────────────────────────────────────────────────────────────────
def _draw_breakpoints_with_labels(
    ax: plt.Axes,
    breakpoint_depths: list[float],
    sec_at_bp: np.ndarray,
    *,
    sec_min: float,
    sec_max: float,
    label_side: str = "right",
    marker_size: int = 55,
    marker_color: str = "#e67e22",
    marker_edge: str = "black",
    marker_label: Optional[str] = None,
    fontsize: float = 8.5,
) -> None:
    """Plot diamond markers AT the breakpoints and labels OUTSIDE the curve.

    The label sits to the right (or left) of the panel, connected to its
    marker by a thin grey leader. This keeps the smoothed-profile line
    and the raw scatter free of clutter.

    Parameters
    ----------
    sec_min, sec_max : float
        Current x-axis range of the panel. Used to compute the label
        anchor position on the chosen side.
    label_side : {"right", "left"}, default "right"
        On which side of the curve the labels sit.
    """
    if len(breakpoint_depths) == 0:
        return

    # Marker (small, simple)
    ax.scatter(
        sec_at_bp, breakpoint_depths,
        s=marker_size, marker="D",
        facecolor=marker_color, edgecolor=marker_edge, linewidth=0.7,
        zorder=6, label=marker_label,
    )

    # Anchor x of the labels: just inside the data axis, on the chosen side.
    span = sec_max - sec_min
    if label_side == "right":
        x_label = sec_max - 0.02 * span
        ha = "right"
    else:
        x_label = sec_min + 0.02 * span
        ha = "left"

    for i, (d, ec) in enumerate(zip(breakpoint_depths, sec_at_bp), start=1):
        if np.isnan(ec):
            continue
        # Leader line from marker → label anchor (thin, grey, low zorder)
        ax.plot(
            [ec, x_label], [d, d],
            color="#7f8c8d", lw=0.6, alpha=0.55, zorder=4,
        )
        ax.text(
            x_label, d, f"BP{i}: {d:.2f} m",
            fontsize=fontsize, fontweight="bold",
            ha=ha, va="center", zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white", edgecolor=marker_color,
                alpha=0.92, linewidth=0.7,
            ),
        )


# ─────────────────────────────────────────────────────────────────────────
#  Single-panel overlay
# ─────────────────────────────────────────────────────────────────────────
def plot_breakpoints_overlay(
    *,
    z_raw: np.ndarray, EC_raw: np.ndarray,
    z_smooth: np.ndarray, EC_smooth: np.ndarray,
    bic_data: dict,
    n_breakpoints: int,
    output_path: str | Path,
    trial: str = "trial_1",
    title: str = "",
    method_label: str = "smoothed",
    figure_size: tuple = (7, 9),
    figure_dpi: int = 130,
    invert_y: bool = True,
    label_side: str = "right",
    vadose_offset_m: float = 0.0,
) -> Path:
    """One-panel plot: raw scatter + smoothed line + BPs as diamonds.

    The breakpoint markers are placed at the (depth, sec) coordinates
    given by the **fitted piecewise-linear model**, NOT by interpolation
    onto the smoothed curve. The SEC value at each breakpoint is computed
    in log10 space (where the fit was performed) and converted to linear
    µS/cm to match the x-axis. If you see the marker visibly off the
    smoothed curve, it means the piecewise model and the smoother
    disagree slightly at that depth — this is diagnostic information,
    not a bug.

    Labels are placed on ``label_side`` (default 'right'), connected to
    each marker by a thin leader line.

    Parameters
    ----------
    bic_data : dict
        Output of :func:`load_bic_json`. Required to reconstruct the
        fitted model and place markers correctly.
    n_breakpoints : int
        Which N to plot.
    trial : str, default "trial_1"
        Which trial to read from the JSON.
    vadose_offset_m : float, default 0.0
        Vertical offset (in metres) added to every depth value before
        plotting, to convert the SEC pipeline's native water-table
        datum to below-ground-level. Pass the well's
        ``vadose_thickness_m`` from ``data/metadata/wells.csv``. The
        default of 0.0 leaves depths in water-table datum and labels
        the y-axis ``"Depth below water table (m)"``; any nonzero
        value labels it ``"Depth below ground level (m)"``. BGL is the
        canonical datum for karst_analysis (see CHANGELOG v17.3).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    breakpoint_depths = extract_breakpoints_for_n(bic_data, n_breakpoints, trial=trial)
    sec_at_bp_linear  = compute_sec_at_breakpoints(
        bic_data, n_breakpoints, trial=trial, output_space="linear",
    )
    bic_value = get_metric_at_n(bic_data, n_breakpoints, metric="bic", trial=trial)

    # ── Datum shift ----------------------------------------------------
    z_raw = np.asarray(z_raw, dtype=float) + vadose_offset_m
    z_smooth = np.asarray(z_smooth, dtype=float) + vadose_offset_m
    breakpoint_depths = [d + vadose_offset_m for d in breakpoint_depths]
    depth_axis_label = (
        "Depth below ground level (m)" if vadose_offset_m > 0
        else "Depth below water table (m)"
    )

    fig, ax = plt.subplots(figsize=figure_size)

    ax.scatter(EC_raw, z_raw, s=4, color="#bdc3c7", alpha=0.45,
               linewidth=0, zorder=1, label=f"Raw ({len(z_raw):,})")
    ax.plot(EC_smooth, z_smooth, color="#1f4e79", lw=1.4, zorder=3,
            label=f"{method_label} ({len(z_smooth):,})")

    # Compute x-axis range AFTER raw + smooth are plotted, so labels
    # attach to the actual visible range.
    x_lo, x_hi = ax.get_xlim()

    if breakpoint_depths:
        _draw_breakpoints_with_labels(
            ax, breakpoint_depths, sec_at_bp_linear,
            sec_min=x_lo, sec_max=x_hi,
            label_side=label_side,
            marker_label=f"BPs (N={len(breakpoint_depths)})",
        )

    ax.text(
        0.02, 0.02, _format_bic(bic_value),
        transform=ax.transAxes, fontsize=9, va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                  edgecolor="#1f4e79", alpha=0.9),
    )

    ax.set_xlabel("Specific electrical conductivity [µS/cm]")
    ax.set_ylabel(depth_axis_label)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Two-panel: savgol vs lowess at the same N
# ─────────────────────────────────────────────────────────────────────────
def plot_breakpoints_compare_methods(
    *,
    z_raw: np.ndarray, EC_raw: np.ndarray,
    # left panel
    z_left: np.ndarray, EC_left: np.ndarray,
    bic_data_left: dict,
    label_left: str = "savgol",
    # right panel
    z_right: np.ndarray, EC_right: np.ndarray,
    bic_data_right: dict,
    label_right: str = "lowess",
    # which N
    n_breakpoints: int,
    trial: str = "trial_1",
    # plumbing
    output_path: str | Path,
    title: str = "",
    figure_size: tuple = (13, 9),
    figure_dpi: int = 130,
    invert_y: bool = True,
    share_x: bool = True,
    vadose_offset_m: float = 0.0,
) -> Path:
    """Two-panel comparison: same well at the same N, two smoothers.

    The breakpoint markers are placed at the (depth, sec) coordinates
    given by the **fitted piecewise-linear model** for each smoother,
    converted from log10 to linear µS/cm to match the x-axis. This is
    the correct visual position; using interpolation onto the smoothed
    curve gives slightly wrong values when the model and the smoother
    don't perfectly agree.

    Labels are placed against the right edge of each panel and connected
    to their markers with leader lines. Markers are small diamonds so
    they don't obscure the curve.

    Parameters
    ----------
    bic_data_left, bic_data_right : dict
        Outputs of :func:`load_bic_json` for each smoother.
    n_breakpoints : int
        N to plot (same in both panels).
    trial : str, default "trial_1"
        Which trial to read in BOTH panels.
    vadose_offset_m : float, default 0.0
        Vertical offset (in metres) added to every depth value (raw,
        smoothed, and breakpoint markers) before plotting, to convert
        the SEC pipeline's native water-table datum to below-ground-
        level. Pass the well's ``vadose_thickness_m`` from
        ``data/metadata/wells.csv``. The default of 0.0 leaves depths
        in water-table datum and labels the y-axis ``"Depth below
        water table (m)"``; any nonzero value labels it ``"Depth below
        ground level (m)"``. BGL is the canonical datum for
        karst_analysis (see CHANGELOG v17.3).
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Extract per-side breakpoints, model SEC values, and BIC.
    bps_left   = extract_breakpoints_for_n(bic_data_left,  n_breakpoints, trial=trial)
    bps_right  = extract_breakpoints_for_n(bic_data_right, n_breakpoints, trial=trial)
    sec_left   = compute_sec_at_breakpoints(
        bic_data_left,  n_breakpoints, trial=trial, output_space="linear",
    )
    sec_right  = compute_sec_at_breakpoints(
        bic_data_right, n_breakpoints, trial=trial, output_space="linear",
    )
    bic_left   = get_metric_at_n(bic_data_left,  n_breakpoints, metric="bic", trial=trial)
    bic_right  = get_metric_at_n(bic_data_right, n_breakpoints, metric="bic", trial=trial)

    # ── Datum shift ----------------------------------------------------
    z_raw = np.asarray(z_raw, dtype=float) + vadose_offset_m
    z_left = np.asarray(z_left, dtype=float) + vadose_offset_m
    z_right = np.asarray(z_right, dtype=float) + vadose_offset_m
    bps_left = [d + vadose_offset_m for d in bps_left]
    bps_right = [d + vadose_offset_m for d in bps_right]
    depth_axis_label = (
        "Depth below ground level (m)" if vadose_offset_m > 0
        else "Depth below water table (m)"
    )

    fig, axes = plt.subplots(
        1, 2, figsize=figure_size, sharey=True, sharex=share_x,
    )

    panels = [
        (axes[0], z_left,  EC_left,  bps_left,  sec_left,  label_left,  bic_left),
        (axes[1], z_right, EC_right, bps_right, sec_right, label_right, bic_right),
    ]

    # First pass: plot raw + smoothed on every panel so x-limits stabilise.
    for ax, zs, ecs, bps, sec_at_bp, lbl, bic in panels:
        ax.scatter(EC_raw, z_raw, s=4, color="#bdc3c7", alpha=0.45,
                   linewidth=0, zorder=1, label=f"Raw ({len(z_raw):,})")
        ax.plot(ecs, zs, color="#1f4e79", lw=1.4, zorder=3,
                label=f"{lbl} ({len(zs):,})")

    # If sharex, the two panels have the SAME x-range — read it once.
    x_lo, x_hi = axes[0].get_xlim()

    # Second pass: BPs, BIC text, axis cosmetics.
    for ax, zs, ecs, bps, sec_at_bp, lbl, bic in panels:
        if bps:
            _draw_breakpoints_with_labels(
                ax, bps, sec_at_bp,
                sec_min=x_lo, sec_max=x_hi,
                label_side="right",
                marker_label=f"BPs (N={len(bps)})",
            )

        ax.text(
            0.02, 0.02, _format_bic(bic),
            transform=ax.transAxes, fontsize=9,
            va="bottom", ha="left",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#1f4e79", alpha=0.9),
        )

        ax.set_xlabel("SEC [µS/cm]")
        ax.set_title(lbl, fontsize=11, fontweight="bold")
        ax.grid(True, ls=":", alpha=0.4)
        ax.legend(loc="lower right", fontsize=8, framealpha=0.92)

    axes[0].set_ylabel(depth_axis_label)

    # Invert y AFTER the panels-loop. With sharey=True the two axes are
    # the same y-axis; calling invert_yaxis() once per panel inverts it
    # twice (cancelling out). Call it once on either panel.
    if invert_y:
        axes[0].invert_yaxis()

    full_title = f"{title} — N={n_breakpoints}" if title else f"N={n_breakpoints}"
    fig.suptitle(full_title, fontsize=13, fontweight="bold", y=0.995)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return out
