"""Visualisation: SEC profile + breakpoints + chord-slope labels.

Single-panel plot intended for thesis/review meetings. Shows, on a SEC
profile (raw scatter under smoothed curve), the breakpoints found by
``piecewise_regression`` plus the chord slopes between them. The two
chord slopes flagged by ``compute_slopes`` as the top and bottom of
the mixing zone are highlighted in a distinct colour and with larger
slope labels.

X-axis scale is selectable via ``axis_scale``:
    - "linear" : SEC in µS/cm. Slope labels in µS/cm·m⁻¹.
    - "log10"  : log10(SEC). Slope labels in decade/m.

Mixing-zone identification is done in log10 space (this is where the
detector and the slope module operate); the TOP and BOTTOM markers
appear at the same physical depths regardless of the chosen axis,
which is the point of plotting both.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Colours ---------------------------------------------------------------
_COLOR_BP_REGULAR  = "#e67e22"   # orange — same hue as breakpoints_overlay
_COLOR_TOP_MZ      = "#c0392b"   # deep red — TOP of mixing zone
_COLOR_BOTTOM_MZ   = "#8e44ad"   # purple   — BOTTOM of mixing zone
_COLOR_RAW         = "#bdc3c7"   # light grey
_COLOR_SMOOTH      = "#1f4e79"   # navy
_COLOR_CHORD       = "#7f8c8d"   # grey for chord lines

# Font sizes ------------------------------------------------------------
_FONT_LABEL_REGULAR = 7.0
_FONT_LABEL_HIGHLIGHT = 11.0
_FONT_BP_LABEL = 8.0


def plot_slopes_overlay(
    *,
    z_raw: np.ndarray,
    EC_raw: np.ndarray,
    z_smooth: np.ndarray,
    EC_smooth: np.ndarray,
    slopes_df: pd.DataFrame,
    output_path: str | Path,
    axis_scale: str = "log10",
    title: str = "",
    method_label: str = "smoothed",
    figure_size: tuple = (9, 11),
    figure_dpi: int = 130,
    invert_y: bool = True,
    vadose_offset_m: float = 0.0,
) -> Path:
    """Plot SEC profile + breakpoints + chord slope labels.

    Parameters
    ----------
    z_raw, EC_raw : np.ndarray
        Raw cast (depth_m, sec_uS_cm). Plotted as a faint scatter.
        ``depth_m`` is in the SEC pipeline's native datum (zero at the
        water table). To plot in BGL, set ``vadose_offset_m`` below.
    z_smooth, EC_smooth : np.ndarray
        Smoothed profile (depth_m, sec_uS_cm), as fed to the breakpoint
        detector. Plotted as a solid line.
    slopes_df : pd.DataFrame
        Output of ``karst_analysis.sec.slopes.compute_slopes``. Must
        contain columns ``depth_top``, ``depth_bottom``,
        ``log10_sec_top``, ``log10_sec_bottom``,
        ``sec_top_uS_cm``, ``sec_bottom_uS_cm``,
        ``slope_log10``, ``slope_linear_uS_cm_per_m``,
        ``is_top_of_mixing``, ``is_bottom_of_mixing``. The depth columns
        are expected in the same datum as ``z_smooth``.
    output_path : str or Path
        Where to save the PNG. Parent directory is created.
    axis_scale : {"linear", "log10"}
        X-axis units. Slope labels match.
    title : str
        Figure title (e.g. "LRS70D · 2022-01-31 · LOWESS · N=15").
    method_label : str
        Legend label for the smoothed curve.
    vadose_offset_m : float, default 0.0
        Vertical offset (in metres) added to every depth value before
        plotting, to convert the SEC pipeline's native water-table
        datum to below-ground-level. Pass the well's
        ``vadose_thickness_m`` from ``data/metadata/wells.csv``. The
        default of 0.0 leaves the data in water-table datum and labels
        the y-axis ``"Depth below water table (m)"`` to be explicit;
        any nonzero value labels the y-axis
        ``"Depth below ground level (m)"``. BGL is the canonical datum
        for the karst_analysis project (see CHANGELOG v17.3).

    Returns
    -------
    Path
        Path to the saved figure.
    """
    if axis_scale not in ("linear", "log10"):
        raise ValueError(
            f"axis_scale must be 'linear' or 'log10', got {axis_scale!r}."
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ── Datum shift -------------------------------------------------
    # The SEC pipeline is in water-table datum; we shift to BGL if a
    # nonzero offset was provided. All Y values (raw, smoothed, and
    # depth columns of slopes_df) are shifted by the same amount, so
    # relative positions are preserved.
    z_raw = np.asarray(z_raw, dtype=float) + vadose_offset_m
    z_smooth = np.asarray(z_smooth, dtype=float) + vadose_offset_m
    if vadose_offset_m != 0.0 and len(slopes_df) > 0:
        slopes_df = slopes_df.copy()
        slopes_df["depth_top"] = slopes_df["depth_top"] + vadose_offset_m
        slopes_df["depth_bottom"] = slopes_df["depth_bottom"] + vadose_offset_m

    depth_axis_label = (
        "Depth below ground level (m)" if vadose_offset_m > 0
        else "Depth below water table (m)"
    )

    # ── X-axis transformation -----------------------------------------
    # Both raw and smoothed curves are in linear µS/cm; transform if
    # log10 is requested. Filter out non-positive raw values to avoid
    # log10 NaNs (the YSI in-air readings produce a few near-zero pts).
    if axis_scale == "log10":
        raw_pos = EC_raw > 0
        x_raw = np.log10(EC_raw[raw_pos])
        z_raw_plot = z_raw[raw_pos]

        sm_pos = EC_smooth > 0
        x_smooth = np.log10(EC_smooth[sm_pos])
        z_smooth_plot = z_smooth[sm_pos]

        x_label = "log₁₀(SEC) [µS/cm]"
        slope_unit = "dec/m"
        bp_top_col = "log10_sec_top"
        bp_bot_col = "log10_sec_bottom"
        slope_col = "slope_log10"
    else:
        x_raw = EC_raw
        z_raw_plot = z_raw
        x_smooth = EC_smooth
        z_smooth_plot = z_smooth
        x_label = "SEC [µS/cm]"
        slope_unit = "µS/cm/m"
        bp_top_col = "sec_top_uS_cm"
        bp_bot_col = "sec_bottom_uS_cm"
        slope_col = "slope_linear_uS_cm_per_m"

    # ── Figure --------------------------------------------------------
    fig, ax = plt.subplots(figsize=figure_size)

    # Layer 1 — raw scatter
    ax.scatter(
        x_raw, z_raw_plot,
        s=4, color=_COLOR_RAW, alpha=0.45, linewidth=0,
        zorder=1, label=f"Raw ({len(z_raw):,})",
    )

    # Layer 2 — smoothed line
    ax.plot(
        x_smooth, z_smooth_plot,
        color=_COLOR_SMOOTH, lw=1.4, zorder=3,
        label=f"{method_label} ({len(z_smooth):,})",
    )

    # Pin the x-range to the smoothed curve so chord labels land in
    # axes coords rather than off-screen.
    x_lo, x_hi = ax.get_xlim()
    span = x_hi - x_lo

    # ── Layers 3, 4, 5 — chords, breakpoints, slope labels ------------
    if len(slopes_df) > 0:
        _draw_chords_and_slopes(
            ax, slopes_df,
            bp_top_col=bp_top_col, bp_bot_col=bp_bot_col,
            slope_col=slope_col, slope_unit=slope_unit,
            x_lo=x_lo, x_hi=x_hi, span=span,
        )

    # ── Cosmetics -----------------------------------------------------
    ax.set_xlabel(x_label, fontsize=11)
    ax.set_ylabel(depth_axis_label, fontsize=11)
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold")
    if invert_y:
        ax.invert_yaxis()
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.92)

    fig.tight_layout()
    fig.savefig(out, dpi=figure_dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────
#  Internals
# ─────────────────────────────────────────────────────────────────────
def _draw_chords_and_slopes(
    ax: plt.Axes,
    slopes_df: pd.DataFrame,
    *,
    bp_top_col: str,
    bp_bot_col: str,
    slope_col: str,
    slope_unit: str,
    x_lo: float,
    x_hi: float,
    span: float,
) -> None:
    """Draw chord lines, breakpoint markers, and slope labels.

    A unique breakpoint set is reconstructed from the slopes_df rows:
    pair i has (depth_top, x_top) and (depth_bottom, x_bottom). Across
    consecutive rows, the bottom of pair i equals the top of pair i+1,
    so the unique breakpoints are simply the tops plus the very last
    bottom.
    """
    # Reconstruct the unique breakpoints in order.
    z_bp = list(slopes_df["depth_top"].to_numpy()) \
         + [float(slopes_df["depth_bottom"].iloc[-1])]
    x_bp = list(slopes_df[bp_top_col].to_numpy()) \
         + [float(slopes_df[bp_bot_col].iloc[-1])]

    is_top    = slopes_df["is_top_of_mixing"].to_numpy(dtype=bool)
    is_bottom = slopes_df["is_bottom_of_mixing"].to_numpy(dtype=bool)
    slopes    = slopes_df[slope_col].to_numpy(dtype=float)

    # Layer 3 — chord lines between consecutive breakpoints
    for i in range(len(slopes_df)):
        x0, x1 = x_bp[i], x_bp[i + 1]
        z0, z1 = z_bp[i], z_bp[i + 1]
        chord_color = _COLOR_CHORD
        chord_lw = 0.8
        if is_top[i]:
            chord_color = _COLOR_TOP_MZ
            chord_lw = 1.6
        elif is_bottom[i]:
            chord_color = _COLOR_BOTTOM_MZ
            chord_lw = 1.6
        ax.plot(
            [x0, x1], [z0, z1],
            color=chord_color, lw=chord_lw, alpha=0.7, zorder=4,
        )

    # Layer 4 — breakpoint markers
    # Build per-breakpoint colour. With the |Δslope|-based mixing-zone
    # criterion (semantic β), a flagged pair sits immediately below the
    # transition we are crossing; the BP that marks that transition is
    # the depth_top of that pair. So both flags point to position `i`
    # in the breakpoint list (the depth_top of pair i is the i-th BP
    # in the reconstructed sequence).
    bp_colors = [_COLOR_BP_REGULAR] * len(z_bp)
    for i in range(len(slopes_df)):
        if is_top[i]:
            bp_colors[i] = _COLOR_TOP_MZ      # depth_top of pair i = TOP MZ BP
        if is_bottom[i]:
            bp_colors[i] = _COLOR_BOTTOM_MZ   # depth_top of pair i = BOTTOM MZ BP

    # Annotate each BP with its index "BPk" placed on the LEFT side of
    # the curve (opposite of the slope labels, which sit on the right).
    x_bp_label = x_lo + 0.02 * span
    for k, (xb, zb, c) in enumerate(zip(x_bp, z_bp, bp_colors), start=1):
        ax.plot(
            [xb, x_bp_label], [zb, zb],
            color="#bdc3c7", lw=0.4, alpha=0.45, zorder=3,
        )
        is_highlight = c != _COLOR_BP_REGULAR
        ax.text(
            x_bp_label, zb, f"BP{k}",
            fontsize=8.5 if is_highlight else 7.0,
            fontweight="bold" if is_highlight else "normal",
            color=c if is_highlight else "#34495e",
            ha="left", va="center", zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=c if is_highlight else "#bdc3c7",
                alpha=0.92, linewidth=0.6,
            ),
        )

    # Plot regular BPs first, then highlights, so the highlights sit on top.
    reg_idx = [i for i, c in enumerate(bp_colors) if c == _COLOR_BP_REGULAR]
    top_idx = [i for i, c in enumerate(bp_colors) if c == _COLOR_TOP_MZ]
    bot_idx = [i for i, c in enumerate(bp_colors) if c == _COLOR_BOTTOM_MZ]

    if reg_idx:
        ax.scatter(
            [x_bp[i] for i in reg_idx], [z_bp[i] for i in reg_idx],
            s=55, marker="D",
            facecolor=_COLOR_BP_REGULAR, edgecolor="black", linewidth=0.7,
            zorder=6, label=f"BPs (N={len(z_bp)})",
        )
    if top_idx:
        ax.scatter(
            [x_bp[i] for i in top_idx], [z_bp[i] for i in top_idx],
            s=120, marker="D",
            facecolor=_COLOR_TOP_MZ, edgecolor="black", linewidth=1.2,
            zorder=7, label="TOP of mixing zone",
        )
    if bot_idx:
        ax.scatter(
            [x_bp[i] for i in bot_idx], [z_bp[i] for i in bot_idx],
            s=120, marker="D",
            facecolor=_COLOR_BOTTOM_MZ, edgecolor="black", linewidth=1.2,
            zorder=7, label="BOTTOM of mixing zone",
        )

    # Layer 5 — slope labels next to each chord midpoint
    # Anchor labels on the right side of the panel, with a leader.
    x_anchor = x_hi - 0.02 * span
    for i in range(len(slopes_df)):
        z_mid = 0.5 * (z_bp[i] + z_bp[i + 1])
        x_mid = 0.5 * (x_bp[i] + x_bp[i + 1])
        slope_val = slopes[i]
        pair_label = f"P{i + 1}"

        if is_top[i]:
            color = _COLOR_TOP_MZ
            fontsize = _FONT_LABEL_HIGHLIGHT
            fontweight = "bold"
            text = f"TOP MZ · {pair_label}: {slope_val:.3g} {slope_unit}"
        elif is_bottom[i]:
            color = _COLOR_BOTTOM_MZ
            fontsize = _FONT_LABEL_HIGHLIGHT
            fontweight = "bold"
            text = f"BOT MZ · {pair_label}: {slope_val:.3g} {slope_unit}"
        else:
            color = "#34495e"
            fontsize = _FONT_LABEL_REGULAR
            fontweight = "normal"
            text = f"{pair_label}: {slope_val:.3g}"

        # Leader line from chord midpoint to label anchor
        ax.plot(
            [x_mid, x_anchor], [z_mid, z_mid],
            color="#bdc3c7", lw=0.5, alpha=0.5, zorder=4,
        )
        ax.text(
            x_anchor, z_mid, text,
            fontsize=fontsize, fontweight=fontweight, color=color,
            ha="right", va="center", zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="white", edgecolor=color,
                alpha=0.92, linewidth=0.7,
            ),
        )
