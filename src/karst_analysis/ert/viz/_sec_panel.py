"""SEC panel renderer for the SEC-vs-ERT comparison figure.

This is a stripped-down port of
``karst_analysis.sec.viz.slopes_overlay.plot_slopes_overlay`` adapted
for two-panel comparison plots. Differences vs the original:

  (a) NO slope labels (no "P1: 590", no "P14: 24.4"). The
      comparison figure is about co-located breakpoints across
      techniques, not about the slope magnitudes per chord.

  (b) Each breakpoint gets a depth label "BPk: z.zz m" placed on the
      RIGHT of the panel (where the slope labels used to live), with
      the same visual style ERT uses on its own panel. This lets the
      reader read both panels with the same convention.

  (c) Accepts an external Axes (does not create its own figure), so
      the caller composes the two-panel figure with sharey=True.

Everything else (raw scatter, smoothed line, chord highlighting for
TOP/BOT MZ, MZ marker styling, BP markers) matches the original
overlay so the SEC panel reads identically to the production figure
in every other respect.

Conventions
-----------
- Depth axis: BGL (positive down). The caller is expected to set
  ax.set_ylim(deepest, 0) to invert.
- The slopes DataFrame is expected to come from
  ``karst_analysis.sec.slopes.compute_slopes`` (same schema as the
  CSVs written by ``scripts/slopes_batch.py``). Column names match
  that schema verbatim.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.axes import Axes


# ── colour palette copied from sec.viz.slopes_overlay ───────────────
_COLOR_BP_REGULAR = "#e67e22"
_COLOR_TOP_MZ     = "#c0392b"
_COLOR_BOTTOM_MZ  = "#8e44ad"
_COLOR_RAW        = "#bdc3c7"
_COLOR_SMOOTH     = "#1f4e79"
_COLOR_CHORD      = "#7f8c8d"

_FONT_LABEL_REGULAR   = 7.0
_FONT_LABEL_HIGHLIGHT = 11.0


def render_sec_panel(
    ax: Axes,
    *,
    z_raw_bgl_m: np.ndarray,
    sec_raw_uS_cm: np.ndarray,
    z_smooth_bgl_m: np.ndarray,
    sec_smooth_uS_cm: np.ndarray,
    slopes_df: pd.DataFrame,
    vadose_m: float,
    axis_scale: str,
    method_label: str = "SEC",
) -> None:
    """Draw the SEC panel onto an existing Axes for the SEC-vs-ERT figure.

    Parameters
    ----------
    ax : Axes
        Target Axes. Caller is responsible for ylim and sharey wiring.
    z_raw_bgl_m, sec_raw_uS_cm : ndarray
        Raw YSI cast, in BGL convention (positive down) and µS/cm.
    z_smooth_bgl_m, sec_smooth_uS_cm : ndarray
        Smoothed cast (LOWESS or Savitzky-Golay), same convention.
    slopes_df : pd.DataFrame
        Output of ``karst_analysis.sec.slopes.compute_slopes`` (or its
        on-disk CSV). Must contain columns ``depth_top``,
        ``depth_bottom``, ``log10_sec_top``, ``log10_sec_bottom``,
        ``sec_top_uS_cm``, ``sec_bottom_uS_cm``, ``is_top_of_mixing``,
        ``is_bottom_of_mixing``. The depth columns are referenced to
        the water table; this function adds ``vadose_m`` on the fly to
        plot in BGL.
    vadose_m : float
        Vadose-zone thickness for this well (depth_bgl - depth_water).
        Constant per cast.
    axis_scale : {"linear", "log10"}
        Which x-scale to plot. SEC display only — the breakpoints
        themselves were detected upstream in log10 regardless.
    method_label : str, default "SEC"
        Used in the legend. Pass e.g. "LOWESS" to name the smoothing.
    """
    if axis_scale == "log10":
        raw_pos = sec_raw_uS_cm > 0
        x_raw = np.log10(sec_raw_uS_cm[raw_pos])
        z_raw_plot = z_raw_bgl_m[raw_pos]

        sm_pos = sec_smooth_uS_cm > 0
        x_smooth = np.log10(sec_smooth_uS_cm[sm_pos])
        z_smooth_plot = z_smooth_bgl_m[sm_pos]

        x_label = r"$\log_{10}$(SEC) [$\mu$S/cm]"
        bp_top_col = "log10_sec_top"
        bp_bot_col = "log10_sec_bottom"
    elif axis_scale == "linear":
        x_raw, z_raw_plot = sec_raw_uS_cm, z_raw_bgl_m
        x_smooth, z_smooth_plot = sec_smooth_uS_cm, z_smooth_bgl_m
        x_label = r"SEC [$\mu$S/cm]"
        bp_top_col = "sec_top_uS_cm"
        bp_bot_col = "sec_bottom_uS_cm"
    else:
        raise ValueError(
            f"axis_scale must be 'linear' or 'log10', got {axis_scale!r}"
        )

    # Layer 1 — raw scatter
    ax.scatter(
        x_raw, z_raw_plot, s=4, color=_COLOR_RAW,
        alpha=0.45, linewidth=0, zorder=1,
        label=f"Raw ({len(z_raw_bgl_m):,})",
    )

    # Layer 2 — smoothed line
    ax.plot(
        x_smooth, z_smooth_plot, color=_COLOR_SMOOTH,
        lw=1.4, zorder=3,
        label=f"{method_label} ({len(z_smooth_bgl_m):,})",
    )

    x_lo, x_hi = ax.get_xlim()
    span = x_hi - x_lo

    # Reconstruct unique BP coordinates (in BGL).
    z_bp = list(slopes_df["depth_top"].to_numpy() + vadose_m) \
         + [float(slopes_df["depth_bottom"].iloc[-1] + vadose_m)]
    x_bp = list(slopes_df[bp_top_col].to_numpy()) \
         + [float(slopes_df[bp_bot_col].iloc[-1])]

    is_top    = slopes_df["is_top_of_mixing"].to_numpy(dtype=bool)
    is_bottom = slopes_df["is_bottom_of_mixing"].to_numpy(dtype=bool)

    # Layer 3 — chord lines between consecutive BPs, highlight TOP/BOT MZ
    for i in range(len(slopes_df)):
        x0, x1 = x_bp[i], x_bp[i + 1]
        z0, z1 = z_bp[i], z_bp[i + 1]
        chord_color, chord_lw = _COLOR_CHORD, 0.8
        if is_top[i]:
            chord_color, chord_lw = _COLOR_TOP_MZ, 1.6
        elif is_bottom[i]:
            chord_color, chord_lw = _COLOR_BOTTOM_MZ, 1.6
        ax.plot(
            [x0, x1], [z0, z1],
            color=chord_color, lw=chord_lw, alpha=0.7, zorder=4,
        )

    # Per-BP colour assignment.
    bp_colors = [_COLOR_BP_REGULAR] * len(z_bp)
    for i in range(len(slopes_df)):
        if is_top[i]:
            bp_colors[i] = _COLOR_TOP_MZ
        if is_bottom[i]:
            bp_colors[i] = _COLOR_BOTTOM_MZ

    # ── CHANGE (b): each BP keeps its left-side label position from
    # the production overlay, but the text now includes the depth in
    # metres ("BPk: z.zz m") instead of just "BPk". TOP MZ and BOT MZ
    # keep their highlighting. NO right-side labels are drawn — the
    # right side, which used to carry the slope labels in the
    # production overlay, is left empty (CHANGE a: no slope labels).
    x_anchor_left = x_lo + 0.02 * span
    for k, (xb, zb, c) in enumerate(zip(x_bp, z_bp, bp_colors), start=1):
        is_highlight = c != _COLOR_BP_REGULAR

        # Leader from BP to the LEFT edge.
        ax.plot(
            [xb, x_anchor_left], [zb, zb],
            color="#bdc3c7", lw=0.4, alpha=0.45, zorder=3,
        )

        # Label text: include depth, mark MZ BPs with prefix.
        if is_top[k - 1] if k - 1 < len(is_top) else False:
            label = f"TOP MZ \u00b7 BP{k}: {zb:.2f} m"
        elif is_bottom[k - 1] if k - 1 < len(is_bottom) else False:
            label = f"BOT MZ \u00b7 BP{k}: {zb:.2f} m"
        else:
            label = f"BP{k}: {zb:.2f} m"

        ax.text(
            x_anchor_left, zb, label,
            fontsize=_FONT_LABEL_HIGHLIGHT if is_highlight else _FONT_LABEL_REGULAR,
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

    # Layer 5 — BP markers (regular first, highlights on top).
    reg_idx = [i for i, c in enumerate(bp_colors) if c == _COLOR_BP_REGULAR]
    top_idx_l = [i for i, c in enumerate(bp_colors) if c == _COLOR_TOP_MZ]
    bot_idx_l = [i for i, c in enumerate(bp_colors) if c == _COLOR_BOTTOM_MZ]

    if reg_idx:
        ax.scatter(
            [x_bp[i] for i in reg_idx], [z_bp[i] for i in reg_idx],
            s=55, marker="D", facecolor=_COLOR_BP_REGULAR,
            edgecolor="black", linewidth=0.7, zorder=6,
            label=f"BPs (N={len(z_bp)})",
        )
    if top_idx_l:
        ax.scatter(
            [x_bp[i] for i in top_idx_l], [z_bp[i] for i in top_idx_l],
            s=120, marker="D", facecolor=_COLOR_TOP_MZ,
            edgecolor="black", linewidth=1.2, zorder=7,
            label="TOP of mixing zone",
        )
    if bot_idx_l:
        ax.scatter(
            [x_bp[i] for i in bot_idx_l], [z_bp[i] for i in bot_idx_l],
            s=120, marker="D", facecolor=_COLOR_BOTTOM_MZ,
            edgecolor="black", linewidth=1.2, zorder=7,
            label="BOTTOM of mixing zone",
        )

    ax.set_xlabel(x_label, fontsize=11)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
