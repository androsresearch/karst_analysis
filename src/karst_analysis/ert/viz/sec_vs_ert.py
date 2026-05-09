"""SEC vs ERT 1D comparison panel — main builder.

Produces a two-panel figure:

    [ SEC panel ]  [ ERT 1D panel ]
    shared Y axis (depth_bgl_m, positive down)

For one (well, transect, x, variant) combination, in either linear or
log10 SEC display. The function is pure (returns the Figure); the
batch script handles writing PNGs.

Cambios respecto al throwaway que validamos:
  (a) NO slope-pair labels (P1, P2, ...). Implemented in
      ``_sec_panel.render_sec_panel``.
  (b) Each SEC BP carries its depth in metres; labels stay on the
      LEFT of the SEC panel (production-overlay convention).
      Implemented in ``_sec_panel.render_sec_panel``.
  (c) NO confidence-interval bands on ERT BP horizontal lines.
      Just the BP marker, the horizontal line, and the right-side
      label. Implemented here in ``_render_ert_panel``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from karst_analysis.ert.io import ErtTrace1D
from karst_analysis.ert.breakpoints import ErtBreakpointFit
from karst_analysis.ert.viz._sec_panel import render_sec_panel


# Colours (kept consistent with _sec_panel for visual coherence).
_COLOR_BP_REGULAR = "#e67e22"
_COLOR_TOP_MZ     = "#c0392b"
_COLOR_BOTTOM_MZ  = "#8e44ad"
_COLOR_ERT_LINE   = "#9467bd"

_FONT_LABEL_REGULAR   = 7.0
_FONT_LABEL_HIGHLIGHT = 11.0


@dataclass(frozen=True)
class SecVsErtInputs:
    """Bundle of inputs for one SEC vs ERT figure.

    Attributes
    ----------
    well_id : str
    sec_date_str : str
        Date string used in figure title (e.g. ``"2022-01-31"``).
    z_raw_bgl_m, sec_raw_uS_cm : ndarray
        Raw YSI cast in BGL convention.
    z_smooth_bgl_m, sec_smooth_uS_cm : ndarray
        Smoothed cast (e.g. LOWESS-PAVA output) in BGL convention.
    slopes_df : pd.DataFrame
        Slopes table from ``karst_analysis.sec.slopes.compute_slopes``,
        as produced by ``scripts/slopes_batch.py`` and stored in
        ``data/slopes/<campaign>/...csv``. Carries the SEC BPs and
        the ``is_top_of_mixing`` / ``is_bottom_of_mixing`` flags.
    vadose_m : float
    sec_method : str
        e.g. ``"lowess"`` — drives panel labels.
    sec_n : int
    sec_trial_idx : int
    ert_trace : ErtTrace1D
    ert_fit : ErtBreakpointFit
    ert_top_mz_idx, ert_bot_mz_idx : Optional[int]
        Indices into ``ert_fit.breakpoints`` (0-based positional).
        Either may be None: top is None only when n_bp < 3; bot is
        None when no interior BP meets the resistivity threshold.
    ert_bot_mz_threshold : float
        Echoed in the title for traceability.
    """

    well_id: str
    sec_date_str: str
    z_raw_bgl_m: np.ndarray
    sec_raw_uS_cm: np.ndarray
    z_smooth_bgl_m: np.ndarray
    sec_smooth_uS_cm: np.ndarray
    slopes_df: pd.DataFrame
    vadose_m: float
    sec_method: str
    sec_n: int
    sec_trial_idx: int

    ert_trace: ErtTrace1D
    ert_fit: ErtBreakpointFit
    ert_top_mz_idx: Optional[int]
    ert_bot_mz_idx: Optional[int]
    ert_bot_mz_threshold: float


# ════════════════════════════════════════════════════════════════════
#  ERT panel renderer (with cambio c)
# ════════════════════════════════════════════════════════════════════
def _render_ert_panel(
    ax: Axes,
    *,
    ert_trace: ErtTrace1D,
    ert_fit: ErtBreakpointFit,
    top_idx: Optional[int],
    bot_idx: Optional[int],
    axis_scale: str,
) -> None:
    """Draw the ERT panel onto an existing Axes.

    Cambio (c): horizontal lines mark BP depths; NO CI shading.
    """
    df = ert_trace.df
    if axis_scale == "log10":
        x_curve = df["resistlog10"].to_numpy()
        x_label = r"$\log_{10}$ resistivity"
    elif axis_scale == "linear":
        x_curve = df["resist_ohm_m"].to_numpy()
        x_label = r"resistivity ($\Omega\cdot$m)"
    else:
        raise ValueError(
            f"axis_scale must be 'linear' or 'log10', got {axis_scale!r}"
        )
    z_curve = df["depth_bgl_m"].to_numpy()

    # Profile line.
    ax.plot(
        x_curve, z_curve, color=_COLOR_ERT_LINE, lw=1.4, zorder=3,
        label="ERT 1D",
    )

    # Per-BP styling.
    bps = ert_fit.breakpoints
    z_bp = bps["Breakpoint X Position"].to_numpy()
    n_bp = len(z_bp)

    bp_colors = [_COLOR_BP_REGULAR] * n_bp
    if top_idx is not None:
        bp_colors[top_idx] = _COLOR_TOP_MZ
    if bot_idx is not None:
        bp_colors[bot_idx] = _COLOR_BOTTOM_MZ

    # Right-edge label anchor.
    x_lo, x_hi = ax.get_xlim()
    span = x_hi - x_lo
    x_anchor = x_hi - 0.02 * span

    # Horizontal lines + right-side labels (NO CI shading).
    for i, (z_bp_i, c) in enumerate(zip(z_bp, bp_colors)):
        is_h = c != _COLOR_BP_REGULAR
        ax.axhline(
            z_bp_i,
            color="black",
            lw=0.4 if not is_h else 0.8,
            alpha=0.4 if not is_h else 0.8,
            zorder=2,
        )

        if i == top_idx:
            label = f"TOP MZ \u00b7 BP{i+1}: {z_bp_i:.2f} m"
        elif i == bot_idx:
            label = f"BOT MZ \u00b7 BP{i+1}: {z_bp_i:.2f} m"
        else:
            label = f"BP{i+1}: {z_bp_i:.2f} m"

        ax.text(
            x_anchor, z_bp_i, label,
            fontsize=_FONT_LABEL_HIGHLIGHT if is_h else _FONT_LABEL_REGULAR,
            fontweight="bold" if is_h else "normal",
            color=c if is_h else "#34495e",
            ha="right", va="center", zorder=7,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor=c if is_h else "0.7",
                alpha=0.92,
                linewidth=0.6 if not is_h else 0.7,
            ),
        )

    # BP markers — regular first, then highlights, so highlights sit on top.
    # X position of the marker = the curve's x-value at that depth
    # (interpolated, so markers sit ON the line).
    x_at_bp = np.interp(z_bp, z_curve, x_curve)

    reg = [i for i, c in enumerate(bp_colors) if c == _COLOR_BP_REGULAR]
    top_l = [i for i, c in enumerate(bp_colors) if c == _COLOR_TOP_MZ]
    bot_l = [i for i, c in enumerate(bp_colors) if c == _COLOR_BOTTOM_MZ]

    if reg:
        ax.scatter(
            [x_at_bp[i] for i in reg], [z_bp[i] for i in reg],
            s=55, marker="D", facecolor=_COLOR_BP_REGULAR,
            edgecolor="black", linewidth=0.7, zorder=6,
            label=f"BPs (N={n_bp})",
        )
    if top_l:
        ax.scatter(
            [x_at_bp[i] for i in top_l], [z_bp[i] for i in top_l],
            s=120, marker="D", facecolor=_COLOR_TOP_MZ,
            edgecolor="black", linewidth=1.2, zorder=7,
            label="TOP of mixing zone",
        )
    if bot_l:
        ax.scatter(
            [x_at_bp[i] for i in bot_l], [z_bp[i] for i in bot_l],
            s=120, marker="D", facecolor=_COLOR_BOTTOM_MZ,
            edgecolor="black", linewidth=1.2, zorder=7,
            label="BOTTOM of mixing zone",
        )

    ax.set_xlabel(x_label, fontsize=11)
    ax.grid(True, ls=":", alpha=0.4)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)


# ════════════════════════════════════════════════════════════════════
#  Public entry point
# ════════════════════════════════════════════════════════════════════
def plot_sec_vs_ert(
    inputs: SecVsErtInputs,
    *,
    axis_scale: str,
    depth_top_m: float = 0.0,
    depth_bottom_m: float = 35.0,
    figsize: tuple[float, float] = (15.0, 11.0),
    width_ratios: tuple[float, float] = (1.0, 0.85),
) -> Figure:
    """Build the SEC-vs-ERT two-panel figure.

    Parameters
    ----------
    inputs : SecVsErtInputs
        All payload (data + decisions).
    axis_scale : {"linear", "log10"}
        Display scale for BOTH panels' x-axis. Breakpoints themselves
        are detected upstream in log10 regardless.
    depth_top_m, depth_bottom_m : float
        Shared Y-axis limits in BGL convention. Default 0..35 m
        matches the SEC fixtures used during v17 validation.
    figsize, width_ratios
        Standard Matplotlib knobs.

    Returns
    -------
    Figure
        Caller is responsible for ``savefig`` / ``close``.
    """
    fig, (ax_sec, ax_ert) = plt.subplots(
        1, 2, figsize=figsize,
        gridspec_kw={"width_ratios": list(width_ratios)},
        sharey=True,
    )
    ax_sec.set_ylim(depth_bottom_m, depth_top_m)
    ax_sec.set_ylabel("depth_bgl_m  (positive down)", fontsize=11)

    render_sec_panel(
        ax_sec,
        z_raw_bgl_m=inputs.z_raw_bgl_m,
        sec_raw_uS_cm=inputs.sec_raw_uS_cm,
        z_smooth_bgl_m=inputs.z_smooth_bgl_m,
        sec_smooth_uS_cm=inputs.sec_smooth_uS_cm,
        slopes_df=inputs.slopes_df,
        vadose_m=inputs.vadose_m,
        axis_scale=axis_scale,
        method_label=inputs.sec_method.upper(),
    )
    ax_sec.set_title(
        f"SEC \u2014 {inputs.well_id} \u2014 {inputs.sec_date_str} \u2014 "
        f"{inputs.sec_method.upper()} N={inputs.sec_n} "
        f"trial_{inputs.sec_trial_idx}",
        fontsize=10, fontweight="bold",
    )

    _render_ert_panel(
        ax_ert,
        ert_trace=inputs.ert_trace,
        ert_fit=inputs.ert_fit,
        top_idx=inputs.ert_top_mz_idx,
        bot_idx=inputs.ert_bot_mz_idx,
        axis_scale=axis_scale,
    )
    ax_ert.set_title(
        f"ERT \u2014 {inputs.ert_trace.transect} "
        f"x={inputs.ert_trace.x_requested:g} \u2014 "
        f"{inputs.ert_trace.variant} \u2014 "
        f"N={inputs.ert_fit.n_breakpoints} "
        f"(detected in log10, seed={inputs.ert_fit.seed_used})\n"
        f"BOT MZ threshold: \u03c1 \u2264 {inputs.ert_bot_mz_threshold:g} "
        f"\u03a9\u00b7m",
        fontsize=10, fontweight="bold",
    )

    fig.suptitle(
        f"SEC vs ERT 1D \u2014 {inputs.well_id} vs "
        f"{inputs.ert_trace.transect}@x{inputs.ert_trace.x_requested:g} "
        f"\u2014 {axis_scale} display \u2014 shared depth axis "
        f"({depth_top_m:.0f}..{depth_bottom_m:.0f} m BGL)",
        fontsize=12,
    )
    fig.tight_layout()
    return fig
