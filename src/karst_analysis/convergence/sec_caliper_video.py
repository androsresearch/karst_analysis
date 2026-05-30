"""SEC + caliper × video-log × Ardaman panel per priority well.

Extends ``caliper_video.plot_caliper_video_panel`` (v5.1) by adding the
SEC profile and its breakpoints to the central panel via a twin x-axis.

Layout (3 columns)
------------------
::

    | BP labels  |  caliper + SEC + severity bands  |  video-log notes  |
    |  width=0.5 |  width=1.6                        |  width=3.2        |

The y-axis ticks are attached to the caliper panel (column 2). The y-axis
title ("Depth below ground level (m)") is rendered as a manual rotated
text on the figure's leftmost edge (not as a matplotlib ylabel) to avoid
collisions with the BP labels.

Key visual decisions
--------------------
* Caliper x-axis: bottom of the panel, brown (``#8e6914``).
* SEC x-axis:    top of the panel, dark blue (``#1d4ed8``), via twin axis.
* Severity bands appear ONLY in the caliper panel (column 2). The video
  panel (column 3) deliberately omits them: video-log notes are placed
  with PAV-isotonic offsets, so they don't actually align with the
  severities at their displayed y-position. Bands behind misaligned
  notes would be misleading.
* BP markers: orange diamonds (``#ff7f0e``) with black edge, plotted on
  the SEC twin axis at ``(sec_at_bp_uS_cm, depth_bgl_m)``.
* BP guide lines: dotted neutral grey (``#9ca3af``), drawn only in the
  caliper panel — distinct from the orange/red severity palette so the
  reader doesn't mistake them for severity hints.
* BP labels: live in column 1, right-aligned, with a short leader
  drawn when the PAV displacement is meaningful.

Outputs
-------
One PNG per (well, smoothing, n) combination. With 5 wells × 2 smoothing
methods × 10 N values, a full batch produces 100 PNGs.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from karst_analysis.caliper.io import (
    DEFAULT_MASTER_CSV, DEFAULT_PERPOINT_CSV, load_perpoint,
)
from karst_analysis.convergence._layout import (
    build_label_text, draw_bracket, minimum_displacement_positions,
)
from karst_analysis.convergence.caliper_video import (
    COMPANION_STYLE, SEVERITY_ALPHAS, SEVERITY_COLORS, WELLS,
    _caliper_from_perpoint, _draw_severity_bands,
    _load_companions_caliper, _site_prefix,
)
from karst_analysis.drilling.io import DEFAULT_ARDAMAN_CSV, load_ardaman
from karst_analysis.sec.export.api import (
    load_breakpoints_at_n, load_sec_profile,
)
from karst_analysis.videolog.io import DEFAULT_VIDEOLOG_XLSX, load_video_notes


# ──────────────────────────────────────────────────────────────────────
#  Visual constants
# ──────────────────────────────────────────────────────────────────────
SEC_COLOR: str = "#1d4ed8"        # dark blue — does not clash with severity yellows/reds
BP_COLOR: str = "#ff7f0e"         # orange — regular BPs (no mixing-zone flag)
BP_COLOR_TOP_MZ: str = "#c0392b"  # deep red — TOP of mixing zone (matches slopes_overlay)
BP_COLOR_BOT_MZ: str = "#8e44ad"  # purple   — BOTTOM of mixing zone (matches slopes_overlay)
BP_GUIDE_COLOR: str = "#9ca3af"   # neutral grey for the dotted depth guide line


# ──────────────────────────────────────────────────────────────────────
#  Mixing-zone flag loader
# ──────────────────────────────────────────────────────────────────────
def _load_mixing_zone_bp_flags(
    *,
    well_id: str,
    campaign: str,
    method: str,
    n: int,
    trial: str,
    project_root: Path,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read the mixing-zone flags for a (well, method, trial, N) job.

    Looks up the slopes CSV produced by ``scripts/slopes_batch.py``::

        data/slopes/<campaign>/{well_id}_{date}__slopes-{method}-N{n}-t{idx}.csv

    and maps its per-pair flags ``is_top_of_mixing`` / ``is_bottom_of_mixing``
    onto a per-breakpoint mask of length ``n``.

    Mapping rule (same as :mod:`karst_analysis.sec.viz.slopes_overlay`):
    pair *i* (0-based) has ``depth_top`` equal to the *i*-th breakpoint in
    depth-ascending order, so ``is_top_of_mixing[i] == True`` flags the
    (*i*+1)-th BP (1-based) as TOP of the mixing zone. The very last
    breakpoint (index ``n``) corresponds to a ``depth_bottom`` and is
    therefore never flagged.

    Returns
    -------
    (mask_top, mask_bot) : tuple of np.ndarray or None
        Boolean arrays of length ``n`` aligned with ``bp_df.bp_index``
        (1..n). ``(None, None)`` if the slopes CSV is missing — the
        caller should then render the panel without mixing-zone colours
        and emit a warning.
    """
    from karst_analysis.sec.jobs_io import trial_index

    idx = trial_index(trial)
    folder = project_root / "data" / "slopes" / campaign
    if not folder.is_dir():
        return None, None

    pattern = f"{well_id}_*__slopes-{method}-N{n}-t{idx}.csv"
    candidates = sorted(folder.glob(pattern))
    if not candidates:
        return None, None

    # Most recent wins (consistent with other resolvers in the package).
    csv_path = max(candidates, key=lambda p: p.stat().st_mtime)
    slopes_df = pd.read_csv(csv_path)

    expected_rows = n - 1
    if len(slopes_df) != expected_rows:
        # Mismatch between N and the chord-pair count — refuse to guess.
        return None, None

    top_pairs = slopes_df["is_top_of_mixing"].to_numpy(dtype=bool)
    bot_pairs = slopes_df["is_bottom_of_mixing"].to_numpy(dtype=bool)

    # Promote pair-aligned flags to per-BP flags of length n.
    mask_top = np.zeros(n, dtype=bool)
    mask_bot = np.zeros(n, dtype=bool)
    mask_top[:expected_rows] = top_pairs
    mask_bot[:expected_rows] = bot_pairs
    return mask_top, mask_bot


# ──────────────────────────────────────────────────────────────────────
#  Plot configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SecCaliperVideoConfig:
    """Visual parameters for the SEC + caliper × video panel.

    The defaults are tuned to look correct at the typical figure size
    (14 × auto-height), but every numeric parameter can be overridden
    when the figure size or aspect ratio is unusual.
    """
    base_figsize: tuple = (14, 14)
    width_ratios: tuple = (0.5, 1.6, 3.2)   # (BP labels, caliper+SEC, video)
    note_wrap: int = 140
    note_fontsize: float = 8.5
    bp_fontsize: float = 8.5
    bracket_lw: float = 1.3
    bracket_tip: float = 0.008
    leader_color: str = "#888888"
    leader_lw: float = 0.55
    note_color: str = "#1a1a1a"
    ardaman_color_lith: str = "#1d4ed8"
    ardaman_color_cond: str = "#0f7a4d"
    grid_alpha: float = 0.22
    outlier_color: str = "#7a1ea8"
    auto_height_factor: float = 0.30
    auto_height_safety:  float = 1.50


# ──────────────────────────────────────────────────────────────────────
#  Internal helper — build the unified video+ardaman entries DataFrame.
#  This is a verbatim copy of the logic in caliper_video.py, but it
#  lives here too because the SEC panel's ``_build_entries`` is called
#  separately. Future refactor: factor it out into _layout.py.
# ──────────────────────────────────────────────────────────────────────
def _build_entries(
    notes_df: pd.DataFrame, ardaman_df: pd.DataFrame,
) -> pd.DataFrame:
    """Concatenate video notes and Ardaman entries into one DataFrame.

    Uses the BGL-positive depth columns (``depth_*_bgl_m``).
    Returns an empty DataFrame if both inputs are empty.
    """
    parts: list[pd.DataFrame] = []
    if not notes_df.empty:
        nd = notes_df.copy()
        nd["kind"] = "note"
        parts.append(nd[["depth_top_m", "depth_bot_m",
                         "depth_top_bgl_m", "depth_bot_bgl_m",
                         "depth_centre_bgl_m", "kind", "note"]]
                     .rename(columns={"note": "text"}))
    if not ardaman_df.empty:
        ad = ardaman_df.copy()
        ad["kind"] = ad["kind"].map({
            "lithology":            "ardaman_lith",
            "conductivity_in_situ": "ardaman_cond",
        })
        parts.append(ad[["depth_top_m", "depth_bot_m",
                         "depth_top_bgl_m", "depth_bot_bgl_m",
                         "depth_centre_bgl_m", "kind", "text"]])
    if not parts:
        return pd.DataFrame()
    return (pd.concat(parts, ignore_index=True)
              .sort_values("depth_centre_bgl_m", ascending=True)
              .reset_index(drop=True))


# ──────────────────────────────────────────────────────────────────────
#  The plot
# ──────────────────────────────────────────────────────────────────────
def plot_sec_caliper_video_panel(
    well_id: str,
    *,
    smoothing: str,
    n: int,
    campaign: str = "2022_02",
    trial: str = "trial_1",
    perpoint_csv: Optional[str | Path] = None,
    video_xlsx: Optional[str | Path] = None,
    ardaman_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path] = None,
    config: Optional[SecCaliperVideoConfig] = None,
    output_path: Optional[str | Path] = None,
    sat_cm: float = 32.5,
) -> plt.Figure:
    """Render the SEC + caliper × video-log panel for one well/N/method.

    Parameters
    ----------
    well_id : str
        Must be a key of ``WELLS`` (e.g. ``"LRS70D"``).
    smoothing : {"savgol", "lowess"}
        Which smoothed SEC profile to overlay.
    n : int
        Which N (number of breakpoints) from the BIC sweep to display.
        Markers and labels are drawn for every BP from 1 to n.
    campaign : str, default ``"2022_02"``
        Field campaign — must match the directory under
        ``data/processed/sec/<campaign>/``.
    trial : str, default ``"trial_1"``
        Which trial of the BIC sweep to read. ``"best_bic"`` lets the
        loader pick the trial with the lowest BIC at this N. When a
        matching slopes CSV exists at
        ``data/slopes/<campaign>/{well}_{date}__slopes-{method}-N{n}-t{idx}.csv``
        the BPs flagged as TOP / BOTTOM of mixing zone are coloured
        red / purple respectively (matching ``slopes_overlay``); if the
        slopes CSV is missing, all BPs render as plain orange diamonds
        and a warning is printed.
    perpoint_csv, video_xlsx, ardaman_csv, master_caliper_csv : path-like, optional
        Override input paths. ``ardaman_csv`` is consulted only for
        wells whose ``WellConfig`` has ``has_ardaman=True``.
    project_root : Path, optional
        Project root for SEC artefact lookup. Defaults to ``Path.cwd()``.
    config : SecCaliperVideoConfig, optional
        Visual parameters override.
    output_path : path-like, optional
        Save the figure to this path (PNG). Parent dirs are created.
    sat_cm : float, default 32.5
        Caliper saturation (drawn as a faint vertical reference).

    Returns
    -------
    matplotlib.figure.Figure
        The caller is responsible for closing it.

    Raises
    ------
    KeyError
        If ``well_id`` is not in ``WELLS``.
    FileNotFoundError
        If the SEC profile or breakpoint JSON is missing for the
        requested (well, campaign, smoothing, n) combination.
    ValueError
        From the underlying loaders if N exceeds the BIC sweep range,
        or if the requested trial is not present.
    """
    cfg = config or SecCaliperVideoConfig()
    if well_id not in WELLS:
        raise KeyError(
            f"Unknown well '{well_id}'. Known wells: {list(WELLS.keys())}"
        )
    wc = WELLS[well_id]

    # Default paths
    if perpoint_csv is None:        perpoint_csv = DEFAULT_PERPOINT_CSV
    if video_xlsx is None:          video_xlsx = DEFAULT_VIDEOLOG_XLSX
    if master_caliper_csv is None:  master_caliper_csv = DEFAULT_MASTER_CSV
    if ardaman_csv is None and wc.has_ardaman:
        ardaman_csv = DEFAULT_ARDAMAN_CSV

    # ── Load all sources ────────────────────────────────────────────
    perpoint_full = load_perpoint(perpoint_csv)
    perpoint_df = (perpoint_full[perpoint_full["well"] == wc.caliper_well]
                   .sort_values("depth_m", ascending=True)
                   .reset_index(drop=True))
    if perpoint_df.empty:
        raise ValueError(
            f"No perpoint rows for '{wc.caliper_well}' in {perpoint_csv}"
        )
    cal_df = _caliper_from_perpoint(perpoint_df, auger_cm=wc.auger_cm)

    notes_df = load_video_notes(video_xlsx, sheet=wc.video_sheet)
    ardaman_df = (load_ardaman(ardaman_csv, well=wc.video_well)
                  if (wc.has_ardaman and ardaman_csv is not None)
                  else pd.DataFrame())
    companions = (_load_companions_caliper(master_caliper_csv,
                                           primary_well=wc.caliper_well)
                  if master_caliper_csv is not None else {})

    sec_df = load_sec_profile(
        well_id=wc.caliper_well, campaign=campaign,
        smoothing=smoothing, project_root=project_root,
    )
    bp_df = load_breakpoints_at_n(
        well_id=wc.caliper_well, campaign=campaign,
        smoothing=smoothing, n=n, trial=trial,
        project_root=project_root,
    )

    # Mixing-zone flags (None,None) when no slopes CSV is on disk —
    # render then degrades gracefully to a single-colour BP panel.
    mz_root = project_root if project_root is not None else Path.cwd()
    mask_top_mz, mask_bot_mz = _load_mixing_zone_bp_flags(
        well_id=wc.caliper_well, campaign=campaign,
        method=smoothing, n=n, trial=trial,
        project_root=mz_root,
    )
    has_mz = mask_top_mz is not None and mask_bot_mz is not None
    if not has_mz:
        print(
            f"  [warn] no slopes CSV found for {wc.caliper_well} "
            f"{smoothing} N={n} {trial} — BPs rendered without "
            f"mixing-zone colours. Run scripts/slopes_batch.py first."
        )

    entries = _build_entries(notes_df, ardaman_df)

    # ── Figure size scales with #notes for readability ──────────────
    n_entries = len(entries)
    base_w, base_h = cfg.base_figsize
    needed_h = cfg.auto_height_factor * cfg.auto_height_safety * n_entries
    fig_h = max(base_h, needed_h)
    figsize = (base_w, fig_h)

    fig, (ax_bp, ax_cal, ax_note) = plt.subplots(
        1, 3, figsize=figsize, sharey=True,
        gridspec_kw=dict(width_ratios=cfg.width_ratios, wspace=0.0),
    )

    # ── Y-limits from union of all sources ──────────────────────────
    ymin_c = [cal_df["depth_m"].min(), float(sec_df["depth_bgl_m"].min())]
    ymax_c = [cal_df["depth_m"].max(), float(sec_df["depth_bgl_m"].max())]
    if not entries.empty:
        ymin_c.append(float(entries["depth_centre_bgl_m"].min()))
        ymax_c.append(float(entries["depth_centre_bgl_m"].max()))
    y_min = min(ymin_c) - 0.8
    y_max = max(ymax_c) + 0.8
    ax_cal.set_ylim(y_min, y_max)

    # ── Column 1: BP labels (no spines, no axis labels) ─────────────
    ax_bp.set_xlim(0, 1)
    ax_bp.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_bp.tick_params(axis="y", which="both", left=False, labelleft=False)
    for spine in ("top", "right", "bottom", "left"):
        ax_bp.spines[spine].set_visible(False)

    # ── Column 2: caliper + SEC ─────────────────────────────────────
    _draw_severity_bands(ax_cal, perpoint_df, alpha_factor=1.0)

    for cmp_name, cmp_df in companions.items():
        suffix = cmp_name[len(_site_prefix(cmp_name)):]
        style = COMPANION_STYLE.get(suffix)
        if style is None:
            continue
        ax_cal.plot(cmp_df["caliper_cm"].to_numpy(),
                    cmp_df["depth_m"].to_numpy(),
                    label=cmp_name, zorder=3, **style)

    z = cal_df["depth_m"].to_numpy()
    raw = cal_df["raw_caliper_cm"].to_numpy()
    out_mask = cal_df["is_outlier"].to_numpy().astype(bool)
    cal_plot = np.where(out_mask, np.nan, raw)
    ax_cal.plot(cal_plot, z, color="#8e6914", lw=0.6, alpha=0.85,
                zorder=4, label=wc.caliper_well)
    if out_mask.any():
        ax_cal.scatter(raw[out_mask], z[out_mask], s=40, marker="x",
                       c=cfg.outlier_color, zorder=5, linewidths=1.0)
    ax_cal.axvline(sat_cm, color="#777777", lw=0.6, ls=":",
                   alpha=0.55, zorder=2)
    ax_cal.set_xlabel("Caliper aperture (cm)", fontsize=10, color="#8e6914")
    ax_cal.tick_params(axis="x", colors="#8e6914", labelsize=8)
    ax_cal.tick_params(axis="y", labelsize=8, left=True, labelleft=True)
    # Suppress the matplotlib ylabel — the y-title is drawn manually as
    # fig.text() to avoid collisions with the BP labels in column 1.
    ax_cal.set_ylabel("")

    all_caliper_mins = [
        np.nanmin(raw[~out_mask]) if (~out_mask).any() else np.nanmax(raw)
    ]
    all_caliper_maxs = [np.nanmax(raw)]
    for cmp_df in companions.values():
        c = cmp_df["caliper_cm"].to_numpy()
        if c.size:
            all_caliper_mins.append(float(np.nanmin(c)))
            all_caliper_maxs.append(float(np.nanmax(c)))
    x_lo = min(min(all_caliper_mins) - 1.0, wc.auger_cm - 1.0)
    x_hi = max(max(all_caliper_maxs), sat_cm) + 1.5
    ax_cal.set_xlim(x_lo, x_hi)
    ax_cal.grid(True, axis="y", alpha=cfg.grid_alpha, linestyle=":")

    # SEC twin axis (top of caliper panel)
    ax_sec = ax_cal.twiny()
    ax_sec.plot(sec_df["sec_uS_cm"].to_numpy(),
                sec_df["depth_bgl_m"].to_numpy(),
                color=SEC_COLOR, lw=1.2, alpha=0.95, zorder=4.5)
    ax_sec.set_xlabel(f"SEC (µS/cm) — {smoothing}", fontsize=10, color=SEC_COLOR)
    ax_sec.tick_params(axis="x", colors=SEC_COLOR, labelsize=8)
    sec_min = float(sec_df["sec_uS_cm"].min())
    sec_max = float(sec_df["sec_uS_cm"].max())
    sec_range = sec_max - sec_min
    ax_sec.set_xlim(sec_min - 0.05*sec_range, sec_max + 0.05*sec_range)

    # BP markers on SEC twin — coloured per mixing-zone flag.
    bp_z = bp_df["depth_bgl_m"].to_numpy()
    bp_sec = bp_df["sec_at_bp_uS_cm"].to_numpy()
    n_bp = len(bp_df)
    if has_mz:
        regular_mask = ~(mask_top_mz | mask_bot_mz)
    else:
        regular_mask = np.ones(n_bp, dtype=bool)

    # Per-BP face colour, used both for the diamonds and (below) for
    # the BP-index label boxes in column 1 — so the chart reads
    # consistently with slopes_overlay (same red / purple / orange).
    bp_facecolor = np.full(n_bp, BP_COLOR, dtype=object)
    if has_mz:
        bp_facecolor[mask_top_mz] = BP_COLOR_TOP_MZ
        bp_facecolor[mask_bot_mz] = BP_COLOR_BOT_MZ

    # Plot regular BPs first, then the flagged ones, so the highlights
    # sit on top (larger marker + thicker edge for emphasis).
    if regular_mask.any():
        ax_sec.scatter(
            bp_sec[regular_mask], bp_z[regular_mask], marker="D", s=60,
            facecolor=BP_COLOR, edgecolor="black", lw=0.8,
            zorder=8, clip_on=False,
        )
    if has_mz and mask_top_mz.any():
        ax_sec.scatter(
            bp_sec[mask_top_mz], bp_z[mask_top_mz], marker="D", s=110,
            facecolor=BP_COLOR_TOP_MZ, edgecolor="black", lw=1.1,
            zorder=9, clip_on=False,
        )
    if has_mz and mask_bot_mz.any():
        ax_sec.scatter(
            bp_sec[mask_bot_mz], bp_z[mask_bot_mz], marker="D", s=110,
            facecolor=BP_COLOR_BOT_MZ, edgecolor="black", lw=1.1,
            zorder=9, clip_on=False,
        )

    # ── BP labels in column 1 (ax_bp) ───────────────────────────────
    if n_bp > 0:
        # Rough y → display unit conversion for label-height calibration.
        # Same reasoning as caliper_video.py.
        y_span = y_max - y_min
        axis_h_in = figsize[1] * 0.90
        pt_per_data = 72 * axis_h_in / y_span
        line_h_data = (cfg.bp_fontsize * 1.30) / pt_per_data
        bp_half_h = 0.5 * (line_h_data + 0.7 * line_h_data)
        anchors = bp_z
        text_y = minimum_displacement_positions(
            anchors, np.full(n_bp, bp_half_h),
            y_lo=y_min + 0.4, y_hi=y_max - 0.4,
            pad=0.04 * line_h_data,
        )
        for i, (bp_idx, z_bp, ty) in enumerate(zip(
            bp_df["bp_index"], bp_z, text_y,
        )):
            # Dotted grey guide line in caliper panel only.
            ax_cal.axhline(z_bp, color=BP_GUIDE_COLOR, lw=0.5, ls=":",
                           alpha=0.55, zorder=2)
            # If PAV pushed the label, draw a short leader from the
            # anchor depth to the displaced label position.
            if abs(ty - z_bp) > 0.3:
                ax_bp.plot([0.92, 0.99], [ty, z_bp],
                           color=BP_GUIDE_COLOR, lw=0.4,
                           alpha=0.7, zorder=3)
            # Inherit the marker colour so the label reads consistently
            # with the diamond it identifies (red / purple / orange).
            label_color = bp_facecolor[i]
            ax_bp.annotate(
                f"BP{bp_idx}: {z_bp:.1f} m",
                xy=(0.95, ty), xycoords=("axes fraction", "data"),
                ha="right", va="center", fontsize=cfg.bp_fontsize,
                color=label_color, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                          edgecolor=label_color, alpha=0.95, lw=0.6),
                zorder=10,
            )

    # ── Legend (lower-left so it doesn't compete with BPs or SEC) ──
    handles = [
        Line2D([0], [0], color="#8e6914", lw=1.2,
               label=f"{wc.caliper_well} caliper"),
    ]
    for cmp_name in companions:
        suffix = cmp_name[len(_site_prefix(cmp_name)):]
        style = COMPANION_STYLE.get(suffix, {})
        handles.append(Line2D([0], [0], color=style.get("color", "#000"),
                              lw=max(style.get("lw", 1.0), 1.0),
                              alpha=style.get("alpha", 1.0),
                              label=cmp_name))
    handles.append(Line2D([0], [0], color=SEC_COLOR, lw=1.2,
                          label=f"SEC ({smoothing})"))
    handles.append(Line2D([0], [0], marker="D", color=BP_COLOR,
                          mec="black", mew=0.8, ms=6, ls="None",
                          label=f"BP (N={n})"))
    if has_mz and mask_top_mz.any():
        handles.append(Line2D([0], [0], marker="D",
                              markerfacecolor=BP_COLOR_TOP_MZ,
                              markeredgecolor="black", mew=1.1, ms=8,
                              ls="None", color="none",
                              label="TOP of mixing zone"))
    if has_mz and mask_bot_mz.any():
        handles.append(Line2D([0], [0], marker="D",
                              markerfacecolor=BP_COLOR_BOT_MZ,
                              markeredgecolor="black", mew=1.1, ms=8,
                              ls="None", color="none",
                              label="BOTTOM of mixing zone"))
    for sev in ("mild", "moderate", "severe"):
        if (perpoint_df["severity_per_sample"] == sev).any():
            handles.append(Patch(facecolor=SEVERITY_COLORS[sev],
                                 alpha=SEVERITY_ALPHAS[sev],
                                 edgecolor="none", label=sev.capitalize()))
    ax_cal.legend(handles=handles, loc="lower left", fontsize=7.5,
                  framealpha=0.92, edgecolor="#cccccc",
                  handlelength=1.5, handleheight=1.0, borderpad=0.4)

    # ── Column 3: video log + Ardaman entries (NO severity bands) ──
    ax_note.set_xlim(0, 1)
    ax_note.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_note.tick_params(axis="y", which="both", left=False, labelleft=False)
    for spine in ("top", "right", "bottom"):
        ax_note.spines[spine].set_visible(False)
    ax_note.spines["left"].set_color("#bbbbbb")
    # Intentionally no _draw_severity_bands call here. Notes are
    # PAV-displaced from their anchor depth, so bands would be
    # misaligned with the apparent text position.

    if not entries.empty:
        labels = [build_label_text(row) for _, row in entries.iterrows()]
        colors_e = entries["kind"].map({
            "note":         cfg.note_color,
            "ardaman_lith": cfg.ardaman_color_lith,
            "ardaman_cond": cfg.ardaman_color_cond,
        }).tolist()
        styles = entries["kind"].map({
            "note":         "normal",
            "ardaman_lith": "italic",
            "ardaman_cond": "italic",
        }).tolist()
        wrapped = [textwrap.fill(s, width=cfg.note_wrap) for s in labels]
        line_counts = np.array([w.count("\n") + 1 for w in wrapped])
        y_span = y_max - y_min
        axis_h_in = figsize[1] * 0.90
        pt_per_data = 72 * axis_h_in / y_span
        line_h_data = (cfg.note_fontsize * 1.30) / pt_per_data
        label_heights = line_counts * line_h_data + 0.7 * line_h_data
        half_heights = 0.5 * label_heights
        anchors = entries["depth_centre_bgl_m"].to_numpy()
        text_y = minimum_displacement_positions(
            anchors, half_heights,
            y_lo=y_min + 0.4, y_hi=y_max - 0.4,
            pad=0.04 * line_h_data,
        )
        x_anchor = 0.005
        x_kink   = 0.025
        x_text   = 0.05
        for i, (_, row) in enumerate(entries.iterrows()):
            e_top = row["depth_top_bgl_m"]
            e_bot = row["depth_bot_bgl_m"]
            anchor_y = row["depth_centre_bgl_m"]
            is_interval = (np.isfinite(e_bot)
                           and abs(e_top - e_bot) > 1e-6)
            if is_interval:
                draw_bracket(ax_note, x_anchor,
                             x_anchor + cfg.bracket_tip,
                             e_top, e_bot,
                             color=colors_e[i], lw=cfg.bracket_lw)
                leader_x_start = x_anchor + cfg.bracket_tip
            else:
                ax_note.plot(x_anchor, anchor_y, marker="o", ms=2.8,
                             color=colors_e[i], zorder=3)
                leader_x_start = x_anchor
            ax_note.plot([leader_x_start, x_kink, x_text],
                         [anchor_y, anchor_y, text_y[i]],
                         color=cfg.leader_color, lw=cfg.leader_lw,
                         alpha=0.85, zorder=2)
            ax_note.text(x_text + 0.005, text_y[i], wrapped[i],
                         ha="left", va="center",
                         fontsize=cfg.note_fontsize,
                         color=colors_e[i], fontstyle=styles[i], zorder=4,
                         bbox=dict(boxstyle="round,pad=0.28",
                                   facecolor="white", edgecolor="#dddddd",
                                   alpha=0.88, lw=0.5))

    right_label = "Video-log observations"
    if wc.has_ardaman:
        right_label += " (black) + Ardaman 2009 (blue / green)"
    ax_note.set_xlabel(right_label, fontsize=10)

    # ── Title (preserves the v5.1 fix for the truncated title bug) ──
    # ``trial`` is rendered humanised: "trial_3" → "trial 3", "best_bic"
    # passes through verbatim.
    trial_label = trial.replace("_", " ") if trial.startswith("trial_") else trial
    if wc.video_well == wc.caliper_well:
        title = (
            f"Well {wc.caliper_well} — caliper + SEC "
            f"({smoothing}, N={n}, {trial_label})"
            f" + video-log"
        )
    else:
        ard_note = (f" + Ardaman 2009 ({wc.video_well})"
                    if wc.has_ardaman else "")
        title = (
            f"Well {wc.caliper_well} caliper + SEC "
            f"({smoothing}, N={n}, {trial_label})"
            f" × video {wc.video_well}{ard_note}"
        )

    title_y = 1.0 - 0.4 / figsize[1]
    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=title_y)
    top_margin = max(0.92, 1.0 - 1.0 / figsize[1])
    fig.subplots_adjust(left=0.06, right=0.985, top=top_margin, bottom=0.05)

    ax_cal.invert_yaxis()  # BGL-positive: 0 at the top of the figure

    # Y-axis title rendered as figure-level rotated text (NOT as
    # ax.set_ylabel) so it lives outside the BP-label column without
    # competing with it.
    fig.text(
        0.012, (top_margin + 0.05) / 2,
        "Depth below ground level (m)",
        rotation=90, ha="center", va="center", fontsize=10,
    )

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────
#  Batch builder
# ──────────────────────────────────────────────────────────────────────
def build_all_sec_caliper_video_panels(
    *,
    wells: Optional[list[str]] = None,
    smoothings: tuple[str, ...] = ("savgol", "lowess"),
    n_min: int = 1,
    n_max: int = 10,
    trial: str = "trial_1",
    jobs: Optional[list[dict]] = None,
    campaign: str = "2022_02",
    output_dir: Optional[str | Path] = None,
    perpoint_csv: Optional[str | Path] = None,
    video_xlsx: Optional[str | Path] = None,
    ardaman_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path] = None,
    config: Optional[SecCaliperVideoConfig] = None,
) -> list[Path]:
    """Render the SEC + caliper × video panel for every (well, smoothing, n, trial).

    Two driving modes
    -----------------

    1. **Jobs mode** (preferred for the thesis chapter) — pass a list of
       dicts via ``jobs``, each with keys ``well``, ``method``,
       ``trial``, ``n``. One panel is produced per job, honouring the
       trial choice made by the user (e.g. via
       ``config/slopes_jobs_2022_02.yml``). When ``jobs`` is given,
       ``wells``, ``smoothings``, ``n_min``, ``n_max``, and ``trial``
       are ignored.

    2. **Legacy grid mode** — pass ``wells`` × ``smoothings`` ×
       ``[n_min..n_max]``, all rendered with the same ``trial`` (default
       ``"trial_1"``). Useful for sensitivity sweeps where the trial is
       held fixed.

    Output files are organised one folder per well, under a campaign
    subfolder (v13). Filenames **always** carry an explicit ``__t{idx}``
    suffix so two trials at the same N never collide::

        results/figures/convergence/sec_caliper_video/<campaign>/
            LRS70D/
                LRS70D_20220131__savgol__N01__t1.png
                ...
                LRS70D_20220131__lowess__N15__t3.png
            AW5D/
                ...

    Failures on individual combinations are logged but don't abort the
    batch — useful when a particular N exceeds the BIC sweep range for
    some wells but not others.

    Parameters
    ----------
    wells : list[str], optional
        (Legacy mode) Subset of wells. Defaults to all keys of ``WELLS``.
    smoothings : tuple[str, ...]
        (Legacy mode) Subset of {"savgol", "lowess"}.
    n_min, n_max : int
        (Legacy mode) Inclusive range of N to render.
    trial : str, default "trial_1"
        (Legacy mode) Trial to use for every (well, smoothing, n).
    jobs : list[dict], optional
        Jobs mode. Each item must have keys ``well``, ``method``,
        ``trial``, ``n``. Other keys are ignored. If given, supersedes
        all four legacy-mode arguments.
    campaign : str
        Field campaign — see ``plot_sec_caliper_video_panel``.
    output_dir : path-like, optional
        Root output directory. Defaults to
        ``results/figures/convergence/sec_caliper_video/<campaign>/``.
    project_root : Path, optional
        Project root for SEC artefact lookup. Defaults to ``Path.cwd()``.
    Other args:
        Passed through to ``plot_sec_caliper_video_panel``.

    Returns
    -------
    list[Path]
        Absolute paths of PNGs actually written.
    """
    from karst_analysis.sec.jobs_io import trial_index

    if output_dir is None:
        from karst_analysis.io import resolve_figure_dir
        output_dir = resolve_figure_dir(
            "convergence/sec_caliper_video",
            campaigns=[campaign],
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build the unified list of (well, method, trial, n) tasks.
    tasks: list[tuple[str, str, str, int]] = []
    if jobs is not None:
        for j in jobs:
            tasks.append((str(j["well"]), str(j["method"]),
                          str(j["trial"]), int(j["n"])))
    else:
        target_wells = wells if wells is not None else list(WELLS.keys())
        for w in target_wells:
            for smoothing in smoothings:
                for n in range(n_min, n_max + 1):
                    tasks.append((w, smoothing, trial, n))

    written: list[Path] = []
    n_skipped = 0

    # Group tasks by well for nicer console output.
    current_well: Optional[str] = None
    date_cache: dict[tuple[str, str], str] = {}  # (well, smoothing) -> date

    for w, smoothing, trial_name, n in tasks:
        if w not in WELLS:
            print(f"[{w}]  unknown well — skipping")
            n_skipped += 1
            continue
        if w != current_well:
            print(f"\n[{w}]")
            current_well = w

        # Resolve the date suffix from a sample SEC profile filename
        # — only used to derive the date stamp in the output filename.
        cache_key = (w, smoothing)
        if cache_key not in date_cache:
            try:
                _sec = load_sec_profile(
                    well_id=w, campaign=campaign, smoothing=smoothing,
                    project_root=project_root,
                )
                date_cache[cache_key] = (
                    _sec["source_file"].iloc[0].split("__")[0].split("_")[-1]
                )
            except Exception as exc:
                print(f"  ✗ ({smoothing}) cannot resolve date: {exc!r}")
                n_skipped += 1
                continue
        date_stamp = date_cache[cache_key]

        well_dir = out_dir / w
        well_dir.mkdir(parents=True, exist_ok=True)
        t_idx = trial_index(trial_name)
        fig_name = (
            f"{w}_{date_stamp}__{smoothing}__N{n:02d}__t{t_idx}.png"
        )
        fig_path = well_dir / fig_name

        try:
            fig = plot_sec_caliper_video_panel(
                w, smoothing=smoothing, n=n, trial=trial_name,
                campaign=campaign,
                perpoint_csv=perpoint_csv,
                video_xlsx=video_xlsx,
                ardaman_csv=(ardaman_csv if WELLS[w].has_ardaman else None),
                master_caliper_csv=master_caliper_csv,
                project_root=project_root,
                config=config,
                output_path=fig_path,
            )
            plt.close(fig)
            written.append(fig_path)
            print(f"  ✓ N={n:02d} {smoothing} {trial_name}")
        except Exception as exc:
            n_skipped += 1
            msg = str(exc).splitlines()[0][:80]
            print(f"  · N={n:02d} {smoothing} {trial_name}  skipped: {msg}")

    print(f"\n{len(written)} panel(s) written, {n_skipped} skipped.")
    return written
