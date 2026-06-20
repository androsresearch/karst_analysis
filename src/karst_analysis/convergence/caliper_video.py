"""Side-by-side caliper + video-log + Ardaman panel per priority well.

Combines:
    * caliper signal + per-sample severity bands  (left)
    * companion-well caliper traces                (left)
    * video-log notes (and Ardaman lithology when applicable)  (right)

Inputs (all loaded by name):
    * per-sample severity CSV from ``karst_analysis.caliper`` pipeline
    * master concatenated caliper CSV (companion traces)
    * video-log xlsx (one sheet per well, names per ``WELLS`` mapping)
    * Ardaman lithology csv (only AW5O / AW6O)

Migration history
-----------------
v5.1: extracted from ``caliper_videolog_panel.py``.

Two fixes were applied during the migration:
    1. Title truncation bug (line ~644 of the original) where
       non-LRS70D wells produced a partial title because the f-string
       was cut short.  The full title formula is restored:
           "Well {caliper_well} caliper × video {video_well}"
           " + Ardaman 2009 ({video_well})"  (when applicable)
    2. The original hardcoded ``data_dir`` and ``out_dir`` in
       ``build_all`` to local windows paths under ``./notebooks/sandbox``.
       Replaced by sensible defaults rooted in the project tree.

No algorithmic changes were made to the loaders, the layout, or the
plotting.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from karst_analysis.caliper.io import (
    DEFAULT_MASTER_CSV, DEFAULT_PERPOINT_CSV,
    load_master_caliper, load_perpoint,
)
from karst_analysis.convergence._layout import (
    minimum_displacement_positions,
    draw_bracket,
    build_label_text,
)
from karst_analysis.drilling.io import DEFAULT_ARDAMAN_CSV, load_ardaman
from karst_analysis.videolog.io import DEFAULT_VIDEOLOG_XLSX, load_video_notes


# ──────────────────────────────────────────────────────────────────────
#  Severity palette — must match karst_analysis.caliper.viz
# ──────────────────────────────────────────────────────────────────────
SEVERITY_COLORS = {"mild": "#fde3a7", "moderate": "#f39c12", "severe": "#c0392b"}
SEVERITY_ALPHAS = {"mild": 0.65,      "moderate": 0.55,      "severe": 0.55}
BAND_ZORDER = 1.5


# ──────────────────────────────────────────────────────────────────────
#  Per-well configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class WellConfig:
    """Mapping between caliper well, video sheet, and Ardaman record.

    For most wells the video came from a *different* borehole at the
    same site — typically the older 2009 well (``O`` suffix) or the
    shallow companion (``S``). Only LRS70D has a video of the same D
    well that was calipered.
    """
    caliper_well: str    # name in the master caliper CSV (always *D)
    video_sheet:  str    # sheet name in the video-log xlsx
    video_well:   str    # well name to display in title (sometimes != sheet)
    auger_cm:     float  # auger diameter for outlier filtering
    has_ardaman:  bool = False


WELLS: dict[str, WellConfig] = {
    "LRS70D": WellConfig("LRS70D", "LRS70D", "LRS70D", 20.32, has_ardaman=False),
    "AW5D":   WellConfig("AW5D",   "AW5",    "AW5O",   15.24, has_ardaman=True),
    "AW6D":   WellConfig("AW6D",   "AW6",    "AW6O",   15.24, has_ardaman=True),
    "BW3D":   WellConfig("BW3D",   "BW3S",   "BW3S",   15.24, has_ardaman=False),
    "LRS69D": WellConfig("LRS69D", "LRS69S", "LRS69S", 15.24, has_ardaman=False),
}


# ──────────────────────────────────────────────────────────────────────
#  Companion-trace styling
# ──────────────────────────────────────────────────────────────────────
COMPANION_STYLE: dict[str, dict] = {
    "O": dict(color="#000000", lw=0.6, alpha=0.85),  # 2009 well, black
    "S": dict(color="#444444", lw=0.6, alpha=0.85),  # shallow, dark grey
}


# ──────────────────────────────────────────────────────────────────────
#  Helper functions used only by the panel
# ──────────────────────────────────────────────────────────────────────
def _site_prefix(well_name: str) -> str:
    """Return the site prefix shared by D/O/S companions.

    >>> _site_prefix("AW5D")
    'AW5'
    >>> _site_prefix("LRS69D")
    'LRS69'
    """
    import re
    m = re.match(r"^([A-Z]+\d+)", well_name)
    if m is None:
        raise ValueError(f"Cannot parse site prefix from '{well_name}'")
    return m.group(1)


def _load_companions_caliper(
    master_csv: str | Path, primary_well: str,
) -> dict[str, pd.DataFrame]:
    """Return all companion caliper traces for the same site as ``primary_well``.

    Excludes ``primary_well`` itself. Each value is a DataFrame with
    columns ``depth_m`` (negative metres, elevation) and ``caliper_cm``,
    sorted by depth.
    """
    import re
    df = load_master_caliper(master_csv)
    site = _site_prefix(primary_well)
    pattern = re.compile(rf"^{re.escape(site)}[A-Z]$")
    out: dict[str, pd.DataFrame] = {}
    for w in sorted(df["well"].unique()):
        if w == primary_well or not pattern.match(w):
            continue
        sub = (df[df["well"] == w][["depth_m", "calibrated_cm"]]
               .rename(columns={"calibrated_cm": "caliper_cm"})
               .sort_values("depth_m")
               .reset_index(drop=True))
        if not sub.empty:
            out[w] = sub
    return out


def _caliper_from_perpoint(
    perpoint_df: pd.DataFrame, auger_cm: float, iqr_k: float = 1.5,
) -> pd.DataFrame:
    """Add an ``is_outlier`` column to the per-sample caliper DataFrame."""
    df = pd.DataFrame({
        "depth_m":        perpoint_df["depth_m"].to_numpy(),
        "raw_caliper_cm": perpoint_df["caliper_cm"].to_numpy(),
    }).sort_values("depth_m").reset_index(drop=True)
    cal = df["raw_caliper_cm"].to_numpy()
    valid = cal >= auger_cm
    if valid.any():
        q1 = float(np.nanpercentile(cal[valid], 25))
        q3 = float(np.nanpercentile(cal[valid], 75))
        lower_fence = q1 - iqr_k * (q3 - q1)
    else:
        lower_fence = -np.inf
    df["is_outlier"] = (cal < auger_cm) & (cal < lower_fence)
    return df


def _load_perpoint_for(perpoint_csv: str | Path, well: str) -> pd.DataFrame:
    """Filter the per-sample CSV to rows for ``well``, ordered top-to-bottom."""
    df = load_perpoint(perpoint_csv)
    sub = df[df["well"] == well].copy()
    if sub.empty:
        raise ValueError(f"No per-sample rows for '{well}' in {perpoint_csv}")
    return sub.sort_values("depth_m", ascending=False).reset_index(drop=True)


def _severity_runs(
    depths: np.ndarray, severities: np.ndarray,
    levels: tuple[str, ...] = ("mild", "moderate", "severe"),
) -> list[tuple[str, float, float]]:
    """Run-length encode contiguous-same-severity samples.

    Returns ``[(severity, z_top, z_bot), ...]`` with z_top, z_bot
    expanded by half-dz so adjacent runs touch visually.
    """
    n = len(depths)
    if n == 0:
        return []
    half_dz = (
        0.5 * float(np.median(np.abs(np.diff(np.sort(depths)))))
        if n > 1 else 0.015
    )
    runs: list[tuple[str, float, float]] = []
    i = 0
    while i < n:
        sev = severities[i]
        if sev not in levels:
            i += 1
            continue
        j = i
        while j + 1 < n and severities[j + 1] == sev:
            j += 1
        runs.append((sev, depths[i] + half_dz, depths[j] - half_dz))
        i = j + 1
    return runs


def _draw_severity_bands(
    ax, perpoint_df: pd.DataFrame, *, alpha_factor: float = 1.0,
) -> None:
    """Render per-sample severity bands behind the data on ``ax``."""
    runs = _severity_runs(
        perpoint_df["depth_m"].to_numpy(),
        perpoint_df["severity_per_sample"].to_numpy(dtype=object),
    )
    draw_order = {"mild": 0, "moderate": 1, "severe": 2}
    for sev, z_top, z_bot in sorted(runs, key=lambda r: draw_order.get(r[0], 0)):
        ax.axhspan(z_bot, z_top,
                   color=SEVERITY_COLORS.get(sev, "#cccccc"),
                   alpha=SEVERITY_ALPHAS.get(sev, 0.35) * alpha_factor,
                   zorder=BAND_ZORDER, linewidth=0)


# ──────────────────────────────────────────────────────────────────────
#  Plot configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class PanelConfig:
    """Visual parameters for the caliper × video panel."""
    base_figsize: tuple = (13, 14)
    width_ratio: tuple = (1.0, 3.2)
    note_wrap: int = 110
    note_fontsize: float = 10.5
    note_x: float = 0.05
    leader_color: str = "#888888"
    leader_lw: float = 0.55
    bracket_lw: float = 1.3
    bracket_tip: float = 0.008
    outlier_color: str = "#7a1ea8"
    grid_alpha: float = 0.22
    note_color: str = "#1a1a1a"
    ardaman_color_lith: str = "#1d4ed8"
    ardaman_color_cond: str = "#0f7a4d"
    auto_height_factor: float = 0.30
    auto_height_safety:  float = 1.50

    # ── v5.2 additions ──────────────────────────────────────────────
    # All three default to the legacy behaviour (Rule 5.3: a new option
    # preserves the previous output unless explicitly opted in).
    #
    # drop_conductivity: when True, Ardaman in-situ-conductivity entries
    #     (kind == "ardaman_cond", the green annotations) are removed
    #     from the right panel. Lithology (blue) and video notes (black)
    #     are unaffected. Default False = legacy (conductivity shown).
    drop_conductivity: bool = False
    #
    # group_boundary_bgl_m: depth (m BGL) of the Group I / Group II
    #     boundary, drawn as a horizontal reference line and used to
    #     decide which group legend entries appear on each half.
    #     DESIGN PARAMETER, not data-derived: hand-set by Mariana from
    #     the Ardaman 2009 report and recorded in thesis table
    #     \label{tab:ardaman_groups}. None = no line, no group legend
    #     (legacy).
    group_boundary_bgl_m: Optional[float] = None
    group_line_color: str = "#333333"
    group_line_lw: float = 1.6
    group_legend_fontsize: float = 17.0   # "large font" group legend
    #
    # split_depth: when True, the well is rendered as two figures split
    #     at the geometric midpoint (total depth / 2): an upper half and
    #     a lower half. Default False = single figure (legacy).
    split_depth: bool = False


# ──────────────────────────────────────────────────────────────────────
#  The plot
# ──────────────────────────────────────────────────────────────────────
def plot_caliper_video_panel(
    well_id: str,
    *,
    perpoint_csv: Optional[str | Path] = None,
    video_xlsx: Optional[str | Path] = None,
    ardaman_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    config: Optional[PanelConfig] = None,
    output_path: Optional[str | Path] = None,
    sat_cm: float = 32.5,
) -> plt.Figure:
    """Build the caliper + video-log panel for one priority well.

    Parameters
    ----------
    well_id : str
        Must be a key of :data:`WELLS`.
    perpoint_csv, video_xlsx, ardaman_csv, master_caliper_csv : path-like, optional
        Override the default input paths. ``ardaman_csv`` is consulted
        only for wells whose ``WellConfig`` has ``has_ardaman=True``.
    config : PanelConfig, optional
        Override visual parameters.
    output_path : path-like, optional
        If given, saves the figure as PNG. When ``cfg.split_depth`` is
        True the well is rendered as two figures; the upper-half file
        keeps ``output_path`` stem with ``_top`` appended and the lower
        half with ``_bottom`` (e.g. ``AW6D_caliper_videolog_panel.png``
        -> ``..._top.png`` and ``..._bottom.png``).
    sat_cm : float, default 32.5
        Caliper saturation level (drawn as a faint vertical reference).

    Returns
    -------
    matplotlib.figure.Figure
        The single figure (legacy mode). When ``cfg.split_depth`` is
        True, returns the upper-half figure; both halves are still
        written to disk if ``output_path`` is given.
    """
    cfg = config or PanelConfig()
    if well_id not in WELLS:
        raise KeyError(
            f"Unknown well '{well_id}'. Known wells: {list(WELLS.keys())}"
        )
    wc = WELLS[well_id]

    # Resolve defaults
    if perpoint_csv is None:        perpoint_csv = DEFAULT_PERPOINT_CSV
    if video_xlsx is None:          video_xlsx = DEFAULT_VIDEOLOG_XLSX
    if master_caliper_csv is None:  master_caliper_csv = DEFAULT_MASTER_CSV
    if ardaman_csv is None and wc.has_ardaman:
        ardaman_csv = DEFAULT_ARDAMAN_CSV

    # ── 1) Load all data ────────────────────────────────────────────
    perpoint_df = _load_perpoint_for(perpoint_csv, well=wc.caliper_well)
    cal_df = _caliper_from_perpoint(perpoint_df, auger_cm=wc.auger_cm)
    notes_df = load_video_notes(video_xlsx, sheet=wc.video_sheet)
    ardaman_df = (load_ardaman(ardaman_csv, well=wc.video_well)
                  if (wc.has_ardaman and ardaman_csv is not None)
                  else pd.DataFrame())
    companions = (_load_companions_caliper(master_caliper_csv,
                                           primary_well=wc.caliper_well)
                  if master_caliper_csv is not None else {})

    # ── 2) Unify entries DataFrame ──────────────────────────────────
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
    entries = (
        pd.concat(parts, ignore_index=True)
          .sort_values("depth_centre_bgl_m", ascending=True)
          .reset_index(drop=True)
        if parts else pd.DataFrame()
    )

    # ── 2b) Drop Ardaman in-situ-conductivity (green) if requested ──
    # Mariana's director considers the conductivity annotations
    # unnecessary. Lithology (blue) and video notes (black) are kept.
    if cfg.drop_conductivity and not entries.empty:
        entries = (entries[entries["kind"] != "ardaman_cond"]
                   .reset_index(drop=True))

    # ── 2c) Depth window for this well (used for the split midpoint) ─
    # y_min ~ surface, y_max ~ deepest sample (BGL-positive). Mirror the
    # same union the single-figure path used so the split midpoint and
    # the legacy y-limits agree.
    ymin_c = [cal_df["depth_m"].min()]
    ymax_c = [cal_df["depth_m"].max()]
    if not entries.empty:
        ymin_c.append(float(entries["depth_centre_bgl_m"].min()))
        if entries["depth_top_bgl_m"].notna().any():
            ymin_c.append(float(np.nanmin(entries["depth_top_bgl_m"])))
        ymax_c.append(float(entries["depth_centre_bgl_m"].max()))
        if entries["depth_bot_bgl_m"].notna().any():
            ymax_c.append(float(np.nanmax(entries["depth_bot_bgl_m"])))
    y_min = min(ymin_c) - 0.8
    y_max = max(ymax_c) + 0.8

    common = dict(
        wc=wc, cfg=cfg, perpoint_df=perpoint_df, cal_df=cal_df,
        entries=entries, companions=companions, sat_cm=sat_cm,
    )

    # ── Legacy single-figure mode ───────────────────────────────────
    if not cfg.split_depth:
        return _render_one(
            y_lo=y_min, y_hi=y_max,
            output_path=output_path,
            half="single",
            **common,
        )

    # ── Split mode: two figures at the geometric midpoint ───────────
    # The midpoint is purely geometric (total depth / 2), independent of
    # the Group I/II boundary. Upper half = [y_min, mid], lower half =
    # [mid, y_max].
    mid = 0.5 * (y_min + y_max)
    top_out, bot_out = _split_output_paths(output_path)
    fig_top = _render_one(
        y_lo=y_min, y_hi=mid, output_path=top_out, half="top", **common,
    )
    fig_bot = _render_one(
        y_lo=mid, y_hi=y_max, output_path=bot_out, half="bottom", **common,
    )
    # Return the upper-half figure for API continuity; both are on disk.
    plt.close(fig_bot)
    return fig_top


def _split_output_paths(output_path):
    """Return (top_path, bottom_path) derived from ``output_path``.

    ``AW6D_caliper_videolog_panel.png`` ->
        (``AW6D_caliper_videolog_panel_top.png``,
         ``AW6D_caliper_videolog_panel_bottom.png``).
    Returns (None, None) when ``output_path`` is None.
    """
    if output_path is None:
        return None, None
    p = Path(output_path)
    return (p.with_name(f"{p.stem}_top{p.suffix}"),
            p.with_name(f"{p.stem}_bottom{p.suffix}"))


def _render_one(
    *,
    wc: "WellConfig",
    cfg: "PanelConfig",
    perpoint_df: pd.DataFrame,
    cal_df: pd.DataFrame,
    entries: pd.DataFrame,
    companions: dict,
    sat_cm: float,
    y_lo: float,
    y_hi: float,
    output_path: Optional[str | Path],
    half: str,
) -> plt.Figure:
    """Render a single figure covering the depth window ``[y_lo, y_hi]``.

    ``half`` is one of ``"single"``, ``"top"``, ``"bottom"`` and only
    affects which Group legend entries are shown and the title suffix.
    All depths are BGL-positive; the y-axis is inverted at the end so 0
    sits at the top.
    """
    # Restrict entries to those whose centre falls inside this window so
    # the right-panel annotations for a half belong to that half.
    if not entries.empty:
        m = ((entries["depth_centre_bgl_m"] >= y_lo)
             & (entries["depth_centre_bgl_m"] <= y_hi))
        entries = entries[m].reset_index(drop=True)

    # ── 3) Figure size scales with number of entries in THIS window ──
    n_entries = len(entries)
    base_w, base_h = cfg.base_figsize
    needed_h = cfg.auto_height_factor * cfg.auto_height_safety * n_entries
    fig_h = max(base_h, needed_h)
    figsize = (base_w, fig_h)

    fig, (ax_cal, ax_note) = plt.subplots(
        1, 2, figsize=figsize, sharey=True,
        gridspec_kw=dict(width_ratios=cfg.width_ratio, wspace=0.02),
    )

    # ── 4) Y-limits are the window passed in ────────────────────────
    # All depths BGL-positive; the axis is inverted afterwards so 0 ends
    # up at the top of the figure.
    y_min, y_max = y_lo, y_hi
    ax_cal.set_ylim(y_min, y_max)

    # ── 5) LEFT panel: severity, companions, primary, references ────
    _draw_severity_bands(ax_cal, perpoint_df, alpha_factor=1.0)

    companion_handles = []
    for cmp_name, cmp_df in companions.items():
        suffix = cmp_name[len(_site_prefix(cmp_name)):]
        style = COMPANION_STYLE.get(suffix)
        if style is None:
            continue
        ax_cal.plot(cmp_df["caliper_cm"].to_numpy(),
                    cmp_df["depth_m"].to_numpy(),
                    label=cmp_name, zorder=3, **style)
        companion_handles.append((cmp_name, style))

    z = cal_df["depth_m"].to_numpy()
    raw = cal_df["raw_caliper_cm"].to_numpy()
    out_mask = cal_df["is_outlier"].to_numpy().astype(bool)
    cal_plot = np.where(out_mask, np.nan, raw)
    ax_cal.plot(cal_plot, z, color="#8e6914", lw=0.6, alpha=0.85,
                zorder=4, label=wc.caliper_well)
    if out_mask.any():
        ax_cal.scatter(raw[out_mask], z[out_mask], s=40, marker="x",
                       c=cfg.outlier_color, zorder=5, linewidths=1.0)
    ax_cal.axvline(sat_cm, color="#777777", lw=0.6, ls=":", alpha=0.55, zorder=2)
    ax_cal.set_xlabel("Caliper aperture (cm)", fontsize=10)
    ax_cal.set_ylabel("Depth below ground level (m)", fontsize=10)
    ax_cal.grid(True, axis="both", alpha=cfg.grid_alpha, linestyle=":")
    ax_cal.tick_params(axis="both", labelsize=8)
    ax_cal.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

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

    # Legend
    handles = [Line2D([0], [0], color="#8e6914", lw=1.2, label=wc.caliper_well)]
    for cmp_name, style in companion_handles:
        handles.append(Line2D([0], [0],
                              color=style["color"],
                              lw=max(style["lw"], 1.0),
                              alpha=style.get("alpha", 1.0),
                              label=cmp_name))
    for sev in ("mild", "moderate", "severe"):
        if (perpoint_df["severity_per_sample"] == sev).any():
            handles.append(Patch(facecolor=SEVERITY_COLORS[sev],
                                 alpha=SEVERITY_ALPHAS[sev],
                                 edgecolor="none",
                                 label=sev.capitalize()))
    if handles:
        ax_cal.legend(handles=handles, loc="upper left", fontsize=7.5,
                      framealpha=0.92, edgecolor="#cccccc",
                      handlelength=1.5, handleheight=1.0,
                      borderpad=0.4, bbox_to_anchor=(0.0, 1.0))

    # ── 5b) Group I / II boundary line + large-font group legend ────
    # The boundary depth is a design parameter set by Mariana from the
    # Ardaman 2009 report (thesis table \label{tab:ardaman_groups}). It
    # is drawn as a horizontal line on both panels wherever it falls
    # inside the current depth window. Group I = above the boundary
    # (shallower), Group II = below it (deeper). Each group label is
    # shown only on a window where that group is actually visible (no
    # Group I on a lower half that lies entirely below the boundary).
    gb = cfg.group_boundary_bgl_m
    line_in_window = (gb is not None) and (y_min <= gb <= y_max)
    if line_in_window:
        # Boundary line on the LEFT panel here; the RIGHT-panel line and
        # the Group text boxes are drawn after ax_note.set_xlim(0, 1)
        # below, so their x-coordinates are in the right panel's final
        # 0..1 data frame.
        ax_cal.axhline(gb, color=cfg.group_line_color,
                       lw=cfg.group_line_lw, ls="-", alpha=0.9, zorder=6)

    # ── 6) RIGHT panel: notes + leaders + brackets ──────────────────
    ax_note.set_xlim(0, 1)

    # Group boundary line + Group text boxes on the RIGHT panel. Drawn
    # here (after set_xlim) so x is in the panel's final 0..1 frame.
    # Director's request: the group names appear as large text boxes at
    # specific depths, not as a corner legend. A box is shown only when
    # (a) its depth is inside this window and (b) it is on the correct
    # side of the boundary for the window. The boxes never overlap each
    # other or the boundary line because their depths (4.8, 6.3) sit on
    # opposite sides of the 5.2 boundary by construction.
    if line_in_window:
        ax_note.axhline(gb, color=cfg.group_line_color,
                        lw=cfg.group_line_lw, ls="-", alpha=0.9, zorder=6)

    if gb is not None:
        # Build the set of "occupied" depth points in this window: note
        # anchors and the boundary itself. Then, for each side of the
        # boundary that the window covers, find the largest gap between
        # consecutive occupied points and centre the group label there.
        note_anchors = (entries["depth_centre_bgl_m"].to_numpy()
                        if not entries.empty else np.array([]))

        def _largest_gap(lo, hi):
            """Return the midpoint of the widest free vertical slot in
            [lo, hi], avoiding existing note anchors. Returns None if
            the interval is degenerate."""
            if hi - lo <= 0.05:
                return None
            inside = note_anchors[(note_anchors > lo) & (note_anchors < hi)]
            pts = np.concatenate(([lo], np.sort(inside), [hi]))
            gaps = np.diff(pts)
            k = int(np.argmax(gaps))
            return 0.5 * (pts[k] + pts[k + 1])

        def _group_box(depth, text):
            ax_note.text(
                0.98, depth, text,
                ha="right", va="center",
                fontsize=cfg.group_legend_fontsize,
                fontweight="bold", color=cfg.group_line_color, zorder=7,
                bbox=dict(boxstyle="round,pad=0.45",
                          facecolor="white", edgecolor=cfg.group_line_color,
                          linewidth=1.4),
            )

        # Group I: window slice shallower than gb (and inside [y_min, y_max]).
        lo_I = y_min
        hi_I = min(gb, y_max)
        if hi_I > lo_I:
            d = _largest_gap(lo_I, hi_I)
            if d is not None:
                _group_box(d, "Group I")

        # Group II: window slice deeper than gb.
        lo_II = max(gb, y_min)
        hi_II = y_max
        if hi_II > lo_II:
            d = _largest_gap(lo_II, hi_II)
            if d is not None:
                _group_box(d, "Group II")

    ax_note.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    ax_note.tick_params(axis="y", which="both", left=False, labelleft=False)
    for spine in ("top", "right", "bottom"):
        ax_note.spines[spine].set_visible(False)
    ax_note.spines["left"].set_color("#bbbbbb")
    _draw_severity_bands(ax_note, perpoint_df, alpha_factor=0.45)

    if not entries.empty:
        labels = [build_label_text(row) for _, row in entries.iterrows()]
        colors = entries["kind"].map({
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
        x_text   = cfg.note_x
        text_x_left = x_text + 0.005

        for i, (_, row) in enumerate(entries.iterrows()):
            e_top = row["depth_top_bgl_m"]
            e_bot = row["depth_bot_bgl_m"]
            anchor_y = row["depth_centre_bgl_m"]
            color = colors[i]
            fontstyle = styles[i]
            wrapped_text = wrapped[i]
            ty = text_y[i]

            is_interval = (np.isfinite(e_bot)
                           and abs(e_top - e_bot) > 1e-6)
            if is_interval:
                draw_bracket(ax_note, x_anchor, x_anchor + cfg.bracket_tip,
                             e_top, e_bot, color=color, lw=cfg.bracket_lw)
                leader_x_start = x_anchor + cfg.bracket_tip
            else:
                ax_note.plot(x_anchor, anchor_y, marker="o", ms=2.8,
                             color=color, zorder=3)
                leader_x_start = x_anchor

            ax_note.plot([leader_x_start, x_kink, x_text],
                         [anchor_y, anchor_y, ty],
                         color=cfg.leader_color, lw=cfg.leader_lw,
                         alpha=0.85, zorder=2)
            ax_note.text(text_x_left, ty, wrapped_text,
                         ha="left", va="center",
                         fontsize=cfg.note_fontsize,
                         color=color, fontstyle=fontstyle, zorder=4,
                         bbox=dict(boxstyle="round,pad=0.28",
                                   facecolor="white", edgecolor="#dddddd",
                                   alpha=0.88, linewidth=0.5))

    right_label = "Video-log observations"
    if wc.has_ardaman:
        # Green = conductivity; only mention it if it's still drawn.
        ard_colors = "blue" if cfg.drop_conductivity else "blue / green"
        right_label += f" (black) + Ardaman 2009 ({ard_colors})"
    ax_note.set_xlabel(right_label, fontsize=10)

    # ── 7) Title  (FIX: full title restored — see migration notes) ──
    if wc.video_well == wc.caliper_well:
        title = (
            f"Well {wc.caliper_well} — caliper breakout zones "
            f"and video-log observations"
        )
    else:
        ard_note = (f" + Ardaman 2009 ({wc.video_well})"
                    if wc.has_ardaman else "")
        title = (
            f"Well {wc.caliper_well} caliper × video {wc.video_well}"
            f"{ard_note}"
        )
    title_y = 1.0 - 0.4 / figsize[1]
    if half == "top":
        title += "  (upper half)"
    elif half == "bottom":
        title += "  (lower half)"
    fig.suptitle(title, fontsize=11.5, fontweight="bold", y=title_y)
    top_margin = max(0.92, 1.0 - 1.0 / figsize[1])
    fig.subplots_adjust(left=0.075, right=0.985, top=top_margin, bottom=0.05)

    # BGL-positive: invert so 0 is at the top of the figure.
    # sharey=True means a single invert on either axis flips both.
    ax_cal.invert_yaxis()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=170, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────
#  Batch builder
# ──────────────────────────────────────────────────────────────────────
def build_all_caliper_video_panels(
    *,
    perpoint_csv: Optional[str | Path] = None,
    video_xlsx: Optional[str | Path] = None,
    ardaman_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    wells: Optional[list[str]] = None,
    config: Optional[PanelConfig] = None,
) -> list[Path]:
    """Render the panel for every well in :data:`WELLS` (or a subset).

    Returns a list of the PNG paths actually written. Failures on
    individual wells are logged but don't abort the batch.

    The default ``output_dir`` is
    ``results/figures/convergence/caliper_video/`` (no campaign
    subfolder — caliper and video are pre-casing techniques and have
    no campaign concept).
    """
    if output_dir is None:
        from karst_analysis.io import resolve_figure_dir
        output_dir = resolve_figure_dir("convergence/caliper_video")
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = wells if wells is not None else list(WELLS.keys())
    written: list[Path] = []
    for w in target:
        if w not in WELLS:
            print(f"[{w}]  unknown — skipping")
            continue
        print(f"[{w}]")
        try:
            fig_path = out_dir / f"{w}_caliper_videolog_panel.png"
            fig = plot_caliper_video_panel(
                w,
                perpoint_csv=perpoint_csv,
                video_xlsx=video_xlsx,
                ardaman_csv=(ardaman_csv if WELLS[w].has_ardaman else None),
                master_caliper_csv=master_caliper_csv,
                config=config,
                output_path=fig_path,
            )
            plt.close(fig)
            # When config.split_depth is set, two files were written
            # (``_top`` / ``_bottom``); otherwise the single base name.
            if config is not None and getattr(config, "split_depth", False):
                top_p, bot_p = _split_output_paths(fig_path)
                print(f"  ✓ {top_p}")
                print(f"  ✓ {bot_p}")
                written.extend([top_p, bot_p])
            else:
                print(f"  ✓ {fig_path}")
                written.append(fig_path)
        except Exception as exc:
            print(f"  ✗ FAILED: {exc!r}")
            import traceback
            traceback.print_exc()
    return written
