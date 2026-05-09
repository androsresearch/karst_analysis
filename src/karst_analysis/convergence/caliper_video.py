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
    note_fontsize: float = 8.5
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
        If given, saves the figure as PNG.
    sat_cm : float, default 32.5
        Caliper saturation level (drawn as a faint vertical reference).

    Returns
    -------
    matplotlib.figure.Figure
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

    # ── 3) Figure size scales with number of entries ─────────────────
    n_entries = len(entries)
    base_w, base_h = cfg.base_figsize
    needed_h = cfg.auto_height_factor * cfg.auto_height_safety * n_entries
    fig_h = max(base_h, needed_h)
    figsize = (base_w, fig_h)

    fig, (ax_cal, ax_note) = plt.subplots(
        1, 2, figsize=figsize, sharey=True,
        gridspec_kw=dict(width_ratios=cfg.width_ratio, wspace=0.02),
    )

    # ── 4) Y-limits from union of all sources ───────────────────────
    # In BGL-positive convention, y_min ~ 0 (surface) and y_max is
    # the deepest point. The axis is inverted afterwards so 0 ends
    # up at the top of the figure.
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

    # ── 6) RIGHT panel: notes + leaders + brackets ──────────────────
    ax_note.set_xlim(0, 1)
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
        right_label += " (black) + Ardaman 2009 (blue / green)"
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
            print(f"  ✓ {fig_path}")
            written.append(fig_path)
        except Exception as exc:
            print(f"  ✗ FAILED: {exc!r}")
            import traceback
            traceback.print_exc()
    return written
