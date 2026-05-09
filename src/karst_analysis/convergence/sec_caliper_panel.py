"""SEC raw traces × caliper panel per priority well  (v10).

A two-column figure for one priority well:

    column 1: caliper aperture (cm)  +  severity bands  +  companion
              traces (O / S wells of the same site)
    column 2: raw SEC traces (μS/cm) for one campaign (one or more
              YSI casts per well).

Differences w.r.t. the existing ``sec_caliper_video.py`` panel:

* No video-log column, no breakpoint-label column, no Ardaman lithology.
* The SEC traces are RAW YSI files (one per cast). For Feb-2022 there is
  one cast per well; for later campaigns this will scale to several.
* No smoothing, no breakpoints overlaid. This is a "data inspection"
  panel that supports the temporal analysis (Idea 2).

The caliper column reuses the helpers from ``caliper_video.py``
verbatim — same severity colours, same companion styling, same
auger reference, same outlier marking. This is intentional so both
panels produce visually-comparable figures.

Outputs
-------
``plot_sec_caliper_panel(well_id, campaign, ...)`` returns a
``matplotlib.Figure``. ``build_all_sec_caliper_panels(...)`` produces
one PNG per well plus an optional master 1×N figure where every
priority well is a column.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from karst_analysis.caliper.io import DEFAULT_MASTER_CSV, DEFAULT_PERPOINT_CSV
from karst_analysis.convergence.caliper_video import (
    COMPANION_STYLE,
    WELLS,
    WellConfig,
    _caliper_from_perpoint,
    _draw_severity_bands,
    _load_companions_caliper,
    _load_perpoint_for,
    _site_prefix,
)
from karst_analysis.sec.io import load_raw_ysi_traces_for_well


# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────

# Plotly's Dark24 categorical palette — 24 distinguishable colours, no
# perceptual ordering (so the eye doesn't impute a sequence among the
# campaigns). Listed verbatim from
# https://plotly.com/python/discrete-color/#color-sequences-in-plotly-express
DARK24_PALETTE: tuple[str, ...] = (
    "#2E91E5", "#E15F99", "#1CA71C", "#FB0D0D", "#DA16FF", "#222A2A",
    "#B68100", "#750D86", "#EB663B", "#511CFB", "#00A08B", "#FB00D1",
    "#FC0080", "#B2828D", "#6C7C32", "#778AAE", "#862A16", "#A777F1",
    "#620042", "#1616A7", "#DA60CA", "#6C4516", "#0D2A63", "#AF0038",
)


@dataclass
class SecCaliperPanelConfig:
    """Visual parameters for the SEC × caliper panel.

    Attributes
    ----------
    figsize : (float, float)
        Single-well panel size in inches.
    width_ratios : tuple[float, float]
        (caliper_width, sec_width). The caliper column is narrower
        because it lives near the auger nominal, the SEC column needs
        more horizontal room to show the full dynamic range.
    sec_alpha : float
    sec_lw : float
    sec_log_x : bool
        Whether to use a log scale on the SEC axis. Default linear,
        per v10 design decision.
    grid_alpha : float
    sat_cm : float
        Caliper saturation marker (drawn as faint vertical line).
    campaign_palette : dict[str, str] or None
        Mapping ``campaign → hex_colour``. If None, colours are
        auto-assigned from the Dark24 palette in the order campaigns
        are passed to the rendering functions. Provide an explicit
        mapping if you want stable colours across plots (so e.g.
        2022_02 is always blue and 2023_06 is always pink).
    fallback_marker : str
        Suffix appended to legend labels when a trace used a fallback
        vadose value. Default ``" *"``.
    """
    figsize: tuple[float, float] = (6.5, 11.0)
    width_ratios: tuple[float, float] = (1.0, 1.4)
    sec_alpha: float = 0.75
    sec_lw: float = 0.7
    sec_log_x: bool = False
    grid_alpha: float = 0.45
    sat_cm: float = 32.5
    campaign_palette: Optional[dict[str, str]] = None
    fallback_marker: str = " *"


def _resolve_campaign_palette(
    campaigns: list[str],
    explicit: Optional[dict[str, str]],
) -> dict[str, str]:
    """Build a final ``{campaign: colour}`` mapping.

    If ``explicit`` is given, missing campaigns are filled from Dark24
    in the order they appear in ``campaigns``. If ``explicit`` is None,
    every campaign is assigned a Dark24 colour in order.
    """
    palette = dict(explicit) if explicit else {}
    pool = [c for c in DARK24_PALETTE if c not in palette.values()]
    pool_iter = iter(pool)
    for c in campaigns:
        if c not in palette:
            try:
                palette[c] = next(pool_iter)
            except StopIteration:
                # More campaigns than palette colours — wrap around.
                palette[c] = DARK24_PALETTE[len(palette) % len(DARK24_PALETTE)]
    return palette


# ──────────────────────────────────────────────────────────────────────
#  Single-axis renderer (used by both per-well and master figures)
# ──────────────────────────────────────────────────────────────────────
def _render_caliper_axis(
    ax: plt.Axes,
    well_id: str,
    *,
    perpoint_csv: str | Path,
    master_caliper_csv: Optional[str | Path],
    cfg: SecCaliperPanelConfig,
    show_xlabel: bool = True,
) -> tuple[float, float]:
    """Draw caliper signal + severity bands + companions onto ``ax``.

    Returns
    -------
    (depth_min, depth_max) : tuple of floats
        The depth range of the caliper data (used to set the shared y-limits).
    """
    wc = WELLS[well_id]
    perpoint_df = _load_perpoint_for(perpoint_csv, well=wc.caliper_well)
    cal_df = _caliper_from_perpoint(perpoint_df, auger_cm=wc.auger_cm)
    companions = (
        _load_companions_caliper(master_caliper_csv, primary_well=wc.caliper_well)
        if master_caliper_csv is not None else {}
    )

    # severity bands first so traces sit on top
    _draw_severity_bands(ax, perpoint_df, alpha_factor=1.0)

    # companions
    for cmp_name, cmp_df in companions.items():
        suffix = cmp_name[len(_site_prefix(cmp_name)):]
        style = COMPANION_STYLE.get(suffix)
        if style is None:
            continue
        ax.plot(
            cmp_df["caliper_cm"].to_numpy(),
            cmp_df["depth_m"].to_numpy(),
            label=cmp_name, zorder=3, **style,
        )

    # primary caliper trace, with outliers masked
    z = cal_df["depth_m"].to_numpy()
    raw = cal_df["raw_caliper_cm"].to_numpy()
    out_mask = cal_df["is_outlier"].to_numpy().astype(bool)
    cal_plot = np.where(out_mask, np.nan, raw)
    ax.plot(cal_plot, z, color="#8e6914", lw=0.6, alpha=0.85,
            zorder=4, label=wc.caliper_well)
    if out_mask.any():
        ax.scatter(raw[out_mask], z[out_mask], s=30, marker="x",
                   c="#c0392b", zorder=5, linewidths=1.0)

    # references: saturation, auger nominal
    ax.axvline(cfg.sat_cm, color="#777777", lw=0.6, ls=":", alpha=0.55, zorder=2)
    ax.axvline(wc.auger_cm, color="#444444", lw=0.5, ls=":", alpha=0.4, zorder=2)

    # x-limits: a little room around min/max + auger
    all_mins = [np.nanmin(raw[~out_mask]) if (~out_mask).any() else np.nanmax(raw)]
    all_maxs = [np.nanmax(raw)]
    for cmp_df in companions.values():
        c = cmp_df["caliper_cm"].to_numpy()
        if c.size:
            all_mins.append(float(np.nanmin(c)))
            all_maxs.append(float(np.nanmax(c)))
    ax.set_xlim(min(min(all_mins) - 1.0, wc.auger_cm - 1.0),
                max(max(all_maxs), cfg.sat_cm) + 1.5)

    if show_xlabel:
        ax.set_xlabel("Caliper (cm)", fontsize=9)
    ax.grid(True, axis="both", alpha=cfg.grid_alpha, linestyle=":")
    ax.tick_params(axis="both", labelsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    return float(np.nanmin(z)), float(np.nanmax(z))


def _render_sec_axis(
    ax: plt.Axes,
    well_id: str,
    campaigns: list[str],
    *,
    project_root: Optional[Path | str],
    cfg: SecCaliperPanelConfig,
    short_xlabel: bool = False,
    show_legend: bool = True,
) -> tuple[float, float]:
    """Draw raw SEC traces of (well_id, campaigns) onto ``ax``.

    Parameters
    ----------
    well_id : str
    campaigns : list[str]
        One or more field-campaign identifiers. All matching CSVs from
        every campaign are drawn on the same axis. Each campaign uses
        a single colour (from the resolved palette) so the eye groups
        traces by campaign.
    short_xlabel : bool, default False
        If True, the SEC axis x-label is just "SEC (μS/cm)" without
        the campaign suffix. Used in master panels where the campaigns
        appear in the figure title.
    show_legend : bool, default True
        If True, an in-axis legend is drawn (one entry per campaign,
        annotated with ``*`` when fallback vadose was used).

    Returns
    -------
    (depth_min, depth_max) : tuple of floats
        The depth range of the SEC traces (used to extend the shared
        y-limits). If no traces are found across all campaigns,
        returns (np.inf, -np.inf).
    """
    palette = _resolve_campaign_palette(campaigns, cfg.campaign_palette)

    z_mins: list[float] = []
    z_maxs: list[float] = []

    # We collect one legend entry per campaign (not per cast). If every
    # cast in a campaign used fallback, the campaign is marked with `*`.
    legend_entries: list[tuple[str, str]] = []  # (label, colour)
    no_data_campaigns: list[str] = []

    for campaign in campaigns:
        try:
            traces = load_raw_ysi_traces_for_well(
                well_id, campaign,
                project_root=project_root,
                add_depth_bgl=True,
            )
        except FileNotFoundError:
            no_data_campaigns.append(campaign)
            continue

        if not traces:
            no_data_campaigns.append(campaign)
            continue

        colour = palette[campaign]
        any_fallback = any(
            tr.vadose_resolution is not None and tr.vadose_resolution.is_fallback
            for tr in traces
        )

        for tr in traces:
            df = tr.df
            if "depth_bgl_m" not in df.columns:
                continue
            z = df["depth_bgl_m"].to_numpy()
            s = df["sec_uS_cm"].to_numpy()
            ax.plot(
                s, z,
                color=colour, lw=cfg.sec_lw, alpha=cfg.sec_alpha,
            )
            z_mins.append(float(np.nanmin(z)))
            z_maxs.append(float(np.nanmax(z)))

        # One legend entry per campaign.
        label = campaign + (cfg.fallback_marker if any_fallback else "")
        legend_entries.append((label, colour))

    if cfg.sec_log_x:
        ax.set_xscale("log")
    if short_xlabel:
        ax.set_xlabel("SEC (μS/cm)", fontsize=9)
    else:
        if len(campaigns) == 1:
            ax.set_xlabel(f"SEC (μS/cm) — campaign {campaigns[0]}", fontsize=9)
        else:
            ax.set_xlabel("SEC (μS/cm)", fontsize=9)
    ax.grid(True, axis="both", alpha=cfg.grid_alpha, linestyle=":")
    ax.tick_params(axis="both", labelsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # No data at all → empty axis with informational text.
    if not z_mins:
        msg = "No raw SEC traces found"
        if no_data_campaigns:
            msg += "\nfor campaigns: " + ", ".join(no_data_campaigns)
        ax.text(
            0.5, 0.5, msg,
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#9ca3af", style="italic",
        )
        return float("inf"), float("-inf")

    # In-axis legend (one entry per campaign), if requested and useful.
    if show_legend and legend_entries:
        handles = [
            Line2D([0], [0], color=clr, lw=2.0, label=lbl)
            for (lbl, clr) in legend_entries
        ]
        ax.legend(
            handles=handles,
            loc="lower right", fontsize=7, framealpha=0.9,
            title="campaign", title_fontsize=7,
        )

    return min(z_mins), max(z_maxs)


# ──────────────────────────────────────────────────────────────────────
#  Public: single-well figure
# ──────────────────────────────────────────────────────────────────────
def plot_sec_caliper_panel(
    well_id: str,
    *,
    campaigns: list[str],
    perpoint_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path | str] = None,
    config: Optional[SecCaliperPanelConfig] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Render the 2-column SEC × caliper panel for one well.

    Parameters
    ----------
    well_id : str
        Must be a key of :data:`WELLS`.
    campaigns : list[str]
        One or more field campaigns to overlay on the SEC axis. Each
        campaign uses a single colour from the configured palette
        (v11 default: Plotly Dark24, categorical). Pass a single-element
        list to reproduce v10 behaviour.
    perpoint_csv : path-like, optional
    master_caliper_csv : path-like, optional
    project_root : Path or str, optional
        Root of the karst_analysis project (defaults to ``Path.cwd()``).
    config : SecCaliperPanelConfig, optional
    output_path : path-like, optional
        If given, save the figure as PNG.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cfg = config or SecCaliperPanelConfig()
    if well_id not in WELLS:
        raise KeyError(f"Unknown well '{well_id}'. Known: {list(WELLS.keys())}")

    if not campaigns:
        raise ValueError(
            "plot_sec_caliper_panel requires `campaigns` to be a non-empty "
            "list of campaign identifiers, e.g. ['2022_02']."
        )

    if perpoint_csv is None:        perpoint_csv = DEFAULT_PERPOINT_CSV
    if master_caliper_csv is None:  master_caliper_csv = DEFAULT_MASTER_CSV

    fig, (ax_cal, ax_sec) = plt.subplots(
        1, 2, figsize=cfg.figsize, sharey=True,
        gridspec_kw=dict(width_ratios=cfg.width_ratios, wspace=0.05),
    )

    z_lo_c, z_hi_c = _render_caliper_axis(
        ax_cal, well_id,
        perpoint_csv=perpoint_csv,
        master_caliper_csv=master_caliper_csv,
        cfg=cfg,
    )
    z_lo_s, z_hi_s = _render_sec_axis(
        ax_sec, well_id, campaigns,
        project_root=project_root,
        cfg=cfg,
    )

    # Shared y-limits: union of caliper and SEC ranges, with small margin.
    z_lo = min(z_lo_c, z_lo_s) - 0.5
    z_hi = max(z_hi_c, z_hi_s) + 0.5
    ax_cal.set_ylim(z_hi, z_lo)   # inverted (BGL grows downward)

    ax_cal.set_ylabel("Depth below ground level (m)", fontsize=10)

    # Title
    if len(campaigns) == 1:
        title = f"{well_id} — caliper × raw SEC ({campaigns[0]})"
    else:
        title = f"{well_id} — caliper × raw SEC ({len(campaigns)} campaigns)"
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.995)

    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.07, wspace=0.05)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────
#  Public: master 1×N figure (all priority wells side by side)
# ──────────────────────────────────────────────────────────────────────
def plot_master_panel(
    well_ids: Iterable[str],
    *,
    campaigns: list[str],
    perpoint_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path | str] = None,
    config: Optional[SecCaliperPanelConfig] = None,
    output_path: Optional[str | Path] = None,
    per_well_width: float = 3.0,
    height: float = 11.0,
) -> plt.Figure:
    """Side-by-side SEC × caliper panel for several wells, shared y-axis.

    Each well occupies TWO sub-axes (caliper + SEC) stacked horizontally.
    All wells share the same depth axis (BGL).

    Parameters
    ----------
    well_ids : iterable of str
        Wells to include, in the desired left-to-right order.
    campaigns : list[str]
        One or more field campaigns to overlay on each SEC axis. Pass a
        single-element list to reproduce v10 behaviour.
    perpoint_csv, master_caliper_csv, project_root : as in single-well version.
    config : SecCaliperPanelConfig, optional
    output_path : path-like, optional
    per_well_width : float, default 3.0
        Horizontal inches allocated to each well (caliper + SEC together).
    height : float, default 11.0

    Returns
    -------
    matplotlib.figure.Figure
    """
    cfg = config or SecCaliperPanelConfig()
    well_list = list(well_ids)
    n = len(well_list)
    if n == 0:
        raise ValueError("well_ids is empty.")

    if not campaigns:
        raise ValueError(
            "plot_master_panel requires `campaigns` to be a non-empty list."
        )
    for w in well_list:
        if w not in WELLS:
            raise KeyError(f"Unknown well '{w}'. Known: {list(WELLS.keys())}")

    if perpoint_csv is None:        perpoint_csv = DEFAULT_PERPOINT_CSV
    if master_caliper_csv is None:  master_caliper_csv = DEFAULT_MASTER_CSV

    # Each well: 2 axes (caliper, sec). Total: 2*n columns.
    figsize = (per_well_width * n, height)
    width_ratios = []
    for _ in well_list:
        width_ratios.extend(cfg.width_ratios)

    fig, axes = plt.subplots(
        1, 2 * n, figsize=figsize, sharey=True,
        gridspec_kw=dict(width_ratios=width_ratios, wspace=0.05),
    )

    z_global_lo = float("inf")
    z_global_hi = float("-inf")

    for i, w in enumerate(well_list):
        ax_cal = axes[2 * i]
        ax_sec = axes[2 * i + 1]

        z_lo_c, z_hi_c = _render_caliper_axis(
            ax_cal, w,
            perpoint_csv=perpoint_csv,
            master_caliper_csv=master_caliper_csv,
            cfg=cfg,
            show_xlabel=True,
        )
        z_lo_s, z_hi_s = _render_sec_axis(
            ax_sec, w, campaigns,
            project_root=project_root,
            cfg=cfg,
            short_xlabel=True,
            show_legend=(i == n - 1),  # only on the rightmost SEC axis
        )
        z_global_lo = min(z_global_lo, z_lo_c, z_lo_s)
        z_global_hi = max(z_global_hi, z_hi_c, z_hi_s)

        # Per-well group title above the caliper sub-axis
        ax_cal.set_title(w, fontsize=11, fontweight="bold", pad=8)

        # Only the leftmost column carries the depth label
        if i == 0:
            ax_cal.set_ylabel("Depth below ground level (m)", fontsize=10)

        # Hide the redundant y-tick labels for non-leftmost axes (sharey
        # already prevents tick mismatch, but the labels look cleaner if
        # we hide them for inner panels).
        if i > 0:
            ax_cal.tick_params(labelleft=False)

    # Apply the global y-limits and invert.
    z_global_lo -= 0.5
    z_global_hi += 0.5
    axes[0].set_ylim(z_global_hi, z_global_lo)

    if len(campaigns) == 1:
        suptitle = f"Priority wells — caliper × raw SEC ({campaigns[0]})"
    else:
        suptitle = (
            f"Priority wells — caliper × raw SEC "
            f"({len(campaigns)} campaigns: {', '.join(campaigns)})"
        )
    fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995)

    # Explicit margins (tight_layout fails on figures with severity bands).
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.08, wspace=0.05)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────
#  Batch driver
# ──────────────────────────────────────────────────────────────────────
def build_all_sec_caliper_panels(
    *,
    campaigns: list[str],
    well_ids: Optional[Iterable[str]] = None,
    perpoint_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path | str] = None,
    output_dir: Optional[str | Path] = None,
    config: Optional[SecCaliperPanelConfig] = None,
    build_master: bool = True,
) -> list[Path]:
    """Render one PNG per well + an optional master figure.

    Output layout (v13)::

        results/figures/convergence/sec_caliper_panel/<campaign-or-multi>/
            <well_id>_sec_caliper.png            (one per well)
            master_panel.png                      (the 1×N master)

    The campaign-subfolder is the campaign name itself when rendering a
    single campaign (e.g. ``2022_02/``) or ``multi_<N>c/`` when
    overlaying several. File names are simple — the subfolder already
    identifies the campaign.

    Returns
    -------
    list[Path]
        Paths of all PNGs written, in order.
    """
    if not campaigns:
        raise ValueError(
            "build_all_sec_caliper_panels requires `campaigns` to be a "
            "non-empty list of campaign identifiers, e.g. ['2022_02']."
        )

    well_list = list(well_ids) if well_ids is not None else list(WELLS.keys())

    # v13 default: results/figures/convergence/sec_caliper_panel/<sub>/
    if output_dir is None:
        from karst_analysis.io import resolve_figure_dir
        output_dir = resolve_figure_dir(
            "convergence/sec_caliper_panel",
            campaigns=campaigns,
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for w in well_list:
        fig_path = out_dir / f"{w}_sec_caliper.png"
        try:
            fig = plot_sec_caliper_panel(
                w, campaigns=campaigns,
                perpoint_csv=perpoint_csv,
                master_caliper_csv=master_caliper_csv,
                project_root=project_root,
                config=config,
                output_path=fig_path,
            )
            plt.close(fig)
            print(f"  wrote {fig_path}")
            written.append(fig_path)
        except Exception as exc:
            print(f"  FAILED {w}: {exc}")

    if build_master and len(well_list) > 1:
        master_path = out_dir / "master_panel.png"
        try:
            fig = plot_master_panel(
                well_list, campaigns=campaigns,
                perpoint_csv=perpoint_csv,
                master_caliper_csv=master_caliper_csv,
                project_root=project_root,
                config=config,
                output_path=master_path,
            )
            plt.close(fig)
            print(f"  wrote {master_path}")
            written.append(master_path)
        except Exception as exc:
            print(f"  FAILED master: {exc}")

    return written
