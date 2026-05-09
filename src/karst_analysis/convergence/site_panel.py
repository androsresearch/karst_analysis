"""SEC raw traces × caliper panel per priority SITE  (v12).

A two-column figure for one priority *site* — one panel for AW5,
one for AW6, one for BW3, one for LRS69, one for LRS70:

    column 1 (caliper) : per-sample severity bands + caliper signal
                         for the site's D well (the only one with a
                         caliper run).
    column 2 (SEC)     : raw YSI traces for ALL wells of the site
                         (D, O, S where they exist) across ALL the
                         requested campaigns. Each trace is identified
                         by:
                            colour    = campaign (Plotly Dark24 palette)
                            line-style= well type (D solid, O dotted, S dashed)

Differences w.r.t. ``sec_caliper_panel.py`` (v11)
-------------------------------------------------
v11 panels are keyed by *well* (one panel per pozo D). v12 panels
are keyed by *site* (one panel per sitio, with all that site's
wells overlaid). Both modules coexist; v11 is preserved verbatim.

When a campaign uses two YSI sondes (suffix ``_R`` / ``_Y`` in the
filename), each probe gets its own legend label and its own colour
slot — they are treated as independent traces of the same campaign.

Vadose handling follows the v11 ``VadoseResolver`` policy: explicit
row in ``data/metadata/wells.csv`` → computation from the YSI CSV's
own columns → fallback to the reference campaign. Campaigns that
end up using a fallback are marked with ``*`` in the legend.

Outputs
-------
``plot_site_panel(site, *, campaigns, well_types)`` returns a
``matplotlib.Figure``.
``plot_master_sites_panel(sites, ...)`` arranges several site panels
side by side with a shared depth axis.
``build_all_site_panels(...)`` is the batch driver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
from karst_analysis.convergence.sec_caliper_panel import (
    DARK24_PALETTE,
    _resolve_campaign_palette,
)
from karst_analysis.sec.io import load_raw_ysi_traces_for_well


# ──────────────────────────────────────────────────────────────────────
#  Site model — one site holds 1..3 wells of types D/O/S
# ──────────────────────────────────────────────────────────────────────
WELL_TYPE_LINESTYLE: dict[str, str] = {
    "D": "-",
    "O": ":",
    "S": "--",
}
"""Linestyle assigned to each well type when overlaying on the SEC axis.

The convention (D solid, O dotted, S dashed) is documented in the
README and matches the visual emphasis in the thesis: the Deep well is
the primary trace; Old and Shallow are subordinate.
"""


# Stable, project-wide colour assignment for the six official campaigns.
# Keeping the same colour for the same campaign across all panels of
# the thesis makes the figures mutually readable. Override per-call
# via SitePanelConfig.campaign_palette if needed.
DEFAULT_CAMPAIGN_PALETTE: dict[str, str] = {
    "2011_05": "#7f7f7f",   # grey   — historical context
    "2022_02": "#1f77b4",   # blue
    "2022_08": "#2ca02c",   # green
    "2023_08": "#ff7f0e",   # orange
    "2025_02": "#9467bd",   # purple
    "2025_11": "#000000",   # black
}
"""Default campaign-to-colour mapping for the v12 site panel.

The list is closed (six entries) to match the six campaigns Mar
defined as the official ones for her thesis. Campaigns NOT in this
mapping fall back to Plotly's Dark24 palette (auto-assigned in the
order they appear in the request) — this is what happens by default
for any new campaign added later.
"""


# Sites are derived from the existing WELLS table (which is keyed by
# the D-well well_id). Each site can host up to three wells (D, O, S);
# the actual presence is decided dynamically by what files exist on
# disk.
def _all_priority_sites() -> list[str]:
    """Return the list of site prefixes for which a D well is configured."""
    return sorted({_site_prefix(wid) for wid in WELLS})


def _well_id_for(site: str, well_type: str) -> str:
    """Build the canonical well_id from a (site, well_type) pair."""
    return f"{site}{well_type}"


# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class SitePanelConfig:
    """Visual parameters for the SEC × caliper panel grouped by site.

    Attributes
    ----------
    figsize : (float, float)
        Single-site panel size in inches. Default is wide (13×11) so
        the SEC column has room to breathe; width_ratios then controls
        how that width is split between caliper and SEC.
    width_ratios : tuple[float, float]
        (caliper_width, sec_width). Default 1:4.2 — the caliper column
        stays narrow (it lives in a small range around the auger
        nominal); the SEC column is wide because it holds many
        overlapping traces.
    sec_alpha : float
    sec_lw : float
        SEC trace line width. Default 1.0 (thicker than v11's 0.7,
        because overlapping many traces from O / S / D needs slightly
        bolder lines to be readable).
    sec_log_x : bool
        Default True for the site panel: in coastal-aquifer SEC the
        dynamic range spans ~3 decades (from ~500 µS/cm freshwater up
        to ~55,000 µS/cm seawater) and a linear axis crushes the
        freshwater portion against the y-axis, hiding the O and S
        traces. Set False to reproduce the v11 linear look.
    sec_min_uS_cm : float
        Render-time floor on the SEC values plotted on this panel.
        Points strictly below this threshold are dropped *visually*;
        the underlying RawYsiTrace.df is NOT modified. The default
        200 µS/cm filters out instrumental "in-air" readings (~1-50
        µS/cm) that some YSI casts contain at the start of the
        descent. Coastal-aquifer freshwater zones legitimately reach
        down to ~500 µS/cm, so 200 is a conservative cut. Set to 0.0
        to disable the filter.
    grid_alpha : float
    sat_cm : float
        Caliper saturation reference line (cm).
    campaign_palette : dict[str, str] or None
        Optional explicit ``campaign -> hex_colour`` mapping. The
        default is :data:`DEFAULT_CAMPAIGN_PALETTE` (six fixed colours
        for the six official campaigns of the thesis). Pass an explicit
        dict to override; campaigns NOT in the mapping are auto-filled
        from Dark24 in call order.
    fallback_marker : str
        Suffix appended to legend labels when a campaign's vadose came
        from the fallback level. Default ``" *"``.
    well_type_linestyle : dict[str, str]
        Mapping ``well_type -> matplotlib linestyle`` used when more
        than one well type appears in the SEC axis. Default is
        ``{"D": "-", "O": ":", "S": "--"}``.
    """
    figsize: tuple[float, float] = (13.0, 11.0)
    width_ratios: tuple[float, float] = (1.0, 4.2)
    sec_alpha: float = 0.7
    sec_lw: float = 1.0
    sec_log_x: bool = True
    sec_min_uS_cm: float = 200.0
    grid_alpha: float = 0.45
    sat_cm: float = 32.5
    campaign_palette: Optional[dict[str, str]] = field(
        default_factory=lambda: dict(DEFAULT_CAMPAIGN_PALETTE)
    )
    fallback_marker: str = " *"
    well_type_linestyle: dict[str, str] = field(
        default_factory=lambda: dict(WELL_TYPE_LINESTYLE)
    )


# ──────────────────────────────────────────────────────────────────────
#  Caliper axis (re-used from the v11 single-well rendering helper)
# ──────────────────────────────────────────────────────────────────────
def _render_site_caliper_axis(
    ax: plt.Axes,
    site: str,
    *,
    perpoint_csv: str | Path,
    master_caliper_csv: Optional[str | Path],
    cfg: SitePanelConfig,
    show_xlabel: bool = True,
) -> tuple[float, float]:
    """Draw the caliper signal of the site's D well, plus severity
    bands and companion (O/S) caliper traces if the master caliper CSV
    has them.

    Returns the (z_min, z_max) range covered by the caliper data so
    that ``plot_site_panel`` can synchronise the y-limits with the SEC
    axis.

    If the site does not have a caliper run (no D well in WELLS), the
    axis displays a placeholder text and returns (np.inf, -np.inf).
    """
    d_well_id = _well_id_for(site, "D")
    if d_well_id not in WELLS:
        ax.text(
            0.5, 0.5,
            f"No caliper data\nfor site {site}",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#9ca3af", style="italic",
        )
        ax.set_xlabel("Caliper (cm)" if show_xlabel else "", fontsize=9)
        return float("inf"), float("-inf")

    wc: WellConfig = WELLS[d_well_id]
    perpoint_df = _load_perpoint_for(perpoint_csv, well=wc.caliper_well)
    cal_df = _caliper_from_perpoint(perpoint_df, auger_cm=wc.auger_cm)
    companions = (
        _load_companions_caliper(master_caliper_csv, primary_well=wc.caliper_well)
        if master_caliper_csv is not None else {}
    )

    # severity bands underneath
    _draw_severity_bands(ax, perpoint_df, alpha_factor=1.0)

    # companion caliper traces (O/S of the same site, if the master CSV has them)
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

    # primary D-well caliper trace, with outliers masked
    z = cal_df["depth_m"].to_numpy()
    raw = cal_df["raw_caliper_cm"].to_numpy()
    out_mask = cal_df["is_outlier"].to_numpy().astype(bool)
    cal_plot = np.where(out_mask, np.nan, raw)
    ax.plot(cal_plot, z, color="#8e6914", lw=0.6, alpha=0.85,
            zorder=4, label=wc.caliper_well)
    if out_mask.any():
        ax.scatter(raw[out_mask], z[out_mask], s=30, marker="x",
                   c="#c0392b", zorder=5, linewidths=1.0)

    # references
    ax.axvline(cfg.sat_cm, color="#777777", lw=0.6, ls=":", alpha=0.55, zorder=2)
    ax.axvline(wc.auger_cm, color="#444444", lw=0.5, ls=":", alpha=0.4, zorder=2)

    # x-limits
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


# ──────────────────────────────────────────────────────────────────────
#  SEC axis — overlay all (campaign, well_type, probe) traces of the site
# ──────────────────────────────────────────────────────────────────────
def _render_site_sec_axis(
    ax: plt.Axes,
    site: str,
    campaigns: list[str],
    well_types: list[str],
    *,
    project_root: Optional[Path | str],
    cfg: SitePanelConfig,
    short_xlabel: bool = False,
    legend_for_campaigns: bool = True,
    legend_for_well_types: bool = True,
) -> tuple[float, float, list[tuple[str, str]], list[tuple[str, str]]]:
    """Draw all SEC traces for the site onto ``ax``.

    Iterates over every (well_type, campaign) combination and loads
    the matching raw YSI files via ``load_raw_ysi_traces_for_well``.

    Returns
    -------
    (z_min, z_max, campaign_legend_entries, well_type_legend_entries)

    The two legend-entries lists are returned in *display* form, so
    the caller can place them in either an in-axis legend or a
    figure-level legend (used by the master panel). Each entry is
    ``(label, dark24_hex_or_linestyle)``.
    """
    palette = _resolve_campaign_palette(campaigns, cfg.campaign_palette)
    z_mins: list[float] = []
    z_maxs: list[float] = []

    # Collect a per-campaign fallback flag so we can mark the legend.
    campaign_used_fallback: dict[str, bool] = {c: False for c in campaigns}
    campaign_drawn: dict[str, bool] = {c: False for c in campaigns}
    well_type_drawn: dict[str, bool] = {wt: False for wt in well_types}

    for wt in well_types:
        well_id = _well_id_for(site, wt)
        ls = cfg.well_type_linestyle.get(wt, "-")
        for campaign in campaigns:
            try:
                traces = load_raw_ysi_traces_for_well(
                    well_id, campaign,
                    project_root=project_root,
                    well_type=wt,
                    add_depth_bgl=True,
                )
            except FileNotFoundError:
                continue

            for tr in traces:
                df = tr.df
                if "depth_bgl_m" not in df.columns:
                    continue
                z = df["depth_bgl_m"].to_numpy()
                s = df["sec_uS_cm"].to_numpy()
                # Render-time filter: drop points below the configured
                # SEC floor (default 200 µS/cm) so instrumental "in-air"
                # readings don't dominate a log-scale axis. The
                # underlying df is unchanged.
                if cfg.sec_min_uS_cm > 0:
                    valid = s >= cfg.sec_min_uS_cm
                    if not valid.any():
                        continue
                    z = z[valid]
                    s = s[valid]
                colour = palette[campaign]
                ax.plot(
                    s, z,
                    color=colour, lw=cfg.sec_lw, alpha=cfg.sec_alpha,
                    linestyle=ls,
                )
                z_mins.append(float(np.nanmin(z)))
                z_maxs.append(float(np.nanmax(z)))
                campaign_drawn[campaign] = True
                well_type_drawn[wt] = True
                if (tr.vadose_resolution is not None
                        and tr.vadose_resolution.is_fallback):
                    campaign_used_fallback[campaign] = True

    if cfg.sec_log_x:
        ax.set_xscale("log")
    if short_xlabel:
        ax.set_xlabel("SEC (μS/cm)", fontsize=9)
    else:
        ax.set_xlabel("SEC (μS/cm)", fontsize=9)
    ax.grid(True, axis="both", alpha=cfg.grid_alpha, linestyle=":")
    ax.tick_params(axis="both", labelsize=8)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    if not z_mins:
        ax.text(
            0.5, 0.5,
            f"No SEC data for site {site}\n"
            f"(campaigns: {', '.join(campaigns)})",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color="#9ca3af", style="italic",
        )
        return float("inf"), float("-inf"), [], []

    # Build legend entries — only for items actually drawn.
    campaign_entries: list[tuple[str, str]] = []
    for c in campaigns:
        if not campaign_drawn[c]:
            continue
        label = c + (cfg.fallback_marker if campaign_used_fallback[c] else "")
        campaign_entries.append((label, palette[c]))

    well_type_entries: list[tuple[str, str]] = []
    for wt in well_types:
        if not well_type_drawn[wt]:
            continue
        well_type_entries.append((wt, cfg.well_type_linestyle.get(wt, "-")))

    return min(z_mins), max(z_maxs), campaign_entries, well_type_entries


# ──────────────────────────────────────────────────────────────────────
#  Public: single-site figure
# ──────────────────────────────────────────────────────────────────────
def plot_site_panel(
    site: str,
    *,
    campaigns: list[str],
    well_types: Optional[list[str]] = None,
    perpoint_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path | str] = None,
    config: Optional[SitePanelConfig] = None,
    output_path: Optional[str | Path] = None,
) -> plt.Figure:
    """Render the 2-column SEC × caliper panel for one site.

    Parameters
    ----------
    site : str
        Site prefix, e.g. ``"AW5"``, ``"LRS70"``. Must correspond to a
        D well configured in :data:`WELLS`.
    campaigns : list[str]
        One or more field campaigns to overlay on the SEC axis.
    well_types : list[str], optional
        Subset of well types to include (default: ``["D", "O", "S"]``).
        Wells that do not have any matching CSV are silently skipped.
    perpoint_csv, master_caliper_csv : path-like, optional
        Caliper inputs.
    project_root : Path or str, optional
        Defaults to ``Path.cwd()``.
    config : SitePanelConfig, optional
    output_path : path-like, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    cfg = config or SitePanelConfig()
    if not campaigns:
        raise ValueError("`campaigns` must be a non-empty list.")
    if well_types is None:
        well_types = ["D", "O", "S"]
    if not well_types:
        raise ValueError("`well_types` must be a non-empty list.")

    if perpoint_csv is None:        perpoint_csv = DEFAULT_PERPOINT_CSV
    if master_caliper_csv is None:  master_caliper_csv = DEFAULT_MASTER_CSV

    fig, (ax_cal, ax_sec) = plt.subplots(
        1, 2, figsize=cfg.figsize, sharey=True,
        gridspec_kw=dict(width_ratios=cfg.width_ratios, wspace=0.05),
    )

    z_lo_c, z_hi_c = _render_site_caliper_axis(
        ax_cal, site,
        perpoint_csv=perpoint_csv,
        master_caliper_csv=master_caliper_csv,
        cfg=cfg,
    )
    z_lo_s, z_hi_s, camp_entries, wt_entries = _render_site_sec_axis(
        ax_sec, site, campaigns, well_types,
        project_root=project_root,
        cfg=cfg,
    )

    # Shared y-limits.
    z_lo = min(z_lo_c, z_lo_s)
    z_hi = max(z_hi_c, z_hi_s)
    if not np.isfinite(z_lo) or not np.isfinite(z_hi):
        # Should not happen unless the site has no caliper AND no SEC.
        z_lo, z_hi = 0.0, 30.0
    z_lo -= 0.5
    z_hi += 0.5
    ax_cal.set_ylim(z_hi, z_lo)
    ax_cal.set_ylabel("Depth below ground level (m)", fontsize=10)

    fig.suptitle(
        f"Site {site} — caliper × raw SEC ({len(campaigns)} campaign"
        f"{'s' if len(campaigns) > 1 else ''})",
        fontsize=12, fontweight="bold", y=0.995,
    )

    # In-axis legends on the SEC axis: campaigns (colour) and well
    # types (linestyle), as two separate small legends so they read
    # independently.
    handles_campaigns = [
        Line2D([0], [0], color=clr, lw=2.0, label=lbl)
        for (lbl, clr) in camp_entries
    ]
    handles_well_types = [
        Line2D([0], [0], color="#222222", lw=1.6, linestyle=ls, label=wt)
        for (wt, ls) in wt_entries
    ]

    if handles_campaigns:
        leg_c = ax_sec.legend(
            handles=handles_campaigns,
            loc="lower left", fontsize=7, framealpha=0.9,
            title="campaign", title_fontsize=7,
        )
        ax_sec.add_artist(leg_c)
    if handles_well_types:
        ax_sec.legend(
            handles=handles_well_types,
            loc="upper right", fontsize=7, framealpha=0.9,
            title="well type", title_fontsize=7,
        )

    fig.subplots_adjust(left=0.10, right=0.97, top=0.94, bottom=0.07, wspace=0.05)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────
#  Public: master 1×N (sites side-by-side, shared depth axis)
# ──────────────────────────────────────────────────────────────────────
def plot_master_sites_panel(
    sites: Iterable[str],
    *,
    campaigns: list[str],
    well_types: Optional[list[str]] = None,
    perpoint_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path | str] = None,
    config: Optional[SitePanelConfig] = None,
    output_path: Optional[str | Path] = None,
    per_site_width: float = 3.0,
    height: float = 11.0,
) -> plt.Figure:
    """Render all the requested sites side-by-side, with a shared y-axis.

    Each site occupies TWO sub-axes (caliper + SEC). The legend appears
    once at the bottom of the figure (campaign colours and well-type
    line styles), to avoid repeating it per site.
    """
    cfg = config or SitePanelConfig()
    site_list = list(sites)
    if not site_list:
        raise ValueError("`sites` is empty.")
    if not campaigns:
        raise ValueError("`campaigns` is empty.")
    if well_types is None:
        well_types = ["D", "O", "S"]

    if perpoint_csv is None:        perpoint_csv = DEFAULT_PERPOINT_CSV
    if master_caliper_csv is None:  master_caliper_csv = DEFAULT_MASTER_CSV

    n = len(site_list)
    figsize = (per_site_width * n, height)
    width_ratios: list[float] = []
    for _ in site_list:
        width_ratios.extend(cfg.width_ratios)

    fig, axes = plt.subplots(
        1, 2 * n, figsize=figsize, sharey=True,
        gridspec_kw=dict(width_ratios=width_ratios, wspace=0.05),
    )

    z_global_lo = float("inf")
    z_global_hi = float("-inf")

    # Build a global palette across all sites so colours are stable.
    palette = _resolve_campaign_palette(campaigns, cfg.campaign_palette)

    # Collect overall fallback status across all sites for the master legend.
    overall_camp_fallback: dict[str, bool] = {c: False for c in campaigns}

    for i, site in enumerate(site_list):
        ax_cal = axes[2 * i]
        ax_sec = axes[2 * i + 1]

        z_lo_c, z_hi_c = _render_site_caliper_axis(
            ax_cal, site,
            perpoint_csv=perpoint_csv,
            master_caliper_csv=master_caliper_csv,
            cfg=cfg,
            show_xlabel=True,
        )
        z_lo_s, z_hi_s, camp_entries, _wt_entries = _render_site_sec_axis(
            ax_sec, site, campaigns, well_types,
            project_root=project_root,
            cfg=cfg,
            short_xlabel=True,
        )

        # Suppress per-axis legends in master view (we make a global one).
        _l = ax_sec.get_legend()
        if _l is not None:
            _l.remove()

        # Track which campaigns hit fallback at any site.
        for (label, _clr) in camp_entries:
            base = label.replace(cfg.fallback_marker, "")
            if cfg.fallback_marker in label:
                overall_camp_fallback[base] = True

        z_global_lo = min(z_global_lo, z_lo_c, z_lo_s)
        z_global_hi = max(z_global_hi, z_hi_c, z_hi_s)

        ax_cal.set_title(site, fontsize=11, fontweight="bold", pad=8)
        if i == 0:
            ax_cal.set_ylabel("Depth below ground level (m)", fontsize=10)
        else:
            ax_cal.tick_params(labelleft=False)

    if not np.isfinite(z_global_lo) or not np.isfinite(z_global_hi):
        z_global_lo, z_global_hi = 0.0, 30.0
    z_global_lo -= 0.5
    z_global_hi += 0.5
    axes[0].set_ylim(z_global_hi, z_global_lo)

    fig.suptitle(
        f"Priority sites — caliper × raw SEC "
        f"({len(campaigns)} campaign{'s' if len(campaigns) > 1 else ''})",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # Global double legend at the bottom of the figure.
    handles_campaigns = []
    for c in campaigns:
        label = c + (cfg.fallback_marker if overall_camp_fallback[c] else "")
        handles_campaigns.append(
            Line2D([0], [0], color=palette[c], lw=2.5, label=label)
        )
    handles_well_types = [
        Line2D([0], [0], color="#222222", lw=1.6,
               linestyle=cfg.well_type_linestyle.get(wt, "-"),
               label=f"well type {wt}")
        for wt in well_types
    ]

    fig.legend(
        handles=handles_campaigns + handles_well_types,
        loc="lower center",
        ncol=min(len(handles_campaigns) + len(handles_well_types), 8),
        fontsize=8, frameon=True, framealpha=0.9,
        bbox_to_anchor=(0.5, 0.005),
        title=(
            f"campaigns ('{cfg.fallback_marker.strip()}' = vadose from "
            f"fallback) and well types"
        ),
        title_fontsize=8,
    )
    fig.subplots_adjust(left=0.05, right=0.99, top=0.93, bottom=0.13,
                        wspace=0.05)

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")

    return fig


# ──────────────────────────────────────────────────────────────────────
#  Batch driver
# ──────────────────────────────────────────────────────────────────────
def build_all_site_panels(
    *,
    campaigns: list[str],
    sites: Optional[Iterable[str]] = None,
    well_types: Optional[list[str]] = None,
    perpoint_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path | str] = None,
    output_dir: Optional[str | Path] = None,
    config: Optional[SitePanelConfig] = None,
    build_master: bool = True,
) -> list[Path]:
    """Render one PNG per site + an optional master figure with all sites.

    Output layout (v13)::

        results/figures/convergence/site_panel/<campaign-or-multi>/
            <site>_site_panel.png       (one per site)
            master_sites.png             (the 1×N master)

    The campaign-subfolder is the campaign name itself when rendering a
    single campaign (e.g. ``2022_02/``) or ``multi_<N>c/`` when
    overlaying several. File names are simple — the subfolder already
    identifies the campaign.

    Returns a list of the paths actually written (in render order).
    """
    if not campaigns:
        raise ValueError("`campaigns` is empty.")

    site_list = list(sites) if sites is not None else _all_priority_sites()

    # v13 default: results/figures/convergence/site_panel/<sub>/
    if output_dir is None:
        from karst_analysis.io import resolve_figure_dir
        output_dir = resolve_figure_dir(
            "convergence/site_panel",
            campaigns=campaigns,
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for site in site_list:
        fig_path = out_dir / f"{site}_site_panel.png"
        try:
            fig = plot_site_panel(
                site, campaigns=campaigns, well_types=well_types,
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
            print(f"  FAILED {site}: {exc}")

    if build_master and len(site_list) > 1:
        master_path = out_dir / "master_sites.png"
        try:
            fig = plot_master_sites_panel(
                site_list, campaigns=campaigns, well_types=well_types,
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
