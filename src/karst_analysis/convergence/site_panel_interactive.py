"""Interactive Plotly version of the site panel  (v14).

Mirror of :mod:`karst_analysis.convergence.site_panel` (v12, matplotlib)
producing a self-contained ``.html`` file per site. Designed to be
opened by anyone in any browser without a Python install — Plotly.js
is embedded into each file so internet access is not required.

The visual conventions match the static version:

* Two columns: caliper (narrow) + raw SEC (wide), with a shared
  depth axis (positive downward, zero at ground level).
* Caliper severity bands rendered as background ``shapes`` (no hover,
  no toggle — they are visual context only).
* SEC traces overlaid with:
      colour      = campaign  (DEFAULT_CAMPAIGN_PALETTE)
      line-style  = well type (D solid, O dotted, S dashed)
* Render-time SEC floor (default 200 µS/cm) drops in-air YSI readings.

Key interactivity
-----------------
* **Synchronised depth zoom.** Y-axes of the two subplots are linked
  so zooming/panning one subplot updates the other. X-axes are
  independent (caliper in cm, SEC in µS/cm — different ranges).
* **Legend toggling.** Click any legend item to hide/show that trace;
  double-click to isolate it. Traces are grouped by campaign via
  ``legendgroup`` so click-on-group hides all wells of that campaign.
* **WebGL rendering.** SEC traces use ``Scattergl`` so the figure
  stays responsive even with tens of thousands of points per cast.

This module does NOT replace the v12 static version. Both coexist
(``site_panel.py`` for static PNGs, ``site_panel_interactive.py`` for
interactive HTMLs). The same data-loading primitives are reused.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from karst_analysis.caliper.io import DEFAULT_MASTER_CSV, DEFAULT_PERPOINT_CSV
from karst_analysis.convergence.caliper_video import (
    SEVERITY_ALPHAS,
    SEVERITY_COLORS,
    WELLS,
    WellConfig,
    _caliper_from_perpoint,
    _load_companions_caliper,
    _load_perpoint_for,
    _severity_runs,
    _site_prefix,
)
from karst_analysis.convergence.site_panel import (
    DEFAULT_CAMPAIGN_PALETTE,
    WELL_TYPE_LINESTYLE,
    _all_priority_sites,
    _well_id_for,
)
from karst_analysis.sec.io import load_raw_ysi_traces_for_well


# ──────────────────────────────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────────────────────────────
@dataclass
class InteractiveSitePanelConfig:
    """Visual parameters for the interactive site panel.

    Defaults intentionally mirror v12's :class:`SitePanelConfig` so the
    interactive figure looks like its static counterpart.

    Attributes
    ----------
    width, height : int
        Figure size in pixels.
    column_widths : tuple[float, float]
        Relative widths (caliper, sec). Must sum to 1.0.
    sec_lw : float
        Width of SEC traces.
    sec_log_x : bool
        Log scale on the SEC axis (default True — see v12 rationale).
    sec_min_uS_cm : float
        Render-time floor on SEC values; points below are dropped from
        the figure (the underlying RawYsiTrace is unchanged).
    sat_cm : float
        Caliper saturation reference line.
    campaign_palette : dict[str, str]
        Stable colour mapping; defaults to
        :data:`DEFAULT_CAMPAIGN_PALETTE`.
    well_type_dash : dict[str, str]
        Plotly dash style per well type. Plotly uses 'solid', 'dot',
        'dash', 'dashdot', 'longdash', 'longdashdot'.
    """
    width: int = 1300
    height: int = 900
    column_widths: tuple[float, float] = (0.18, 0.82)
    sec_lw: float = 1.0
    sec_log_x: bool = True
    sec_min_uS_cm: float = 200.0
    sat_cm: float = 32.5
    campaign_palette: dict[str, str] = field(
        default_factory=lambda: dict(DEFAULT_CAMPAIGN_PALETTE)
    )
    well_type_dash: dict[str, str] = field(
        default_factory=lambda: {"D": "solid", "O": "dot", "S": "dash"}
    )
    fallback_marker: str = " *"


# ──────────────────────────────────────────────────────────────────────
#  Plotly conversion helpers
# ──────────────────────────────────────────────────────────────────────
def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Plotly shape `fillcolor` accepts ``rgba(r,g,b,a)`` strings."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _severity_band_shapes(
    perpoint_df: pd.DataFrame, *, xref: str = "x", yref: str = "y",
) -> list[dict]:
    """Build the list of `layout.shapes` dicts for severity bands.

    Mirrors v12's matplotlib ``_draw_severity_bands`` but emits Plotly
    rectangles instead. Bands are drawn behind the data via
    ``layer='below'``.
    """
    runs = _severity_runs(
        perpoint_df["depth_m"].to_numpy(),
        perpoint_df["severity_per_sample"].to_numpy(dtype=object),
    )
    shapes = []
    draw_order = {"mild": 0, "moderate": 1, "severe": 2}
    for sev, z_top, z_bot in sorted(runs, key=lambda r: draw_order.get(r[0], 0)):
        colour = SEVERITY_COLORS.get(sev, "#cccccc")
        alpha = SEVERITY_ALPHAS.get(sev, 0.35)
        # In our depth convention y increases downward; z_top < z_bot
        # numerically, but in the plot the "top" of a band is at
        # z_top (smaller depth) and the "bottom" at z_bot (larger
        # depth). For Plotly we just give y0 and y1 in either order.
        shapes.append(dict(
            type="rect",
            xref=f"{xref} domain", yref=yref,  # span full x of caliper
            x0=0, x1=1, y0=z_bot, y1=z_top,
            fillcolor=_hex_to_rgba(colour, alpha),
            line=dict(width=0),
            layer="below",
        ))
    return shapes


# ──────────────────────────────────────────────────────────────────────
#  Caliper data loading (reuses v12 primitives, returns DataFrames only)
# ──────────────────────────────────────────────────────────────────────
def _load_caliper_for_site(
    site: str,
    *,
    perpoint_csv: str | Path,
    master_caliper_csv: Optional[str | Path],
) -> Optional[dict]:
    """Pure-data version of v12's ``_render_site_caliper_axis``.

    Returns a dict with keys ``perpoint_df``, ``cal_df``, ``companions``,
    ``well_config``. None if the site has no D well in WELLS.
    """
    d_well_id = _well_id_for(site, "D")
    if d_well_id not in WELLS:
        return None
    wc: WellConfig = WELLS[d_well_id]
    perpoint_df = _load_perpoint_for(perpoint_csv, well=wc.caliper_well)
    cal_df = _caliper_from_perpoint(perpoint_df, auger_cm=wc.auger_cm)
    companions = (
        _load_companions_caliper(master_caliper_csv, primary_well=wc.caliper_well)
        if master_caliper_csv is not None else {}
    )
    return dict(
        perpoint_df=perpoint_df,
        cal_df=cal_df,
        companions=companions,
        well_config=wc,
    )


# ──────────────────────────────────────────────────────────────────────
#  Public: one site panel as Plotly Figure
# ──────────────────────────────────────────────────────────────────────
def plot_site_panel_interactive(
    site: str,
    *,
    campaigns: list[str],
    well_types: Optional[list[str]] = None,
    perpoint_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path | str] = None,
    config: Optional[InteractiveSitePanelConfig] = None,
    output_path: Optional[str | Path] = None,
) -> go.Figure:
    """Build the interactive site panel and (optionally) write it to disk.

    Parameters mirror those of :func:`plot_site_panel` (the v12 static
    counterpart). The returned object is a ``plotly.graph_objects.Figure``.

    If ``output_path`` is given, the figure is written as a
    self-contained HTML file (Plotly.js embedded, ~3 MB). Open the
    file in any modern browser; no server or Python required.
    """
    cfg = config or InteractiveSitePanelConfig()
    if not campaigns:
        raise ValueError("`campaigns` must be a non-empty list.")
    if well_types is None:
        well_types = ["D", "O", "S"]
    if not well_types:
        raise ValueError("`well_types` must be a non-empty list.")

    if perpoint_csv is None:        perpoint_csv = DEFAULT_PERPOINT_CSV
    if master_caliper_csv is None:  master_caliper_csv = DEFAULT_MASTER_CSV

    # ── Build the figure shell ──────────────────────────────────────
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=list(cfg.column_widths),
        shared_yaxes=True,           # ← synchronised depth zoom
        horizontal_spacing=0.04,
        subplot_titles=("Caliper (cm)", "SEC (µS/cm)"),
    )

    # Track depth range across panels so we can clip y-axis sensibly.
    z_mins: list[float] = []
    z_maxs: list[float] = []

    # ── Caliper column ──────────────────────────────────────────────
    cal = _load_caliper_for_site(
        site, perpoint_csv=perpoint_csv, master_caliper_csv=master_caliper_csv,
    )
    if cal is None:
        # Annotate the caliper subplot with a placeholder.
        fig.add_annotation(
            xref="x domain", yref="y domain", x=0.5, y=0.5,
            text=f"No caliper data<br>for site {site}",
            showarrow=False, font=dict(size=11, color="#9ca3af"),
            row=1, col=1,
        )
    else:
        wc = cal["well_config"]

        # Severity bands as background shapes (decision: shapes, not
        # interactive traces).
        for shape in _severity_band_shapes(
            cal["perpoint_df"], xref="x", yref="y",
        ):
            fig.add_shape(**shape, row=1, col=1)

        # Companion (O / S) caliper traces, if present.
        for cmp_name, cmp_df in cal["companions"].items():
            suffix = cmp_name[len(_site_prefix(cmp_name)):]
            dash = cfg.well_type_dash.get(suffix, "dash")
            fig.add_trace(
                go.Scatter(
                    x=cmp_df["caliper_cm"].to_numpy(),
                    y=cmp_df["depth_m"].to_numpy(),
                    mode="lines",
                    line=dict(color="#666666", width=0.7, dash=dash),
                    name=cmp_name,
                    legendgroup="caliper",
                    showlegend=True,
                    hovertemplate=(
                        f"<b>{cmp_name}</b><br>"
                        "Depth: %{y:.2f} m BGL<br>"
                        "Caliper: %{x:.2f} cm<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        # Primary D-well caliper trace.
        z = cal["cal_df"]["depth_m"].to_numpy()
        raw = cal["cal_df"]["raw_caliper_cm"].to_numpy()
        out_mask = cal["cal_df"]["is_outlier"].to_numpy().astype(bool)
        cal_plot = np.where(out_mask, np.nan, raw)
        fig.add_trace(
            go.Scatter(
                x=cal_plot, y=z,
                mode="lines",
                line=dict(color="#8e6914", width=0.9),
                name=wc.caliper_well,
                legendgroup="caliper",
                showlegend=True,
                hovertemplate=(
                    f"<b>{wc.caliper_well}</b><br>"
                    "Depth: %{y:.2f} m BGL<br>"
                    "Caliper: %{x:.2f} cm<extra></extra>"
                ),
            ),
            row=1, col=1,
        )
        # Outlier markers (the X's in the static version).
        if out_mask.any():
            fig.add_trace(
                go.Scatter(
                    x=raw[out_mask], y=z[out_mask],
                    mode="markers",
                    marker=dict(symbol="x", size=8, color="#c0392b",
                                line=dict(width=1)),
                    name=f"{wc.caliper_well} outliers",
                    legendgroup="caliper",
                    showlegend=True,
                    hovertemplate=(
                        "<b>outlier</b><br>"
                        "Depth: %{y:.2f} m BGL<br>"
                        "Caliper: %{x:.2f} cm<extra></extra>"
                    ),
                ),
                row=1, col=1,
            )

        # Saturation and auger reference vertical lines on the caliper axis.
        fig.add_vline(
            x=cfg.sat_cm, line=dict(color="#777777", width=0.7, dash="dot"),
            row=1, col=1,
        )
        fig.add_vline(
            x=wc.auger_cm, line=dict(color="#444444", width=0.6, dash="dot"),
            row=1, col=1,
        )

        z_mins.append(float(np.nanmin(z)))
        z_maxs.append(float(np.nanmax(z)))

    # ── SEC column ──────────────────────────────────────────────────
    palette = dict(cfg.campaign_palette)

    for campaign in campaigns:
        for wt in well_types:
            well_id = _well_id_for(site, wt)
            try:
                traces = load_raw_ysi_traces_for_well(
                    well_id, campaign, project_root=project_root,
                )
            except (FileNotFoundError, KeyError):
                continue
            if not traces:
                continue

            dash = cfg.well_type_dash.get(wt, "solid")

            for tr in traces:
                df = tr.df
                if "depth_bgl_m" not in df.columns:
                    continue
                z = df["depth_bgl_m"].to_numpy()
                s = df["sec_uS_cm"].to_numpy()

                # Render-time floor (drops in-air readings).
                if cfg.sec_min_uS_cm > 0:
                    valid = s >= cfg.sec_min_uS_cm
                    if not valid.any():
                        continue
                    z = z[valid]
                    s = s[valid]

                # Probe marker, e.g. "_R" / "_Y" — surfaces in trace
                # name but does not change the campaign colour.
                probe_suffix = (
                    f" probe {tr.probe}" if tr.probe is not None else ""
                )
                trace_name = f"{well_id} {campaign}{probe_suffix}"

                # Mark fallback in legend if applicable.
                fallback_suffix = (
                    cfg.fallback_marker
                    if (tr.vadose_resolution is not None
                        and tr.vadose_resolution.is_fallback)
                    else ""
                )
                trace_name_legend = f"{trace_name}{fallback_suffix}"

                colour = palette.get(campaign, "#888888")

                # customdata for hover: campaign + well type, broadcast
                # to every point.
                customdata = np.tile(
                    np.array([[campaign, wt]], dtype=object),
                    (z.size, 1),
                )

                fig.add_trace(
                    go.Scattergl(
                        x=s, y=z,
                        mode="lines",
                        line=dict(
                            color=colour,
                            width=cfg.sec_lw,
                            dash=dash,
                        ),
                        name=trace_name_legend,
                        legendgroup=campaign,    # campaign-level toggle
                        showlegend=True,
                        customdata=customdata,
                        hovertemplate=(
                            "<b>%{fullData.name}</b><br>"
                            "Depth: %{y:.2f} m BGL<br>"
                            "SEC: %{x:.0f} µS/cm<br>"
                            "Campaign: %{customdata[0]}<br>"
                            "Well type: %{customdata[1]}"
                            "<extra></extra>"
                        ),
                    ),
                    row=1, col=2,
                )
                z_mins.append(float(np.nanmin(z)))
                z_maxs.append(float(np.nanmax(z)))

    # ── Y axis (shared depth) ───────────────────────────────────────
    if z_mins and z_maxs:
        z_lo = min(z_mins) - 0.5
        z_hi = max(z_maxs) + 0.5
    else:
        z_lo, z_hi = 0.0, 30.0
    # Both subplots share y, so we set the range on the (shared) y1.
    fig.update_yaxes(
        range=[z_hi, z_lo],   # reversed: depth positive downward
        title_text="Depth below ground level (m)",
        row=1, col=1,
    )

    # ── X axes ──────────────────────────────────────────────────────
    # Caliper: linear, generous range around auger / saturation refs.
    fig.update_xaxes(
        title_text="Caliper (cm)",
        row=1, col=1,
    )
    # SEC: log by default (decision in v12).
    if cfg.sec_log_x:
        fig.update_xaxes(
            type="log",
            title_text="SEC (µS/cm, log scale)",
            row=1, col=2,
        )
    else:
        fig.update_xaxes(
            title_text="SEC (µS/cm)",
            row=1, col=2,
        )

    # ── Title and layout ────────────────────────────────────────────
    n_camp = len(campaigns)
    fig.update_layout(
        title=dict(
            text=(f"<b>Site {site}</b> — caliper × raw SEC "
                  f"({n_camp} campaign{'s' if n_camp > 1 else ''})"),
            x=0.5, xanchor="center",
            font=dict(size=14),
        ),
        width=cfg.width,
        height=cfg.height,
        legend=dict(
            groupclick="toggleitem",   # click on group toggles the item
            orientation="v",
            yanchor="top", y=1.0,
            xanchor="left", x=1.02,
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#cccccc",
            borderwidth=0.5,
            font=dict(size=9),
        ),
        margin=dict(l=70, r=200, t=70, b=60),
        hovermode="closest",
        plot_bgcolor="#fafafa",
    )

    # ── Write HTML ─────────────────────────────────────────────────
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # include_plotlyjs=True embeds plotly.js into the file (~3 MB)
        # so the file works offline on any machine.
        fig.write_html(
            str(out),
            include_plotlyjs=True,
            full_html=True,
            config={"displaylogo": False, "responsive": True},
        )

    return fig


# ──────────────────────────────────────────────────────────────────────
#  Public: batch driver
# ──────────────────────────────────────────────────────────────────────
def build_all_site_panels_interactive(
    *,
    campaigns: list[str],
    sites: Optional[Iterable[str]] = None,
    well_types: Optional[list[str]] = None,
    perpoint_csv: Optional[str | Path] = None,
    master_caliper_csv: Optional[str | Path] = None,
    project_root: Optional[Path | str] = None,
    output_dir: Optional[str | Path] = None,
    config: Optional[InteractiveSitePanelConfig] = None,
) -> list[Path]:
    """Render one HTML per site (no master figure in the interactive view).

    Output layout (v13/v14)::

        results/figures/convergence/site_panel_interactive/<sub>/
            <site>_site_panel.html

    The campaign-subfolder is the campaign name when rendering a single
    campaign, or ``multi_<N>c`` when overlaying several.

    Returns a list of the paths actually written.
    """
    if not campaigns:
        raise ValueError("`campaigns` is empty.")

    site_list = list(sites) if sites is not None else _all_priority_sites()

    # Resolve default output directory via the v13 helper.
    if output_dir is None:
        from karst_analysis.io import resolve_figure_dir
        output_dir = resolve_figure_dir(
            "convergence/site_panel_interactive",
            campaigns=campaigns,
        )
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for site in site_list:
        out_path = out_dir / f"{site}_site_panel.html"
        try:
            plot_site_panel_interactive(
                site, campaigns=campaigns, well_types=well_types,
                perpoint_csv=perpoint_csv,
                master_caliper_csv=master_caliper_csv,
                project_root=project_root,
                config=config,
                output_path=out_path,
            )
            print(f"  wrote {out_path}")
            written.append(out_path)
        except Exception as exc:
            print(f"  FAILED {site}: {exc}")
    return written
