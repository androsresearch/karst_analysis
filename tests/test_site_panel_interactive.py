"""Smoke tests for the v14 interactive site panel.

Coverage:
    * imports resolve correctly and the public API is reachable
    * configuration defaults match v12 (so figures look like their
      static counterpart)
    * helpers behave (hex_to_rgba, severity_band_shapes)
    * end-to-end: a real call returns a Plotly Figure with the
      expected number of subplots and writes a valid HTML file
"""

from __future__ import annotations

from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
HAS_FEB_2022 = (PROJECT_ROOT / "data" / "raw" / "sec" / "2022_02").exists()


# ──────────────────────────────────────────────────────────────────────
# Imports + module wiring
# ──────────────────────────────────────────────────────────────────────
def test_imports_are_wired():
    """The v14 public API is reachable from karst_analysis.convergence."""
    from karst_analysis.convergence import (
        InteractiveSitePanelConfig,
        plot_site_panel_interactive,
        build_all_site_panels_interactive,
    )
    cfg = InteractiveSitePanelConfig()
    # Defaults should mirror the v12 static config.
    assert cfg.sec_log_x is True
    assert cfg.sec_lw == 1.0
    assert cfg.sec_min_uS_cm == 200.0
    assert cfg.column_widths == (0.18, 0.82)
    # Plotly dash convention.
    assert cfg.well_type_dash["D"] == "solid"
    assert cfg.well_type_dash["O"] == "dot"
    assert cfg.well_type_dash["S"] == "dash"
    # Six official campaigns in palette.
    assert set(cfg.campaign_palette) == {
        "2011_05", "2022_02", "2022_08", "2023_08", "2025_02", "2025_11",
    }


# ──────────────────────────────────────────────────────────────────────
# _hex_to_rgba helper
# ──────────────────────────────────────────────────────────────────────
def test_hex_to_rgba_basic():
    from karst_analysis.convergence.site_panel_interactive import _hex_to_rgba

    assert _hex_to_rgba("#000000", 0.5) == "rgba(0,0,0,0.500)"
    assert _hex_to_rgba("#ffffff", 1.0) == "rgba(255,255,255,1.000)"
    assert _hex_to_rgba("#1f77b4", 0.35) == "rgba(31,119,180,0.350)"


def test_hex_to_rgba_handles_no_hash():
    from karst_analysis.convergence.site_panel_interactive import _hex_to_rgba
    # Lstrip handles the missing # gracefully.
    assert _hex_to_rgba("ff0000", 0.7) == "rgba(255,0,0,0.700)"


# ──────────────────────────────────────────────────────────────────────
# Input validation (no data needed)
# ──────────────────────────────────────────────────────────────────────
def test_plot_site_panel_interactive_validates_inputs():
    from karst_analysis.convergence import plot_site_panel_interactive

    with pytest.raises(ValueError, match="campaigns"):
        plot_site_panel_interactive("AW5", campaigns=[])

    with pytest.raises(ValueError, match="well_types"):
        plot_site_panel_interactive(
            "AW5", campaigns=["2022_02"], well_types=[],
        )


def test_build_all_site_panels_interactive_validates_inputs():
    from karst_analysis.convergence import build_all_site_panels_interactive

    with pytest.raises(ValueError, match="empty"):
        build_all_site_panels_interactive(campaigns=[])


# ──────────────────────────────────────────────────────────────────────
# End-to-end: real call producing a Plotly Figure + HTML
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_plot_site_panel_interactive_end_to_end(tmp_path):
    """Real call: AW5 with one campaign produces a Figure and an HTML."""
    import plotly.graph_objects as go
    from karst_analysis.convergence import plot_site_panel_interactive

    out = tmp_path / "AW5.html"
    fig = plot_site_panel_interactive(
        "AW5", campaigns=["2022_02"],
        project_root=PROJECT_ROOT,
        output_path=out,
    )
    assert isinstance(fig, go.Figure)
    # At least the D-well caliper trace + 1 SEC trace (AW5D in 2022_02)
    assert len(fig.data) >= 2
    # File written and non-trivial in size.
    assert out.exists()
    size_bytes = out.stat().st_size
    # Embedded plotly.js makes minimum ~3 MB.
    assert size_bytes > 1_000_000, f"HTML too small: {size_bytes} bytes"


@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_subplots_share_y_axis(tmp_path):
    """The two subplots share the y-axis → Plotly's `matches='y'` is set."""
    from karst_analysis.convergence import plot_site_panel_interactive

    fig = plot_site_panel_interactive(
        "AW5", campaigns=["2022_02"],
        project_root=PROJECT_ROOT,
    )
    # y-axes from make_subplots(shared_yaxes=True): yaxis is the
    # primary, yaxis2 has matches='y'.
    yaxis2 = fig.layout.yaxis2
    assert yaxis2.matches == "y", (
        "yaxis2 must match yaxis for synchronised depth zoom"
    )


@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_sec_traces_are_scattergl(tmp_path):
    """SEC traces use Scattergl for WebGL acceleration."""
    import plotly.graph_objects as go
    from karst_analysis.convergence import plot_site_panel_interactive

    fig = plot_site_panel_interactive(
        "AW5", campaigns=["2022_02", "2022_08"],
        project_root=PROJECT_ROOT,
    )
    # SEC traces are on subplot col=2 → xaxis='x2', yaxis='y2'.
    sec_traces = [
        t for t in fig.data
        if getattr(t, "xaxis", None) == "x2"
    ]
    assert len(sec_traces) > 0, "no SEC traces found"
    for t in sec_traces:
        assert isinstance(t, go.Scattergl), (
            f"SEC trace {t.name!r} is {type(t).__name__}, expected Scattergl"
        )


@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_severity_bands_drawn_as_shapes(tmp_path):
    """Severity bands are layout shapes, not traces (decision: shapes)."""
    from karst_analysis.convergence import plot_site_panel_interactive

    fig = plot_site_panel_interactive(
        "AW5", campaigns=["2022_02"],
        project_root=PROJECT_ROOT,
    )
    # AW5D has severity bands in its perpoint CSV → shapes should exist.
    assert len(fig.layout.shapes) > 0


@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_legendgroup_by_campaign(tmp_path):
    """SEC traces of the same campaign share a legendgroup (group toggle)."""
    from karst_analysis.convergence import plot_site_panel_interactive

    fig = plot_site_panel_interactive(
        "AW5", campaigns=["2022_02", "2022_08"],
        well_types=["D"],
        project_root=PROJECT_ROOT,
    )
    sec_traces = [t for t in fig.data if getattr(t, "xaxis", None) == "x2"]
    # All 2022_02 traces share legendgroup="2022_02"
    groups = {t.name.split()[1].rstrip(" *"): t.legendgroup
              for t in sec_traces if t.name and t.legendgroup}
    for name, lg in groups.items():
        # legendgroup should be the campaign id
        assert lg in ("2022_02", "2022_08"), (
            f"unexpected legendgroup {lg!r} for trace {name!r}"
        )


# ──────────────────────────────────────────────────────────────────────
# Batch driver
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_build_all_writes_one_html_per_site(tmp_path):
    from karst_analysis.convergence import build_all_site_panels_interactive

    written = build_all_site_panels_interactive(
        campaigns=["2022_02"],
        sites=["AW5", "AW6"],
        project_root=PROJECT_ROOT,
        output_dir=tmp_path,
    )
    # 2 sites → 2 HTML files (no master in interactive mode).
    assert len(written) == 2
    for p in written:
        assert p.exists()
        assert p.suffix == ".html"
    names = {p.name for p in written}
    assert "AW5_site_panel.html" in names
    assert "AW6_site_panel.html" in names
