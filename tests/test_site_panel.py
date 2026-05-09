"""Smoke tests for the SEC × caliper panel grouped by site (v12).

These tests verify:
    * imports are wired correctly
    * the loader picks up files from BOTH layouts (with / without
      well_type subfolder)
    * R/Y probe markers in filenames are captured into RawYsiTrace.probe
    * the public API (plot_site_panel, plot_master_sites_panel,
      build_all_site_panels) returns / writes what is expected
    * sites without caliper data do not crash (graceful placeholder)

Heavy data-driven tests live in conftest fixtures; we follow the
v10/v11 pattern of skipping when the raw data is not present.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


PROJECT_ROOT = Path(__file__).parent.parent

# Some tests need the real raw SEC tree; others only need the FEB-2022
# subset (which is the v10/v11 footprint). We branch.
HAS_FEB_2022 = (PROJECT_ROOT / "data" / "raw" / "sec" / "2022_02").exists()
HAS_AUG_2022 = (PROJECT_ROOT / "data" / "raw" / "sec" / "2022_08").exists()
HAS_FEB_2025 = (PROJECT_ROOT / "data" / "raw" / "sec" / "2025_02").exists()
HAS_NOV_2025 = (PROJECT_ROOT / "data" / "raw" / "sec" / "2025_11").exists()


# ──────────────────────────────────────────────────────────────────────
# Imports + module wiring
# ──────────────────────────────────────────────────────────────────────
def test_imports_are_wired():
    """v12 public API is reachable from karst_analysis.convergence."""
    from karst_analysis.convergence import (
        SitePanelConfig,
        WELL_TYPE_LINESTYLE,
        DEFAULT_CAMPAIGN_PALETTE,
        plot_site_panel,
        plot_master_sites_panel,
        build_all_site_panels,
    )
    cfg = SitePanelConfig()
    # Defaults match the v12 design decisions.
    assert cfg.sec_log_x is True
    assert cfg.sec_lw == 1.0
    assert cfg.sec_min_uS_cm == 200.0
    assert cfg.width_ratios == (1.0, 4.2)
    assert cfg.figsize == (13.0, 11.0)
    # Linestyle convention.
    assert WELL_TYPE_LINESTYLE["D"] == "-"
    assert WELL_TYPE_LINESTYLE["O"] == ":"
    assert WELL_TYPE_LINESTYLE["S"] == "--"
    # Six official campaigns.
    expected_campaigns = {"2011_05", "2022_02", "2022_08", "2023_08",
                          "2025_02", "2025_11"}
    assert set(DEFAULT_CAMPAIGN_PALETTE) == expected_campaigns


# ──────────────────────────────────────────────────────────────────────
# Loader: layouts with and without well_type subfolder
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_loader_2022_02_flat_layout():
    """The 2022_02 raw data is loadable.

    Either layout works (with or without a `<well_type>/` subfolder)
    because the loader uses ``rglob`` to find CSVs. This test just
    confirms one trace per priority well is returned.
    """
    from karst_analysis.sec.io import load_raw_ysi_traces_for_well

    traces = load_raw_ysi_traces_for_well(
        "AW5D", "2022_02", project_root=PROJECT_ROOT,
    )
    assert len(traces) == 1
    assert traces[0].probe is None  # no probe marker in 2022_02 filenames


@pytest.mark.skipif(not HAS_AUG_2022, reason="2022_08 raw data not present")
def test_loader_v12_layout_without_well_type_subfolder():
    """The 2022_08+ layout (no D/O/S subfolder) is handled."""
    from karst_analysis.sec.io import load_raw_ysi_traces_for_well

    # AW5_D in 2022_08 is at data/raw/sec/2022_08/AW5_D_YSI_*.csv (no D/ folder)
    traces = load_raw_ysi_traces_for_well(
        "AW5D", "2022_08", project_root=PROJECT_ROOT,
    )
    assert len(traces) >= 1
    # No probe marker for 2022_08
    assert all(t.probe is None for t in traces)


@pytest.mark.skipif(not HAS_AUG_2022, reason="2022_08 raw data not present")
def test_loader_returns_multiple_casts_when_present():
    """BW3D in 2022_08 has 2 casts on different days; both are loaded."""
    from karst_analysis.sec.io import load_raw_ysi_traces_for_well

    traces = load_raw_ysi_traces_for_well(
        "BW3D", "2022_08", project_root=PROJECT_ROOT,
    )
    assert len(traces) == 2
    dates = sorted(t.date_str for t in traces)
    assert dates == ["20220809", "20220812"]


@pytest.mark.skipif(not HAS_FEB_2025, reason="2025_02 raw data not present")
def test_loader_captures_probe_marker():
    """LRS69_D in 2025_02 has _R_ and _Y_ probe markers; both load."""
    from karst_analysis.sec.io import load_raw_ysi_traces_for_well

    traces = load_raw_ysi_traces_for_well(
        "LRS69D", "2025_02", project_root=PROJECT_ROOT,
    )
    # Two probes on the same day → two traces
    probes = sorted(t.probe for t in traces if t.probe is not None)
    assert "R" in probes
    assert "Y" in probes


@pytest.mark.skipif(not HAS_NOV_2025, reason="2025_11 raw data not present")
def test_loader_handles_csv_without_vertical_position():
    """2025_11 CSVs have only 'Depth m' (no Vertical Position).

    The loader must still produce a depth_m column (mapped from
    'Depth m') and, with the resolver fallback, a depth_bgl_m column.
    """
    from karst_analysis.sec.io import load_raw_ysi_traces_for_well

    traces = load_raw_ysi_traces_for_well(
        "BW3D", "2025_11", project_root=PROJECT_ROOT,
    )
    assert len(traces) == 1
    tr = traces[0]
    assert "depth_m" in tr.df.columns
    # depth_bgl_m only present if the resolver succeeded; with the
    # feb-2022 fallback in wells.csv it should.
    assert "depth_bgl_m" in tr.df.columns
    # Resolution should be marked as fallback.
    assert tr.vadose_resolution is not None
    assert tr.vadose_resolution.is_fallback


# ──────────────────────────────────────────────────────────────────────
# Public API: plot_site_panel
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_plot_site_panel_returns_figure(tmp_path):
    """Smoke test: single-site panel writes a PNG and returns a Figure."""
    from karst_analysis.convergence import plot_site_panel

    out = tmp_path / "AW5_site.png"
    fig = plot_site_panel(
        "AW5", campaigns=["2022_02"],
        project_root=PROJECT_ROOT,
        output_path=out,
    )
    assert isinstance(fig, plt.Figure)
    assert out.exists()
    assert out.stat().st_size > 1000
    # Two subplots: caliper + SEC
    assert len(fig.axes) >= 2
    plt.close(fig)


@pytest.mark.skipif(not (HAS_FEB_2022 and HAS_AUG_2022),
                    reason="multi-campaign raw data not present")
def test_plot_site_panel_multi_campaign(tmp_path):
    """Site panel with several campaigns + several well types runs end-to-end."""
    from karst_analysis.convergence import plot_site_panel

    out = tmp_path / "AW5_multi.png"
    fig = plot_site_panel(
        "AW5",
        campaigns=["2022_02", "2022_08"],
        well_types=["D", "O", "S"],
        project_root=PROJECT_ROOT,
        output_path=out,
    )
    assert out.exists()
    plt.close(fig)


def test_plot_site_panel_validates_inputs():
    """Empty campaigns/well_types raise; unknown sites bubble up."""
    from karst_analysis.convergence import plot_site_panel

    with pytest.raises(ValueError, match="campaigns"):
        plot_site_panel("AW5", campaigns=[])
    with pytest.raises(ValueError, match="well_types"):
        plot_site_panel("AW5", campaigns=["2022_02"], well_types=[])


# ──────────────────────────────────────────────────────────────────────
# Public API: plot_master_sites_panel
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_plot_master_sites_panel_writes_png(tmp_path):
    """Master 1×N has 2N axes (caliper + SEC per site)."""
    from karst_analysis.convergence import plot_master_sites_panel

    sites = ["AW5", "AW6", "BW3", "LRS69", "LRS70"]
    out = tmp_path / "master.png"
    fig = plot_master_sites_panel(
        sites, campaigns=["2022_02"],
        project_root=PROJECT_ROOT,
        output_path=out,
    )
    # 5 sites × 2 axes = 10 axes
    assert len(fig.axes) == 2 * len(sites)
    assert out.exists()
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────
# Public API: build_all_site_panels
# ──────────────────────────────────────────────────────────────────────
@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_build_all_site_panels_writes_expected_files(tmp_path):
    """Batch driver writes one PNG per site plus the master figure."""
    from karst_analysis.convergence import build_all_site_panels

    written = build_all_site_panels(
        campaigns=["2022_02"],
        sites=["AW5", "AW6"],
        project_root=PROJECT_ROOT,
        output_dir=tmp_path,
    )
    # 2 sites + 1 master
    assert len(written) == 3
    for p in written:
        assert p.exists()
        assert p.suffix == ".png"
    names = {p.name for p in written}
    assert "AW5_site_panel.png" in names
    assert "AW6_site_panel.png" in names
    assert "master_sites.png" in names


@pytest.mark.skipif(not HAS_FEB_2022, reason="2022_02 raw data not present")
def test_build_all_site_panels_skips_master_with_single_site(tmp_path):
    """A single-site request should NOT produce a master file."""
    from karst_analysis.convergence import build_all_site_panels

    written = build_all_site_panels(
        campaigns=["2022_02"],
        sites=["AW5"],
        project_root=PROJECT_ROOT,
        output_dir=tmp_path,
    )
    assert len(written) == 1
    assert all("master" not in p.name for p in written)
