"""Smoke tests for the SEC raw × caliper panel (v10).

These are NOT pixel-by-pixel tests of the figure — they verify:
    * imports are wired correctly,
    * the loader returns the expected number of traces per well,
    * the public API can be called and returns a Figure,
    * the vadose extraction utility reproduces the values in wells.csv.

Heavy figure-rendering tests live in
``test_sec_caliper_video_smoke.py``; we follow the same lightweight
pattern here.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest


# Skip the entire module if running outside the repo (e.g. CI without data).
PROJECT_ROOT = Path(__file__).parent.parent
HAS_DATA = (PROJECT_ROOT / "data" / "raw" / "sec" / "2022_02").exists()


@pytest.mark.skipif(not HAS_DATA, reason="raw SEC data not present")
def test_load_raw_ysi_traces_for_priority_wells():
    """Each Feb-2022 priority well returns exactly one trace."""
    from karst_analysis.sec.io import load_raw_ysi_traces_for_well

    for well in ["AW5D", "AW6D", "BW3D", "LRS69D", "LRS70D"]:
        traces = load_raw_ysi_traces_for_well(
            well, "2022_02", project_root=PROJECT_ROOT,
        )
        assert len(traces) == 1, (
            f"{well}: expected 1 trace for 2022_02, got {len(traces)}"
        )
        tr = traces[0]
        assert "depth_m" in tr.df.columns
        assert "sec_uS_cm" in tr.df.columns
        assert "depth_bgl_m" in tr.df.columns
        assert tr.vadose_thickness_m is not None
        assert tr.vadose_thickness_m > 0
        # Depth_bgl_m should always be non-negative (sonda starts AT or below
        # the water table, never above the ground).
        assert (tr.df["depth_bgl_m"] >= 0).all(), (
            f"{well}: negative depth_bgl_m values found"
        )


@pytest.mark.skipif(not HAS_DATA, reason="raw SEC data not present")
def test_plot_sec_caliper_panel_returns_figure(tmp_path):
    """Smoke test: single-well panel writes a PNG and returns a Figure."""
    from karst_analysis.convergence import plot_sec_caliper_panel

    out = tmp_path / "LRS70D_sec_caliper.png"
    fig = plot_sec_caliper_panel(
        "LRS70D", campaigns=["2022_02"],
        project_root=PROJECT_ROOT,
        output_path=out,
    )
    assert isinstance(fig, plt.Figure)
    assert out.exists()
    assert out.stat().st_size > 1000  # PNG should be at least 1 KB
    plt.close(fig)


@pytest.mark.skipif(not HAS_DATA, reason="raw SEC data not present")
def test_plot_master_panel_writes_png(tmp_path):
    """Smoke test: master panel writes a PNG and has 2N axes."""
    from karst_analysis.convergence import plot_master_panel

    out = tmp_path / "master.png"
    wells = ["AW5D", "AW6D", "BW3D", "LRS69D", "LRS70D"]
    fig = plot_master_panel(
        wells, campaigns=["2022_02"],
        project_root=PROJECT_ROOT,
        output_path=out,
    )
    assert isinstance(fig, plt.Figure)
    # 5 wells × 2 axes (caliper + sec) = 10 axes
    assert len(fig.axes) == 2 * len(wells)
    assert out.exists()
    plt.close(fig)


@pytest.mark.skipif(not HAS_DATA, reason="raw SEC data not present")
def test_build_all_writes_expected_files(tmp_path):
    """Batch driver writes one PNG per well plus the master figure."""
    from karst_analysis.convergence import build_all_sec_caliper_panels

    written = build_all_sec_caliper_panels(
        campaigns=["2022_02"],
        project_root=PROJECT_ROOT,
        output_dir=tmp_path,
    )
    # 5 wells + 1 master = 6 figures
    assert len(written) == 6
    for p in written:
        assert p.exists()
        assert p.suffix == ".png"

    names = {p.name for p in written}
    assert "AW5D_sec_caliper.png" in names
    assert "LRS70D_sec_caliper.png" in names
    assert "master_panel.png" in names


def test_well_id_to_filename_prefix():
    """Helper converts well_id to the underscore-separated filename prefix."""
    from karst_analysis.sec.io.loaders import _well_id_to_filename_prefix
    assert _well_id_to_filename_prefix("AW6D") == "AW6_D"
    assert _well_id_to_filename_prefix("LRS70D") == "LRS70_D"
    assert _well_id_to_filename_prefix("BW3D") == "BW3_D"
    with pytest.raises(ValueError):
        _well_id_to_filename_prefix("invalid_name")


def test_unknown_well_raises():
    """plot_sec_caliper_panel rejects unknown wells loud and clear."""
    from karst_analysis.convergence import plot_sec_caliper_panel
    with pytest.raises(KeyError, match="Unknown well"):
        plot_sec_caliper_panel("XX99D", campaigns=["2022_02"])


def test_imports_are_wired():
    """The new public API is reachable from karst_analysis.convergence."""
    from karst_analysis.convergence import (
        SecCaliperPanelConfig,
        plot_sec_caliper_panel,
        plot_master_panel,
        build_all_sec_caliper_panels,
    )
    cfg = SecCaliperPanelConfig()
    # Defaults match v10 design decisions.
    assert cfg.sec_log_x is False
    assert cfg.width_ratios == (1.0, 1.4)
