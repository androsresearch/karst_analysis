"""Tests for the v13 figure-path resolver.

These cover the three canonical cases of the v13 convention:

* pre-casing techniques  → no campaign subfolder
* single-campaign        → ``<technique>/<campaign>/``
* multi-campaign         → ``<technique>/multi_<N>c/``

and the override / edge-case behaviour (custom figures_root,
``None`` or empty campaigns, single-element list, etc.).
"""

from __future__ import annotations

from pathlib import Path

import pytest


# ──────────────────────────────────────────────────────────────────────
#  Imports + module wiring
# ──────────────────────────────────────────────────────────────────────
def test_imports_exposed_from_io():
    """Public API is reachable from karst_analysis.io."""
    from karst_analysis.io import (
        FIGURES_ROOT,
        campaign_subdir_label,
        resolve_figure_dir,
    )
    assert FIGURES_ROOT == Path("results") / "figures"
    assert callable(resolve_figure_dir)
    assert callable(campaign_subdir_label)


# ──────────────────────────────────────────────────────────────────────
#  Pre-casing: no campaign subfolder
# ──────────────────────────────────────────────────────────────────────
def test_pre_casing_no_campaign():
    """When no campaigns are given, no subfolder is appended."""
    from karst_analysis.io import resolve_figure_dir

    assert resolve_figure_dir("caliper") == Path("results/figures/caliper")
    assert resolve_figure_dir(
        "convergence/caliper_video"
    ) == Path("results/figures/convergence/caliper_video")


def test_pre_casing_with_none_campaigns():
    """Explicit None campaigns is equivalent to omission."""
    from karst_analysis.io import resolve_figure_dir

    assert resolve_figure_dir(
        "caliper", campaigns=None
    ) == Path("results/figures/caliper")


def test_pre_casing_with_empty_campaigns():
    """Empty list of campaigns also yields no subfolder."""
    from karst_analysis.io import resolve_figure_dir

    assert resolve_figure_dir(
        "caliper", campaigns=[]
    ) == Path("results/figures/caliper")


# ──────────────────────────────────────────────────────────────────────
#  Single-campaign: <technique>/<campaign>/
# ──────────────────────────────────────────────────────────────────────
def test_single_campaign_appends_campaign_name():
    """One campaign → subfolder is the campaign name itself."""
    from karst_analysis.io import resolve_figure_dir

    assert resolve_figure_dir(
        "breakpoints", campaigns=["2022_02"]
    ) == Path("results/figures/breakpoints/2022_02")

    assert resolve_figure_dir(
        "convergence/sec_caliper_panel", campaigns=["2025_11"]
    ) == Path("results/figures/convergence/sec_caliper_panel/2025_11")


# ──────────────────────────────────────────────────────────────────────
#  Multi-campaign: <technique>/multi_<N>c/
# ──────────────────────────────────────────────────────────────────────
def test_multi_campaign_uses_multi_label():
    """Several campaigns → subfolder is multi_<N>c."""
    from karst_analysis.io import resolve_figure_dir

    out = resolve_figure_dir(
        "convergence/site_panel",
        campaigns=["2022_02", "2022_08", "2023_08"],
    )
    assert out == Path("results/figures/convergence/site_panel/multi_3c")


def test_multi_campaign_with_six_campaigns():
    """v12 default case: six campaigns → multi_6c."""
    from karst_analysis.io import resolve_figure_dir

    out = resolve_figure_dir(
        "convergence/site_panel",
        campaigns=["2011_05", "2022_02", "2022_08", "2023_08",
                   "2025_02", "2025_11"],
    )
    assert out == Path("results/figures/convergence/site_panel/multi_6c")


# ──────────────────────────────────────────────────────────────────────
#  Override of figures_root (used by tests that write to tmp_path)
# ──────────────────────────────────────────────────────────────────────
def test_figures_root_override(tmp_path):
    """Custom figures_root replaces the default `results/figures`."""
    from karst_analysis.io import resolve_figure_dir

    out = resolve_figure_dir(
        "breakpoints",
        campaigns=["2022_02"],
        figures_root=tmp_path,
    )
    assert out == tmp_path / "breakpoints" / "2022_02"


# ──────────────────────────────────────────────────────────────────────
#  campaign_subdir_label: pure helper
# ──────────────────────────────────────────────────────────────────────
def test_campaign_subdir_label_none():
    from karst_analysis.io import campaign_subdir_label
    assert campaign_subdir_label(None) is None
    assert campaign_subdir_label([]) is None


def test_campaign_subdir_label_single():
    from karst_analysis.io import campaign_subdir_label
    assert campaign_subdir_label(["2022_02"]) == "2022_02"


def test_campaign_subdir_label_multi():
    from karst_analysis.io import campaign_subdir_label
    assert campaign_subdir_label(["a", "b", "c", "d"]) == "multi_4c"


# ──────────────────────────────────────────────────────────────────────
#  Integration: drivers honour the helper for their default output_dir
# ──────────────────────────────────────────────────────────────────────
def test_site_panel_default_output_uses_helper(monkeypatch, tmp_path):
    """build_all_site_panels with no output_dir computes the path via
    resolve_figure_dir (which we monkeypatch to redirect to tmp_path).
    """
    import matplotlib
    matplotlib.use("Agg")

    PROJECT_ROOT = Path(__file__).parent.parent
    if not (PROJECT_ROOT / "data" / "raw" / "sec" / "2022_02").exists():
        pytest.skip("raw SEC data not present")

    from karst_analysis.convergence import build_all_site_panels
    import karst_analysis.io.figure_paths as fp_module

    # Redirect the helper's root so the test stays sandboxed in tmp_path.
    monkeypatch.setattr(fp_module, "FIGURES_ROOT", tmp_path)

    written = build_all_site_panels(
        campaigns=["2022_02"],
        sites=["AW5"],
        project_root=PROJECT_ROOT,
        # output_dir omitted on purpose: helper resolves it
        build_master=False,
    )
    # The helper places single-campaign output under <root>/<technique>/<campaign>/.
    expected_subdir = tmp_path / "convergence" / "site_panel" / "2022_02"
    for p in written:
        assert p.is_relative_to(expected_subdir), (
            f"{p} not under {expected_subdir}"
        )
