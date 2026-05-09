"""Smoke test for the SEC + caliper × video panel.

Visual figures are difficult to unit-test without committing many
golden PNGs to the repo. Instead this test:

    * Verifies the module imports cleanly.
    * Checks ``WELLS`` and the public functions are exposed.
    * Renders one panel for one (well, smoothing, n) combination
      and asserts a non-zero PNG was written.

The full visual inspection is performed in notebook 08, where Mariana
can open the resulting PNGs interactively.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

# Use a non-interactive backend so the test runs in headless CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from karst_analysis.convergence import (
    SecCaliperVideoConfig, WELLS,
    build_all_sec_caliper_video_panels,
    plot_sec_caliper_video_panel,
)


ROOT = Path(__file__).resolve().parent.parent
SEC_DIR = ROOT / "data" / "processed" / "sec" / "2022_02"
BP_DIR = ROOT / "data" / "breakpoints" / "2022_02"
PERPOINT_CSV = (ROOT / "data" / "processed" / "caliper"
                / "priority_wells_cumulative_min_v2_perpoint.csv")


pytestmark = pytest.mark.skipif(
    not (SEC_DIR.exists() and BP_DIR.exists() and PERPOINT_CSV.exists()),
    reason="SEC processed / breakpoints / caliper perpoint outputs not available "
           "(run preprocess_batch.py + breakpoints_batch.py + caliper_run_pipeline.py)",
)


class TestPanelImports:
    def test_well_keys(self):
        assert set(WELLS.keys()) == {"LRS70D", "AW5D", "AW6D", "BW3D", "LRS69D"}

    def test_config_defaults(self):
        cfg = SecCaliperVideoConfig()
        # Sanity-check the 3-column layout
        assert len(cfg.width_ratios) == 3
        # BP labels column should be the narrowest
        assert cfg.width_ratios[0] < cfg.width_ratios[1] < cfg.width_ratios[2]


class TestPanelSmoke:
    def test_renders_one_panel(self, tmp_path):
        out = tmp_path / "LRS70D_smoke.png"
        # Cd into project root because the SEC export API uses Path.cwd()
        # to locate processed artefacts.
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(ROOT)
            fig = plot_sec_caliper_video_panel(
                "LRS70D", smoothing="savgol", n=3,
                output_path=out,
            )
            plt.close(fig)
        finally:
            os.chdir(old_cwd)
        assert out.exists(), "PNG file was not created"
        assert out.stat().st_size > 50_000, "PNG suspiciously small"

    def test_invalid_well_raises(self):
        with pytest.raises(KeyError, match="Unknown well"):
            plot_sec_caliper_video_panel(
                "ZZZ99X", smoothing="savgol", n=3,
            )


class TestBatchSmoke:
    def test_batch_writes_files(self, tmp_path):
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(ROOT)
            paths = build_all_sec_caliper_video_panels(
                wells=["LRS70D"],
                smoothings=("savgol",),
                n_min=2, n_max=3,
                output_dir=tmp_path,
            )
        finally:
            os.chdir(old_cwd)
        # Expect 2 files (N=2 and N=3)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 50_000
