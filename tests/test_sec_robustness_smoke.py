"""Smoke test for the SEC robustness end-to-end pipeline.

Skipped automatically if the SEC processed / breakpoint outputs aren't
present (they're produced by ``preprocess_batch.py`` and
``breakpoints_batch.py`` which run before this analysis).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from karst_analysis.sec.robustness import (
    compute_robustness, compute_robustness_sensitivity,
    plot_robustness_panel, plot_delta_sensitivity,
)


ROOT = Path(__file__).resolve().parent.parent
SEC_DIR = ROOT / "data" / "processed" / "sec" / "2022_02"
BP_DIR = ROOT / "data" / "breakpoints" / "2022_02"

pytestmark = pytest.mark.skipif(
    not (SEC_DIR.exists() and BP_DIR.exists()),
    reason="SEC processed / breakpoints not available",
)


class TestRobustnessE2E:
    def test_compute_returns_nonempty_for_lrs70d(self):
        import os
        cwd0 = os.getcwd()
        try:
            os.chdir(ROOT)
            res = compute_robustness("LRS70D", campaign="2022_02")
        finally:
            os.chdir(cwd0)
        assert not res.bp_records.empty
        assert not res.clusters.empty
        # 110 BPs is the expected total: 1+2+...+10 = 55 per smoothing × 2
        assert len(res.bp_records) == 110

    def test_top_cluster_has_high_agreement(self):
        """For LRS70D the top cluster should have agreement >= 8."""
        import os
        cwd0 = os.getcwd()
        try:
            os.chdir(ROOT)
            res = compute_robustness("LRS70D", campaign="2022_02")
        finally:
            os.chdir(cwd0)
        assert res.clusters.iloc[0]["agreement"] >= 8

    def test_unknown_well_raises(self):
        import os
        cwd0 = os.getcwd()
        try:
            os.chdir(ROOT)
            with pytest.raises(ValueError, match="No breakpoints"):
                compute_robustness("ZZZ99X", campaign="2022_02")
        finally:
            os.chdir(cwd0)

    def test_sensitivity_returns_three_deltas(self):
        import os
        cwd0 = os.getcwd()
        try:
            os.chdir(ROOT)
            df = compute_robustness_sensitivity(
                "LRS70D", campaign="2022_02",
                deltas_m=(0.3, 0.5, 1.0),
            )
        finally:
            os.chdir(cwd0)
        assert set(df["delta_m"].unique()) == {0.3, 0.5, 1.0}

    def test_panel_renders(self, tmp_path):
        import os
        cwd0 = os.getcwd()
        try:
            os.chdir(ROOT)
            res = compute_robustness("LRS70D", campaign="2022_02")
            out = tmp_path / "panel.png"
            fig = plot_robustness_panel(res, well_id="LRS70D", output_path=out)
            plt.close(fig)
        finally:
            os.chdir(cwd0)
        assert out.exists()
        assert out.stat().st_size > 30_000
