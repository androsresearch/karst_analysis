"""Tests for ``karst_analysis.ert.viz.sec_vs_ert``.

Unit tests use synthetic SEC + ERT inputs so they run in milliseconds
without depending on real CSVs. They verify the figure builds and
contains the expected number of axes / lines / scatter collections.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")  # noqa: E402

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from karst_analysis.ert.io import ErtTrace1D
from karst_analysis.ert.breakpoints import ErtBreakpointFit
from karst_analysis.ert.viz import SecVsErtInputs, plot_sec_vs_ert


def _synthetic_sec_inputs() -> dict:
    """Build a minimal but valid SEC payload."""
    n_raw = 200
    z_raw = np.linspace(1.0, 28.0, n_raw)
    sec_raw = 1000 + 49000 / (1 + np.exp(-(z_raw - 10) / 1.5))
    z_sm = np.linspace(1.0, 28.0, 60)
    sec_sm = 1000 + 49000 / (1 + np.exp(-(z_sm - 10) / 1.5))

    # Slopes-CSV-like DataFrame (4 chord pairs => 5 BPs).
    z_bp_w = np.array([0.5, 4.0, 9.0, 13.0, 22.0])  # water-table reference
    sec_bp = np.interp(z_bp_w + 1.0, z_sm, sec_sm)
    log_bp = np.log10(sec_bp)
    n_pairs = len(z_bp_w) - 1
    sl = pd.DataFrame({
        "depth_top":          z_bp_w[:-1],
        "depth_bottom":       z_bp_w[1:],
        "log10_sec_top":      log_bp[:-1],
        "log10_sec_bottom":   log_bp[1:],
        "sec_top_uS_cm":      sec_bp[:-1],
        "sec_bottom_uS_cm":   sec_bp[1:],
        "is_top_of_mixing":   [False, True, False, False],
        "is_bottom_of_mixing":[False, False, True, False],
    })
    return dict(
        z_raw_bgl=z_raw, sec_raw=sec_raw,
        z_sm_bgl=z_sm, sec_sm=sec_sm,
        slopes_df=sl, vadose_m=1.0,
    )


def _synthetic_ert() -> tuple[ErtTrace1D, ErtBreakpointFit, int, int]:
    """Build a minimal ErtTrace1D + ErtBreakpointFit + MZ indices."""
    z = np.linspace(0.5, 30.0, 100)
    rho = 80 / (1 + np.exp((z - 10) / 1.5))      # decreasing sigmoid
    rho = np.clip(rho, 1.0, 100.0)
    df = pd.DataFrame({
        "depth_bgl_m": z,
        "resist_ohm_m": rho,
        "resistlog10": np.log10(rho),
    })
    trace = ErtTrace1D(
        transect="T_test",
        x_requested=100.0,
        x_extracted=100.25,
        variant="synthetic",
        source_path=Path("/dev/null"),
        df=df,
    )

    # Breakpoints DataFrame in the format extract_breakpoints emits.
    z_bp = np.array([5.0, 9.0, 13.0, 21.0])
    fit = ErtBreakpointFit(
        breakpoints=pd.DataFrame({
            "Breakpoint X Position": z_bp,
            "Breakpoint Y Position": np.interp(z_bp, z, np.log10(rho)),
            "Confidence Interval (X)": [(b - 0.3, b + 0.3) for b in z_bp],
        }),
        n_breakpoints=len(z_bp),
        seed_used=1,
        seeds_tried=(0, 1),
        bic=-1234.5,
        rss=0.005,
    )
    return trace, fit, 1, 2  # top idx=1 (z=9), bot idx=2 (z=13)


def _build_inputs() -> SecVsErtInputs:
    sec = _synthetic_sec_inputs()
    trace, fit, top_idx, bot_idx = _synthetic_ert()
    return SecVsErtInputs(
        well_id="WELL_T",
        sec_date_str="2022-01-31",
        z_raw_bgl_m=sec["z_raw_bgl"],
        sec_raw_uS_cm=sec["sec_raw"],
        z_smooth_bgl_m=sec["z_sm_bgl"],
        sec_smooth_uS_cm=sec["sec_sm"],
        slopes_df=sec["slopes_df"],
        vadose_m=sec["vadose_m"],
        sec_method="lowess",
        sec_n=4,
        sec_trial_idx=1,
        ert_trace=trace,
        ert_fit=fit,
        ert_top_mz_idx=top_idx,
        ert_bot_mz_idx=bot_idx,
        ert_bot_mz_threshold=25.0,
    )


# ════════════════════════════════════════════════════════════════════
class TestPlotSecVsErt:
    def test_builds_two_panels(self):
        fig = plot_sec_vs_ert(_build_inputs(), axis_scale="log10")
        assert len(fig.axes) == 2
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_shared_y_axis(self):
        fig = plot_sec_vs_ert(_build_inputs(), axis_scale="log10",
                              depth_top_m=0.0, depth_bottom_m=35.0)
        ax_sec, ax_ert = fig.axes
        assert ax_sec.get_ylim() == ax_ert.get_ylim()
        # Y axis must be inverted (deepest at bottom).
        ymin, ymax = ax_sec.get_ylim()
        assert ymin > ymax
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_both_scales_run(self):
        for scale in ("linear", "log10"):
            fig = plot_sec_vs_ert(_build_inputs(), axis_scale=scale)
            assert len(fig.axes) == 2
            import matplotlib.pyplot as plt
            plt.close(fig)

    def test_invalid_scale_raises(self):
        with pytest.raises(ValueError, match="axis_scale"):
            plot_sec_vs_ert(_build_inputs(), axis_scale="xyz")

    def test_ert_panel_no_ci_bands(self):
        """Cambio (c): ERT panel must NOT contain axhspan patches
        (the grey CI bands of v15-prototype); only axhlines."""
        fig = plot_sec_vs_ert(_build_inputs(), axis_scale="log10")
        ax_ert = fig.axes[1]
        # axhspan adds Polygon patches; axhline adds Line2D objects.
        from matplotlib.patches import Polygon
        polygons = [p for p in ax_ert.patches if isinstance(p, Polygon)]
        # No CI shading polygons on ERT axis.
        assert len(polygons) == 0
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_top_only_marks_only_top(self):
        """If bot_idx is None, BOT MZ marker must not appear."""
        inputs = _build_inputs()
        # Replace with bot_idx=None
        inputs_no_bot = SecVsErtInputs(
            **{**inputs.__dict__, "ert_bot_mz_idx": None}
        )
        fig = plot_sec_vs_ert(inputs_no_bot, axis_scale="log10")
        ax_ert = fig.axes[1]
        labels = [c.get_label() for c in ax_ert.collections]
        assert "BOTTOM of mixing zone" not in labels
        assert "TOP of mixing zone" in labels
        import matplotlib.pyplot as plt
        plt.close(fig)
