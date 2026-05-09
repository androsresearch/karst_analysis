"""Visualisation for ERT 1D analyses.

Public API
----------
plot_sec_vs_ert(inputs, *, axis_scale) -> Figure
SecVsErtInputs                        # dataclass for the inputs bundle

The internal SEC panel renderer is in ``_sec_panel.py`` (private).
"""

from karst_analysis.ert.viz.sec_vs_ert import (
    SecVsErtInputs,
    plot_sec_vs_ert,
)

__all__ = ["SecVsErtInputs", "plot_sec_vs_ert"]
