"""Visualisations for SEC pipelines and breakpoint workflow."""

from karst_analysis.sec.viz.diagnostic import (
    plot_diagnostic,
    plot_balance_histogram,
)
from karst_analysis.sec.viz.comparison import plot_smoothing_comparison
from karst_analysis.sec.viz.profiles import plot_profile_plotly
from karst_analysis.sec.viz.segments import (
    plot_segments,
    interactive_segmented_regression,
)
from karst_analysis.sec.viz.breakpoints_overlay import (
    load_bic_json,
    extract_breakpoints_for_n,
    get_metric_at_n,
    compute_sec_at_breakpoints,
    plot_breakpoints_overlay,
    plot_breakpoints_compare_methods,
)
from karst_analysis.sec.viz.slopes_overlay import (
    plot_slopes_overlay,
)

__all__ = [
    "plot_diagnostic",
    "plot_balance_histogram",
    "plot_smoothing_comparison",
    "plot_profile_plotly",
    "plot_segments",
    "interactive_segmented_regression",
    "load_bic_json",
    "extract_breakpoints_for_n",
    "get_metric_at_n",
    "compute_sec_at_breakpoints",
    "plot_breakpoints_overlay",
    "plot_breakpoints_compare_methods",
    "plot_slopes_overlay",
]
