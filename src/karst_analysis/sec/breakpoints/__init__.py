"""Breakpoint detection for SEC profiles.

Workflow (notebook-driven, intentionally manual):
    1. ``best_n_breakpoints`` : compute BIC for n=1..N over multiple trials.
    2. ``select_best_trial`` + ``elbow_max_distance`` : pick a candidate.
    3. ``rebuild_model`` : reconstruct the piecewise fit with chosen N.
    4. ``extract_segments`` / ``extract_breakpoints`` : extract for plotting.
"""

from karst_analysis.sec.breakpoints.detection import (
    best_n_breakpoints,
    elbow_max_distance,
    extract_breakpoints,
)
from karst_analysis.sec.breakpoints.segments import (
    extract_segments,
    segment_data,
    fit_linear_models,
    calculate_metrics_per_segment,
)
from karst_analysis.sec.breakpoints.selection import (
    select_best_trial,
    get_global_metrics,
    rebuild_model,
    get_breakpoint_data,
)

__all__ = [
    # detection
    "best_n_breakpoints",
    "elbow_max_distance",
    "extract_breakpoints",
    # segments
    "extract_segments",
    "segment_data",
    "fit_linear_models",
    "calculate_metrics_per_segment",
    # selection
    "select_best_trial",
    "get_global_metrics",
    "rebuild_model",
    "get_breakpoint_data",
]
