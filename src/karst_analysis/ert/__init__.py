"""ERT 1D resistivity profile analysis.

Public API
----------
I/O:
    ErtTrace1D, ErtWellAssoc, ErtWellMap
    load_ert_1d_csv, load_ert_1d_traces
    parse_ert_filename

Breakpoint detection (with seed discovery):
    ErtBreakpointFit
    detect_breakpoints_with_seed_discovery

Mixing zone:
    select_ert_mixing_zone   (array-pure)
    mark_ert_mixing_zone     (DataFrame wrapper)

A 2D ERT module is planned but not implemented in this iteration.
"""

from karst_analysis.ert.breakpoints import (
    ErtBreakpointFit,
    detect_breakpoints_with_seed_discovery,
)
from karst_analysis.ert.io import (
    ErtTrace1D,
    ErtWellAssoc,
    ErtWellMap,
    load_ert_1d_csv,
    load_ert_1d_traces,
    parse_ert_filename,
)
from karst_analysis.ert.mixing_zone import (
    mark_ert_mixing_zone,
    select_ert_mixing_zone,
)
from karst_analysis.ert.viz import (
    SecVsErtInputs,
    plot_sec_vs_ert,
)

__all__ = [
    # io
    "ErtTrace1D",
    "ErtWellAssoc",
    "ErtWellMap",
    "load_ert_1d_csv",
    "load_ert_1d_traces",
    "parse_ert_filename",
    # breakpoints
    "ErtBreakpointFit",
    "detect_breakpoints_with_seed_discovery",
    # mixing zone
    "select_ert_mixing_zone",
    "mark_ert_mixing_zone",
    # viz
    "SecVsErtInputs",
    "plot_sec_vs_ert",
]
