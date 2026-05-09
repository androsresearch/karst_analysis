"""High-level data accessors designed for consumption by *other* projects.

This sub-package is the "stable API" for SEC outputs: callers from
external repositories (e.g. ``caliper_video``) should import from here,
not from internal modules. The functions hide the on-disk layout
(filenames, JSON shape, datum conventions) behind a small, documented
interface.

Typical usage from another project
----------------------------------
>>> from karst_analysis.sec.export import (
...     load_sec_profile, load_breakpoints_at_n, list_available_runs,
... )
>>> sec  = load_sec_profile(well_id="LRS70D", campaign="2022_02",
...                         smoothing="lowess")
>>> bps5 = load_breakpoints_at_n(well_id="LRS70D", campaign="2022_02",
...                              smoothing="lowess", n=5)
>>> # both DataFrames carry depth_m AND depth_bgl_m so the caller picks.

The functions DO NOT re-fit any model. They read the artefacts produced
by ``scripts/breakpoints_batch.py`` (or notebook 03) and return tidy
DataFrames.
"""

from karst_analysis.sec.export.api import (
    list_available_runs,
    load_sec_profile,
    load_breakpoints_at_n,
    load_bic_curve,
)

__all__ = [
    "list_available_runs",
    "load_sec_profile",
    "load_breakpoints_at_n",
    "load_bic_curve",
]
