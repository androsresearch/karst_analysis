"""Caliper-log analysis sub-package.

Submodules
----------
config       : Centralised constants (OFFSET_CM, TRIM_DEPTHS_M, etc.).
io           : Loaders for the master and per-sample CSVs.
noise        : Instrumental-noise estimation (AW5O vs AW5D).
baseline     : Cumulative-minimum baseline construction.
detection    : Breakout detection and per-sample severity classification.
pipeline     : End-to-end orchestration over multiple wells.
viz          : Multi-well panel figure.

Typical usage
-------------
>>> from karst_analysis.caliper.io import load_master_caliper
>>> from karst_analysis.caliper.noise import estimate_noise_aw5o_vs_aw5d
>>> from karst_analysis.caliper.pipeline import (
...     process_many_wells, perpoint_dataframe, zones_dataframe,
... )
>>> from karst_analysis.caliper.viz import plot_priority_wells_panel
>>>
>>> df_master = load_master_caliper()
>>> noise_report = estimate_noise_aw5o_vs_aw5d(df_master)
>>> sigma_inst = noise_report["AW5O"]["sigma_MAD_cm"]
>>> results = process_many_wells(df_master, sigma_inst)
>>> perpoint_dataframe(results).to_csv("perpoint.csv", index=False)
>>> plot_priority_wells_panel(results, sigma_inst, output_path="panel.png")

For end-to-end CLIs, see ``scripts/caliper_estimate_noise.py`` and
``scripts/caliper_run_pipeline.py``.
"""

from karst_analysis.caliper.io import load_master_caliper, load_perpoint
from karst_analysis.caliper.noise import (
    estimate_noise_aw5o_vs_aw5d,
    measure_noise_in_interval,
    compare_drilling_methods,
)
from karst_analysis.caliper.baseline import (
    fit_cumulative_min_split,
    fit_cumulative_min_single_zone,
    CumulativeMinResult,
    SplitCumulativeMinResult,
)
from karst_analysis.caliper.detection import detect_breakouts_cumulative_min
from karst_analysis.caliper.pipeline import (
    process_one_well,
    process_many_wells,
    perpoint_dataframe,
    zones_dataframe,
)
from karst_analysis.caliper.viz import plot_priority_wells_panel

__all__ = [
    "load_master_caliper", "load_perpoint",
    "estimate_noise_aw5o_vs_aw5d", "measure_noise_in_interval",
    "compare_drilling_methods",
    "fit_cumulative_min_split", "fit_cumulative_min_single_zone",
    "CumulativeMinResult", "SplitCumulativeMinResult",
    "detect_breakouts_cumulative_min",
    "process_one_well", "process_many_wells",
    "perpoint_dataframe", "zones_dataframe",
    "plot_priority_wells_panel",
]
