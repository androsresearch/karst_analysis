"""SEC breakpoint robustness analysis.

Quantifies which breakpoints survive perturbation of the smoothing
method (savgol vs lowess) and the BIC sweep N (1..10), pooling all
detected BPs and clustering them by depth. Output is a per-cluster
score:

    persistence = #N_savgol_seeing_it + #N_lowess_seeing_it     (0..20)
    agreement   = min(n_savgol, n_lowess)                       (0..10)

High ``agreement`` means the cluster is detected by BOTH smoothings,
which is the strongest evidence that the breakpoint is a real
physical interface rather than a smoothing artefact.

Submodules
----------
clustering : single-linkage clustering on 1-D depths.
scoring    : ``compute_robustness(well_id, ...)`` orchestrator.
viz        : per-well diagnostic panel + δ-sensitivity + BIC curves.

Limitations
-----------
* Single-linkage chains: a long string of BPs spaced just under δ can
  collapse into one wide cluster. The CSV outputs preserve all
  original BP records (with ``cluster_id`` annotation) so post-hoc
  reanalysis is always possible.
* The ``wide_flag`` uses (depth_max - depth_min). This is sensitive
  to outliers; ``depth_iqr`` is a more robust alternative reported
  alongside.
* The "BIC-optimal N" reported per smoothing is informative only:
  this analysis does NOT pick a winning method. The point is to find
  breakpoints that survive both.
"""

from karst_analysis.sec.robustness.clustering import (
    cluster_depths_single_linkage,
    summarize_clusters,
)
from karst_analysis.sec.robustness.scoring import (
    DEFAULT_DELTA_M,
    SENSITIVITY_DELTAS_M,
    DEFAULT_SMOOTHINGS,
    DEFAULT_N_RANGE,
    RobustnessResult,
    compute_robustness,
    compute_robustness_sensitivity,
)
from karst_analysis.sec.robustness.viz import (
    plot_robustness_panel,
    plot_delta_sensitivity,
    plot_bic_curves,
    SAVGOL_COLOR,
    LOWESS_COLOR,
    CLUSTER_BAR_COLOR,
)

__all__ = [
    # clustering
    "cluster_depths_single_linkage",
    "summarize_clusters",
    # scoring
    "DEFAULT_DELTA_M",
    "SENSITIVITY_DELTAS_M",
    "DEFAULT_SMOOTHINGS",
    "DEFAULT_N_RANGE",
    "RobustnessResult",
    "compute_robustness",
    "compute_robustness_sensitivity",
    # viz
    "plot_robustness_panel",
    "plot_delta_sensitivity",
    "plot_bic_curves",
    "SAVGOL_COLOR",
    "LOWESS_COLOR",
    "CLUSTER_BAR_COLOR",
]
