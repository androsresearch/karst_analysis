"""1-D single-linkage clustering of breakpoint depths.

The robustness analysis pools breakpoints from many (smoothing, N)
combinations into a single bag of (depth, source) records. Two records
are considered "the same physical breakpoint" if they are within δ
metres of each other.

Single-linkage clustering means: any two records within δ end up in
the same cluster — even if the cluster's diameter ends up much larger
than δ. This is the *intentional* behaviour: in continuous karst
zones, individual breakpoints chain together into wide clusters that
correctly represent transition zones rather than discrete features.
The downside is "chaining" — a long string of BPs spaced just under δ
collapses into one big cluster. The CSV outputs preserve the original
BP-to-cluster assignments so post-hoc reanalysis is always possible.

This module is deliberately small and dependency-free (just numpy +
pandas) so it can be unit-tested cheaply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cluster_depths_single_linkage(
    depths: np.ndarray,
    delta_m: float,
) -> np.ndarray:
    """Assign cluster IDs to a 1-D array of depths via single-linkage.

    Two depths belong to the same cluster if the distance between
    consecutive depths (after sorting) is less than ``delta_m``.

    Parameters
    ----------
    depths : np.ndarray
        1-D array of depth values (m, BGL-positive). Order is preserved
        in the output (cluster IDs returned in the input order).
    delta_m : float
        Linkage threshold in metres. Two consecutive sorted depths
        with gap < delta_m get fused into the same cluster.
        ``delta_m`` must be strictly positive.

    Returns
    -------
    np.ndarray
        Integer array of cluster IDs, same length as ``depths``.
        Cluster IDs are 0-indexed and assigned in order of increasing
        depth (cluster 0 is the shallowest).

    Notes
    -----
    Single-linkage chains: depths {5.0, 5.4, 5.8, 6.2} with delta=0.5
    all end up in one cluster (gaps are 0.4 < 0.5). The resulting
    cluster has diameter 1.2 m, larger than delta. This is intended.

    Examples
    --------
    >>> import numpy as np
    >>> cluster_depths_single_linkage(np.array([5.0, 5.3, 9.1, 9.5]), 0.5)
    array([0, 0, 1, 1])
    >>> cluster_depths_single_linkage(np.array([5.0, 5.6, 9.1]), 0.5)
    array([0, 1, 2])
    """
    if delta_m <= 0:
        raise ValueError(f"delta_m must be > 0, got {delta_m}")

    n = len(depths)
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([0], dtype=int)

    # Sort, but remember original positions so we can put cluster IDs
    # back in input order.
    order = np.argsort(depths, kind="stable")
    sorted_d = depths[order]

    sorted_cluster = np.zeros(n, dtype=int)
    current_id = 0
    for i in range(1, n):
        gap = sorted_d[i] - sorted_d[i - 1]
        if gap >= delta_m:
            current_id += 1
        sorted_cluster[i] = current_id

    # Restore input order
    out = np.empty(n, dtype=int)
    out[order] = sorted_cluster
    return out


def summarize_clusters(
    bp_records: pd.DataFrame,
    cluster_ids: np.ndarray,
    n_max_smoothing: dict[str, int],
) -> pd.DataFrame:
    """Build per-cluster summary statistics.

    Parameters
    ----------
    bp_records : pd.DataFrame
        Long-format DataFrame with one row per detected BP. Required
        columns: ``smoothing``, ``N``, ``depth_bgl_m``.
    cluster_ids : np.ndarray
        Cluster ID for each row of ``bp_records`` (same length).
    n_max_smoothing : dict[str, int]
        Maximum N considered per smoothing method. Used to compute
        the persistence denominator. For example
        ``{"savgol": 10, "lowess": 10}`` means each smoothing
        contributes up to 10 N values to the analysis.

    Returns
    -------
    pd.DataFrame
        One row per cluster, columns:
            cluster_id           : 0-indexed, ordered by increasing depth
            depth_median         : median of all BPs in the cluster
            depth_min, depth_max : extent of the cluster
            depth_iqr            : interquartile range (robustness inside)
            n_savgol             : how many distinct N values (savgol) place
                                   a BP in this cluster
            n_lowess             : same, for lowess
            persistence          : n_savgol + n_lowess, range 0..(N_max_savgol + N_max_lowess)
            agreement            : min(n_savgol, n_lowess), penalises one-sided
            cluster_diameter_m   : depth_max - depth_min
            wide_flag            : True if diameter > 2 m (transition-zone marker)
            n_bp_total           : total raw BP entries in the cluster
        Sorted by ``agreement`` desc, then ``persistence`` desc.
    """
    df = bp_records.copy()
    df["cluster_id"] = cluster_ids
    rows: list[dict] = []
    for cid, sub in df.groupby("cluster_id"):
        depths = sub["depth_bgl_m"].to_numpy()
        # Distinct N values per smoothing: a single (smoothing, N) combo can
        # contribute multiple BPs to the SAME cluster if all its BPs fall
        # in there. We count it once per cluster — that's what "the BP
        # appears in run (smoothing, N)" means.
        distinct_runs = sub.groupby("smoothing")["N"].nunique()
        n_savgol = int(distinct_runs.get("savgol", 0))
        n_lowess = int(distinct_runs.get("lowess", 0))
        diameter = float(depths.max() - depths.min())
        rows.append(dict(
            cluster_id=int(cid),
            depth_median=float(np.median(depths)),
            depth_min=float(depths.min()),
            depth_max=float(depths.max()),
            depth_iqr=float(np.percentile(depths, 75) - np.percentile(depths, 25)),
            n_savgol=n_savgol,
            n_lowess=n_lowess,
            persistence=n_savgol + n_lowess,
            agreement=min(n_savgol, n_lowess),
            cluster_diameter_m=diameter,
            wide_flag=bool(diameter > 2.0),
            n_bp_total=int(len(sub)),
        ))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return (out.sort_values(["agreement", "persistence"], ascending=[False, False])
              .reset_index(drop=True))
