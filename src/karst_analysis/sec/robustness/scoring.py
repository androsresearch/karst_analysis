"""High-level API for the robustness analysis.

Orchestrates the per-well computation:

    1. Pool BPs across (smoothing, N) combinations.
    2. Cluster pooled depths via single-linkage with threshold delta.
    3. Compute persistence + agreement scores per cluster.
    4. Identify the N that minimises BIC for each smoothing
       (a per-method reference point — NOT a "winner").
    5. Optionally run the same analysis at multiple delta values for
       a sensitivity test.

Outputs are returned as DataFrames; CSV writing happens in the API
``compute_robustness`` function or in the CLI script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from karst_analysis.sec.export.api import (
    load_bic_curve, load_breakpoints_at_n,
)
from karst_analysis.sec.robustness.clustering import (
    cluster_depths_single_linkage, summarize_clusters,
)


# ──────────────────────────────────────────────────────────────────────
#  Defaults
# ──────────────────────────────────────────────────────────────────────
DEFAULT_DELTA_M: float = 0.5
SENSITIVITY_DELTAS_M: tuple[float, ...] = (0.3, 0.5, 1.0)
DEFAULT_SMOOTHINGS: tuple[str, ...] = ("savgol", "lowess")
DEFAULT_N_RANGE: tuple[int, int] = (1, 10)


@dataclass
class RobustnessResult:
    """Container for per-well robustness output.

    Attributes
    ----------
    bp_records : pd.DataFrame
        One row per detected BP. Columns:
            well_id, smoothing, N, bp_index, depth_bgl_m, sec_at_bp_uS_cm,
            cluster_id (assigned at the chosen delta).
    clusters : pd.DataFrame
        One row per cluster with persistence/agreement scores.
        See ``clustering.summarize_clusters`` for column docs.
    bic_summary : pd.DataFrame
        One row per smoothing method with the N that minimises BIC.
        Columns: well_id, smoothing, n_optimal_bic, bic_at_optimal,
        delta_bic_vs_n_minus_1, top_3_robust_depths_m
    delta_m : float
        Linkage threshold used for the main result.
    n_max_smoothing : dict[str, int]
        Maximum N actually present in the data per smoothing.
        Used to compute the persistence denominator.
    """
    bp_records: pd.DataFrame
    clusters: pd.DataFrame
    bic_summary: pd.DataFrame
    delta_m: float
    n_max_smoothing: dict[str, int]


# ──────────────────────────────────────────────────────────────────────
#  Pooling
# ──────────────────────────────────────────────────────────────────────
def _pool_breakpoints(
    well_id: str,
    *,
    campaign: str,
    smoothings: tuple[str, ...],
    n_range: tuple[int, int],
    project_root: Optional[Path | str] = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Load every (smoothing, N) BP set and pool them into one DataFrame.

    A combination (smoothing, N) is silently skipped if its breakpoint
    JSON is missing or the BIC sweep doesn't include that N. Skipping
    is the right behaviour for the outer loop (the analysis just
    proceeds with whatever was found).

    Returns
    -------
    pooled : pd.DataFrame
        Long-format with columns: well_id, smoothing, N, bp_index,
        depth_bgl_m, sec_at_bp_uS_cm. May be empty.
    n_max_per_smoothing : dict[str, int]
        Highest N successfully loaded per smoothing.
    """
    n_min, n_max = n_range
    rows: list[pd.DataFrame] = []
    n_max_observed: dict[str, int] = {s: 0 for s in smoothings}

    for smoothing in smoothings:
        for n in range(n_min, n_max + 1):
            try:
                bp_df = load_breakpoints_at_n(
                    well_id=well_id, campaign=campaign,
                    smoothing=smoothing, n=n,
                    project_root=project_root,
                )
            except Exception:
                # Missing JSON or N out of range — silently skip.
                continue
            if bp_df.empty:
                continue
            sub = bp_df[["bp_index", "depth_bgl_m", "sec_at_bp_uS_cm"]].copy()
            sub["well_id"] = well_id
            sub["smoothing"] = smoothing
            sub["N"] = n
            rows.append(sub[["well_id", "smoothing", "N",
                             "bp_index", "depth_bgl_m", "sec_at_bp_uS_cm"]])
            n_max_observed[smoothing] = max(n_max_observed[smoothing], n)

    if not rows:
        return pd.DataFrame(columns=["well_id", "smoothing", "N",
                                      "bp_index", "depth_bgl_m",
                                      "sec_at_bp_uS_cm"]), n_max_observed
    return pd.concat(rows, ignore_index=True), n_max_observed


# ──────────────────────────────────────────────────────────────────────
#  BIC summary
# ──────────────────────────────────────────────────────────────────────
def _bic_summary_for_well(
    well_id: str,
    *,
    campaign: str,
    smoothings: tuple[str, ...],
    project_root: Optional[Path | str] = None,
) -> pd.DataFrame:
    """For each smoothing, find the N that minimises BIC.

    Returns
    -------
    pd.DataFrame
        Columns: well_id, smoothing, n_optimal_bic, bic_at_optimal,
        delta_bic_n_plus_1 (BIC at n_optimal+1 minus BIC at n_optimal,
        useful to gauge how sharp the BIC minimum is).
    """
    rows = []
    for smoothing in smoothings:
        try:
            bic_df = load_bic_curve(
                well_id=well_id, campaign=campaign, smoothing=smoothing,
                project_root=project_root,
            )
        except Exception:
            continue
        if bic_df.empty:
            continue
        # NaN BICs (non-converged fits) are excluded from the argmin.
        valid = bic_df.dropna(subset=["bic"])
        if valid.empty:
            continue
        idx_min = valid["bic"].idxmin()
        n_opt = int(valid.loc[idx_min, "n_breakpoints"])
        bic_opt = float(valid.loc[idx_min, "bic"])
        # Compare to BIC at n_opt+1 (sharpness of minimum)
        next_row = valid[valid["n_breakpoints"] == n_opt + 1]
        delta_next = (
            float(next_row["bic"].iloc[0] - bic_opt)
            if not next_row.empty else np.nan
        )
        rows.append(dict(
            well_id=well_id,
            smoothing=smoothing,
            n_optimal_bic=n_opt,
            bic_at_optimal=bic_opt,
            delta_bic_n_plus_1=delta_next,
        ))
    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────────────────
#  Main API
# ──────────────────────────────────────────────────────────────────────
def compute_robustness(
    well_id: str,
    *,
    campaign: str = "2022_02",
    smoothings: tuple[str, ...] = DEFAULT_SMOOTHINGS,
    n_range: tuple[int, int] = DEFAULT_N_RANGE,
    delta_m: float = DEFAULT_DELTA_M,
    project_root: Optional[Path | str] = None,
) -> RobustnessResult:
    """Compute the robustness analysis for a single well.

    Parameters
    ----------
    well_id : str
        Well identifier (e.g. ``"LRS70D"``).
    campaign : str
        Field campaign — points to ``data/processed/sec/<campaign>/``.
    smoothings : tuple[str, ...]
        Smoothings to pool. Default: both savgol and lowess.
    n_range : tuple[int, int]
        Inclusive N range to sweep. Default: 1..10.
    delta_m : float
        Single-linkage threshold in metres.
    project_root : Path, optional
        Project root for SEC artefact lookup.

    Returns
    -------
    RobustnessResult

    Raises
    ------
    ValueError
        If no breakpoints could be loaded for the well at all.
    """
    pooled, n_max_per_smoothing = _pool_breakpoints(
        well_id, campaign=campaign, smoothings=smoothings,
        n_range=n_range, project_root=project_root,
    )
    if pooled.empty:
        raise ValueError(
            f"No breakpoints found for well '{well_id}' "
            f"in campaign '{campaign}' across smoothings={smoothings}, "
            f"n_range={n_range}. Has the BP detection batch been run?"
        )

    # Cluster
    cluster_ids = cluster_depths_single_linkage(
        pooled["depth_bgl_m"].to_numpy(), delta_m=delta_m,
    )
    pooled = pooled.copy()
    pooled["cluster_id"] = cluster_ids

    clusters = summarize_clusters(
        pooled, cluster_ids, n_max_smoothing=n_max_per_smoothing,
    )

    # BIC summary
    bic_summary = _bic_summary_for_well(
        well_id, campaign=campaign, smoothings=smoothings,
        project_root=project_root,
    )

    # Annotate the BIC summary with the top-3 robust depths
    if not clusters.empty:
        top3 = clusters.head(3)["depth_median"].to_list()
        top3_str = ", ".join(f"{d:.2f}" for d in top3)
    else:
        top3_str = ""
    bic_summary["top_3_robust_depths_m"] = top3_str

    return RobustnessResult(
        bp_records=pooled,
        clusters=clusters,
        bic_summary=bic_summary,
        delta_m=delta_m,
        n_max_smoothing=n_max_per_smoothing,
    )


def compute_robustness_sensitivity(
    well_id: str,
    *,
    campaign: str = "2022_02",
    smoothings: tuple[str, ...] = DEFAULT_SMOOTHINGS,
    n_range: tuple[int, int] = DEFAULT_N_RANGE,
    deltas_m: tuple[float, ...] = SENSITIVITY_DELTAS_M,
    project_root: Optional[Path | str] = None,
) -> pd.DataFrame:
    """Run the clustering at multiple delta values and report stability.

    For each delta in ``deltas_m`` returns the cluster summary. The
    output is a long-format DataFrame with an extra ``delta_m`` column
    so plots can show "how does the number of robust clusters change
    with delta?".

    Parameters
    ----------
    Same as ``compute_robustness`` plus ``deltas_m`` (a tuple).

    Returns
    -------
    pd.DataFrame
        Columns: delta_m, well_id, plus the columns from
        ``summarize_clusters``. Sorted by ``delta_m`` then ``cluster_id``.
    """
    parts: list[pd.DataFrame] = []
    for d in deltas_m:
        try:
            res = compute_robustness(
                well_id, campaign=campaign, smoothings=smoothings,
                n_range=n_range, delta_m=d, project_root=project_root,
            )
        except ValueError:
            continue
        sub = res.clusters.copy()
        sub.insert(0, "delta_m", d)
        sub.insert(1, "well_id", well_id)
        parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)
