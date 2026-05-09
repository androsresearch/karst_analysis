"""SEC ↔ caliper quantitative convergence (Idea 3).

For each robust SEC cluster of a given well, find the anomalous caliper
zones it matches and produce a per-cluster convergence score. Aggregates
to a per-well summary. Also reports caliper zones that have no SEC
counterpart, which is itself useful information (e.g. dry cavities that
do not produce a SEC step, or caliper false positives).

Inputs
------
* SEC robust clusters CSV (from ``sec.robustness``):
  ``well_id, depth_median, depth_min, depth_max, agreement, persistence,
  cluster_diameter_m, wide_flag, ...``
* Caliper zones CSV (from the caliper pipeline):
  ``well, z_top, z_bot, z_centre, thickness_m, severity, peak_excess_cm,
  peak_excess_over_threshold_cm, ...``

All depths in both inputs are in metres, BGL-positive (depth below
ground level, increasing downward).

Outputs (built by the public ``compute_convergence`` function)
--------------------------------------------------------------
``cluster_matches`` : DataFrame, one row per analysed SEC cluster.
``well_summary``    : DataFrame, one row per well.
``unmatched_zones`` : DataFrame, anomalous caliper zones without a
                      matching SEC cluster (filtered by min-severity).

Design
------
The module is **purely functional and config-driven**. Every behavioural
parameter (matching rule, tolerance, severity weights, filters, best-
match strategy) is read from the config dict that the caller passes in.
No magic numbers are hardcoded in the matching logic. Tests can stub a
config dict directly without going through YAML.

Convention reminders
--------------------
* SEC robustness CSV uses column ``well_id``.
* Caliper zones CSV uses column ``well``.
  We rename caliper to ``well_id`` on load to keep downstream uniform.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────────────────────────────
#  I/O loaders
# ──────────────────────────────────────────────────────────────────────
def load_sec_clusters(path: str | Path) -> pd.DataFrame:
    """Read the SEC robustness clusters CSV.

    Required columns
    ----------------
    well_id, cluster_id, depth_median, depth_min, depth_max, depth_iqr,
    persistence, agreement, cluster_diameter_m, wide_flag.

    Returns
    -------
    DataFrame with the columns above, dtypes left as read by pandas.
    """
    df = pd.read_csv(path)
    required = {
        "well_id", "cluster_id", "depth_median", "depth_min", "depth_max",
        "agreement", "persistence", "cluster_diameter_m", "wide_flag",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"SEC clusters CSV {path} is missing columns: {sorted(missing)}"
        )
    return df


def load_caliper_zones(path: str | Path) -> pd.DataFrame:
    """Read the caliper anomalous-zones CSV and normalise.

    Renames the leading ``well`` column to ``well_id`` so downstream
    code can join on a single key name.

    Required columns (in the source file)
    -------------------------------------
    well, z_top, z_bot, z_centre, thickness_m, severity, peak_excess_cm.
    """
    df = pd.read_csv(path)
    required = {
        "well", "z_top", "z_bot", "z_centre", "thickness_m", "severity",
        "peak_excess_cm",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Caliper zones CSV {path} is missing columns: {sorted(missing)}"
        )
    df = df.rename(columns={"well": "well_id"}).copy()
    # Reset the index after the rename so each zone has a stable integer id.
    df = df.reset_index(drop=True)
    df["zone_id"] = df.index
    return df


# ──────────────────────────────────────────────────────────────────────
#  Geometric primitives
# ──────────────────────────────────────────────────────────────────────
def compute_overlap_m(
    cluster_min: float, cluster_max: float,
    zone_bot: float, zone_top: float,
) -> float:
    """Length of the depth interval common to a cluster and a zone (m).

    Both intervals are given in BGL coordinates (depth increases
    downward), so we treat them as ``[lo, hi]`` with
    ``lo = min(depth_min, depth_max)`` etc., to be robust to whichever
    direction the source CSV uses.

    Returns 0 if the intervals are disjoint.
    """
    c_lo, c_hi = sorted([cluster_min, cluster_max])
    z_lo, z_hi = sorted([zone_bot, zone_top])
    return max(0.0, min(c_hi, z_hi) - max(c_lo, z_lo))


def compute_center_distance_m(
    cluster_center: float, zone_center: float,
) -> float:
    """Absolute distance between cluster median depth and zone centre."""
    return float(abs(cluster_center - zone_center))


# ──────────────────────────────────────────────────────────────────────
#  Matching
# ──────────────────────────────────────────────────────────────────────
def find_matches(
    cluster_row: pd.Series,
    zones_df: pd.DataFrame,
    *,
    matching_rule: str,
    tolerance_m: float,
) -> pd.DataFrame:
    """Return the subset of zones that match a single cluster.

    The returned DataFrame includes the columns of ``zones_df`` plus
    two extras that quantify the match:

    * ``overlap_m`` — depth overlap of the intervals (>= 0).
    * ``center_distance_m`` — |cluster.depth_median − zone.z_centre|.

    Parameters
    ----------
    cluster_row : pd.Series
        One row from the SEC clusters DataFrame.
    zones_df : pd.DataFrame
        Caliper zones for the same well (caller filters by well_id).
    matching_rule : {"overlap_only", "center_distance_only", "hybrid"}
    tolerance_m : float
        Centre-to-centre tolerance, used by the rules that include
        distance.

    Notes
    -----
    Rules are evaluated row-wise on ``zones_df``. The function never
    mutates its inputs.
    """
    if zones_df.empty:
        cols = list(zones_df.columns) + ["overlap_m", "center_distance_m"]
        return pd.DataFrame(columns=cols)

    overlap = zones_df.apply(
        lambda r: compute_overlap_m(
            cluster_row["depth_min"], cluster_row["depth_max"],
            r["z_bot"], r["z_top"],
        ),
        axis=1,
    )
    center_d = zones_df.apply(
        lambda r: compute_center_distance_m(
            cluster_row["depth_median"], r["z_centre"],
        ),
        axis=1,
    )

    if matching_rule == "overlap_only":
        keep = overlap > 0
    elif matching_rule == "center_distance_only":
        keep = center_d <= tolerance_m
    elif matching_rule == "hybrid":
        keep = (overlap > 0) | (center_d <= tolerance_m)
    else:
        raise ValueError(
            f"Unknown matching_rule {matching_rule!r}. Expected one of "
            f"'overlap_only', 'center_distance_only', 'hybrid'."
        )

    out = zones_df.loc[keep].copy()
    out["overlap_m"] = overlap.loc[keep].values
    out["center_distance_m"] = center_d.loc[keep].values
    return out


def select_best_match(
    matches_df: pd.DataFrame,
    *,
    priority: str,
    severity_weights: dict[str, int | float],
) -> Optional[pd.Series]:
    """Pick the single 'best' match according to the priority strategy.

    Returns ``None`` if ``matches_df`` is empty. Tie-breaking is built
    in so each strategy yields a deterministic single row.

    Parameters
    ----------
    matches_df : DataFrame returned by ``find_matches``.
    priority : {"overlap_then_distance", "max_severity", "max_peak_excess"}
    severity_weights : dict
        Mapping severity label → numeric weight, used by
        ``max_severity`` priority.
    """
    if matches_df.empty:
        return None

    df = matches_df.copy()

    if priority == "overlap_then_distance":
        # max overlap; tie-break: min center distance
        df = df.sort_values(
            by=["overlap_m", "center_distance_m"],
            ascending=[False, True],
            kind="mergesort",  # stable, keeps source order on full ties
        )
        return df.iloc[0]

    if priority == "max_severity":
        df["_sev_w"] = df["severity"].map(severity_weights).astype(float)
        df = df.sort_values(
            by=["_sev_w", "overlap_m", "center_distance_m"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        return df.iloc[0].drop(labels=["_sev_w"])

    if priority == "max_peak_excess":
        df = df.sort_values(
            by=["peak_excess_cm", "overlap_m", "center_distance_m"],
            ascending=[False, False, True],
            kind="mergesort",
        )
        return df.iloc[0]

    raise ValueError(
        f"Unknown best_match_priority {priority!r}. Expected one of "
        f"'overlap_then_distance', 'max_severity', 'max_peak_excess'."
    )


# ──────────────────────────────────────────────────────────────────────
#  Scoring
# ──────────────────────────────────────────────────────────────────────
def score_cluster(
    cluster_row: pd.Series,
    matches_df: pd.DataFrame,
    *,
    severity_weights: dict[str, int | float],
) -> dict:
    """Compute scoring fields for one cluster given its matches.

    Returns a dict with the convergence score and aggregate
    descriptors of the match set. Designed to be merged with the
    cluster row to form one output row.
    """
    n_matched = len(matches_df)
    has_match = n_matched > 0

    if has_match:
        weights = matches_df["severity"].map(severity_weights).astype(float)
        sev_max = float(weights.max())
        # Severity label corresponding to the max weight (deterministic via
        # the inverse mapping; if multiple severities share the max weight,
        # take the alphabetically first to be reproducible).
        max_sev_labels = sorted(
            s for s, w in severity_weights.items() if w == sev_max
        )
        max_severity_label = max_sev_labels[0] if max_sev_labels else None
        all_sev = ",".join(sorted(matches_df["severity"].unique()))
    else:
        sev_max = 0.0
        max_severity_label = None
        all_sev = ""

    convergence_score = float(cluster_row["agreement"]) * sev_max

    return {
        "has_caliper_match": bool(has_match),
        "is_converging": bool(has_match),  # alias kept for explicit binary read
        "n_caliper_zones_matched": int(n_matched),
        "multi_match_flag": bool(n_matched > 1),
        "max_matched_severity": max_severity_label,
        "all_matched_severities": all_sev,
        "severity_weight_max": sev_max,
        "convergence_score": convergence_score,
    }


# ──────────────────────────────────────────────────────────────────────
#  Public top-level orchestrator
# ──────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ConvergenceResult:
    """Container for the three output tables of ``compute_convergence``."""
    cluster_matches: pd.DataFrame
    well_summary: pd.DataFrame
    unmatched_zones: pd.DataFrame


def compute_convergence(
    sec_clusters: pd.DataFrame,
    caliper_zones: pd.DataFrame,
    *,
    config: dict,
) -> ConvergenceResult:
    """Run the full SEC ↔ caliper convergence analysis.

    Parameters
    ----------
    sec_clusters : DataFrame
        Output of ``load_sec_clusters`` (or equivalent).
    caliper_zones : DataFrame
        Output of ``load_caliper_zones`` (with ``well_id`` column).
    config : dict
        The ``convergence.sec_caliper`` sub-config (already validated
        by ``karst_analysis.config.load_config``).

    Returns
    -------
    ConvergenceResult
    """
    # Read config keys with explicit names so the function reads top-down.
    agreement_min: int = config["sec_agreement_min"]
    severity_filter: Optional[list[str]] = config["caliper_severity_filter"]
    matching_rule: str = config["matching_rule"]
    tolerance_m: float = config["tolerance_m"]
    best_match_priority: str = config["best_match_priority"]
    severity_weights: dict = config["severity_weights"]
    unmatched_min_sev: Optional[str] = config["unmatched_zones_min_severity"]

    # ── Filter inputs ──
    clusters = sec_clusters.loc[
        sec_clusters["agreement"] >= agreement_min
    ].copy()

    if severity_filter is not None:
        zones = caliper_zones.loc[
            caliper_zones["severity"].isin(severity_filter)
        ].copy()
    else:
        zones = caliper_zones.copy()

    # ── Per-cluster matching loop ──
    rows: list[dict] = []
    matched_zone_ids: set[tuple[str, int]] = set()  # (well_id, zone_id)

    for _, cluster_row in clusters.iterrows():
        well = cluster_row["well_id"]
        zones_well = zones.loc[zones["well_id"] == well]

        matches = find_matches(
            cluster_row, zones_well,
            matching_rule=matching_rule,
            tolerance_m=tolerance_m,
        )

        # Track which zones matched anything (for the unmatched-zones report).
        for zid in matches["zone_id"].tolist():
            matched_zone_ids.add((well, int(zid)))

        best = select_best_match(
            matches,
            priority=best_match_priority,
            severity_weights=severity_weights,
        )
        scores = score_cluster(
            cluster_row, matches,
            severity_weights=severity_weights,
        )

        # Compose the output row: cluster identity + scores + best-match
        # descriptors (NaN if no match).
        row = {
            "well_id": well,
            "cluster_id": cluster_row["cluster_id"],
            "depth_median": cluster_row["depth_median"],
            "depth_min": cluster_row["depth_min"],
            "depth_max": cluster_row["depth_max"],
            "agreement": cluster_row["agreement"],
            "persistence": cluster_row["persistence"],
            "cluster_diameter_m": cluster_row["cluster_diameter_m"],
            "wide_flag": cluster_row["wide_flag"],
            **scores,
        }

        if best is not None:
            row.update({
                "best_match_zone_id": int(best["zone_id"]),
                "best_match_severity": best["severity"],
                "best_match_overlap_m": float(best["overlap_m"]),
                "best_match_center_distance_m": float(best["center_distance_m"]),
                "best_match_thickness_m": float(best["thickness_m"]),
                "best_match_peak_excess_cm": float(best["peak_excess_cm"]),
                "best_match_z_top": float(best["z_top"]),
                "best_match_z_bot": float(best["z_bot"]),
            })
        else:
            row.update({
                "best_match_zone_id": pd.NA,
                "best_match_severity": pd.NA,
                "best_match_overlap_m": np.nan,
                "best_match_center_distance_m": np.nan,
                "best_match_thickness_m": np.nan,
                "best_match_peak_excess_cm": np.nan,
                "best_match_z_top": np.nan,
                "best_match_z_bot": np.nan,
            })

        rows.append(row)

    cluster_matches = pd.DataFrame(rows)

    # ── Per-well summary ──
    well_summary = _build_well_summary(cluster_matches)

    # ── Unmatched zones report ──
    unmatched_zones = _build_unmatched_zones(
        zones,
        matched_zone_ids,
        min_severity=unmatched_min_sev,
        severity_weights=severity_weights,
    )

    return ConvergenceResult(
        cluster_matches=cluster_matches,
        well_summary=well_summary,
        unmatched_zones=unmatched_zones,
    )


def _build_well_summary(cluster_matches: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cluster_matches to one row per well."""
    if cluster_matches.empty:
        return pd.DataFrame(columns=[
            "well_id", "n_clusters_analyzed", "n_converging",
            "fraction_converging", "n_with_severe_match",
            "n_with_moderate_match", "n_with_mild_match",
            "max_convergence_score", "mean_convergence_score_converging",
        ])

    def _agg(g: pd.DataFrame) -> pd.Series:
        n_total = len(g)
        n_conv = int(g["is_converging"].sum())
        sev = g["max_matched_severity"]
        return pd.Series({
            "n_clusters_analyzed": n_total,
            "n_converging": n_conv,
            "fraction_converging": n_conv / n_total if n_total else 0.0,
            "n_with_severe_match": int((sev == "severe").sum()),
            "n_with_moderate_match": int((sev == "moderate").sum()),
            "n_with_mild_match": int((sev == "mild").sum()),
            "max_convergence_score": float(g["convergence_score"].max()),
            "mean_convergence_score_converging": float(
                g.loc[g["is_converging"], "convergence_score"].mean()
            ) if n_conv > 0 else 0.0,
        })

    summary = (
        cluster_matches.groupby("well_id", sort=True)
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    return summary


def _build_unmatched_zones(
    zones: pd.DataFrame,
    matched_ids: set[tuple[str, int]],
    *,
    min_severity: Optional[str],
    severity_weights: dict[str, int | float],
) -> pd.DataFrame:
    """Return zones that were not matched by any SEC cluster.

    Filtered by min_severity (if not None): only zones whose severity
    weight is >= the weight of min_severity are kept.
    """
    if zones.empty:
        return zones.iloc[0:0].copy()

    is_unmatched = ~zones.apply(
        lambda r: (r["well_id"], int(r["zone_id"])) in matched_ids,
        axis=1,
    )
    out = zones.loc[is_unmatched].copy()

    if min_severity is not None:
        threshold = severity_weights[min_severity]
        keep = out["severity"].map(severity_weights).astype(float) >= threshold
        out = out.loc[keep].copy()

    # Project to a tidy subset of columns.
    cols = [
        "well_id", "z_top", "z_bot", "z_centre", "thickness_m",
        "severity", "peak_excess_cm",
    ]
    return out[cols].reset_index(drop=True)
