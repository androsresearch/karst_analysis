"""Cleaning operations for SEC profiles.

These are the deterministic, non-smoothing steps that prepare a raw YSI
profile for analysis. Each function takes a DataFrame, returns a
DataFrame (and optionally stats), and is independently testable.

Operations included:
    ensure_chronological_order  : sort by acquisition time if available
    filter_negative_depths      : drop above-water-table noise
    filter_monotonic_descent    : remove probe-bounce reversals
    average_duplicate_depths    : collapse repeated depth readings
    resample_pchip              : interpolate to uniform Δz (PCHIP)
    resample_linear             : interpolate to uniform Δz (linear)
    enforce_monotonic_ec        : PAVA isotonic regression on EC

Note: ``filter_monotonic_descent`` and ``enforce_monotonic_ec`` enforce
DIFFERENT monotonicities:
    - filter_monotonic_descent : depth must increase with sample order
      (the probe is going down). Filters PROBE BEHAVIOUR.
    - enforce_monotonic_ec     : conductivity must not decrease with depth.
      Filters POST-SMOOTHING ARTEFACTS in a freshwater-lens setting.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator

from karst_analysis.sec.io.columns import find_column_name


# ─────────────────────────────────────────────────────────────────────────
#  1. Chronological order
# ─────────────────────────────────────────────────────────────────────────
def ensure_chronological_order(
    df: pd.DataFrame,
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Sort by time columns if present, otherwise return a copy unchanged."""
    time_col = find_column_name(df, "time", column_mappings)
    time_frac_col = find_column_name(df, "time_frac", column_mappings)

    sort_cols = [c for c in (time_col, time_frac_col) if c is not None]

    if sort_cols:
        return df.sort_values(by=sort_cols).reset_index(drop=True)

    if logger:
        logger.warning("No time columns detected — returning unsorted copy.")
    return df.copy()


# ─────────────────────────────────────────────────────────────────────────
#  2. Filter negative depths (above water table noise / surface)
# ─────────────────────────────────────────────────────────────────────────
def filter_negative_depths(
    df: pd.DataFrame,
    depth_col: Optional[str] = None,
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Drop rows where depth is negative."""
    if depth_col is None:
        depth_col = find_column_name(df, "depth", column_mappings)
        if depth_col is None:
            raise ValueError("No depth column found in DataFrame.")

    n0 = len(df)
    out = df[df[depth_col] >= 0].copy()
    removed = n0 - len(out)

    if logger:
        logger.info(f"Removed {removed} rows with depth < 0.")
    return out


# ─────────────────────────────────────────────────────────────────────────
#  3. Filter probe-bounce (non-monotonic descent)
# ─────────────────────────────────────────────────────────────────────────
def filter_monotonic_descent(
    df: pd.DataFrame,
    depth_col: Optional[str] = None,
    tolerance: float = 0.002,
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, dict]:
    """Drop readings where the probe moved upward beyond ``tolerance``.

    Walks the time-ordered samples and keeps each row only if its depth
    is at least ``max_depth_so_far - tolerance``. Useful to remove
    spurious upticks during descent.

    Returns
    -------
    (DataFrame, stats dict)
        Stats keys: total_readings, kept_readings, removed_readings,
        removal_pct, max_reversal_m.
    """
    if depth_col is None:
        depth_col = find_column_name(df, "depth", column_mappings)
        if depth_col is None:
            raise ValueError("No depth column found in DataFrame.")

    depths = df[depth_col].values
    keep = np.ones(len(df), dtype=bool)
    max_d = -np.inf
    max_reversal = 0.0

    for i, d in enumerate(depths):
        if d >= max_d - tolerance:
            max_d = max(max_d, d)
        else:
            keep[i] = False
            max_reversal = max(max_reversal, max_d - d)

    out = df[keep].copy()
    n_total = len(df)
    n_kept = int(keep.sum())
    n_removed = n_total - n_kept

    stats = {
        "total_readings": n_total,
        "kept_readings": n_kept,
        "removed_readings": n_removed,
        "removal_pct": (n_removed / n_total * 100.0) if n_total else 0.0,
        "max_reversal_m": float(max_reversal),
    }

    if logger and n_removed:
        logger.info(
            f"Monotonic-descent filter removed {n_removed} rows "
            f"({stats['removal_pct']:.1f}%); max reversal {max_reversal:.3f} m."
        )

    return out, stats


# ─────────────────────────────────────────────────────────────────────────
#  4. Average duplicate depths
# ─────────────────────────────────────────────────────────────────────────
def average_duplicate_depths(
    df: pd.DataFrame,
    depth_col: Optional[str] = None,
    value_col: Optional[str] = None,
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse rows sharing the same depth by averaging their values.

    Returns
    -------
    (averaged DataFrame, duplicates DataFrame)
        The duplicates DataFrame has columns ``Duplicated Depth`` and
        ``Frequency`` for inspection.
    """
    if depth_col is None:
        depth_col = find_column_name(df, "depth", column_mappings)
        if depth_col is None:
            raise ValueError("No depth column found in DataFrame.")
    if value_col is None:
        value_col = find_column_name(df, "conductivity", column_mappings)
        if value_col is None:
            raise ValueError("No conductivity column found in DataFrame.")

    grouped = df.groupby(depth_col)[value_col].agg(["mean", "count"]).reset_index()
    duplicates = grouped[grouped["count"] > 1][[depth_col, "count"]].copy()
    duplicates.columns = ["Duplicated Depth", "Frequency"]

    out = pd.DataFrame({depth_col: grouped[depth_col], value_col: grouped["mean"]})

    if logger and len(duplicates):
        logger.info(f"Averaged {len(duplicates)} duplicate-depth groups.")
    return out, duplicates


# ─────────────────────────────────────────────────────────────────────────
#  5a. Resample to uniform grid — PCHIP (preserves shape, no overshoot)
# ─────────────────────────────────────────────────────────────────────────
def resample_pchip(
    df: pd.DataFrame,
    depth_col: Optional[str] = None,
    value_col: Optional[str] = None,
    dz: Optional[float] = None,
    dz_method: str = "percentile95",
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Resample depth profile to uniform spacing using PCHIP interpolation.

    PCHIP is a piecewise-cubic Hermite interpolant that does not overshoot
    on monotonic data — a good default for SEC profiles.
    """
    return _resample_generic(
        df, kind="pchip",
        depth_col=depth_col, value_col=value_col,
        dz=dz, dz_method=dz_method,
        column_mappings=column_mappings, logger=logger,
    )


# ─────────────────────────────────────────────────────────────────────────
#  5b. Resample to uniform grid — linear (faster, simpler)
# ─────────────────────────────────────────────────────────────────────────
def resample_linear(
    df: pd.DataFrame,
    depth_col: Optional[str] = None,
    value_col: Optional[str] = None,
    dz: Optional[float] = None,
    dz_method: str = "percentile95",
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Resample to uniform spacing using linear interpolation."""
    return _resample_generic(
        df, kind="linear",
        depth_col=depth_col, value_col=value_col,
        dz=dz, dz_method=dz_method,
        column_mappings=column_mappings, logger=logger,
    )


def _resample_generic(
    df, *, kind,
    depth_col, value_col,
    dz, dz_method,
    column_mappings, logger,
) -> pd.DataFrame:
    if depth_col is None:
        depth_col = find_column_name(df, "depth", column_mappings)
        if depth_col is None:
            raise ValueError("No depth column found in DataFrame.")
    if value_col is None:
        value_col = find_column_name(df, "conductivity", column_mappings)
        if value_col is None:
            raise ValueError("No conductivity column found in DataFrame.")

    sorted_df = df.sort_values(depth_col).copy()
    depths = sorted_df[depth_col].values
    values = sorted_df[value_col].values

    if dz is None:
        delta = np.diff(depths)
        if len(delta) == 0:
            raise ValueError("Need at least 2 distinct depths to resample.")
        dz = {
            "percentile95": np.percentile(delta, 95),
            "median": np.median(delta),
            "mean": np.mean(delta),
            "min": np.min(delta),
        }.get(dz_method)
        if dz is None:
            raise ValueError(f"Unknown dz_method: {dz_method}")

    z_min, z_max = depths.min(), depths.max()
    grid = np.arange(z_min, z_max + dz, dz)
    grid = grid[grid <= z_max]

    if kind == "pchip":
        interp = PchipInterpolator(depths, values)
        v_new = interp(grid)
    elif kind == "linear":
        v_new = np.interp(grid, depths, values)
    else:
        raise ValueError(f"Unknown interpolation kind: {kind}")

    if logger:
        logger.info(f"Resampled ({kind}): {len(df)} → {len(grid)} pts (dz={dz:.4f}).")

    return pd.DataFrame({depth_col: grid, value_col: v_new})


# ─────────────────────────────────────────────────────────────────────────
#  6. PAVA — enforce monotonic EC with depth
# ─────────────────────────────────────────────────────────────────────────
def enforce_monotonic_ec(
    df: pd.DataFrame,
    depth_col: Optional[str] = None,
    value_col: Optional[str] = None,
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, int]:
    """Project EC onto the cone of sequences non-decreasing with depth.

    Pool Adjacent Violators Algorithm (PAVA). Removes physically
    impossible reversals (EC dropping with depth in a coastal aquifer)
    while leaving already-monotonic stretches untouched.

    Returns
    -------
    (DataFrame with corrected EC, number of points actually adjusted)
    """
    if depth_col is None:
        depth_col = find_column_name(df, "depth", column_mappings)
        if depth_col is None:
            raise ValueError("No depth column found in DataFrame.")
    if value_col is None:
        value_col = find_column_name(df, "conductivity", column_mappings)
        if value_col is None:
            raise ValueError("No conductivity column found in DataFrame.")

    # Sort shallow→deep so "non-decreasing with depth" maps to
    # "non-decreasing along the array index".
    out = df.sort_values(depth_col).reset_index(drop=True).copy()
    ec = out[value_col].astype(float).to_numpy().copy()

    means = [float(v) for v in ec]
    sizes = [1] * len(ec)
    i = 0
    while i < len(means) - 1:
        if means[i] > means[i + 1]:
            new_size = sizes[i] + sizes[i + 1]
            new_mean = (means[i] * sizes[i] + means[i + 1] * sizes[i + 1]) / new_size
            means[i] = new_mean
            sizes[i] = new_size
            del means[i + 1], sizes[i + 1]
            while i > 0 and means[i - 1] > means[i]:
                new_size = sizes[i - 1] + sizes[i]
                new_mean = (
                    means[i - 1] * sizes[i - 1] + means[i] * sizes[i]
                ) / new_size
                means[i - 1] = new_mean
                sizes[i - 1] = new_size
                del means[i], sizes[i]
                i -= 1
        else:
            i += 1

    monotonic = np.empty(len(ec), dtype=float)
    idx = 0
    for m, s in zip(means, sizes):
        monotonic[idx:idx + s] = m
        idx += s

    n_corrected = int(np.sum(np.abs(monotonic - ec) > 1e-9))
    out[value_col] = monotonic

    if logger:
        logger.info(f"PAVA adjusted {n_corrected}/{len(ec)} points.")

    return out, n_corrected
