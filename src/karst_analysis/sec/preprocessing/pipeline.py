"""End-to-end SEC preprocessing pipelines.

Two parallel pipelines, each producing a smoothed profile from a raw YSI
DataFrame. They share six cleaning steps and differ only in the smoothing
back-end (and an optional PAVA monotonicity enforcement after LOWESS).

Both pipelines also append a ``depth_bgl_m`` column when a vadose-zone
thickness is provided, so the output is directly comparable with caliper
logs (which use depth-below-ground-level).

Pipeline shape
--------------
    raw → chronological → adjust → filter neg → monotonic descent
        → average dups → resample → SMOOTH → [PAVA] → log10
        → [+ depth_bgl_m]
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from karst_analysis.corrections import ysi_to_depth_below_ground
from karst_analysis.sec.io.columns import find_column_name
from karst_analysis.sec.preprocessing.adjustments import adjust_vertical_position
from karst_analysis.sec.preprocessing.cleaning import (
    average_duplicate_depths,
    enforce_monotonic_ec,
    ensure_chronological_order,
    filter_monotonic_descent,
    filter_negative_depths,
    resample_pchip,
)
from karst_analysis.sec.preprocessing.smoothing import (
    apply_savgol_filter,
    lowess_smooth,
)
from karst_analysis.sec.preprocessing.transforms import apply_log10_conductivity


# ─────────────────────────────────────────────────────────────────────────
#  Shared cleaning prelude (steps 1–6, identical for both pipelines)
# ─────────────────────────────────────────────────────────────────────────
def _shared_cleaning(
    df: pd.DataFrame,
    *,
    apply_depth_adjustment: bool,
    depth_adjustment: float,
    depth_adjustment_method: str,
    apply_monotonic_descent_filter: bool,
    monotonic_descent_tolerance: float,
    dz: Optional[float],
    dz_method: str,
    column_mappings: Optional[dict],
    logger: Optional[logging.Logger],
) -> tuple[pd.DataFrame, dict]:
    """Run the cleaning steps that precede smoothing.

    Returns (cleaned DataFrame, stats dict).
    """
    stats: dict = {"original_rows": len(df)}

    df = ensure_chronological_order(df, column_mappings=column_mappings, logger=logger)

    if apply_depth_adjustment:
        df = adjust_vertical_position(
            df,
            adjustment=depth_adjustment,
            method=depth_adjustment_method,
            column_mappings=column_mappings,
            logger=logger,
        )
        stats["depth_adjustment_applied"] = True
        stats["depth_adjustment_method"] = depth_adjustment_method
    else:
        stats["depth_adjustment_applied"] = False

    df = filter_negative_depths(df, column_mappings=column_mappings, logger=logger)
    stats["after_negative_filter"] = len(df)

    if apply_monotonic_descent_filter:
        df, mono_stats = filter_monotonic_descent(
            df,
            tolerance=monotonic_descent_tolerance,
            column_mappings=column_mappings,
            logger=logger,
        )
        stats["monotonic_descent"] = mono_stats
    else:
        stats["monotonic_descent"] = None

    df, duplicates = average_duplicate_depths(
        df, column_mappings=column_mappings, logger=logger
    )
    stats["duplicates_collapsed"] = len(duplicates)

    df = resample_pchip(
        df,
        dz=dz,
        dz_method=dz_method,
        column_mappings=column_mappings,
        logger=logger,
    )
    stats["after_resample"] = len(df)

    return df, stats


def _add_depth_bgl(
    df: pd.DataFrame,
    vadose_thickness_m: Optional[float],
    column_mappings: Optional[dict],
    logger: Optional[logging.Logger],
) -> pd.DataFrame:
    """Append a `depth_bgl_m` column if vadose thickness is provided."""
    if vadose_thickness_m is None:
        return df

    depth_col = find_column_name(df, "depth", column_mappings)
    if depth_col is None:
        if logger:
            logger.warning("Cannot add depth_bgl_m: no depth column.")
        return df

    out = df.copy()
    out["depth_bgl_m"] = ysi_to_depth_below_ground(
        out[depth_col].values, vadose_thickness_m
    )
    if logger:
        logger.info(f"Added depth_bgl_m (vadose = {vadose_thickness_m} m).")
    return out


# ─────────────────────────────────────────────────────────────────────────
#  Public pipelines
# ─────────────────────────────────────────────────────────────────────────
def process_savgol(
    df: pd.DataFrame,
    *,
    # cleaning
    apply_depth_adjustment: bool = False,
    depth_adjustment: float = 0.272,
    depth_adjustment_method: str = "TOM",
    apply_monotonic_descent_filter: bool = True,
    monotonic_descent_tolerance: float = 0.002,
    dz: Optional[float] = None,
    dz_method: str = "percentile95",
    # smoothing
    savgol_window: int = 11,
    savgol_order: int = 3,
    savgol_segmented: bool = True,
    savgol_gradient_factor: float = 20.0,
    savgol_min_gradient_threshold: float = 1000.0,
    # transforms
    apply_log10: bool = True,
    # vadose correction
    vadose_thickness_m: Optional[float] = None,
    # plumbing
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, dict]:
    """Full pipeline with Savitzky-Golay smoothing.

    Returns
    -------
    (DataFrame, stats dict)
    """
    df, stats = _shared_cleaning(
        df,
        apply_depth_adjustment=apply_depth_adjustment,
        depth_adjustment=depth_adjustment,
        depth_adjustment_method=depth_adjustment_method,
        apply_monotonic_descent_filter=apply_monotonic_descent_filter,
        monotonic_descent_tolerance=monotonic_descent_tolerance,
        dz=dz,
        dz_method=dz_method,
        column_mappings=column_mappings,
        logger=logger,
    )

    df = apply_savgol_filter(
        df,
        window_length=savgol_window,
        poly_order=savgol_order,
        segmented=savgol_segmented,
        gradient_factor=savgol_gradient_factor,
        min_gradient_threshold=savgol_min_gradient_threshold,
        column_mappings=column_mappings,
        logger=logger,
    )
    stats["smoothing"] = {
        "method": "savgol",
        "window": savgol_window,
        "order": savgol_order,
        "segmented": savgol_segmented,
        "gradient_factor": savgol_gradient_factor,
        "min_gradient_threshold": savgol_min_gradient_threshold,
    }

    if apply_log10:
        df = apply_log10_conductivity(df, column_mappings=column_mappings, logger=logger)
        stats["log10_applied"] = True
    else:
        stats["log10_applied"] = False

    df = _add_depth_bgl(df, vadose_thickness_m, column_mappings, logger)
    stats["vadose_thickness_m"] = vadose_thickness_m
    stats["final_rows"] = len(df)

    return df, stats


def process_lowess(
    df: pd.DataFrame,
    *,
    # cleaning
    apply_depth_adjustment: bool = False,
    depth_adjustment: float = 0.272,
    depth_adjustment_method: str = "TOM",
    apply_monotonic_descent_filter: bool = True,
    monotonic_descent_tolerance: float = 0.002,
    dz: Optional[float] = None,
    dz_method: str = "percentile95",
    # smoothing
    lowess_frac: float = 0.05,
    lowess_degree: int = 1,
    lowess_iter: int = 2,
    # PAVA monotonicity (post-smoothing)
    apply_pava: bool = True,
    # transforms
    apply_log10: bool = True,
    # vadose correction
    vadose_thickness_m: Optional[float] = None,
    # plumbing
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> tuple[pd.DataFrame, dict]:
    """Full pipeline with LOWESS smoothing and optional PAVA monotonicity.

    Returns
    -------
    (DataFrame, stats dict)
    """
    df, stats = _shared_cleaning(
        df,
        apply_depth_adjustment=apply_depth_adjustment,
        depth_adjustment=depth_adjustment,
        depth_adjustment_method=depth_adjustment_method,
        apply_monotonic_descent_filter=apply_monotonic_descent_filter,
        monotonic_descent_tolerance=monotonic_descent_tolerance,
        dz=dz,
        dz_method=dz_method,
        column_mappings=column_mappings,
        logger=logger,
    )

    df = lowess_smooth(
        df,
        frac=lowess_frac,
        degree=lowess_degree,
        n_robust_iter=lowess_iter,
        column_mappings=column_mappings,
        logger=logger,
    )
    stats["smoothing"] = {
        "method": "lowess",
        "frac": lowess_frac,
        "degree": lowess_degree,
        "iter": lowess_iter,
    }

    if apply_pava:
        df, n_corrected = enforce_monotonic_ec(
            df, column_mappings=column_mappings, logger=logger
        )
        stats["pava_applied"] = True
        stats["pava_n_corrected"] = n_corrected
    else:
        stats["pava_applied"] = False

    if apply_log10:
        df = apply_log10_conductivity(df, column_mappings=column_mappings, logger=logger)
        stats["log10_applied"] = True
    else:
        stats["log10_applied"] = False

    df = _add_depth_bgl(df, vadose_thickness_m, column_mappings, logger)
    stats["vadose_thickness_m"] = vadose_thickness_m
    stats["final_rows"] = len(df)

    return df, stats
