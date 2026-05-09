"""Savitzky-Golay smoothing — segmented and standard variants.

The segmented variant detects gradient discontinuities (sharp transitions)
and avoids smoothing across them, preserving step-like features that are
diagnostic of interface zones.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from karst_analysis.sec.io.columns import find_column_name


# ─────────────────────────────────────────────────────────────────────────
def _savgol_segmented(
    values: np.ndarray,
    window_length: int = 11,
    poly_order: int = 3,
    gradient_factor: float = 20.0,
    min_gradient_threshold: float = 1000.0,
    logger: Optional[logging.Logger] = None,
) -> np.ndarray:
    """Savitzky-Golay applied per segment between detected transitions."""
    gradients = np.abs(np.diff(values))
    threshold = max(np.median(gradients) * gradient_factor, min_gradient_threshold)
    discontinuities = np.where(gradients > threshold)[0]

    if len(discontinuities) == 0:
        return savgol_filter(values, window_length=window_length, polyorder=poly_order)

    if logger:
        logger.info(f"Detected {len(discontinuities)} gradient discontinuities.")

    half_window = window_length // 2
    is_transition = np.zeros(len(values), dtype=bool)
    for idx in discontinuities:
        start_idx = max(0, idx - half_window)
        end_idx = min(len(values), idx + half_window + 2)
        is_transition[start_idx:end_idx] = True

    out = values.copy()
    padded = np.concatenate(([True], is_transition, [True]))
    transitions = np.where(padded[:-1] != padded[1:])[0]

    segments = []
    for i in range(0, len(transitions), 2):
        start = transitions[i]
        end = transitions[i + 1]
        segments.append((start, end))

    smoothed = 0
    for start, end in segments:
        if end - start >= window_length:
            out[start:end] = savgol_filter(
                values[start:end], window_length=window_length, polyorder=poly_order
            )
            smoothed += 1

    if logger:
        logger.info(
            f"Smoothed {smoothed} segments; {int(is_transition.sum())} transition "
            f"points preserved."
        )
    return out


# ─────────────────────────────────────────────────────────────────────────
def apply_savgol_filter(
    df: pd.DataFrame,
    value_col: Optional[str] = None,
    window_length: int = 11,
    poly_order: int = 3,
    segmented: bool = True,
    gradient_factor: float = 20.0,
    min_gradient_threshold: float = 1000.0,
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Smooth the conductivity column with Savitzky-Golay.

    Parameters
    ----------
    df : pd.DataFrame
    value_col : str, optional
    window_length : int, default 11
        Filter window length (must be odd; auto-corrected if not).
    poly_order : int, default 3
        Polynomial order. Must be < window_length.
    segmented : bool, default True
        If True, use the segmented variant that preserves transitions.
    gradient_factor : float, default 20.0
        Multiplier on the median gradient to define a discontinuity.
    min_gradient_threshold : float, default 1000.0
        Floor on the discontinuity threshold (absolute units of EC).
    column_mappings : dict, optional
    logger : logging.Logger, optional
    """
    if value_col is None:
        value_col = find_column_name(df, "conductivity", column_mappings)
        if value_col is None:
            raise ValueError("No conductivity column found in DataFrame.")

    out = df.copy()

    if window_length % 2 == 0:
        window_length += 1
        if logger:
            logger.warning(f"window_length must be odd; adjusted to {window_length}.")

    if len(df) < window_length:
        if logger:
            logger.warning(
                f"Not enough points ({len(df)}) for window={window_length}; "
                f"returning unsmoothed."
            )
        return out

    if segmented:
        smoothed = _savgol_segmented(
            out[value_col].values,
            window_length=window_length,
            poly_order=poly_order,
            gradient_factor=gradient_factor,
            min_gradient_threshold=min_gradient_threshold,
            logger=logger,
        )
    else:
        smoothed = savgol_filter(
            out[value_col].values, window_length=window_length, polyorder=poly_order
        )
        if logger:
            logger.info(f"Applied SavGol (window={window_length}, order={poly_order}).")

    out[value_col] = smoothed
    return out
