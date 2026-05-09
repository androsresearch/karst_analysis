"""Vertical-position adjustments for SEC profiles.

Different YSI configurations and operators introduce small offsets in
the recorded depth that need correction before further processing.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from karst_analysis.sec.io.columns import find_column_name


VALID_METHODS = ("TOM", "YSI")


def adjust_vertical_position(
    df: pd.DataFrame,
    adjustment: float = 0.272,
    method: str = "TOM",
    depth_col: Optional[str] = None,
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Add an offset to the depth column.

    Parameters
    ----------
    df : pd.DataFrame
    adjustment : float, default 0.272
        Metres to add to the depth values.
    method : str, default "TOM"
        - ``"TOM"`` : add the offset only to depths ≥ 0.001 m, leave
          values ≤ 0 unchanged.
        - ``"YSI"`` : add the offset to all depths.
    depth_col : str, optional
        Override depth column auto-detection.
    column_mappings : dict, optional
    logger : logging.Logger, optional

    Returns
    -------
    pd.DataFrame
        Copy with adjusted depth values.
    """
    if depth_col is None:
        depth_col = find_column_name(df, "depth", column_mappings)
        if depth_col is None:
            raise ValueError("No depth column found in DataFrame.")

    if method not in VALID_METHODS:
        raise ValueError(f"Invalid method '{method}'. Must be one of {VALID_METHODS}.")

    out = df.copy()

    if method == "TOM":
        mask = out[depth_col] >= 0.001
        out.loc[mask, depth_col] = out.loc[mask, depth_col] + adjustment
        if logger:
            adjusted, untouched = int(mask.sum()), int((~mask).sum())
            logger.info(
                f"TOM adjustment (+{adjustment} m): {adjusted} adjusted, "
                f"{untouched} unchanged."
            )
    else:  # method == "YSI"
        out[depth_col] = out[depth_col] + adjustment
        if logger:
            logger.info(f"YSI adjustment (+{adjustment} m) applied to all rows.")

    return out
