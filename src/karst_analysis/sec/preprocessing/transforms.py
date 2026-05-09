"""Value-space transforms for SEC profiles."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from karst_analysis.sec.io.columns import find_column_name


def apply_log10_conductivity(
    df: pd.DataFrame,
    value_col: Optional[str] = None,
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Add a log10-transformed copy of the conductivity column.

    The new column is named ``"log10_<value_col>"``. Non-positive values
    become NaN.
    """
    if value_col is None:
        value_col = find_column_name(df, "conductivity", column_mappings)
        if value_col is None:
            raise ValueError("No conductivity column found in DataFrame.")

    out = df.copy()
    log_col = f"log10_{value_col}"

    values = out[value_col].astype(float).values
    non_positive = values <= 0
    n_bad = int(non_positive.sum())

    if n_bad and logger:
        logger.warning(
            f"{n_bad} non-positive conductivity values replaced with NaN in {log_col}."
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        log_vals = np.log10(values)
    log_vals[non_positive] = np.nan

    out[log_col] = log_vals

    if logger:
        logger.info(f"Added log10 column: {log_col}")
    return out
