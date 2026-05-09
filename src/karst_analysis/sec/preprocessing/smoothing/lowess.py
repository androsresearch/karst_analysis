"""LOWESS / LOESS smoother with IRLS bisquare robustness.

Sliding-window two-pointer implementation: O(n·k) instead of O(n² log n).
Adapted from the Huang et al. (2024) framework.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from karst_analysis.sec.io.columns import find_column_name


def _lowess_smooth_array(
    z: np.ndarray,
    EC: np.ndarray,
    frac: float = 0.05,
    degree: int = 1,
    n_robust_iter: int = 2,
) -> np.ndarray:
    """Core LOWESS routine on raw arrays. Preserves input ordering."""
    order = np.argsort(z)
    z_s, EC_s = z[order], EC[order]
    n = len(z_s)
    k = max(int(frac * n), degree + 2)
    k = min(k, n)

    y_out = np.zeros(n, dtype=float)
    robust_w = np.ones(n, dtype=float)

    for it in range(n_robust_iter + 1):
        l = 0
        for i in range(n):
            while (l + k < n) and ((z_s[l + k] - z_s[i]) < (z_s[i] - z_s[l])):
                l += 1
            r = l + k
            dx = z_s[l:r] - z_s[i]
            d = np.abs(dx)
            h = max(d[0], d[-1], 1e-12)
            w = (1.0 - (d / h) ** 3) ** 3
            w = np.maximum(w, 0.0) * robust_w[l:r]

            X = np.empty((k, degree + 1))
            X[:, 0] = 1.0
            for p in range(1, degree + 1):
                X[:, p] = dx ** p
            WX = w[:, None] * X
            try:
                beta = np.linalg.solve(X.T @ WX, X.T @ (w * EC_s[l:r]))
                y_out[i] = beta[0]
            except np.linalg.LinAlgError:
                y_out[i] = EC_s[i]

        if it < n_robust_iter:
            resid = EC_s - y_out
            s = np.median(np.abs(resid)) + 1e-9
            u = np.clip(resid / (6.0 * s), -1.0, 1.0)
            robust_w = (1.0 - u ** 2) ** 2

    out = np.empty(n, dtype=float)
    out[order] = y_out
    return out


def lowess_smooth(
    df: pd.DataFrame,
    value_col: Optional[str] = None,
    depth_col: Optional[str] = None,
    frac: float = 0.05,
    degree: int = 1,
    n_robust_iter: int = 2,
    column_mappings: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """Smooth the conductivity column with LOWESS + IRLS bisquare.

    Parameters
    ----------
    df : pd.DataFrame
    value_col : str, optional
    depth_col : str, optional
    frac : float, default 0.05
        Fraction of points per local fit (0.03–0.10 typical).
    degree : int, default 1
        Local polynomial degree (1 = locally linear).
    n_robust_iter : int, default 2
        Bisquare re-weighting passes after the initial WLS.
    column_mappings : dict, optional
    logger : logging.Logger, optional
    """
    if value_col is None:
        value_col = find_column_name(df, "conductivity", column_mappings)
        if value_col is None:
            raise ValueError("No conductivity column found in DataFrame.")
    if depth_col is None:
        depth_col = find_column_name(df, "depth", column_mappings)
        if depth_col is None:
            raise ValueError("No depth column found in DataFrame.")

    out = df.copy()
    z = out[depth_col].astype(float).to_numpy()
    EC = out[value_col].astype(float).to_numpy()

    if logger:
        logger.info(f"LOWESS (frac={frac}, iter={n_robust_iter}) on {len(z)} points.")

    smoothed = _lowess_smooth_array(z, EC, frac=frac, degree=degree, n_robust_iter=n_robust_iter)
    out[value_col] = smoothed
    return out
