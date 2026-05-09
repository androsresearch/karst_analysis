"""Breakpoint detection: BIC sweep and elbow selection.

Wraps `piecewise_regression.ModelSelection` with the multi-trial pattern
needed for stable results (the underlying optimiser uses random
initialisation, so a single run is not reproducible enough).
"""

from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np
import pandas as pd
import piecewise_regression as pw
from piecewise_regression.main import Fit


def elbow_max_distance(metric: Union[pd.Series, np.ndarray]) -> int:
    """Find the elbow of a monotonic-ish curve using max-distance-to-line.

    Connects the first and last points with a straight line, then returns
    the index whose perpendicular distance to that line is largest.
    Robust default for "knee" detection on BIC vs N curves.

    Parameters
    ----------
    metric : array-like
        Sequence of metric values (e.g. BIC for n_breakpoints=1..N).

    Returns
    -------
    int
        Index (0-based) of the elbow.
    """
    if isinstance(metric, pd.Series):
        metric = metric.values
    if not isinstance(metric, np.ndarray):
        raise ValueError("Input must be a pandas.Series or numpy.ndarray.")
    if len(metric) < 2:
        raise ValueError("Need at least two elements to compute an elbow.")

    start_idx, end_idx = 0, len(metric) - 1
    start_value, end_value = metric[start_idx], metric[end_idx]

    line_vec = np.array([end_idx - start_idx, end_value - start_value])
    line_vec = line_vec / np.linalg.norm(line_vec)

    distances = []
    for i in range(len(metric)):
        point_vec = np.array([i - start_idx, metric[i] - start_value])
        proj_length = np.dot(point_vec, line_vec)
        proj_vec = proj_length * line_vec
        distance_vec = point_vec - proj_vec
        distances.append(np.linalg.norm(distance_vec))

    return int(np.argmax(distances))


def best_n_breakpoints(
    x,
    y,
    max_breakpoints: int = 10,
    n_trials: int = 3,
    tolerance: float = 1e-5,
    min_distance: float = 0.01,
) -> Dict[str, Any]:
    """Run a multi-trial BIC sweep for breakpoint selection.

    Because ``piecewise_regression`` uses random initialisation, each
    trial may converge to a slightly different optimum. Running multiple
    trials and inspecting their agreement provides a stability check.

    Parameters
    ----------
    x, y : array-like
        Profile coordinates (e.g. depth and EC).
    max_breakpoints : int, default 10
    n_trials : int, default 3
    tolerance : float, default 1e-5
    min_distance : float, default 0.01

    Returns
    -------
    dict
        Keys: ``trial_1``, ``trial_2``, ... Each value is a dict with:
            df                          : full ModelSelection DataFrame
            best_n_breakpoint_bic       : index of elbow on BIC
            min_bic_n_breakpoint        : N with the absolute lowest BIC
            best_n_breakpoint_rss       : index of elbow on RSS
    """
    results: Dict[str, Any] = {}

    for i in range(n_trials):
        ms = pw.ModelSelection(
            x, y, max_breakpoints,
            tolerance=tolerance,
            min_distance_between_breakpoints=min_distance,
        )
        ms_df = pd.DataFrame(ms.model_summaries)

        y_bic = ms_df["bic"]
        y_rss = ms_df["rss"]

        best_n_bic = elbow_max_distance(y_bic)
        best_n_rss = elbow_max_distance(y_rss)

        min_bic_index = y_bic.idxmin()
        min_bic_n = ms_df.loc[min_bic_index, "n_breakpoints"]

        results[f"trial_{i+1}"] = {
            "df": ms_df,
            "best_n_breakpoint_bic": best_n_bic,
            "min_bic_n_breakpoint": min_bic_n,
            "best_n_breakpoint_rss": best_n_rss,
        }

    return results


def extract_breakpoints(model: Fit) -> pd.DataFrame:
    """Extract breakpoint positions and confidence intervals from a Fit.

    Returns
    -------
    pd.DataFrame
        Columns: ``Breakpoint X Position``, ``Breakpoint Y Position``,
        ``Confidence Interval (X)``. Index starts at 1.
    """
    if not isinstance(model, Fit):
        raise TypeError("Model must be a piecewise_regression.main.Fit instance.")
    if not model.best_muggeo:
        raise ValueError("Model has not converged or has no valid breakpoints.")

    estimates = model.best_muggeo.best_fit.estimates

    bx, by, conf = [], [], []
    for i in range(1, model.best_muggeo.n_breakpoints + 1):
        key = f"breakpoint{i}"
        x_bp = estimates[key]["estimate"]
        ci = estimates[key]["confidence_interval"]
        y_bp = model.predict(np.array([x_bp]))[0]
        bx.append(x_bp)
        by.append(y_bp)
        conf.append(ci)

    df = pd.DataFrame({
        "Breakpoint X Position": bx,
        "Breakpoint Y Position": by,
        "Confidence Interval (X)": conf,
    })
    df.index = range(1, len(df) + 1)
    return df
