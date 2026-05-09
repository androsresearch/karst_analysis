"""Trial selection, model rebuild, and global fit metrics."""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
from piecewise_regression import Fit


def get_global_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> tuple:
    """Compute RSS, TSS, R², adjusted R².

    Parameters
    ----------
    y_true, y_pred : array-like
    p : int
        Number of parameters in the model (used for the adjusted R²).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rss = float(np.sum((y_true - y_pred) ** 2))
    tss = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - rss / tss if tss != 0 else np.nan

    n = len(y_true)
    if (n - p - 1) != 0:
        r2_adj = 1.0 - (1.0 - r2) * (n - 1) / (n - p - 1)
    else:
        r2_adj = float("nan")

    return rss, tss, r2, r2_adj


def select_best_trial(file_path: str, key: str = "best_n_breakpoint_bic"):
    """Select the most-frequent N across trials, breaking ties by lowest mean.

    Parameters
    ----------
    file_path : str
        Path to a JSON file produced by saving the output of
        :func:`best_n_breakpoints`.
    key : str
        Either ``"best_n_breakpoint_bic"`` or ``"best_n_breakpoint_rss"``.

    Returns
    -------
    (best_trial_name, trial_data, lowest_average_metric)
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    counts = Counter([trial[key] for trial in data.values()])
    most_common = counts.most_common(1)[0][0]

    filtered = {
        name: trial for name, trial in data.items() if trial[key] == most_common
    }

    metric_key = "bic" if key == "best_n_breakpoint_bic" else "rss"
    best_name = None
    lowest = float("inf")
    for name, trial in filtered.items():
        values = list(trial["df"][metric_key].values())
        avg = sum(values) / len(values)
        if avg < lowest:
            lowest = avg
            best_name = name

    return best_name, data[best_name], lowest


def get_breakpoint_data(data: dict, n_breakpoints: int):
    """Extract the BIC, n_breakpoints, and estimates entries for a given N.

    Used after loading a saved JSON to inspect a specific n_breakpoints.
    """
    key = str(n_breakpoints)
    if key in data["n_breakpoints"]:
        return {
            "bic": data["bic"][key],
            "n_breakpoints": data["n_breakpoints"][key],
            "estimates": data["estimates"][key],
        }
    return f"No information for n_breakpoints = {n_breakpoints}"


def rebuild_model(
    xx,
    yy,
    params_dict: dict,
    tolerance: float = 1e-5,
    min_distance: float = 0.01,
):
    """Reconstruct a `Fit` from saved breakpoint estimates.

    Parameters
    ----------
    xx, yy : array-like
        Original (x, y) data.
    params_dict : dict
        Must contain ``n_breakpoints`` and ``estimates``.
    """
    if "n_breakpoints" not in params_dict or "estimates" not in params_dict:
        raise ValueError("params_dict must contain 'n_breakpoints' and 'estimates'.")

    n_bp = params_dict["n_breakpoints"]
    estimates = params_dict["estimates"]
    breakpoints = [estimates[f"breakpoint{i+1}"]["estimate"] for i in range(n_bp)]

    return Fit(
        xx, yy,
        start_values=breakpoints,
        n_breakpoints=n_bp,
        n_boot=0,
        tolerance=tolerance,
        min_distance_between_breakpoints=min_distance,
    )
