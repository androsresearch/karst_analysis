"""Segment extraction and per-segment metrics for fitted piecewise models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def extract_segments(fit_object) -> dict:
    """Extract segment data and per-segment fitted line from a Fit.

    Returns
    -------
    dict
        ``{"segments": [{"segment": int, "data_x": np.ndarray,
        "data_y": np.ndarray, "fitted_model": {"slope": float,
        "intercept": float, "fitted_y": np.ndarray}}, ...]}``
    """
    if not hasattr(fit_object, "best_muggeo") or not fit_object.best_muggeo:
        raise ValueError("Fit object did not converge or has no valid model.")

    best_fit = fit_object.best_muggeo.best_fit
    breakpoints = best_fit.next_breakpoints
    xx = np.array(fit_object.xx)
    yy = np.array(fit_object.yy)

    edges = [min(xx)] + list(breakpoints) + [max(xx)]
    intercept = best_fit.raw_params[0]
    alpha = best_fit.raw_params[1]
    beta_hats = best_fit.raw_params[2:2 + len(breakpoints)]

    segments = []
    for i in range(len(edges) - 1):
        seg_start, seg_end = edges[i], edges[i + 1]
        mask = (xx >= seg_start) & (xx <= seg_end)
        seg_x, seg_y = xx[mask], yy[mask]

        y_fit = intercept + alpha * seg_x
        for j, bp in enumerate(breakpoints):
            y_fit += beta_hats[j] * np.maximum(0, seg_x - bp)

        segments.append({
            "segment": i + 1,
            "data_x": seg_x,
            "data_y": seg_y,
            "fitted_model": {
                "slope": alpha,
                "intercept": intercept,
                "fitted_y": y_fit,
            },
        })

    return {"segments": segments}


def segment_data(x: np.ndarray, y: np.ndarray, df: dict, num_breakpoints: int) -> dict:
    """Split (x, y) into segments using breakpoints stored in a JSON-like dict.

    Used for re-loading saved BIC results and re-segmenting without re-fitting.
    """
    key = str(num_breakpoints)
    if key not in df["estimates"]:
        raise ValueError(f"n_breakpoints={num_breakpoints} not in 'estimates'.")

    breakpoints = []
    for i in range(1, num_breakpoints + 1):
        bk = f"breakpoint{i}"
        if bk in df["estimates"][key]:
            breakpoints.append(df["estimates"][key][bk]["estimate"])
        else:
            raise ValueError(f"breakpoint{i} missing for n={num_breakpoints}.")
    breakpoints = sorted(breakpoints)

    segments = {}
    start_idx = 0
    for i, bp in enumerate(breakpoints):
        end_idx = np.searchsorted(x, bp, side="right")
        segments[str(i + 1)] = [x[start_idx:end_idx], y[start_idx:end_idx]]
        start_idx = end_idx
    segments[str(len(breakpoints) + 1)] = [x[start_idx:], y[start_idx:]]
    return segments


def fit_linear_models(segments: dict) -> dict:
    """Fit a linear model to each segment and compute basic metrics."""
    results = {}

    for key, (x_seg, y_seg) in segments.items():
        if len(x_seg) == 0 or len(y_seg) == 0:
            results[key] = {"error": "empty or insufficient segment"}
            continue

        x_seg = np.array(x_seg).reshape(-1, 1)
        y_seg = np.array(y_seg)

        model = LinearRegression()
        model.fit(x_seg, y_seg)
        y_pred = model.predict(x_seg)

        rms = np.sqrt(mean_squared_error(y_seg, y_pred))
        rng = float(np.max(y_seg) - np.min(y_seg))
        rms_pct_minmax = (rms / rng * 100.0) if rng != 0 else 0.0

        if np.any(y_seg == 0):
            rms_pct_meas = "Indefinido (y_segment contains zeros)"
        else:
            rms_pct_meas = float(np.sqrt(np.mean(((y_pred - y_seg) / y_seg) ** 2)) * 100.0)

        r2 = r2_score(y_seg, y_pred)

        results[key] = {
            "model": model,
            "RMS": rms,
            "RMS%_min_max": rms_pct_minmax,
            "RMS%": rms_pct_meas,
            "R^2": r2,
        }

    return results


def calculate_metrics_per_segment(fit_model) -> list[dict]:
    """Compute R² and RMS% for each segment of a fitted Fit model."""
    if not fit_model.best_muggeo:
        raise ValueError("Model is not properly fitted.")

    metrics = []
    xx = np.array(fit_model.xx)
    yy = np.array(fit_model.yy)
    edges = (
        [min(xx)] + list(fit_model.best_muggeo.best_fit.next_breakpoints) + [max(xx)]
    )

    for i in range(len(edges) - 1):
        mask = (xx >= edges[i]) & (xx < edges[i + 1])
        x_seg, y_seg = xx[mask], yy[mask]
        if len(x_seg) == 0:
            continue
        y_pred = fit_model.predict(x_seg)

        rss = np.sum((y_seg - y_pred) ** 2)
        tss = np.sum((y_seg - np.mean(y_seg)) ** 2)
        r2 = 1 - rss / tss if tss != 0 else np.nan

        rms = np.sqrt(mean_squared_error(y_seg, y_pred))
        rng = float(np.max(y_seg) - np.min(y_seg))
        rms_pct_minmax = (rms / rng * 100.0) if rng != 0 else 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            rms_pct = np.sqrt(np.mean(((y_seg - y_pred) / y_seg) ** 2)) * 100.0

        metrics.append({
            "Segment": i + 1,
            "R^2": r2,
            "RMS%": rms_pct,
            "RMS% (min-max)": rms_pct_minmax,
        })

    return metrics
