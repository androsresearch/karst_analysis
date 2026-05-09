"""Instrumental-noise estimation for caliper logs.

Separates the *transducer noise* (caliper hardware) from the
*drilling-induced roughness* (sonic coring vs rotary auger), by
comparing residuals from a smooth detrending applied to two wells:

    * AW5O — smooth sonic-cored borehole (transducer noise dominates)
    * AW5D — rotary-auger borehole       (transducer + drilling roughness)

Variance decomposition under the assumption of independent contributions:

    sigma_AW5D^2  =  sigma_inst^2  +  sigma_drilling^2

The migration of the original ``estimate_noise_aw5o_vs_aw5d.py`` separates:

    * The reusable, well-tested numerical routines (this module).
    * The CLI orchestrator that loads data, calls the routines, and
      writes the JSON report (``scripts/caliper_estimate_noise.py``).

Migration history
-----------------
v5: extracted from ``estimate_noise_aw5o_vs_aw5d.py`` with no algorithmic
changes. The byte-for-byte equivalence of the migrated pipeline against
the original is enforced by ``tests/test_caliper_noise.py``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


# ──────────────────────────────────────────────────────────────────────
#  Numerical primitives
# ──────────────────────────────────────────────────────────────────────
def moving_average_centered(y: np.ndarray, win_pts: int) -> np.ndarray:
    """Centred moving-average with reflective padding.

    Identical to the original implementation. Returns an array of the
    same length as ``y``. If ``win_pts`` is even it is bumped to the
    next odd integer so the window is exactly centred.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if win_pts < 2 or n < 2:
        return y.copy()
    if win_pts % 2 == 0:
        win_pts += 1
    half = win_pts // 2
    pad_left = y[half:0:-1] if half > 0 else y[:0]
    pad_right = y[-2:-half - 2:-1] if half > 0 else y[:0]
    padded = np.concatenate([pad_left, y, pad_right])
    kernel = np.ones(win_pts) / win_pts
    return np.convolve(padded, kernel, mode="valid")[:n]


def lag1_autocorrelation(residuals: np.ndarray) -> float:
    """Sample lag-1 autocorrelation of a 1-D residual series."""
    r = residuals - residuals.mean()
    return float(np.sum(r[:-1] * r[1:]) / np.sum(r ** 2))


# ──────────────────────────────────────────────────────────────────────
#  Per-interval noise estimate
# ──────────────────────────────────────────────────────────────────────
def measure_noise_in_interval(
    z: np.ndarray,
    cal: np.ndarray,
    z_lo: float,
    z_hi: float,
    detrend_window_m: float,
) -> dict:
    """Estimate noise on a single well over a fixed depth interval.

    The procedure detrends the caliper signal with a centred moving
    average over ``detrend_window_m`` and reports the standard deviation
    and robust MAD-based scale of the residuals.

    Parameters
    ----------
    z, cal : np.ndarray
        Depth (negative-elevation convention) and caliper (cm) arrays
        for one well, sorted by depth.
    z_lo, z_hi : float
        Inclusive bounds of the interval to use (in metres).
    detrend_window_m : float
        Window length of the moving average, in metres.

    Returns
    -------
    dict
        Same fields as in the original ``noise_comparison.json`` schema:
        ``well_interval``, ``n``, ``median_dz_m``, ``detrend_window_m``,
        ``detrend_window_pts``, ``cal_mean_cm``, ``cal_std_cm``,
        ``sigma_std_cm``, ``sigma_MAD_cm``, ``residual_mean_cm``,
        ``lag1_autocorr``.
    """
    mask = (z >= z_lo) & (z <= z_hi)
    z_sel = z[mask]
    cal_sel = cal[mask]

    if len(z_sel) < 3:
        raise ValueError(
            f"Interval [{z_lo}, {z_hi}] has only {len(z_sel)} samples; "
            f"need at least 3."
        )

    dz = float(np.median(np.diff(z_sel)))
    win_pts = max(3, int(round(detrend_window_m / dz)))
    if win_pts % 2 == 0:
        win_pts += 1

    trend = moving_average_centered(cal_sel, win_pts)
    resid = cal_sel - trend
    mad = float(np.median(np.abs(resid - np.median(resid))))

    return dict(
        well_interval=[z_lo, z_hi],
        n=int(len(z_sel)),
        median_dz_m=dz,
        detrend_window_m=detrend_window_m,
        detrend_window_pts=int(win_pts),
        cal_mean_cm=float(np.mean(cal_sel)),
        cal_std_cm=float(np.std(cal_sel, ddof=1)),
        sigma_std_cm=float(np.std(resid, ddof=1)),
        sigma_MAD_cm=1.4826 * mad,
        residual_mean_cm=float(np.mean(resid)),
        lag1_autocorr=lag1_autocorrelation(resid),
    )


# ──────────────────────────────────────────────────────────────────────
#  Two-well drilling-method comparison
# ──────────────────────────────────────────────────────────────────────
def compare_drilling_methods(
    aw5o_result: dict, aw5d_result: dict,
) -> dict:
    """Variance decomposition of the AW5D residual into transducer + drilling.

    Under the assumption of independent Gaussian contributions:

        sigma_drilling = sqrt(max(0, sigma_AW5D^2 - sigma_AW5O^2))

    Parameters
    ----------
    aw5o_result, aw5d_result : dict
        Outputs of :func:`measure_noise_in_interval` for AW5O and AW5D.

    Returns
    -------
    dict
        Schema matches the ``"comparison"`` block of the original JSON:
        ``sigma_drilling_from_std_cm``, ``sigma_drilling_from_MAD_cm``,
        ``interpretation``.
    """
    o_std = aw5o_result["sigma_std_cm"]
    d_std = aw5d_result["sigma_std_cm"]
    o_mad = aw5o_result["sigma_MAD_cm"]
    d_mad = aw5d_result["sigma_MAD_cm"]

    sd_drill_std = float(np.sqrt(max(0.0, d_std ** 2 - o_std ** 2)))
    sd_drill_mad = float(np.sqrt(max(0.0, d_mad ** 2 - o_mad ** 2)))

    return dict(
        sigma_drilling_from_std_cm=sd_drill_std,
        sigma_drilling_from_MAD_cm=sd_drill_mad,
        interpretation=(
            "AW5O sigma is dominated by caliper transducer noise (smooth "
            "sonic-cored hole). AW5D sigma is the quadrature sum of "
            "transducer noise and rotary-auger drilling roughness."
        ),
    )


# ──────────────────────────────────────────────────────────────────────
#  High-level estimator (used by both the CLI and the tests)
# ──────────────────────────────────────────────────────────────────────
def estimate_noise_aw5o_vs_aw5d(
    df_master,
    aw5o_source: str = "AW5O_caliper_20210910.LAS",
    aw5d_source: str = "AW5D_caliper_20210910.LAS",
    aw5o_interval: Optional[tuple[float, float]] = None,
    aw5d_interval: Optional[tuple[float, float]] = None,
    detrend_window_m: Optional[float] = None,
) -> dict:
    """Run the full noise-comparison pipeline on a master caliper DataFrame.

    Builds the full report dict that matches the ``noise_comparison.json``
    schema. The returned dict can be ``json.dump``-ed directly.

    Parameters
    ----------
    df_master : pd.DataFrame
        Master caliper table with columns ``source_file``, ``Depth [m]``,
        ``calibrated_cm``, ``Diameter_auger_in``.
    aw5o_source, aw5d_source : str
        ``source_file`` values to filter for AW5O and AW5D respectively.
    aw5o_interval, aw5d_interval : tuple of (z_lo, z_hi), optional
        Override the depth intervals from
        :data:`karst_analysis.caliper.config.NOISE_INTERVAL_AW5O` and
        ``..._AW5D``.
    detrend_window_m : float, optional
        Override the moving-average window from
        :data:`karst_analysis.caliper.config.DETREND_WINDOW_M`.
    """
    from karst_analysis.caliper.config import (
        DETREND_WINDOW_M, NOISE_INTERVAL_AW5O, NOISE_INTERVAL_AW5D,
    )

    if aw5o_interval is None:
        aw5o_interval = NOISE_INTERVAL_AW5O
    if aw5d_interval is None:
        aw5d_interval = NOISE_INTERVAL_AW5D
    if detrend_window_m is None:
        detrend_window_m = DETREND_WINDOW_M

    results: dict = {}
    for well, source, (z_lo, z_hi), drilling in [
        ("AW5O", aw5o_source, aw5o_interval, "sonic_coring"),
        ("AW5D", aw5d_source, aw5d_interval, "rotary_auger"),
    ]:
        sub = df_master[df_master["source_file"] == source].sort_values("depth_m")
        z = sub["depth_m"].to_numpy()
        cal = sub["calibrated_cm"].to_numpy()
        auger_in = float(sub["Diameter_auger_in"].iloc[0])
        auger_cm = auger_in * 2.54

        res = measure_noise_in_interval(z, cal, z_lo, z_hi, detrend_window_m)
        res["well"] = well
        res["auger_in"] = auger_in
        res["auger_cm"] = auger_cm
        res["drilling_method"] = drilling
        results[well] = res

    results["comparison"] = compare_drilling_methods(results["AW5O"], results["AW5D"])
    return results
