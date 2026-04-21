"""
Salinity profile fitting following Huang et al. (2024, WRR, 10.1029/2024WR037244).

Fits a modified van Genuchten sigmoidal function to an electrical-conductivity (EC)
versus elevation profile measured in a coastal aquifer, using a regularized
non-linear least-squares optimization with the Trust-Region method.

Model (Eq. 2 in Huang et al., 2024):
        C(z*) = Cs - (Cs - Cf) * [1 + (alpha * z*)^n]^(-m)
with the scaled variables
        C  = EC / lambda_C          (scaled EC,       lambda_C = 100 mS/cm)
        z* = z  / lambda_z          (scaled elevation, lambda_z = -10 m)
and fitting parameters b = (alpha, n, m, Cf, Cs).

Regularized objective function (Eq. 6):
        O(b) = sum_i w_i * [C_i - C_hat_i(b)]^2
             + lam * sum_j v_j * (b_j - b_j*)^2
with preferred values b_j* only applied to (alpha, n, m).

Goodness-of-fit:  R^2 (Eq. 7) and RMSE (Eq. 8).

INPUT CSV columns (as provided by the user):
    - "Vertical Position m"          : depth below reference, positive, in metres
    - "Corrected sp Cond [µS/cm]"    : specific conductivity, IN mS/cm
      (header label mentions µS/cm but values are mS/cm per user specification)

The script assumes the CSV has already been cleaned and resampled.

Usage:
    python fit_salinity_profile.py path/to/profile.csv [--output plot.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Configuration: values from Huang et al. (2024), Section 3.1 (page 8)
# ---------------------------------------------------------------------------
LAMBDA_C: float = 100.0        # EC scaling factor [mS/cm]
LAMBDA_Z: float = -10.0        # Elevation scaling factor [m]

# Regularization weights v_j for (alpha, n, m)
V_ALPHA: float = 10.0
V_N:     float = 1.0e-5
V_M:     float = 0.1
LAM_REG: float = 1.0e-2        # Global regularization weight lambda

# Preferred values b_j* for (alpha, n, m)
ALPHA_STAR: float = 0.4
N_STAR:     float = 100.0
M_STAR:     float = 0.01

# Parameter bounds for fit_profile. Two presets are available:
#   "open"    -- original non-negativity bounds used in Huang's reference
#                implementation. The optimizer can drift toward the
#                "step-function" regime (m << 1e-2, n >> 100), which is
#                numerically finite but gives physically absurd mixing-zone
#                attributes (|z_mid| >> well depth, W ~ 1e30 m).
#   "bounded" -- upper bounds that keep the fit inside a physically
#                reasonable region for coastal freshwater-lens profiles.
#                m_low = 5e-3 is below Huang's preferred m* = 1e-2 but well
#                above the degenerate regime (~3e-4 observed in LRS69).
_FIT_BOUNDS_OPEN:    list[tuple[float, float | None]] = [(0.0, None)] * 5
_FIT_BOUNDS_BOUNDED: list[tuple[float, float]] = [
    (1.0e-3, 1.0e2),   # alpha
    (1.0e-1, 1.0e3),   # n     (Huang's n* = 100 sits in the middle)
    (5.0e-3, 5.0e1),   # m     (Huang's m* = 1e-2 inside; cuts off degenerate m << 1e-3)
    (0.0,    1.5),     # Cf    (scaled EC; seawater ~ 0.55)
    (0.0,    1.5),     # Cs
]

# CSV column names (as supplied by the user)
COL_POSITION: str = "Vertical Position m"
COL_EC:       str = "Corrected sp Cond [µS/cm]"   # unit auto-detected from header


# ---------------------------------------------------------------------------
# Model, objective function, and derived quantities
# ---------------------------------------------------------------------------
def sigmoid_C(z: np.ndarray, alpha: float, n: float, m: float,
              Cf: float, Cs: float,
              lambda_z: float = LAMBDA_Z) -> np.ndarray:
    """Scaled-salinity profile C(z*) from Eq. 2 of Huang et al. (2024).

    Parameters
    ----------
    z : elevation [m] (negative below mean sea level).
    alpha, n, m, Cf, Cs : fitting parameters.
    lambda_z : elevation scaling factor.

    Returns
    -------
    C : scaled salinity (dimensionless, = EC / lambda_C).
    """
    z_star = z / lambda_z
    # Guard against tiny negative bases that arise numerically when
    # elevation is above sea level (z_star < 0); the paper's data are
    # taken over the freshwater-to-seawater transition where z_star > 0.
    base = np.clip(alpha * z_star, 1e-12, None)
    return Cs - (Cs - Cf) * (1.0 + base ** n) ** (-m)


def sigmoid_EC(z: np.ndarray, alpha: float, n: float, m: float,
               Cf: float, Cs: float) -> np.ndarray:
    """Fitted EC [mS/cm] as a function of elevation."""
    return LAMBDA_C * sigmoid_C(z, alpha, n, m, Cf, Cs)


def objective(params: np.ndarray, z: np.ndarray, C_obs: np.ndarray,
              weights: np.ndarray) -> float:
    """Regularized OLS objective, Eq. 6 of Huang et al. (2024).

    The data-mismatch term is computed in scaled-salinity units (C = EC/lambda_C)
    to keep weights on the same order as in the paper.
    """
    alpha, n, m, Cf, Cs = params
    C_pred = sigmoid_C(z, alpha, n, m, Cf, Cs)

    # Data mismatch (Eq. 6, first sum)
    data_term = np.sum(weights * (C_obs - C_pred) ** 2)

    # Regularization (Eq. 6, second sum) -- only for alpha, n, m
    reg_term = LAM_REG * (
        V_ALPHA * (alpha - ALPHA_STAR) ** 2
        + V_N * (n - N_STAR) ** 2
        + V_M * (m - M_STAR) ** 2
    )
    return data_term + reg_term


def _log_expm1(u: float) -> float:
    # log(exp(u) - 1) for u > 0, stable even when exp(u) would overflow float64.
    # For u > 40 we use the identity log(exp(u) - 1) = u + log1p(-exp(-u)).
    if u > 40.0:
        return float(u + np.log1p(-np.exp(-u)))
    return float(np.log(np.expm1(u)))


def elevation_of_scaled_salinity(C_star: float, alpha: float, n: float,
                                 m: float, lambda_z: float = LAMBDA_Z) -> float:
    """Inverse of Eq. 2, used to find z(C*) -- Eq. 3 in Huang et al. (2024).

    C* = (C - Cs) / (Cf - Cs), so C* = 0.5 gives the mid-salinity elevation.
    """
    # Pathological parameters (optimizer may reach the non-negative bounds
    # closed end) or out-of-domain C_star produce no finite elevation.
    if alpha <= 0.0 or n <= 0.0 or m <= 0.0 or not (0.0 < C_star < 1.0):
        return float("nan")
    # z = lambda_z * (C_star^(-1/m) - 1)^(1/n) / alpha evaluated in log-space
    # to avoid double-exponential overflow when m is small.
    u = -np.log(C_star) / m                      # u > 0
    log_z_abs = _log_expm1(u) / n                # log(|z*|)
    if not np.isfinite(log_z_abs) or log_z_abs > 700.0:
        return float("nan")
    return float(lambda_z * np.exp(log_z_abs) / alpha)


def salinity_gradient_at(C_star: float, alpha: float, n: float, m: float,
                         Cf: float, Cs: float,
                         lambda_c: float = LAMBDA_C,
                         lambda_z: float = LAMBDA_Z) -> float:
    """Salinity gradient s = dEC/dz at a given scaled salinity -- Eq. 5."""
    if alpha <= 0.0 or n <= 0.0 or m <= 0.0 or not (0.0 < C_star < 1.0):
        return float("nan")
    u = -np.log(C_star) / m
    log_z_abs = _log_expm1(u) / n
    if not np.isfinite(log_z_abs) or log_z_abs > 700.0:
        return float("nan")
    z_star_abs = np.exp(log_z_abs)
    one_minus = -np.expm1(-u)                    # 1 - C_star^(1/m), stable
    numer = lambda_c * (Cs - Cf) * alpha * m * n * C_star * one_minus
    denom = lambda_z * z_star_abs
    if denom == 0.0:
        return float("nan")
    return float(numer / denom)


# ---------------------------------------------------------------------------
# Circle fitting for the fresh-brackish and brackish-salt transitions,
# following Huang (2024) Zenodo reference implementation exactly:
#   1) Sample the fitted sigmoid densely.
#   2) Compute arctan(dz/dEC) at each point (local tangent angle).
#   3) |Δangle| between consecutive samples is a curvature proxy.
#   4) Keep the points whose |Δangle| exceeds a high percentile (98 / 95)
#      separately for the salty (EC > mid_EC) and fresh (EC < mid_EC) halves.
#   5) Least-squares circle fit to those high-curvature points, in the
#      physical (EC [mS/cm], elevation [m]) space.
# ---------------------------------------------------------------------------
from scipy.optimize import least_squares  # noqa: E402  (paired with transition_radii)


def _fit_circle_least_squares(x_pts: np.ndarray, y_pts: np.ndarray,
                              x0_init: float, y0_init: float, r_init: float
                              ) -> tuple[float, float, float]:
    """Fit a circle (x0, y0, r) to (x_pts, y_pts) via scipy.least_squares.
    The residual is the radial distance from each point to the circle.
    """
    def residuals(params):
        x0, y0, r = params
        return np.sqrt((x_pts - x0) ** 2 + (y_pts - y0) ** 2) - r

    res = least_squares(residuals, x0=[x0_init, y0_init, r_init])
    x0, y0, r = res.x
    return float(x0), float(y0), float(abs(r))


def transition_radii(alpha: float, n: float, m: float, Cf: float, Cs: float,
                     z_range: tuple[float, float] = (-85.0, 0.0),
                     n_samples: int = 1000,
                     perc_s: float = 98.0,
                     perc_f: float = 95.0) -> dict:
    """Replicate the author's method (Huang, 2024; DOI: 10.5281/zenodo.10591418)
    for obtaining r_f and r_s from a fitted EC-vs-elevation profile.

    Parameters
    ----------
    alpha, n, m, Cf, Cs : fitted sigmoid parameters.
    z_range : (z_min, z_max) elevation window used to sample the curve.
    n_samples : density of the sampling of the fitted curve.
    perc_s, perc_f : percentile thresholds on |Δ arctan(dz/dEC)| used to
        select points at the saltwater knee and the freshwater knee.
        Defaults (98, 95) are those used by the authors for the LBD data.

    Returns
    -------
    dict with r_f, r_s, the circle centres and the high-curvature point sets
    used (handy for plotting, exactly like Figure 5 of the paper).
    """
    z_min, z_max = sorted(z_range)
    # Dense samples from the shallowest to the deepest elevation, like the
    # authors use (matric*-10 = np.linspace(0, -85, 1000)).
    elev = np.linspace(z_max, z_min, n_samples)
    EC = sigmoid_EC(elev, alpha, n, m, Cf, Cs)           # mS/cm
    mid_EC = 0.5 * (Cf + Cs) * LAMBDA_C                  # mS/cm

    # Derivatives dz/dEC and arctan (in degrees) between consecutive samples
    d_EL = np.diff(elev)
    d_EC = np.diff(EC)
    with np.errstate(divide="ignore", invalid="ignore"):
        deriv = np.where(d_EC != 0.0, d_EL / d_EC, -1e20)
    arctans = np.degrees(np.arctan(deriv))               # length n_samples - 1

    # |Δangle| between consecutive tangent orientations (curvature proxy)
    d_ang = np.abs(np.diff(arctans))                     # length n_samples - 2

    # Split this array into saltwater-side and freshwater-side, using the
    # EC value at the "second-neighbour" point i+2 in the original elev array,
    # matching the author's indexing convention.
    EC_ref = EC[2:]                                      # length n_samples - 2
    salty_mask = EC_ref > mid_EC
    fresh_mask = ~salty_mask

    if d_ang[salty_mask].size == 0 or d_ang[fresh_mask].size == 0:
        return {"r_f": float("nan"), "r_s": float("nan"),
                "center_f": (np.nan, np.nan), "center_s": (np.nan, np.nan),
                "salty_points": (np.array([]), np.array([])),
                "fresh_points": (np.array([]), np.array([]))}

    thr_s = np.percentile(d_ang[salty_mask], perc_s)
    thr_f = np.percentile(d_ang[fresh_mask], perc_f)

    # Points whose local bend is above the percentile threshold
    sel_salty = salty_mask & (d_ang > thr_s)
    sel_fresh = fresh_mask & (d_ang > thr_f)
    EL_salty, EC_salty = elev[2:][sel_salty], EC[2:][sel_salty]
    EL_fresh, EC_fresh = elev[2:][sel_fresh], EC[2:][sel_fresh]

    # Reasonable initial guesses for the circle fit, derived from the
    # centroid of the selected points (rather than hard-coded like in the
    # reference implementation, so this works on arbitrary data).
    def _init_from_pts(xpts, ypts):
        if xpts.size < 3:
            return None
        x0 = float(np.mean(xpts))
        y0 = float(np.mean(ypts))
        r0 = float(np.mean(np.sqrt((xpts - x0) ** 2 + (ypts - y0) ** 2))) or 1.0
        return x0, y0, r0

    def _safe_circle(xpts, ypts):
        init = _init_from_pts(xpts, ypts)
        if init is None:
            return float("nan"), (float("nan"), float("nan"))
        x0, y0, r = _fit_circle_least_squares(xpts, ypts, *init)
        return r, (x0, y0)

    r_s, center_s = _safe_circle(EC_salty, EL_salty)
    r_f, center_f = _safe_circle(EC_fresh, EL_fresh)

    return {
        "r_f": r_f, "r_s": r_s,
        "center_f": center_f, "center_s": center_s,
        "salty_points": (EC_salty, EL_salty),
        "fresh_points": (EC_fresh, EL_fresh),
    }


def isochlor_elevation(EC_target: float, alpha: float, n: float, m: float,
                       Cf: float, Cs: float) -> float:
    """Elevation at which the fitted profile reaches a given EC value, e.g.
    the 5 mS/cm isochlor used by Huang et al. (Section 2.2, z_5)."""
    C_target = EC_target / LAMBDA_C
    Cstar = (C_target - Cs) / (Cf - Cs)
    # Guard: requested EC is outside the fitted range
    if not (0.0 < Cstar < 1.0):
        return float("nan")
    return float(elevation_of_scaled_salinity(Cstar, alpha, n, m))


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
def load_profile(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the cleaned CSV and return (elevation [m], EC [mS/cm]).

    Elevation is computed as the negative of the 'Vertical Position m' column
    so that below-datum points have negative elevation (as in the paper).

    The EC column is expected to be named 'Corrected sp Cond [...]' and its
    unit is *auto-detected from the header*:
        - header contains 'µS/cm', 'uS/cm' or 'us/cm'  → divide by 1000
        - header contains 'mS/cm' (or nothing readable) → leave as-is
    The model and all attributes (including the 5 mS/cm isochlor) are
    defined in mS/cm, so EC is converted internally to that unit.
    """
    df = pd.read_csv(csv_path)

    missing = [c for c in (COL_POSITION, COL_EC) if c not in df.columns]
    if missing:
        raise KeyError(f"CSV is missing required column(s): {missing}. "
                       f"Found: {list(df.columns)}")

    z = -df[COL_POSITION].to_numpy(dtype=float)     # elevation [m]
    EC_raw = df[COL_EC].to_numpy(dtype=float)

    # Unit auto-detection from the column header
    header = COL_EC.lower().replace("μ", "µ")       # normalize Greek mu
    if "µs/cm" in header or "us/cm" in header:
        EC = EC_raw / 1000.0                        # µS/cm → mS/cm
        print(f"[load_profile] Header says µS/cm → converting to mS/cm "
              f"(EC_min={EC_raw.min():.1f} µS/cm = {EC.min():.3f} mS/cm, "
              f"EC_max={EC_raw.max():.1f} µS/cm = {EC.max():.2f} mS/cm)")
    else:
        EC = EC_raw
        print(f"[load_profile] Using EC as-is (assumed mS/cm).")

    mask = np.isfinite(z) & np.isfinite(EC)
    z, EC = z[mask], EC[mask]
    order = np.argsort(-z)                          # from shallow to deep
    return z[order], EC[order]


# ---------------------------------------------------------------------------
# Fitting routine
# ---------------------------------------------------------------------------
def fit_profile(z: np.ndarray, EC: np.ndarray,
                initial_guess: tuple[float, float, float, float, float] | None = None,
                bounds: str = "open",
                ) -> dict:
    """Fit Eq. 2 to (z, EC) and return optimal parameters and fit statistics.

    Parameters
    ----------
    z, EC : profile arrays.
    initial_guess : optional (alpha, n, m, Cf, Cs); defaults to Huang's
        preferred values with Cf0/Cs0 inferred from the data range.
    bounds : "open" (default) keeps Huang's reference-implementation
        non-negativity bounds -- preserves backward compatibility and lets
        the optimizer roam freely. "bounded" activates physically motivated
        upper bounds (see ``_FIT_BOUNDS_BOUNDED``) that prevent the
        optimizer from drifting into the step-function regime where the
        mixing-zone attributes become physically meaningless.
    """
    C_obs = EC / LAMBDA_C                            # scale observations
    weights = np.ones_like(C_obs)                    # w_i = 1 (per paper, page 8)

    # Reasonable initial guess from the data + paper's preferred values
    if initial_guess is None:
        Cf0 = float(np.min(EC) / LAMBDA_C)
        Cs0 = float(np.max(EC) / LAMBDA_C)
        initial_guess = (ALPHA_STAR, N_STAR, M_STAR, Cf0, Cs0)

    if bounds == "open":
        fit_bounds = _FIT_BOUNDS_OPEN
    elif bounds == "bounded":
        fit_bounds = _FIT_BOUNDS_BOUNDED
        # Nudge the initial guess strictly inside the bounds if it sits on
        # or below a lower edge (avoids trust-constr starting on a face).
        ig = list(initial_guess)
        for i, (lo, hi) in enumerate(fit_bounds):
            if ig[i] <= lo:
                ig[i] = lo + 0.1 * (hi - lo) if np.isfinite(hi) else lo * 1.1 + 1e-6
            if np.isfinite(hi) and ig[i] >= hi:
                ig[i] = hi - 0.1 * (hi - lo)
        initial_guess = tuple(ig)
    else:
        raise ValueError(f"bounds must be 'open' or 'bounded', got {bounds!r}")

    result = minimize(
        fun=objective,
        x0=np.asarray(initial_guess, dtype=float),
        args=(z, C_obs, weights),
        method="trust-constr",
        bounds=fit_bounds,
        options={"xtol": 1e-10, "gtol": 1e-10, "maxiter": 5000},
    )

    alpha, n, m, Cf, Cs = result.x
    EC_pred = sigmoid_EC(z, alpha, n, m, Cf, Cs)

    # Goodness of fit (Eqs. 7 and 8), computed on raw EC to be meaningful
    ss_res = np.sum((EC - EC_pred) ** 2)
    ss_tot = np.sum((EC - np.mean(EC)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / len(EC)))

    return {
        "alpha": float(alpha),
        "n": float(n),
        "m": float(m),
        "Cf": float(Cf),
        "Cs": float(Cs),
        "EC_min_fit": float(LAMBDA_C * Cf),          # fitted freshwater EC
        "EC_max_fit": float(LAMBDA_C * Cs),          # fitted seawater EC
        "R2": float(r2),
        "RMSE": rmse,
        "converged": bool(result.success),
        "message": result.message,
        "nfev": int(result.nfev),
    }


def mixing_zone_attributes(fit: dict,
                           z_range: tuple[float, float] | None = None) -> dict:
    """Compute the five key interface attributes from the fit (Section 2.2).

    Returns z_m, s_m, W (mixing-zone thickness), the two circle-fit radii
    r_f and r_s (with centres and high-curvature point sets for plotting),
    plus the 5 mS/cm isochlor elevation and the min/max EC.

    Parameters
    ----------
    fit : dict returned by ``fit_profile``.
    z_range : (z_min, z_max) elevation window passed to ``transition_radii``.
        If None, use (-85, 0) as in the reference implementation.
    """
    alpha, n, m, Cf, Cs = fit["alpha"], fit["n"], fit["m"], fit["Cf"], fit["Cs"]
    z_m   = elevation_of_scaled_salinity(0.5, alpha, n, m)
    z_05  = elevation_of_scaled_salinity(0.95, alpha, n, m)   # 5% salinity
    z_95  = elevation_of_scaled_salinity(0.05, alpha, n, m)   # 95% salinity
    s_m   = salinity_gradient_at(0.5, alpha, n, m, Cf, Cs)
    radii = transition_radii(alpha, n, m, Cf, Cs,
                             z_range=z_range if z_range is not None else (-85.0, 0.0))
    z_5mS = isochlor_elevation(5.0, alpha, n, m, Cf, Cs)
    W = abs(z_05 - z_95) if (np.isfinite(z_05) and np.isfinite(z_95)) else float("nan")
    return {
        "z_mid [m]": float(z_m),
        "s_mid [mS/cm / m]": float(s_m),
        "Mixing zone thickness W [m]": float(W),
        "r_f (fresh->brackish)": float(radii["r_f"]),
        "r_s (brackish->salt)": float(radii["r_s"]),
        "z_5mS/cm isochlor [m]": float(z_5mS),
        "Minimum EC [mS/cm]": fit["EC_min_fit"],
        "Maximum EC [mS/cm]": fit["EC_max_fit"],
        "_elev_5pct": float(z_05),
        "_elev_95pct": float(z_95),
        "_circle_f": {"center": radii["center_f"], "r": radii["r_f"],
                      "points": radii["fresh_points"]},
        "_circle_s": {"center": radii["center_s"], "r": radii["r_s"],
                      "points": radii["salty_points"]},
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_fit(z: np.ndarray, EC: np.ndarray, fit: dict,
             attrs: dict | None = None,
             raw_overlay: tuple[np.ndarray, np.ndarray] | None = None,
             data_label: str = "Observed (post-processed)",
             output_path: Path | None = None, show: bool = True) -> None:
    """Plot raw profile, fitted sigmoid, optimal parameters, R^2 and attributes.

    If ``attrs`` is supplied, also overlay the two fitted circles (r_f, r_s)
    at the salinity transitions, matching the visual style of Figure 5 in
    Huang et al. (2024).

    If ``raw_overlay`` is supplied as (z_raw, EC_raw), those points are
    drawn as a pale background layer behind the fitted data. This is
    useful when EC is a cleaned or LOWESS-smoothed version of the actual
    raw measurements, and you want to show both.
    """
    z_plot = np.linspace(z.min(), z.max(), 500)
    EC_plot = sigmoid_EC(z_plot, fit["alpha"], fit["n"], fit["m"],
                         fit["Cf"], fit["Cs"])

    fig, ax = plt.subplots(figsize=(7.0, 9.0))

    # Optional: truly raw data (e.g., original YSI file with 10⁴+ samples)
    if raw_overlay is not None:
        z_raw, EC_raw = raw_overlay
        ax.scatter(EC_raw, z_raw, s=4, color="#bdc3c7",
                   alpha=0.55, linewidth=0,
                   label=f"Raw ({len(z_raw)} pts)", zorder=1)

    ax.scatter(EC, z, s=4, facecolor="white", edgecolor="#1f4e79",
               linewidth=0.5, label=data_label, zorder=3)
    ax.plot(EC_plot, z_plot, color="#c0392b", lw=2.0,
            label="Fitted sigmoid (Eq. 2)", zorder=2)

    if attrs is not None:
        ec_min = fit["EC_min_fit"]
        ec_max = fit["EC_max_fit"]
        ec_mid = 0.5 * (ec_min + ec_max)
        z_m = attrs["z_mid [m]"]
        z_5pct = attrs["_elev_5pct"]
        z_95pct = attrs["_elev_95pct"]

        # Mid-salinity marker
        ax.axhline(z_m, color="#7f8c8d", lw=0.8, ls="--", alpha=0.7)
        ax.plot([ec_mid], [z_m], marker="o", ms=6, color="#2c3e50",
                zorder=4, label=f"Mid-salinity (z_m = {z_m:.2f} m)")

        # 5% and 95% salinity bounds (define W)
        ax.axhline(z_5pct,  color="#27ae60", lw=0.8, ls=":", alpha=0.8)
        ax.axhline(z_95pct, color="#27ae60", lw=0.8, ls=":", alpha=0.8,
                   label=f"5% / 95% salinity (W = "
                         f"{attrs['Mixing zone thickness W [m]']:.2f} m)")

        # 5 mS/cm isochlor if inside the fitted range
        z_5mS = attrs["z_5mS/cm isochlor [m]"]
        if np.isfinite(z_5mS) and z.min() <= z_5mS <= z.max():
            ax.axhline(z_5mS, color="#8e44ad", lw=0.8, ls="-.", alpha=0.7,
                       label=f"z_5 (5 mS/cm isochlor = {z_5mS:.2f} m)")

        # Fitted circles at the two transitions (Huang et al., 2024 Fig. 5)
        theta = np.linspace(0, 2 * np.pi, 200)
        for key, color, label in (("_circle_f", "#16a085",
                                    f"r_f = {attrs['r_f (fresh->brackish)']:.2f}"),
                                  ("_circle_s", "#16a085",
                                    f"r_s = {attrs['r_s (brackish->salt)']:.2f}")):
            c = attrs.get(key)
            if c is None or not np.isfinite(c["r"]):
                continue
            x0, y0 = c["center"]
            r = c["r"]
            ax.plot(x0 + r * np.cos(theta), y0 + r * np.sin(theta),
                    color=color, ls="--", lw=1.1, label=label, zorder=2)
            ax.plot([x0], [y0], marker="o", ms=4, color="#e67e22", zorder=4)

    ax.set_xlabel("Electrical conductivity  EC  [mS/cm]")
    ax.set_ylabel("Elevation  z  [m]")
    ax.set_title("Salinity profile fit\n(Huang et al., 2024 framework)")
    ax.grid(True, linestyle=":", alpha=0.5)

    # Clamp the visible area to the data range (fitted circles may be
    # larger than the profile and are allowed to clip off-frame).
    ec_pad = 0.05 * (EC.max() - EC.min() + 1e-9)
    z_pad  = 0.05 * (z.max() - z.min() + 1e-9)
    ax.set_xlim(max(0.0, EC.min() - ec_pad), EC.max() + ec_pad)
    ax.set_ylim(z.min() - z_pad, z.max() + z_pad)

    # Parameter / goodness-of-fit annotation box
    param_lines = [
        f"α  = {fit['alpha']:.4g}",
        f"n  = {fit['n']:.4g}",
        f"m  = {fit['m']:.4g}",
        f"C_f = {fit['Cf']:.4g}   (EC_min ≈ {fit['EC_min_fit']:.2f} mS/cm)",
        f"C_s = {fit['Cs']:.4g}   (EC_max ≈ {fit['EC_max_fit']:.2f} mS/cm)",
        "",
        f"R² = {fit['R2']:.4f}",
        f"RMSE = {fit['RMSE']:.3f} mS/cm",
    ]
    if attrs is not None:
        param_lines += [
            "",
            f"s_m = {attrs['s_mid [mS/cm / m]']:.3g} mS/cm/m",
            f"r_f = {attrs['r_f (fresh->brackish)']:.3g}   "
            f"r_s = {attrs['r_s (brackish->salt)']:.3g}",
        ]
    ax.text(0.03, 0.03, "\n".join(param_lines),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9, family="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                      edgecolor="#1f4e79", alpha=0.92))

    ax.legend(loc="upper right", frameon=True, fontsize=8.5)
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=200)
        print(f"Figure saved to: {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Optional pre-processing utilities
# ---------------------------------------------------------------------------
def lowess_smooth(z: np.ndarray, EC: np.ndarray,
                  frac: float = 0.08, degree: int = 1,
                  n_robust_iter: int = 2) -> np.ndarray:
    """Locally-weighted polynomial regression (LOWESS) smoother with
    iterative re-weighting (IRLS) for robustness to local outliers.

    Recommended for removing "staircase" artefacts caused by stationary
    measurement periods in YSI-style field profiling: each 3-min stop
    equilibrates the probe to a local value, producing flat plateaus
    that jump when the probe resumes descent. LOWESS smooths across the
    stairs while preserving the overall sigmoid shape because the IRLS
    bisquare re-weighting treats the repetitive plateau values as
    downweighted points with respect to the local trend.

    This implementation exploits the 1-D sorted structure: the k nearest
    neighbours of a point form a contiguous window that slides
    monotonically as the target advances, reducing the per-iteration
    cost from O(n² log n) to O(n · k).

    Parameters
    ----------
    z, EC : profile arrays (any order). The smoother operates in the
        depth-ordered space but preserves the input ordering.
    frac : fraction of points used in each local fit. For a probe
        stopping every 0.5–1 m in a 28 m profile with ~1000 samples,
        frac ≈ 0.05–0.10 spans two plateaus at once and erases them.
    degree : polynomial degree of the local fit (1 = locally linear,
        usually sufficient; 2 can be needed for strongly curved regions).
    n_robust_iter : number of IRLS robust re-weighting passes
        (2 is a good default; the first pass is ordinary weighted regression).

    Returns
    -------
    EC_smooth : array of the same length and ordering as EC.
    """
    order = np.argsort(z)
    z_s, EC_s = z[order], EC[order]
    n = len(z_s)
    k = max(int(frac * n), degree + 2)
    k = min(k, n)

    y_out = np.zeros(n, dtype=float)
    robust_w = np.ones(n, dtype=float)

    for it in range(n_robust_iter + 1):
        # Sliding-window two-pointer: the k nearest neighbours of z_s[i]
        # are a contiguous slice [l, l+k) that advances monotonically.
        l = 0
        for i in range(n):
            while (l + k < n) and ((z_s[l + k] - z_s[i]) < (z_s[i] - z_s[l])):
                l += 1
            r = l + k
            dx = z_s[l:r] - z_s[i]
            d  = np.abs(dx)
            h  = max(d[0], d[-1], 1e-12)       # farthest point in the window
            w  = (1.0 - (d / h) ** 3) ** 3
            w  = np.maximum(w, 0.0) * robust_w[l:r]

            # Weighted least squares solve — centered design matrix so that
            # the value at z_s[i] is simply the intercept beta[0].
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

    # Restore original ordering
    out = np.empty(n, dtype=float)
    out[order] = y_out
    return out


def enforce_monotonic_with_depth(z: np.ndarray, EC: np.ndarray
                                  ) -> np.ndarray:
    """Isotonic regression (Pool Adjacent Violators Algorithm, PAVA).

    Returns the sequence of EC values that:
      - is non-decreasing as depth increases (equivalently, non-decreasing
        as elevation z decreases — freshwater up, seawater down),
      - minimizes the sum of squared deviations from the input EC.

    Motivation: LOWESS can produce small non-monotonic oscillations
    ("S-curves") near stair-step transitions, because the local linear
    fit mean-reverts the ends of each plateau. Enforcing monotonicity
    a-posteriori is the least-assumption, L2-optimal way to remove
    those physically impossible reversals without touching the rest
    of the profile (where there are no violations, PAVA is the identity).

    Algorithm (O(n) amortised):
      Walk the sequence left-to-right (deep-to-shallow). Merge any
      adjacent pair of blocks whose means violate the monotone
      constraint, recursing backwards if the merge triggers a new
      violation. Each block stores its weighted-average value and size.

    Parameters
    ----------
    z, EC : elevation [m] and EC [mS/cm]. Any order.

    Returns
    -------
    EC_mono : array of the same length and ordering as EC, monotone
        non-decreasing as depth increases.
    """
    # Sort by depth (ascending depth = descending elevation)
    order = np.argsort(-z)                       # deepest-first? No: argsort(-z) gives elev descending, i.e., shallowest first
    # We want depth ascending = elevation descending = z going from high to low
    # argsort(-z) returns indices that make -z ascending, i.e., z descending. Good.
    EC_sorted = EC[order].astype(float).copy()
    n = len(EC_sorted)

    # Block storage: parallel lists of (mean, size)
    means = [float(v) for v in EC_sorted]
    sizes = [1] * n

    i = 0
    while i < len(means) - 1:
        if means[i] > means[i + 1]:               # violation
            new_size = sizes[i] + sizes[i + 1]
            new_mean = (means[i] * sizes[i]
                        + means[i + 1] * sizes[i + 1]) / new_size
            means[i]  = new_mean
            sizes[i]  = new_size
            del means[i + 1], sizes[i + 1]
            # Propagate backwards in case this merge violates with the prior block
            while i > 0 and means[i - 1] > means[i]:
                new_size = sizes[i - 1] + sizes[i]
                new_mean = (means[i - 1] * sizes[i - 1]
                            + means[i] * sizes[i]) / new_size
                means[i - 1] = new_mean
                sizes[i - 1] = new_size
                del means[i], sizes[i]
                i -= 1
        else:
            i += 1

    # Expand the blocks back to a full-length array (still in sorted order)
    out_sorted = np.empty(n, dtype=float)
    idx = 0
    for mean, size in zip(means, sizes):
        out_sorted[idx:idx + size] = mean
        idx += size

    # Unsort back to the original ordering
    out = np.empty(n, dtype=float)
    out[order] = out_sorted
    return out


def resample_uniform_z(z: np.ndarray, EC: np.ndarray,
                       step_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Linearly interpolate the profile onto an evenly-spaced elevation grid
    at the given step (metres). Input arrays may be in any order.

    When the same depth appears many times (typical of field profiling where
    the probe rests for several minutes at each depth), those repeats are
    first aggregated into a single mean-EC value before interpolation,
    so that linear interpolation between unique-depth nodes is well-defined.
    """
    order = np.argsort(z)                            # ascending z
    z_a, EC_a = z[order], EC[order]

    # Aggregate near-duplicate depths (resolution 1 mm) into their mean EC,
    # so that np.interp has a strictly increasing x axis.
    bin_key = np.round(z_a / 1e-3).astype(np.int64)
    unique_keys, inverse = np.unique(bin_key, return_inverse=True)
    z_u  = np.zeros(len(unique_keys))
    EC_u = np.zeros(len(unique_keys))
    cnt  = np.zeros(len(unique_keys))
    np.add.at(z_u,  inverse, z_a)
    np.add.at(EC_u, inverse, EC_a)
    np.add.at(cnt,  inverse, 1.0)
    z_u  /= cnt
    EC_u /= cnt

    z_new  = np.arange(z_u[0], z_u[-1], step_m)
    EC_new = np.interp(z_new, z_u, EC_u)
    order = np.argsort(-z_new)                       # back to shallow→deep
    return z_new[order], EC_new[order]


def resample_uniform_EC(z: np.ndarray, EC: np.ndarray,
                        step_mS: float) -> tuple[np.ndarray, np.ndarray]:
    """Resample the profile at uniform EC increments [mS/cm].

    Rationale: OLS on depth-uniform data gives very low weight to the
    transition because most points live on the freshwater / seawater
    plateaus. Spacing points evenly along the EC axis rebalances the
    fit toward the shape-defining region.
    """
    order = np.argsort(EC)
    EC_s, z_s = EC[order], z[order]
    keep = np.concatenate([[True], np.diff(EC_s) > 1e-9])
    EC_s, z_s = EC_s[keep], z_s[keep]
    EC_new = np.arange(EC_s[0], EC_s[-1], step_mS)
    z_new = np.interp(EC_new, EC_s, z_s)
    order = np.argsort(-z_new)
    return z_new[order], EC_new[order]


def trim_shallow(z: np.ndarray, EC: np.ndarray,
                 min_elev_m: float = -1.0) -> tuple[np.ndarray, np.ndarray]:
    """Drop points above ``min_elev_m`` to remove surface effects
    (evaporation, tidal flushing) as in Huang et al. (2024) Section 2.4."""
    mask = z < min_elev_m
    return z[mask], EC[mask]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    global LAM_REG, ALPHA_STAR, N_STAR, M_STAR

    parser = argparse.ArgumentParser(
        description="Fit a modified van Genuchten sigmoid to a salinity profile "
                    "following Huang et al. (2024).")
    parser.add_argument("csv", type=Path, help="Path to the cleaned profile CSV.")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Optional path to save the figure (e.g. plot.png).")
    parser.add_argument("--no-show", action="store_true",
                        help="Do not open an interactive plot window.")

    parser.add_argument("--raw-csv", type=Path, default=None, metavar="RAW_CSV",
                        help="Optional path to an additional CSV containing the "
                             "truly raw (unprocessed) profile — overlaid as a "
                             "pale background on the final figure, so you can "
                             "see how the processed / LOWESS data compares to "
                             "the original measurements.")

    # Optional pre-processing
    parser.add_argument("--trim-above", type=float, default=None, metavar="ELEV",
                        help="Drop points shallower than ELEV (negative, e.g. -1.0) "
                             "to remove surface effects.")
    parser.add_argument("--preresample-z", type=float, default=None, metavar="STEP_M",
                        help="BEFORE smoothing, re-sample the profile onto a "
                             "uniform elevation grid of this step [m]. Near-"
                             "duplicate depths (from stationary probe periods) "
                             "are first averaged, then linear interpolation "
                             "fills the gaps between stops. Use this when raw "
                             "data has highly uneven depth sampling (dense "
                             "plateaus + sparse jumps). Example: 0.01 for 1 cm.")
    parser.add_argument("--smooth-lowess", type=float, default=None, metavar="FRAC",
                        help="Apply LOWESS smoothing with the given fraction "
                             "(recommended 0.05-0.10 to remove stationary-probe "
                             "staircase artefacts). Example: --smooth-lowess 0.08")
    parser.add_argument("--monotonic", action="store_true",
                        help="Enforce monotonicity after smoothing: EC must be "
                             "non-decreasing with depth (equivalently, non-"
                             "decreasing as elevation decreases). Removes any "
                             "physically impossible reversals LOWESS may have "
                             "introduced at step edges. Recommended together "
                             "with --smooth-lowess.")
    parser.add_argument("--resample-z", type=float, default=None, metavar="STEP_M",
                        help="Resample at uniform elevation step [m]. "
                             "Example: --resample-z 0.10 for 10 cm.")
    parser.add_argument("--resample-ec", type=float, default=None, metavar="STEP_MS",
                        help="Resample at uniform EC step [mS/cm] — "
                             "recommended for sharp transitions. "
                             "Example: --resample-ec 0.2 for 200 µS/cm.")
    parser.add_argument("--bounds", choices=("open", "bounded"), default="open",
                        help="Optimizer bounds. 'open' (default) keeps the "
                             "non-negativity-only bounds of Huang's reference "
                             "implementation. 'bounded' activates physically "
                             "motivated upper limits that prevent the fit from "
                             "drifting into the step-function regime (where the "
                             "mixing-zone attributes become meaningless).")

    # Optional overrides for the Huang regularization defaults
    parser.add_argument("--lam-reg", type=float, default=None,
                        help=f"Override regularization weight lambda "
                             f"(default {LAM_REG}; use 0 to disable).")
    parser.add_argument("--alpha-star", type=float, default=None,
                        help=f"Override preferred alpha (default {ALPHA_STAR}).")
    parser.add_argument("--n-star", type=float, default=None,
                        help=f"Override preferred n (default {N_STAR}).")
    parser.add_argument("--m-star", type=float, default=None,
                        help=f"Override preferred m (default {M_STAR}).")

    args = parser.parse_args()

    # Apply any override to the module-level constants used by the objective
    if args.lam_reg    is not None: LAM_REG    = args.lam_reg
    if args.alpha_star is not None: ALPHA_STAR = args.alpha_star
    if args.n_star     is not None: N_STAR     = args.n_star
    if args.m_star     is not None: M_STAR     = args.m_star

    z, EC = load_profile(args.csv)
    print(f"Loaded {len(z)} points from {args.csv}")
    print(f"Elevation range: {z.min():.2f} m to {z.max():.2f} m")
    print(f"EC range:        {EC.min():.2f} mS/cm to {EC.max():.2f} mS/cm\n")

    # Optional pre-processing, in order: trim → smooth → resample
    if args.trim_above is not None:
        n0 = len(z)
        z, EC = trim_shallow(z, EC, min_elev_m=args.trim_above)
        print(f"[trim] kept {len(z)} / {n0} points below z = {args.trim_above} m")

    if args.preresample_z is not None:
        n0 = len(z)
        z, EC = resample_uniform_z(z, EC, step_m=args.preresample_z)
        print(f"[pre-resample] {n0} → {len(z)} points, "
              f"uniform {args.preresample_z} m grid (gaps filled by interpolation)")

    if args.smooth_lowess is not None:
        print(f"[smooth] applying LOWESS with frac={args.smooth_lowess}")
        EC = lowess_smooth(z, EC, frac=args.smooth_lowess)

    if args.monotonic:
        EC_before = EC.copy()
        EC = enforce_monotonic_with_depth(z, EC)
        n_changed = int(np.sum(np.abs(EC - EC_before) > 1e-9))
        print(f"[monotonic] PAVA adjusted {n_changed} / {len(EC)} points "
              f"to enforce EC non-decreasing with depth")

    if args.resample_ec is not None:
        n0 = len(z)
        z, EC = resample_uniform_EC(z, EC, step_mS=args.resample_ec)
        print(f"[resample EC] {n0} → {len(z)} points "
              f"(step = {args.resample_ec} mS/cm)")
    elif args.resample_z is not None:
        n0 = len(z)
        z, EC = resample_uniform_z(z, EC, step_m=args.resample_z)
        print(f"[resample z] {n0} → {len(z)} points "
              f"(step = {args.resample_z} m)")

    # Print active regularization settings for reproducibility
    print(f"\nRegularization: λ={LAM_REG}, preferred α*={ALPHA_STAR}, "
          f"n*={N_STAR}, m*={M_STAR}")
    print(f"Bounds mode: {args.bounds}\n")

    fit = fit_profile(z, EC, bounds=args.bounds)

    print("Optimization result")
    print("-------------------")
    print(f"  converged : {fit['converged']}  ({fit['message']})")
    print(f"  function evaluations : {fit['nfev']}")
    print()
    print("Optimal parameters")
    print("------------------")
    for k in ("alpha", "n", "m", "Cf", "Cs"):
        print(f"  {k:<5s} = {fit[k]:.6g}")
    print(f"  R^2   = {fit['R2']:.6f}")
    print(f"  RMSE  = {fit['RMSE']:.6f}  mS/cm")
    print()

    attrs = mixing_zone_attributes(fit, z_range=(float(z.min()), float(z.max())))
    print("Derived interface attributes")
    print("----------------------------")
    for k, v in attrs.items():
        if k.startswith("_"):            # skip private helper keys
            continue
        print(f"  {k:<30s} = {v:.4g}")

    # If the user supplied a raw CSV for overlay, load it (without any
    # processing) and pass it to plot_fit as a background layer.
    raw_overlay = None
    if args.raw_csv is not None:
        z_raw_ovl, EC_raw_ovl = load_profile(args.raw_csv)
        raw_overlay = (z_raw_ovl, EC_raw_ovl)
        print(f"[raw overlay] loaded {len(z_raw_ovl)} points from {args.raw_csv}")

    # Decide a clear label depending on what processing was actually done
    if args.smooth_lowess is not None:
        suffix = " + monotonic" if args.monotonic else ""
        data_label = f"LOWESS-smoothed (frac={args.smooth_lowess}){suffix}"
    elif args.monotonic:
        data_label = "Observed (monotonic)"
    elif args.resample_ec is not None or args.resample_z is not None:
        data_label = "Observed (resampled)"
    elif args.trim_above is not None:
        data_label = "Observed (trimmed)"
    else:
        data_label = "Observed EC"

    plot_fit(z, EC, fit, attrs=attrs,
             raw_overlay=raw_overlay,
             data_label=data_label,
             output_path=args.output, show=not args.no_show)


if __name__ == "__main__":
    main()
