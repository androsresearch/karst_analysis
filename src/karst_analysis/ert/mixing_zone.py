"""Mixing-zone identification for ERT 1D resistivity profiles.

This module is the ERT analogue of
``karst_analysis.sec.slopes._mark_mixing_zone``. The mathematical
backbone — turning angle on a [0, 1]-normalised piecewise trajectory —
is identical. Two things differ:

1. **Y axis.** SEC uses log10(SEC); ERT uses log10(resistivity). Both
   work in log space because resistivity (like conductivity) varies
   over orders of magnitude across a coastal-aquifer profile.

2. **Threshold direction.** Saltwater means HIGH SEC and LOW
   resistivity. So the SEC eligibility rule ``sec_bp >= 40_000`` flips
   to ``rho_bp <= bot_mz_rho_threshold`` for ERT. Everything else
   (curvature ranking, asymmetric TOP-vs-BOT logic, "honest gap" when
   no BP qualifies) is preserved.

Why a separate module from SEC
-------------------------------
Mariana asked that each technique own its mixing-zone code so that
SEC's logic is not perturbed when ERT-specific decisions evolve. The
two functions stay near-identical for now; if they diverge later, no
shared base needs to be maintained.

Threshold is a required parameter
---------------------------------
``select_ert_mixing_zone`` does NOT carry a default for
``bot_mz_rho_threshold`` — it is a required keyword argument. The
threshold is a scientific decision, not a software default. The batch
script reads it from ``config/pipeline.yml`` (key
``ert.bot_mz_rho_threshold``); tests and ad-hoc callers pass the
number explicitly.

A working placeholder of 25.0 Ω·m is recorded in the config with a
``PROVISIONAL`` comment, awaiting Mariana's final scientific call.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════
#  Pure-array entry point
# ════════════════════════════════════════════════════════════════════
def select_ert_mixing_zone(
    z_bp: np.ndarray,
    rho_bp: np.ndarray,
    *,
    bot_mz_rho_threshold: float,
) -> tuple[int | None, int | None]:
    """Identify TOP and BOT MZ from a sequence of breakpoints.

    Parameters
    ----------
    z_bp : 1-D ndarray, length n_bp
        Breakpoint depths (BGL convention, positive down). Must include
        the two endpoints — TOP and BOT are chosen only from interior
        breakpoints.
    rho_bp : 1-D ndarray, length n_bp
        Resistivity at each breakpoint, in Ω·m (linear, NOT log10).
    bot_mz_rho_threshold : float
        BPs with ``rho_bp[i] <= threshold`` are eligible for BOT MZ.

    Returns
    -------
    (top_idx, bot_idx) : tuple of int or None
        Indices into ``z_bp`` / ``rho_bp``. Either may be ``None``:
          - both ``None``: too few BPs (<3) or degenerate ranges.
          - ``top_idx`` set, ``bot_idx`` None: no interior BP meets
            the threshold (honest gap, same convention as SEC when no
            BP reaches 40 000 µS/cm).

    Method
    ------
    1. Map (z, log10(rho)) to (zn, yn) where each axis is rescaled
       to [0, 1] over the breakpoint range. This makes the curvature
       metric robust to aspect-ratio differences across wells.
    2. Compute the turning angle at each interior BP:
       ``theta_i = arccos((u_i · v_i) / (|u_i| |v_i|))``
       with u_i = (zn[i]-zn[i-1], yn[i]-yn[i-1]) and v_i analogous
       for the outgoing chord.
    3. TOP MZ = argmax(theta) over interior BPs. Purely geometric.
    4. BOT MZ = argmax(theta) over interior BPs that satisfy
       ``rho_bp[i] <= threshold`` AND i != top_idx. If no such BP
       exists, BOT is left unmarked.

    Edge cases
    ----------
    - ``n_bp < 3``: no interior BPs; returns ``(None, None)``.
    - ``z_bp.max() == z_bp.min()`` or ``log10(rho_bp).max() ==
      log10(rho_bp).min()``: degenerate range; returns ``(None, None)``.
    - All ``rho_bp`` are above threshold: returns ``(top_idx, None)``.

    Tie warning
    -----------
    Emits ``UserWarning`` if the 1st and 2nd ranked turning angles
    among the BOT-eligible BPs are exactly equal (ambiguous BOT
    selection). This mirrors SEC behaviour.
    """
    z_bp = np.asarray(z_bp, dtype=float)
    rho_bp = np.asarray(rho_bp, dtype=float)
    if z_bp.shape != rho_bp.shape:
        raise ValueError(
            f"z_bp and rho_bp must have the same shape, got "
            f"{z_bp.shape} and {rho_bp.shape}"
        )
    n_bp = len(z_bp)
    if n_bp < 3:
        return None, None

    if (rho_bp <= 0).any():
        raise ValueError(
            "rho_bp must be strictly positive (log10 is taken)."
        )
    log_rho_bp = np.log10(rho_bp)

    # Normalise both axes to [0, 1] over the breakpoint range.
    z_range = z_bp.max() - z_bp.min()
    y_range = log_rho_bp.max() - log_rho_bp.min()
    if z_range == 0 or y_range == 0:
        return None, None
    zn = (z_bp - z_bp.min()) / z_range
    yn = (log_rho_bp - log_rho_bp.min()) / y_range

    # Turning angle at each interior breakpoint.
    turning = np.zeros(n_bp)
    for i in range(1, n_bp - 1):
        u = np.array([zn[i] - zn[i - 1], yn[i] - yn[i - 1]])
        v = np.array([zn[i + 1] - zn[i], yn[i + 1] - yn[i]])
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu == 0 or nv == 0:
            continue
        cos_theta = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        turning[i] = np.arccos(cos_theta)

    interior_idx = np.arange(1, n_bp - 1)
    interior_turning = turning[interior_idx]

    # TOP MZ: largest turning angle, no constraint.
    order = np.argsort(-interior_turning, kind="stable")
    top_idx = int(interior_idx[order[0]])

    # BOT MZ: largest turning angle among interior BPs whose
    # resistivity <= threshold, excluding the BP already picked as TOP.
    bot_eligible_mask = (
        (rho_bp[interior_idx] <= bot_mz_rho_threshold)
        & (interior_idx != top_idx)
    )
    if not bot_eligible_mask.any():
        return top_idx, None

    elig_pos = np.where(bot_eligible_mask)[0]
    elig_turning = interior_turning[elig_pos]
    elig_order = np.argsort(-elig_turning, kind="stable")
    bot_idx = int(interior_idx[elig_pos[elig_order[0]]])

    if len(elig_turning) >= 2:
        m1 = elig_turning[elig_order[0]]
        m2 = elig_turning[elig_order[1]]
        if np.isclose(m1, m2, rtol=0.0, atol=0.0):
            i_first = int(interior_idx[elig_pos[elig_order[0]]] + 1)
            i_second = int(interior_idx[elig_pos[elig_order[1]]] + 1)
            warnings.warn(
                f"Tie in BOT-MZ turning-angle ranking among threshold-"
                f"eligible BPs (BP{i_first} and BP{i_second} have equal "
                f"curvature). Selecting by first-occurrence order.",
                UserWarning,
                stacklevel=2,
            )

    return top_idx, bot_idx


# ════════════════════════════════════════════════════════════════════
#  DataFrame entry point
# ════════════════════════════════════════════════════════════════════
def mark_ert_mixing_zone(
    breakpoints_df: pd.DataFrame,
    *,
    bot_mz_rho_threshold: float,
    depth_col: str = "Breakpoint X Position",
    rho_col: str = "resist_ohm_m",
) -> pd.DataFrame:
    """Add ``is_top_of_mixing`` / ``is_bottom_of_mixing`` flags to a
    breakpoints DataFrame.

    Parameters
    ----------
    breakpoints_df : pd.DataFrame
        One row per breakpoint, including (at least) a depth column
        and a resistivity column. The ``Breakpoint X Position``
        default matches the output of
        ``karst_analysis.sec.breakpoints.extract_breakpoints``.
    bot_mz_rho_threshold : float
        See ``select_ert_mixing_zone``.
    depth_col, rho_col : str
        Names of the depth and resistivity columns to use.

    Returns
    -------
    pd.DataFrame
        A COPY of the input with two extra boolean columns. The input
        is not mutated.

    Notes
    -----
    Resistivity at each breakpoint is typically obtained by linear
    interpolation of the ERT trace's ``resist_ohm_m`` column at the
    breakpoint depth (since ``extract_breakpoints`` only stores the
    fitted value in log10 space). Callers should pass a DataFrame
    that already has the resistivity column populated.
    """
    if depth_col not in breakpoints_df.columns:
        raise ValueError(
            f"breakpoints_df missing depth column {depth_col!r}. "
            f"Available: {list(breakpoints_df.columns)}"
        )
    if rho_col not in breakpoints_df.columns:
        raise ValueError(
            f"breakpoints_df missing resistivity column {rho_col!r}. "
            f"Available: {list(breakpoints_df.columns)}"
        )

    out = breakpoints_df.copy()
    out["is_top_of_mixing"] = False
    out["is_bottom_of_mixing"] = False

    z_bp = out[depth_col].to_numpy(dtype=float)
    rho_bp = out[rho_col].to_numpy(dtype=float)

    top_idx, bot_idx = select_ert_mixing_zone(
        z_bp, rho_bp,
        bot_mz_rho_threshold=bot_mz_rho_threshold,
    )

    if top_idx is not None:
        out.iloc[top_idx, out.columns.get_loc("is_top_of_mixing")] = True
    if bot_idx is not None:
        out.iloc[bot_idx, out.columns.get_loc("is_bottom_of_mixing")] = True

    return out
