"""Chord slopes between consecutive SEC breakpoints.

Given the breakpoints produced by ``karst_analysis.sec.breakpoints``
(i.e. the (X, Y) corner points of a piecewise-linear fit on
log10(SEC) vs depth), this module computes the slope of the chord
joining each consecutive pair and identifies the freshwater-saltwater
mixing zone as the two breakpoints of largest discrete curvature in
the (depth, log10(SEC)) plane.

The slopes computed here are the chord slopes BETWEEN breakpoints,
NOT the per-segment slopes of the underlying piecewise model. For N
breakpoints this yields N-1 chord slopes (no edge segments included).

Conventions
-----------
- Depth is positive downward, "top" = shallower, "bottom" = deeper.
- Breakpoint Y is the model's predicted log10(SEC) at the breakpoint
  X position, exactly as returned by
  ``karst_analysis.sec.breakpoints.extract_breakpoints``.
- For a freshwater-over-saltwater profile, log10(SEC) increases with
  depth and slopes are positive. Negative slopes are preserved with
  their sign (not absolute-valued away).

Mixing-zone identification
--------------------------
A SEC profile through a coastal aquifer transitions from a freshwater
plateau (upper asymptote) through a steep mixing region into a
saltwater plateau (lower asymptote). The two boundaries of the mixing
zone are the points where the curve "knees" — i.e. where it bends
most sharply between asymptote and steep regime. Geometrically, those
are the two points of maximum discrete curvature along the
piecewise-linear breakpoint trajectory.

Curvature is measured as the turning angle at each interior breakpoint
in the (z, log10(SEC)) plane, with both axes rescaled to [0, 1] over
the breakpoint range. Normalisation ensures the angle reflects what an
analyst sees on a balanced figure rather than the raw aspect ratio of
the data.

The two breakpoints of largest turning angle become BP_TOP_MZ (the
shallower one) and BP_BOT_MZ (the deeper one). Following the existing
DataFrame schema, the flag ``is_top_of_mixing`` is placed on the chord
pair whose ``depth_top`` equals BP_TOP_MZ; ``is_bottom_of_mixing`` on
the pair whose ``depth_top`` equals BP_BOT_MZ.

Public API
----------
compute_slopes(breakpoints_df) -> pd.DataFrame
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd


# Column names of the input DataFrame, matching `extract_breakpoints`
# in karst_analysis.sec.breakpoints.detection.
_X_COL = "Breakpoint X Position"
_Y_COL = "Breakpoint Y Position"


def compute_slopes(
    breakpoints_df: pd.DataFrame,
    *,
    bot_mz_sec_threshold: float = 40_000.0,
) -> pd.DataFrame:
    """Compute chord slopes between consecutive breakpoints.

    Parameters
    ----------
    breakpoints_df : pd.DataFrame
        Output of ``karst_analysis.sec.breakpoints.extract_breakpoints``
        (or any DataFrame with the columns ``Breakpoint X Position``
        and ``Breakpoint Y Position``). X must be strictly ascending
        (depth, in metres). Y is the piecewise model's predicted value
        at X, in log10(µS/cm).
    bot_mz_sec_threshold : float, default 40 000
        Minimum SEC value (µS/cm) a breakpoint must have to qualify as
        the BOT MZ. The breakpoint must reach the saltwater regime, not
        sit on a freshwater plateau or middle wiggle. If no breakpoint
        in the profile reaches this SEC, BOT MZ is left unmarked
        (the boolean column stays False on every row), which is honest
        information: that well does not reach established saltwater.
        TOP MZ is unaffected by this threshold.

    Returns
    -------
    pd.DataFrame
        One row per consecutive breakpoint pair. Empty if fewer than 2
        breakpoints are provided. Columns:

        - ``pair_idx``               : 1-based index of the pair.
        - ``depth_top``              : shallower breakpoint depth (m).
        - ``depth_bottom``           : deeper breakpoint depth (m).
        - ``log10_sec_top``          : Y at the top breakpoint.
        - ``log10_sec_bottom``       : Y at the bottom breakpoint.
        - ``sec_top_uS_cm``          : 10 ** log10_sec_top.
        - ``sec_bottom_uS_cm``       : 10 ** log10_sec_bottom.
        - ``slope_log10``            : (Y_bot - Y_top) / (z_bot - z_top),
                                       in log10(µS/cm) per metre.
        - ``slope_linear_uS_cm_per_m``: linear-space chord slope, in
                                       µS/cm per metre.
        - ``slope_sign``             : np.sign(slope_log10) as int.
        - ``is_top_of_mixing``       : True for the chord pair whose
                                       ``depth_top`` is the breakpoint
                                       of largest discrete curvature
                                       (purely geometric criterion).
        - ``is_bottom_of_mixing``    : True for the chord pair whose
                                       ``depth_top`` is the breakpoint
                                       of largest discrete curvature
                                       AMONG those with sec_top ≥
                                       bot_mz_sec_threshold (geometric
                                       criterion + physical constraint).
                                       May be unmarked everywhere if
                                       no breakpoint reaches the
                                       threshold.

    Raises
    ------
    KeyError
        If the input DataFrame lacks the required columns.
    AssertionError
        If ``Breakpoint X Position`` is not strictly ascending.

    Notes
    -----
    Curvature at an interior breakpoint is the turning angle between
    the chord arriving at it and the chord leaving it, computed in
    coordinates where both axes have been rescaled to [0, 1] over the
    breakpoint range.

    The asymmetric treatment of TOP and BOT mixing zone is physical:
    the upper boundary is a purely geometric "knee" leaving the
    freshwater asymptote, while the lower boundary marks where the
    profile reaches saltwater establishment, which requires the
    conductivity to actually be saltwater-like (hence the SEC
    threshold).

    Edge cases:
        - If the largest-curvature breakpoint is the very first
          interior breakpoint AND no other interior breakpoint stands
          out, both flags collapse onto a single pair (only when the
          BP also passes the SEC threshold for BOT MZ).
        - With fewer than 3 breakpoints (2 or fewer chord pairs) there
          are no interior turning angles; the existing trivial
          flagging applies (1 pair → both flags on it; 2 pairs → both
          flags on the downstream pair).
    """
    # ── 1) Validate input ────────────────────────────────────────────
    missing = [c for c in (_X_COL, _Y_COL) if c not in breakpoints_df.columns]
    if missing:
        raise KeyError(
            f"breakpoints_df is missing required column(s): {missing}. "
            f"Expected output of "
            f"karst_analysis.sec.breakpoints.extract_breakpoints."
        )

    x = breakpoints_df[_X_COL].to_numpy(dtype=float)
    y = breakpoints_df[_Y_COL].to_numpy(dtype=float)

    # ── 2) Trivial cases ─────────────────────────────────────────────
    if len(x) < 2:
        return _empty_slopes_frame()

    # Strict ascending order in depth — this is how
    # extract_breakpoints emits them; we don't silently reorder.
    if not np.all(np.diff(x) > 0):
        raise AssertionError(
            "'Breakpoint X Position' must be strictly ascending. "
            "Got x = {!r}. Sort the DataFrame by this column before "
            "calling compute_slopes.".format(x)
        )

    # ── 3) Chord slopes (N-1 rows for N breakpoints) ─────────────────
    z_top = x[:-1]
    z_bot = x[1:]
    y_top = y[:-1]
    y_bot = y[1:]

    dz = z_bot - z_top  # all > 0 by the ascending check above
    dy_log = y_bot - y_top
    slope_log10 = dy_log / dz

    sec_top = np.power(10.0, y_top)
    sec_bot = np.power(10.0, y_bot)
    slope_lin = (sec_bot - sec_top) / dz

    n_pairs = len(slope_log10)
    out = pd.DataFrame({
        "pair_idx": np.arange(1, n_pairs + 1, dtype=int),
        "depth_top": z_top,
        "depth_bottom": z_bot,
        "log10_sec_top": y_top,
        "log10_sec_bottom": y_bot,
        "sec_top_uS_cm": sec_top,
        "sec_bottom_uS_cm": sec_bot,
        "slope_log10": slope_log10,
        "slope_linear_uS_cm_per_m": slope_lin,
        "slope_sign": np.sign(slope_log10).astype(int),
        "is_top_of_mixing": False,
        "is_bottom_of_mixing": False,
    })

    # ── 4) Identify TOP MZ (curvature) and BOT MZ (curvature + threshold) ──
    out = _mark_mixing_zone(out, bot_mz_sec_threshold=bot_mz_sec_threshold)

    return out


# ────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────
def _empty_slopes_frame() -> pd.DataFrame:
    """Return a slopes DataFrame with the right schema and zero rows."""
    return pd.DataFrame({
        "pair_idx": pd.Series(dtype=int),
        "depth_top": pd.Series(dtype=float),
        "depth_bottom": pd.Series(dtype=float),
        "log10_sec_top": pd.Series(dtype=float),
        "log10_sec_bottom": pd.Series(dtype=float),
        "sec_top_uS_cm": pd.Series(dtype=float),
        "sec_bottom_uS_cm": pd.Series(dtype=float),
        "slope_log10": pd.Series(dtype=float),
        "slope_linear_uS_cm_per_m": pd.Series(dtype=float),
        "slope_sign": pd.Series(dtype=int),
        "is_top_of_mixing": pd.Series(dtype=bool),
        "is_bottom_of_mixing": pd.Series(dtype=bool),
    })


def _mark_mixing_zone(
    df: pd.DataFrame,
    *,
    bot_mz_sec_threshold: float,
) -> pd.DataFrame:
    """Set is_top_of_mixing / is_bottom_of_mixing.

    TOP MZ: purely geometric — the interior breakpoint with the
    largest discrete curvature.

    BOT MZ: geometric AND physical — the interior breakpoint with the
    largest discrete curvature, AMONG those whose SEC ≥ threshold,
    and excluding the BP already chosen as TOP MZ. If no interior BP
    meets the threshold, BOT MZ is left unmarked.

    Curvature is the turning angle between the incoming and outgoing
    chords at each interior breakpoint, computed in coordinates where
    both axes are rescaled to [0, 1] over the breakpoint range.

    Edge cases
    ----------
    - 1 pair (2 BPs): no interior BPs → both flags on the only pair.
    - 2 pairs (3 BPs): one interior BP. TOP gets it. BOT gets it only
      if its SEC ≥ threshold; otherwise BOT stays unmarked.
    - >=3 pairs: standard logic with separate selection for TOP and BOT.

    Tie warning
    -----------
    Emits ``UserWarning`` if the 1st and 2nd ranked turning angles
    among the BOT-eligible BPs are exactly equal (ambiguous selection
    of BOT MZ).
    """
    n_pairs = len(df)
    if n_pairs == 0:
        return df

    if n_pairs == 1:
        # Trivial: only one chord; flag it as both ends.
        idx = df.index[0]
        df.at[idx, "is_top_of_mixing"] = True
        # BOT only if it meets the threshold
        if df["sec_top_uS_cm"].iloc[0] >= bot_mz_sec_threshold:
            df.at[idx, "is_bottom_of_mixing"] = True
        return df

    # ── Reconstruct unique breakpoint coordinates from the pairs ───────
    z_bp = np.concatenate([
        df["depth_top"].to_numpy(),
        [df["depth_bottom"].iloc[-1]],
    ])
    y_bp = np.concatenate([
        df["log10_sec_top"].to_numpy(),
        [df["log10_sec_bottom"].iloc[-1]],
    ])
    sec_bp = np.power(10.0, y_bp)
    n_bp = len(z_bp)  # = n_pairs + 1

    if n_pairs == 2:
        # 3 breakpoints, one interior at z_bp[1].
        idx_downstream = df.index[1]
        df.at[idx_downstream, "is_top_of_mixing"] = True
        # BOT only if the interior BP's SEC meets the threshold
        if sec_bp[1] >= bot_mz_sec_threshold:
            df.at[idx_downstream, "is_bottom_of_mixing"] = True
        return df

    # ── Normalise both axes to [0, 1] over the breakpoint range ────────
    z_range = z_bp.max() - z_bp.min()
    y_range = y_bp.max() - y_bp.min()
    if z_range == 0 or y_range == 0:
        return df
    zn = (z_bp - z_bp.min()) / z_range
    yn = (y_bp - y_bp.min()) / y_range

    # ── Turning angle at each interior breakpoint (i = 1..n_bp-2) ──────
    turning = np.zeros(n_bp)  # interior values; endpoints stay 0
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

    # ── TOP MZ: largest turning angle, no constraint ───────────────────
    order = np.argsort(-interior_turning, kind="stable")
    bp_top_mz_idx = int(interior_idx[order[0]])

    # ── BOT MZ: largest turning angle among BPs with SEC ≥ threshold,
    #            excluding the BP already chosen as TOP MZ ──────────────
    bot_eligible_mask = (sec_bp[interior_idx] >= bot_mz_sec_threshold) & \
                        (interior_idx != bp_top_mz_idx)

    if not bot_eligible_mask.any():
        # No BP qualifies. Honest gap: leave BOT unmarked.
        df.at[df.index[bp_top_mz_idx], "is_top_of_mixing"] = True
        return df

    # Restrict the ranking to eligible BPs
    elig_local_positions = np.where(bot_eligible_mask)[0]
    elig_turning = interior_turning[elig_local_positions]
    elig_order = np.argsort(-elig_turning, kind="stable")
    bp_bot_mz_idx = int(interior_idx[elig_local_positions[elig_order[0]]])

    # Tie warning for BOT (1st vs 2nd among eligible)
    if len(elig_turning) >= 2:
        mag_1st = elig_turning[elig_order[0]]
        mag_2nd = elig_turning[elig_order[1]]
        if np.isclose(mag_1st, mag_2nd, rtol=0.0, atol=0.0):
            warnings.warn(
                "Tie in BOT-MZ turning-angle ranking among threshold-eligible "
                "BPs (BP{} and BP{} have equal curvature). Selecting by "
                "first-occurrence order.".format(
                    int(interior_idx[elig_local_positions[elig_order[0]]] + 1),
                    int(interior_idx[elig_local_positions[elig_order[1]]] + 1),
                ),
                UserWarning,
                stacklevel=3,
            )

    # Sanity: BOT must be deeper than TOP (otherwise something is off)
    # We do NOT enforce ordering here; instead, the caller can inspect
    # depth_top of each flagged row to verify. In a sigmoid-like profile
    # with SEC monotone increasing with depth, BOT will naturally be
    # below TOP because SEC threshold ≥ 40k is only satisfied at depths
    # where the profile has already passed the freshwater asymptote.

    df.at[df.index[bp_top_mz_idx], "is_top_of_mixing"] = True
    df.at[df.index[bp_bot_mz_idx], "is_bottom_of_mixing"] = True

    return df
