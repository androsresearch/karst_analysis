"""Tests for ``karst_analysis.sec.slopes``.

Verifies the chord-slope computation, the mixing-zone flagging, and
the contract guarantees (input validation, empty cases, tie warning).

Mixing-zone semantics:
- TOP MZ: largest discrete curvature among interior breakpoints,
  no constraint.
- BOT MZ: largest discrete curvature among interior breakpoints with
  sec_top ≥ ``bot_mz_sec_threshold``, excluding the BP chosen as TOP MZ.
  Unmarked if no BP qualifies.

The default threshold is 40 000 µS/cm. Tests that don't care about
the threshold use a low value (e.g. 0.0) to disable it.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from karst_analysis.sec.slopes import compute_slopes


# ────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────
def _make_bp_df(x_list, y_list) -> pd.DataFrame:
    """Build a breakpoints DataFrame in the format extract_breakpoints emits."""
    return pd.DataFrame(
        {
            "Breakpoint X Position": list(x_list),
            "Breakpoint Y Position": list(y_list),
            "Confidence Interval (X)": [(np.nan, np.nan)] * len(x_list),
        },
        index=range(1, len(x_list) + 1),
    )


# ────────────────────────────────────────────────────────────────────
#  Empty / trivial cases
# ────────────────────────────────────────────────────────────────────
def test_empty_input_returns_empty_frame_with_schema():
    bp = _make_bp_df([], [])
    out = compute_slopes(bp)
    assert len(out) == 0
    expected_cols = {
        "pair_idx", "depth_top", "depth_bottom",
        "log10_sec_top", "log10_sec_bottom",
        "sec_top_uS_cm", "sec_bottom_uS_cm",
        "slope_log10", "slope_linear_uS_cm_per_m",
        "slope_sign", "is_top_of_mixing", "is_bottom_of_mixing",
    }
    assert set(out.columns) == expected_cols


def test_single_breakpoint_returns_empty_frame():
    bp = _make_bp_df([5.0], [3.5])
    out = compute_slopes(bp)
    assert len(out) == 0


# ────────────────────────────────────────────────────────────────────
#  Basic slope arithmetic
# ────────────────────────────────────────────────────────────────────
def test_two_breakpoints_one_slope_top_only_when_below_threshold():
    """N=2 → 1 pair. With threshold 40k and sec_top=1000, BOT does NOT
    get marked (only TOP)."""
    bp = _make_bp_df([2.0, 4.0], [3.0, 4.0])  # sec_top=1000, sec_bot=10000
    out = compute_slopes(bp)
    assert len(out) == 1
    row = out.iloc[0]
    assert bool(row["is_top_of_mixing"]) is True
    assert bool(row["is_bottom_of_mixing"]) is False  # 1000 < 40k


def test_two_breakpoints_both_flags_when_above_threshold():
    """N=2 → 1 pair. With sec_top=50000, BOT gets marked too."""
    bp = _make_bp_df([2.0, 4.0], [4.7, 4.71])  # sec_top ~= 50000
    out = compute_slopes(bp)
    row = out.iloc[0]
    assert bool(row["is_top_of_mixing"]) is True
    assert bool(row["is_bottom_of_mixing"]) is True


def test_two_breakpoints_threshold_disabled():
    """With threshold=0, BOT always marked even when sec is freshwater."""
    bp = _make_bp_df([2.0, 4.0], [3.0, 4.0])  # sec_top=1000
    out = compute_slopes(bp, bot_mz_sec_threshold=0.0)
    row = out.iloc[0]
    assert bool(row["is_bottom_of_mixing"]) is True


def test_three_breakpoints_below_threshold_no_bot():
    """N=3, interior BP has sec=1100, below 40k → BOT unmarked."""
    bp = _make_bp_df([1.0, 2.0, 3.0], [3.0, 3.05, 4.0])
    out = compute_slopes(bp)
    assert len(out) == 2
    assert bool(out.iloc[1]["is_top_of_mixing"]) is True
    assert bool(out.iloc[1]["is_bottom_of_mixing"]) is False  # 10**3.05 ~ 1100


def test_slope_signs_preserved_for_decreasing_y():
    bp = _make_bp_df([1.0, 3.0], [4.0, 3.5])
    out = compute_slopes(bp)
    assert out.iloc[0]["slope_log10"] == pytest.approx(-0.25)
    assert out.iloc[0]["slope_sign"] == -1


# ────────────────────────────────────────────────────────────────────
#  Mixing-zone identification with threshold
# ────────────────────────────────────────────────────────────────────
def test_sigmoid_like_profile_picks_two_knees_above_threshold():
    """Synthetic flat-steep-flat profile that reaches 50k.

        BP1 (0, 3.000)  freshwater plateau (1000)
        BP2 (1, 3.010)  freshwater plateau (1023)
        BP3 (2, 3.020)  TOP knee (1047)
        BP4 (3, 3.800)  middle (6310)
        BP5 (4, 4.580)  BOT knee (38019) — below 40k threshold!
        BP6 (5, 4.700)  saltwater (50119)
        BP7 (6, 4.700)  saltwater (50119)

    With default threshold=40k, BP5 sec=38019 does NOT pass.
    BOT MZ has to be chosen among {BP6} (only BP above threshold among
    interior); BP6 has small turning angle but is the only candidate.
    """
    bp = _make_bp_df(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 3.01, 3.02, 3.8, 4.58, 4.7, 4.7],
    )
    out = compute_slopes(bp)

    # TOP: BP3 by curvature (depth_top of pair 3 = 2.0)
    top_row = out.loc[out["is_top_of_mixing"]].iloc[0]
    assert top_row["depth_top"] == pytest.approx(2.0)

    # BOT: must have sec_top >= 40k. BP5 has sec=38019 (just under), BP6
    # has 50119 (over). So BOT is BP6 → pair 6 → depth_top = 5.0
    bot_rows = out.loc[out["is_bottom_of_mixing"]]
    assert len(bot_rows) == 1
    assert bot_rows.iloc[0]["depth_top"] == pytest.approx(5.0)


def test_sigmoid_with_lower_threshold_picks_curvature_winner():
    """Same profile, but threshold=30k so BP5 qualifies. Now BOT is BP5
    (higher curvature than BP6)."""
    bp = _make_bp_df(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [3.0, 3.01, 3.02, 3.8, 4.58, 4.7, 4.7],
    )
    out = compute_slopes(bp, bot_mz_sec_threshold=30_000.0)

    bot_row = out.loc[out["is_bottom_of_mixing"]].iloc[0]
    assert bot_row["depth_top"] == pytest.approx(4.0)  # BP5


def test_no_bp_above_threshold_leaves_bot_unmarked():
    """A profile that never reaches saltwater. BOT MZ stays unmarked."""
    bp = _make_bp_df(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        [3.0, 3.01, 3.5, 3.6, 3.7],   # max sec ~= 5012 << 40k
    )
    out = compute_slopes(bp)

    # TOP still gets marked (curvature criterion)
    assert out["is_top_of_mixing"].sum() == 1
    # BOT is left completely unmarked
    assert out["is_bottom_of_mixing"].sum() == 0


def test_lrs70d_real_pattern():
    """Real LRS70D LOWESS N=15 trial_1: validates BP8 = TOP, BP12 = BOT
    when using the default threshold of 40k."""
    bp = _make_bp_df(
        [1.088875, 2.331340, 3.223473, 3.920148, 4.540055, 5.444627,
         7.179577, 8.297417, 9.053570, 10.839219, 11.799181, 12.697619,
         20.861298, 23.734505, 25.585277],
        [3.343077, 3.468078, 3.638172, 3.727791, 3.742533, 3.757014,
         3.771367, 3.796156, 4.050722, 4.270141, 4.590433, 4.679732,
         4.699115, 4.702587, 4.702990],
    )
    out = compute_slopes(bp)

    top_row = out.loc[out["is_top_of_mixing"]].iloc[0]
    bot_row = out.loc[out["is_bottom_of_mixing"]].iloc[0]

    # BP8 by curvature (sec ~6260, below 40k, but curvature is highest)
    assert top_row["depth_top"] == pytest.approx(8.297417, rel=1e-5)
    # BP12 has sec ~47832 (above 40k) and is the curvature winner among
    # eligible BPs (BP12, BP13, BP14 all >40k; BP12 has highest curvature)
    assert bot_row["depth_top"] == pytest.approx(12.697619, rel=1e-5)


def test_threshold_invariance_when_all_bps_high():
    """If every BP has sec ≥ threshold, threshold has no effect: BOT is
    just the largest curvature among interior, excluding TOP."""
    z = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    # All log10 ≥ 4.7 → all sec ≥ 50k
    y = [4.70, 4.71, 4.72, 5.5, 6.28, 6.40, 6.41]
    out_low = compute_slopes(_make_bp_df(z, y), bot_mz_sec_threshold=0.0)
    out_high = compute_slopes(_make_bp_df(z, y), bot_mz_sec_threshold=40_000.0)
    assert (out_low["is_top_of_mixing"].to_numpy()
            == out_high["is_top_of_mixing"].to_numpy()).all()
    assert (out_low["is_bottom_of_mixing"].to_numpy()
            == out_high["is_bottom_of_mixing"].to_numpy()).all()


# ────────────────────────────────────────────────────────────────────
#  Input validation
# ────────────────────────────────────────────────────────────────────
def test_missing_columns_raises_keyerror():
    bad = pd.DataFrame({"foo": [1, 2], "bar": [3, 4]})
    with pytest.raises(KeyError, match="missing required column"):
        compute_slopes(bad)


def test_non_ascending_x_raises_assertionerror():
    bp = _make_bp_df([2.0, 1.0, 3.0], [3.0, 3.5, 4.0])
    with pytest.raises(AssertionError, match="strictly ascending"):
        compute_slopes(bp)


def test_duplicate_x_raises_assertionerror():
    bp = _make_bp_df([1.0, 1.0, 3.0], [3.0, 3.5, 4.0])
    with pytest.raises(AssertionError, match="strictly ascending"):
        compute_slopes(bp)


# ────────────────────────────────────────────────────────────────────
#  Tie-break warning
# ────────────────────────────────────────────────────────────────────
def test_tie_in_bot_eligible_emits_warning():
    """When the 1st and 2nd ranked turning angles among threshold-eligible
    BOT candidates are exactly equal, warn and pick by first-occurrence."""
    # Build a profile where multiple BPs above threshold have equal angles.
    # Use log10 values all ≥ 4.7 so threshold is not the constraint.
    bp = _make_bp_df(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        [4.70, 4.70, 4.80, 4.80, 4.70, 4.70, 4.80],
    )
    # The geometry creates several interior BPs with equal curvature.
    with pytest.warns(UserWarning, match=r"Tie in BOT-MZ"):
        compute_slopes(bp, bot_mz_sec_threshold=0.0)


def test_no_tie_no_warning():
    """No warning when curvatures are well separated."""
    bp = _make_bp_df([0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                     [3.0, 3.05, 3.06, 4.5, 4.6, 4.61])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = compute_slopes(bp, bot_mz_sec_threshold=0.0)
    assert len(out) == 5
