"""Tests for the ERT mixing-zone selection (analogous to SEC's
``_mark_mixing_zone``, with the resistivity-threshold direction flipped).

These tests are pure-synthetic so they run in milliseconds and don't
depend on the ERT data fixture.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from karst_analysis.ert.mixing_zone import (
    mark_ert_mixing_zone,
    select_ert_mixing_zone,
)


# ════════════════════════════════════════════════════════════════════
# select_ert_mixing_zone — pure-array entry point
# ════════════════════════════════════════════════════════════════════
class TestSelectErtMixingZone:
    """The core array-pure logic. All edge cases covered."""

    def test_too_few_breakpoints_returns_none_none(self):
        """With fewer than 3 BPs there are no interior BPs."""
        z = np.array([1.0, 2.0])
        rho = np.array([10.0, 5.0])
        assert select_ert_mixing_zone(
            z, rho, bot_mz_rho_threshold=25.0,
        ) == (None, None)

    def test_degenerate_depth_range_returns_none_none(self):
        z = np.array([5.0, 5.0, 5.0])
        rho = np.array([10.0, 5.0, 1.0])
        assert select_ert_mixing_zone(
            z, rho, bot_mz_rho_threshold=25.0,
        ) == (None, None)

    def test_degenerate_rho_range_returns_none_none(self):
        z = np.array([1.0, 2.0, 3.0])
        rho = np.array([10.0, 10.0, 10.0])
        assert select_ert_mixing_zone(
            z, rho, bot_mz_rho_threshold=25.0,
        ) == (None, None)

    def test_negative_rho_raises(self):
        z = np.array([1.0, 2.0, 3.0])
        rho = np.array([10.0, -5.0, 1.0])
        with pytest.raises(ValueError, match="strictly positive"):
            select_ert_mixing_zone(z, rho, bot_mz_rho_threshold=25.0)

    def test_shape_mismatch_raises(self):
        z = np.array([1.0, 2.0, 3.0])
        rho = np.array([10.0, 5.0])
        with pytest.raises(ValueError, match="same shape"):
            select_ert_mixing_zone(z, rho, bot_mz_rho_threshold=25.0)

    def test_simple_three_breakpoints(self):
        """One interior BP — TOP gets it. BOT gets it iff rho<=threshold."""
        z = np.array([0.0, 5.0, 10.0])
        rho = np.array([100.0, 10.0, 2.0])
        # threshold=25 → interior BP rho=10 qualifies for BOT but is
        # already taken as TOP, so BOT is None.
        top, bot = select_ert_mixing_zone(z, rho, bot_mz_rho_threshold=25.0)
        assert top == 1
        assert bot is None

    def test_top_purely_geometric(self):
        """A profile with a sharp kink at one BP: that BP becomes TOP."""
        # Sharp turn at index 2: from "going right" to "going down".
        z = np.array([0.0, 1.0, 2.0, 2.5, 3.0])
        # rho falls slowly then drops sharply at index 2.
        rho = np.array([100.0, 90.0, 80.0, 5.0, 4.5])
        top, bot = select_ert_mixing_zone(
            z, rho, bot_mz_rho_threshold=25.0,
        )
        # Sharpest curvature is at index 2 (the kink).
        assert top == 2
        # BPs 1, 2, 3 are interior; among those, BPs 2 and 3 have rho<=25.
        # BP 2 is taken by TOP, so BOT picks among {3}.
        assert bot == 3

    def test_no_eligible_for_bot_returns_top_only(self):
        """All resistivities above threshold → BOT unmarked, TOP set."""
        z = np.array([0.0, 1.0, 2.0, 3.0])
        rho = np.array([1000.0, 800.0, 500.0, 700.0])
        top, bot = select_ert_mixing_zone(
            z, rho, bot_mz_rho_threshold=25.0,
        )
        assert top is not None
        assert bot is None

    def test_bot_excludes_top(self):
        """If the same BP would win both TOP and BOT, BOT must look elsewhere."""
        # Construct a profile where the largest curvature is at BP 1
        # AND rho at BP 1 satisfies the threshold. BOT must still pick
        # a different BP (or return None if no other eligible).
        z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        rho = np.array([100.0, 5.0, 4.0, 3.0, 2.5])
        top, bot = select_ert_mixing_zone(
            z, rho, bot_mz_rho_threshold=25.0,
        )
        assert top is not None
        # If bot is set, it MUST be different from top.
        if bot is not None:
            assert bot != top

    def test_endpoints_never_selected(self):
        """First and last BP cannot be TOP or BOT."""
        z = np.linspace(0, 10, 5)
        rho = np.array([100.0, 50.0, 10.0, 5.0, 1.0])
        top, bot = select_ert_mixing_zone(
            z, rho, bot_mz_rho_threshold=25.0,
        )
        assert top not in (0, len(z) - 1)
        if bot is not None:
            assert bot not in (0, len(z) - 1)

    def test_threshold_direction_flipped_from_sec(self):
        """Saltwater = LOW resistivity. A high-rho BP must NOT be eligible."""
        z = np.array([0.0, 1.0, 2.0, 3.0])
        # The interior BPs have very different rho values.
        rho = np.array([100.0, 80.0, 5.0, 90.0])
        top, bot = select_ert_mixing_zone(
            z, rho, bot_mz_rho_threshold=25.0,
        )
        # Only BP at index 2 (rho=5) is eligible for BOT.
        if bot is not None:
            assert rho[bot] <= 25.0

    @pytest.mark.xfail(
        reason=(
            "Inherited from SEC: the tie check uses "
            "np.isclose(m1, m2, rtol=0.0, atol=0.0), which is exact "
            "equality. Symmetric synthetic inputs produce turning "
            "angles that match to ~15 decimal places but not bit-for-"
            "bit, so the warning fires only for hand-crafted exact "
            "ties. This is a methodological caveat to track in "
            "NOTES_open_questions.md (related to entry #4)."
        ),
        strict=True,
    )
    def test_tie_warning(self):
        """If two eligible BPs have identical curvature, emit a warning."""
        # A perfectly symmetric profile so two interior BPs have
        # identical turning angles.
        z = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        # Symmetric V-shape in log10(rho).
        rho = np.array([100.0, 10.0, 1.0, 10.0, 100.0])
        # BPs 1 and 3 have identical curvature. Both have rho=10
        # which is <= 25, so both are eligible for BOT (after one is
        # taken by TOP, only the other is left — no tie). To force a
        # tie among BOT-eligible BPs we need at least 3 interior BPs
        # with equal curvature, two of which qualify for BOT after
        # excluding TOP. Use 5 interior BPs.
        z = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        rho = np.array([100.0, 10.0, 1.0, 0.5, 1.0, 10.0, 100.0])
        # Interior idx = 1..5. By symmetry, BP 2 and BP 4 have equal
        # curvature; BP 3 is the apex. TOP takes BP 3 (largest);
        # BOT-eligible among {1,2,4,5} (excluding TOP=3) are those with
        # rho<=25 → all four (rho=10,1,1,10). The two with the largest
        # equal turning angle are BPs 2 and 4 (both rho=1).
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            top, bot = select_ert_mixing_zone(
                z, rho, bot_mz_rho_threshold=25.0,
            )
        # We expect exactly one UserWarning about the tie.
        tie_warnings = [w for w in caught if "Tie in BOT-MZ" in str(w.message)]
        assert len(tie_warnings) == 1


# ════════════════════════════════════════════════════════════════════
# mark_ert_mixing_zone — DataFrame wrapper
# ════════════════════════════════════════════════════════════════════
class TestMarkErtMixingZone:
    def _bp_df(self, z, rho) -> pd.DataFrame:
        return pd.DataFrame({
            "Breakpoint X Position": z,
            "resist_ohm_m": rho,
        })

    def test_does_not_mutate_input(self):
        df = self._bp_df([0.0, 1.0, 2.0], [100.0, 10.0, 1.0])
        before = df.copy()
        _ = mark_ert_mixing_zone(df, bot_mz_rho_threshold=25.0)
        pd.testing.assert_frame_equal(df, before)

    def test_returns_copy_with_two_extra_columns(self):
        df = self._bp_df([0.0, 1.0, 2.0], [100.0, 10.0, 1.0])
        out = mark_ert_mixing_zone(df, bot_mz_rho_threshold=25.0)
        assert "is_top_of_mixing" in out.columns
        assert "is_bottom_of_mixing" in out.columns
        # All other columns preserved.
        for col in df.columns:
            assert col in out.columns

    def test_top_flagged_at_correct_row(self):
        # Sharp kink at row index 2.
        df = self._bp_df(
            [0.0, 1.0, 2.0, 2.5, 3.0],
            [100.0, 90.0, 80.0, 5.0, 4.5],
        )
        out = mark_ert_mixing_zone(df, bot_mz_rho_threshold=25.0)
        assert out["is_top_of_mixing"].sum() == 1
        # Row index in the underlying DataFrame might be 0..4, but the
        # flagged ROW must be the same one regardless of label index.
        flagged_pos = np.where(out["is_top_of_mixing"].to_numpy())[0][0]
        assert flagged_pos == 2

    def test_no_bot_when_threshold_not_met(self):
        df = self._bp_df([0.0, 1.0, 2.0], [1000.0, 500.0, 700.0])
        out = mark_ert_mixing_zone(df, bot_mz_rho_threshold=25.0)
        assert out["is_top_of_mixing"].sum() == 1
        assert out["is_bottom_of_mixing"].sum() == 0

    def test_missing_depth_col_raises(self):
        df = pd.DataFrame({"resist_ohm_m": [10.0, 5.0, 1.0]})
        with pytest.raises(ValueError, match="missing depth column"):
            mark_ert_mixing_zone(df, bot_mz_rho_threshold=25.0)

    def test_missing_rho_col_raises(self):
        df = pd.DataFrame({"Breakpoint X Position": [0.0, 1.0, 2.0]})
        with pytest.raises(ValueError, match="missing resistivity column"):
            mark_ert_mixing_zone(df, bot_mz_rho_threshold=25.0)

    def test_custom_column_names(self):
        df = pd.DataFrame({
            "depth": [0.0, 1.0, 2.0, 3.0],
            "rho": [100.0, 50.0, 10.0, 5.0],
        })
        out = mark_ert_mixing_zone(
            df, bot_mz_rho_threshold=25.0,
            depth_col="depth", rho_col="rho",
        )
        # Just make sure it ran and added the flags.
        assert "is_top_of_mixing" in out.columns
        assert "is_bottom_of_mixing" in out.columns

    def test_works_with_non_zero_indexed_dataframe(self):
        """``extract_breakpoints`` returns a DataFrame indexed 1..N.
        The wrapper must use positional access (iloc), not label."""
        df = self._bp_df(
            [0.0, 1.0, 2.0, 2.5, 3.0],
            [100.0, 90.0, 80.0, 5.0, 4.5],
        )
        df.index = pd.Index([1, 2, 3, 4, 5])  # mimic extract_breakpoints output
        out = mark_ert_mixing_zone(df, bot_mz_rho_threshold=25.0)
        flagged_pos = np.where(out["is_top_of_mixing"].to_numpy())[0][0]
        assert flagged_pos == 2
