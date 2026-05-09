"""Unit tests for the cumulative-minimum baseline primitives.

These don't require any data on disk — they exercise the algorithm on
synthetic inputs to verify the documented behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

from karst_analysis.caliper.baseline import (
    fit_cumulative_min_single_zone,
    fit_cumulative_min_split,
    _running_cumulative_minimum,
    _find_anchors,
)


# ──────────────────────────────────────────────────────────────────────
#  Internal primitives
# ──────────────────────────────────────────────────────────────────────
class TestRunningCumulativeMinimum:
    def test_monotone_decreasing_input(self):
        """A strictly decreasing series: every value is a new min."""
        v = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        result = _running_cumulative_minimum(v)
        np.testing.assert_array_equal(result, v)

    def test_monotone_increasing_input(self):
        """A strictly increasing series: only the first value is a min."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _running_cumulative_minimum(v)
        np.testing.assert_array_equal(result, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_alternating(self):
        v = np.array([3.0, 5.0, 2.0, 8.0, 1.0, 4.0])
        result = _running_cumulative_minimum(v)
        np.testing.assert_array_equal(result, [3.0, 3.0, 2.0, 2.0, 1.0, 1.0])


class TestFindAnchors:
    def test_first_sample_is_always_anchor(self):
        v = np.array([5.0, 6.0, 7.0])
        anchors = _find_anchors(v)
        assert 0 in anchors

    def test_anchors_correspond_to_strict_drops(self):
        # 5, 6, 4, 4, 3, 8, 2  → cum_min: 5, 5, 4, 4, 3, 3, 2
        # anchors at indices: 0 (always), 2 (5→4), 4 (4→3), 6 (3→2)
        v = np.array([5.0, 6.0, 4.0, 4.0, 3.0, 8.0, 2.0])
        anchors = _find_anchors(v)
        np.testing.assert_array_equal(anchors, [0, 2, 4, 6])


# ──────────────────────────────────────────────────────────────────────
#  Single-zone fit
# ──────────────────────────────────────────────────────────────────────
class TestSingleZoneFit:
    def test_baseline_le_caliper_pointwise(self):
        """By construction, M(z) <= C(z) at every z."""
        rng = np.random.default_rng(0)
        z = np.linspace(1.0, 30.0, 200)   # BGL-positive
        cal = 16.0 + rng.normal(0, 0.5, 200) + 0.1 * (30 - z)
        fit = fit_cumulative_min_single_zone(z, cal, interp_kind="step")
        assert (fit.baseline <= fit.cal + 1e-9).all()

    def test_baseline_monotone_top_down(self):
        """Top-down: baseline must be non-increasing from surface to depth.
        With BGL-positive convention, surface = SMALLEST z."""
        rng = np.random.default_rng(1)
        z = np.linspace(1.0, 30.0, 100)
        cal = 16.0 + rng.normal(0, 0.5, 100)
        fit = fit_cumulative_min_single_zone(
            z, cal, interp_kind="step", direction="top_down",
        )
        # Sort surface-first (ascending z in BGL-positive)
        order = np.argsort(fit.z)
        baseline_top_down = fit.baseline[order]
        assert (np.diff(baseline_top_down) <= 1e-9).all()

    def test_anchor_count_matches(self):
        v = np.array([5.0, 6.0, 4.0, 4.0, 3.0, 8.0, 2.0])
        # BGL-positive: surface at z=0, deeper toward larger z.
        z = np.arange(len(v), dtype=float)
        fit = fit_cumulative_min_single_zone(z, v, interp_kind="step")
        # Surface-first iteration with z=0,1,2,... is the same as iterating
        # v in its original order (5,6,4,4,3,8,2).
        # cum_min: 5,5,4,4,3,3,2 → strict drops at 0 (always), 2, 4, 6.
        assert fit.n_anchors == 4

    def test_interp_kinds_agree_at_anchors(self):
        """Step, linear and PCHIP must all produce the same y-values
        AT the anchor depths (where the baseline equals the caliper)."""
        rng = np.random.default_rng(2)
        z = np.linspace(1.0, 30.0, 100)   # BGL-positive
        cal = 16.0 + rng.normal(0, 0.5, 100)
        fits = {
            kind: fit_cumulative_min_single_zone(z, cal, interp_kind=kind)
            for kind in ("step", "linear", "pchip")
        }
        anchors = fits["step"].anchor_indices
        for kind in ("linear", "pchip"):
            np.testing.assert_allclose(
                fits[kind].baseline[anchors],
                fits["step"].baseline[anchors],
                atol=1e-9,
            )


# ──────────────────────────────────────────────────────────────────────
#  Split fit (shallow + deep)
# ──────────────────────────────────────────────────────────────────────
class TestSplitFit:
    def test_zone_labels_assigned(self):
        z = np.linspace(1.0, 30.0, 100)   # BGL-positive
        cal = 16.0 + np.zeros(100)
        fit = fit_cumulative_min_split(
            z, cal, trim_depth_m=5.0,   # BGL-positive trim
            interp_kind="linear", direction="top_down",
            analyse_shallow=True, floor_cm=10.0, iqr_k=None,
        )
        assert (fit.zone_label == "shallow").any()
        assert (fit.zone_label == "deep").any()

    def test_floor_filter_drops_anchors_below_floor(self):
        """Setting floor_cm = 16 removes any anchor below 16."""
        z = np.linspace(1.0, 20.0, 50)   # BGL-positive
        cal = np.full(50, 16.0)
        cal[20] = 12.0  # spurious narrow value
        fit = fit_cumulative_min_split(
            z, cal, trim_depth_m=5.0,   # BGL-positive trim
            interp_kind="linear", direction="top_down",
            analyse_shallow=True, floor_cm=16.0, iqr_k=None,
        )
        assert (fit.baseline >= 16.0 - 1e-9).all()
