"""Unit tests for the geometric layout helpers in convergence._layout."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from karst_analysis.convergence._layout import (
    minimum_displacement_positions,
    build_label_text,
)


class TestMinimumDisplacementPositions:
    def test_no_overlap_no_change(self):
        """When labels are far apart, positions equal anchors."""
        anchors = np.array([10.0, 5.0, 0.0])
        half_h = np.array([0.5, 0.5, 0.5])
        out = minimum_displacement_positions(anchors, half_h, y_lo=-10, y_hi=20)
        np.testing.assert_allclose(out, anchors, atol=1e-9)

    def test_single_anchor_unchanged(self):
        anchors = np.array([5.0])
        half_h = np.array([1.0])
        out = minimum_displacement_positions(anchors, half_h, y_lo=0, y_hi=10)
        assert out[0] == 5.0

    def test_empty_anchors(self):
        anchors = np.array([])
        half_h = np.array([])
        out = minimum_displacement_positions(anchors, half_h, y_lo=0, y_hi=10)
        assert len(out) == 0

    def test_overlapping_get_separated(self):
        """Two labels at the same anchor with finite half-height must
        end up separated by at least their combined half-heights."""
        anchors = np.array([5.0, 5.0])
        half_h = np.array([0.5, 0.5])
        out = minimum_displacement_positions(anchors, half_h, y_lo=0, y_hi=10)
        gap = abs(out[0] - out[1])
        assert gap >= half_h[0] + half_h[1] - 1e-9

    def test_clamped_to_bounds(self):
        """Labels that would fall outside [y_lo, y_hi] are clamped."""
        anchors = np.array([15.0, 14.5, 14.0])  # close to y_hi=15
        half_h = np.array([1.0, 1.0, 1.0])
        out = minimum_displacement_positions(anchors, half_h, y_lo=0, y_hi=15)
        # All centres must be within [y_lo + h, y_hi - h]
        for y, h in zip(out, half_h):
            assert y - h >= 0 - 1e-9
            assert y + h <= 15 + 1e-9


class TestBuildLabelText:
    def test_single_point(self):
        row = pd.Series({
            "depth_top_m": 5.0, "depth_bot_m": 5.0,
            "text": "Some note", "kind": "note",
        })
        s = build_label_text(row)
        assert s == "(5.0 m) Some note"

    def test_interval(self):
        row = pd.Series({
            "depth_top_m": 5.0, "depth_bot_m": 7.5,
            "text": "Wide unit", "kind": "note",
        })
        s = build_label_text(row)
        assert s == "(5.0–7.5 m) Wide unit"

    def test_ardaman_lithology_prefix(self):
        row = pd.Series({
            "depth_top_m": 0.0, "depth_bot_m": 6.1,
            "text": "Limestone", "kind": "ardaman_lith",
        })
        s = build_label_text(row)
        assert s.startswith("[Ardaman]")

    def test_ardaman_cond_prefix(self):
        row = pd.Series({
            "depth_top_m": 3.7, "depth_bot_m": 3.7,
            "text": "0.85 ms/cm", "kind": "ardaman_cond",
        })
        s = build_label_text(row)
        assert s.startswith("[Ardaman]")

    def test_nan_bot_treated_as_point(self):
        row = pd.Series({
            "depth_top_m": 20.4, "depth_bot_m": np.nan,
            "text": "Deepest", "kind": "ardaman_lith",
        })
        s = build_label_text(row)
        # NaN should be treated as a point (no range)
        assert "20.4–" not in s
        assert "(20.4 m)" in s
