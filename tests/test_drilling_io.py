"""Unit tests for the Ardaman drilling-record loader."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from karst_analysis.drilling.io import load_ardaman, DEFAULT_ARDAMAN_CSV


ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / DEFAULT_ARDAMAN_CSV


@pytest.mark.skipif(not CSV_PATH.exists(),
                    reason=f"ardaman csv not found at {CSV_PATH}")
class TestLoadArdaman:
    def test_aw5o_returns_20_rows(self):
        df = load_ardaman(CSV_PATH, well="AW5O")
        assert len(df) == 20

    def test_aw6o_returns_16_rows(self):
        df = load_ardaman(CSV_PATH, well="AW6O")
        assert len(df) == 16

    def test_required_columns(self):
        df = load_ardaman(CSV_PATH, well="AW5O")
        required = {"depth_top_m", "depth_bot_m", "kind", "text",
                    "depth_top_bgl_m", "depth_bot_bgl_m", "depth_centre_bgl_m"}
        assert required.issubset(df.columns)

    def test_unknown_well_returns_empty(self):
        df = load_ardaman(CSV_PATH, well="DoesNotExist")
        assert df.empty

    def test_kind_values(self):
        df = load_ardaman(CSV_PATH, well="AW5O")
        kinds = set(df["kind"].unique())
        assert kinds <= {"lithology", "conductivity_in_situ"}

    def test_lithology_intervals_extend_to_next(self):
        """Each lithology row's depth_bot_m equals the next lithology
        row's depth_top_m (except the last, which has NaN)."""
        df = load_ardaman(CSV_PATH, well="AW5O")
        # Sort by top-down (most negative elev first means deepest first;
        # for this test we want shallowest first → ascending depth_top_m)
        df = df.sort_values("depth_top_m").reset_index(drop=True)
        is_lith = df["kind"] == "lithology"
        lith = df[is_lith].reset_index(drop=True)
        for i in range(len(lith) - 1):
            assert lith.loc[i, "depth_bot_m"] == lith.loc[i + 1, "depth_top_m"]
        # Last lithology entry has NaN bottom
        assert pd.isna(lith.iloc[-1]["depth_bot_m"])

    def test_conductivity_is_point(self):
        """Conductivity rows have top == bottom."""
        df = load_ardaman(CSV_PATH, well="AW5O")
        cond = df[df["kind"] == "conductivity_in_situ"]
        for _, row in cond.iterrows():
            assert row["depth_top_m"] == row["depth_bot_m"]

    def test_bgl_columns_match_depth(self):
        df = load_ardaman(CSV_PATH, well="AW5O")
        # BGL-positive: depth_top_bgl_m == depth_top_m (kept for clarity).
        np.testing.assert_array_equal(
            df["depth_top_bgl_m"].values, df["depth_top_m"].values,
        )

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_ardaman("/nonexistent/path.csv", well="AW5O")
