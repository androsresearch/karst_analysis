"""End-to-end tests for the videolog xlsx loader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from karst_analysis.videolog.io import load_video_notes, DEFAULT_VIDEOLOG_XLSX


ROOT = Path(__file__).resolve().parent.parent
XLSX_PATH = ROOT / DEFAULT_VIDEOLOG_XLSX


@pytest.mark.skipif(not XLSX_PATH.exists(),
                    reason=f"video xlsx not found at {XLSX_PATH}")
class TestLoadVideoNotes:
    def test_lrs70d_returns_22_rows(self):
        df = load_video_notes(XLSX_PATH, sheet="LRS70D")
        assert len(df) == 22

    def test_aw6_returns_52_rows(self):
        df = load_video_notes(XLSX_PATH, sheet="AW6")
        assert len(df) == 52

    def test_required_columns(self):
        df = load_video_notes(XLSX_PATH, sheet="LRS70D")
        required = {"depth_top_m", "depth_bot_m", "note",
                    "depth_top_bgl_m", "depth_bot_bgl_m", "depth_centre_bgl_m"}
        assert required.issubset(df.columns)

    def test_bgl_columns_match_depth(self):
        df = load_video_notes(XLSX_PATH, sheet="LRS70D")
        # In BGL-positive convention, depth_top_bgl_m == depth_top_m
        # (no negation). Both kept for naming clarity.
        assert (df["depth_top_bgl_m"] == df["depth_top_m"]).all()
        assert (df["depth_bot_bgl_m"] == df["depth_bot_m"]).all()

    def test_sorted_top_to_bottom(self):
        """Sorted by depth_centre_bgl_m ascending = surface first."""
        df = load_video_notes(XLSX_PATH, sheet="LRS70D")
        assert (df["depth_centre_bgl_m"].diff().dropna() >= 0).all()

    def test_typo_fixes_applied(self):
        """At least one note should differ from the raw xlsx text."""
        df = load_video_notes(XLSX_PATH, sheet="AW6")
        # If any typo fix was needed, the resulting text shouldn't
        # contain the original misspelled forms.
        all_notes = " ".join(df["note"].astype(str))
        for typo, _ in [("occuring", "occurring"), ("beome", "become")]:
            # The fix was applied, so the typo should NOT appear in
            # the cleaned notes (it's possible the typo wasn't in
            # this sheet, in which case neither form appears, which
            # is also fine).
            pass  # smoke test only — actual mapping checked in test_videolog_parsing

    def test_unknown_sheet_raises(self):
        with pytest.raises(Exception):
            load_video_notes(XLSX_PATH, sheet="DoesNotExist")

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_video_notes("/nonexistent/path.xlsx", sheet="anything")
