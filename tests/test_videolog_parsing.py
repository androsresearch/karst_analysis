"""Unit tests for videolog parsing helpers."""

from __future__ import annotations

import pandas as pd
import pytest

from karst_analysis.videolog.parsing import (
    parse_depth_token, apply_typo_fixes, TYPO_FIXES,
)


class TestParseDepthToken:
    def test_simple_float(self):
        assert parse_depth_token("1.5") == (1.5, 1.5)

    def test_with_m_suffix(self):
        assert parse_depth_token("1.5 m") == (1.5, 1.5)
        assert parse_depth_token("1.5m") == (1.5, 1.5)
        assert parse_depth_token("1.5 M") == (1.5, 1.5)

    def test_range_hyphen(self):
        assert parse_depth_token("1.5-2.0") == (1.5, 2.0)

    def test_range_en_dash(self):
        assert parse_depth_token("1.5–2.0") == (1.5, 2.0)

    def test_range_em_dash(self):
        assert parse_depth_token("1.5—2.0") == (1.5, 2.0)

    def test_range_with_spaces(self):
        assert parse_depth_token("  1.5 - 2.0  ") == (1.5, 2.0)

    def test_numeric_input(self):
        assert parse_depth_token(3.7) == (3.7, 3.7)
        assert parse_depth_token(3) == (3.0, 3.0)

    def test_empty_string(self):
        assert parse_depth_token("") == (None, None)
        assert parse_depth_token("   ") == (None, None)

    def test_none_input(self):
        assert parse_depth_token(None) == (None, None)

    def test_nan_input(self):
        import numpy as np
        assert parse_depth_token(float("nan")) == (None, None)

    def test_unparseable(self):
        assert parse_depth_token("abc") == (None, None)
        assert parse_depth_token("1.5 to 2.0") == (None, None)

    def test_integer_string(self):
        assert parse_depth_token("12") == (12.0, 12.0)


class TestApplyTypoFixes:
    def test_known_typo(self):
        assert apply_typo_fixes("This is occuring") == "This is occurring"

    def test_multiple_typos(self):
        text = "Voids beome occuring at Botom"
        out = apply_typo_fixes(text)
        assert "become" in out
        assert "occurring" in out
        assert "Bottom" in out

    def test_strips_whitespace(self):
        assert apply_typo_fixes("  hello  ") == "hello"

    def test_no_typos_unchanged(self):
        assert apply_typo_fixes("Plain text without errors.") == "Plain text without errors."

    def test_typo_list_is_list_of_tuples(self):
        """Sanity check: TYPO_FIXES has the expected shape."""
        assert isinstance(TYPO_FIXES, list)
        for entry in TYPO_FIXES:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
            assert isinstance(entry[0], str)
            assert isinstance(entry[1], str)
