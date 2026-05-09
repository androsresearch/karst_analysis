"""End-to-end validation of the caliper pipeline against the golden outputs.

These tests run the full migrated pipeline on
``data/raw/caliper/concatenate_caliper_all.csv`` and verify that the
outputs match the golden references in ``tests/golden/`` bit-for-bit
(within 1e-9 absolute tolerance for floats; exact equality for strings).

If any of these tests fail, the migration introduced a regression and
must be debugged before continuing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from karst_analysis.caliper.io import load_master_caliper
from karst_analysis.caliper.noise import estimate_noise_aw5o_vs_aw5d
from karst_analysis.caliper.pipeline import (
    process_many_wells, perpoint_dataframe, zones_dataframe,
)


# Paths
ROOT = Path(__file__).resolve().parent.parent
MASTER_CSV = ROOT / "data" / "raw" / "caliper" / "concatenate_caliper_all.csv"
GOLDEN_DIR = ROOT / "tests" / "golden"
GOLDEN_NOISE = GOLDEN_DIR / "noise_comparison.json"
GOLDEN_PERPOINT = GOLDEN_DIR / "priority_wells_cumulative_min_v2_perpoint.csv"
GOLDEN_ZONES = GOLDEN_DIR / "priority_wells_cumulative_min_v2_zones.csv"

ATOL = 1e-9


# ──────────────────────────────────────────────────────────────────────
#  Module-scoped fixtures: compute the migrated outputs once
# ──────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def df_master():
    if not MASTER_CSV.exists():
        pytest.skip(f"Master caliper CSV not found at {MASTER_CSV}")
    return load_master_caliper(MASTER_CSV)


@pytest.fixture(scope="module")
def noise_report(df_master):
    return estimate_noise_aw5o_vs_aw5d(df_master)


@pytest.fixture(scope="module")
def sigma_inst(noise_report):
    return noise_report["AW5O"]["sigma_MAD_cm"]


@pytest.fixture(scope="module")
def results(df_master, sigma_inst):
    return process_many_wells(df_master, sigma_inst)


@pytest.fixture(scope="module")
def perpoint_df(results):
    return perpoint_dataframe(results)


@pytest.fixture(scope="module")
def zones_df(results):
    return zones_dataframe(results)


# ──────────────────────────────────────────────────────────────────────
#  Tests
# ──────────────────────────────────────────────────────────────────────
def test_noise_json_matches_golden(noise_report):
    """The migrated noise estimation must reproduce noise_comparison.json
    bit-for-bit (within 1e-12)."""
    if not GOLDEN_NOISE.exists():
        pytest.skip(f"Golden file not found at {GOLDEN_NOISE}")
    with open(GOLDEN_NOISE) as f:
        golden = json.load(f)

    def _walk(a, b, path=""):
        for k in set(a) | set(b):
            full = f"{path}.{k}" if path else k
            assert k in a, f"Missing in result: {full}"
            assert k in b, f"Missing in golden: {full}"
            va, vb = a[k], b[k]
            if isinstance(va, dict):
                _walk(va, vb, full)
            elif isinstance(va, list):
                assert va == vb, f"List mismatch at {full}: {va} vs {vb}"
            elif isinstance(va, float):
                assert np.isclose(va, vb, atol=1e-12, rtol=0), (
                    f"Float mismatch at {full}: {va} vs {vb} "
                    f"(delta {va - vb:.2e})"
                )
            else:
                assert va == vb, f"Mismatch at {full}: {va!r} vs {vb!r}"

    _walk(noise_report, golden)


def test_perpoint_csv_matches_golden(perpoint_df):
    """The migrated per-sample CSV must equal the golden one.

    Note on tolerance: the golden CSV was saved by the original
    ``export_perpoint_breakouts.py`` with ``float_format="%.4f"``,
    so its float values are rounded to 4 decimals. The migrated
    pipeline keeps full precision. We therefore use a tolerance of
    5e-5 (half of the last decimal at 4 places). When the migrated
    DataFrame is saved with the same ``float_format="%.4f"`` and
    reloaded, the diff drops to 0.00e+00 — confirming this is purely
    a serialisation difference, not an algorithmic regression.
    """
    if not GOLDEN_PERPOINT.exists():
        pytest.skip(f"Golden file not found at {GOLDEN_PERPOINT}")
    golden = pd.read_csv(GOLDEN_PERPOINT)

    # Tolerance accounting for golden's 4-decimal rounding
    PERPOINT_ATOL = 5e-5

    assert list(perpoint_df.columns) == list(golden.columns), (
        f"Column order differs.\nresult: {list(perpoint_df.columns)}\n"
        f"golden: {list(golden.columns)}"
    )
    assert len(perpoint_df) == len(golden), (
        f"Row count differs: result={len(perpoint_df)} golden={len(golden)}"
    )

    for col in perpoint_df.columns:
        if perpoint_df[col].dtype.kind in "fc":
            max_abs = float(np.nanmax(np.abs(
                perpoint_df[col].values - golden[col].values
            )))
            assert max_abs <= PERPOINT_ATOL, (
                f"Column {col!r} differs by up to {max_abs:.3e} (atol={PERPOINT_ATOL})"
            )
        else:
            assert (perpoint_df[col].values == golden[col].values).all(), (
                f"Column {col!r} (string/object) differs"
            )


def test_perpoint_csv_matches_golden_after_4f_roundtrip(perpoint_df, tmp_path):
    """Stronger version: save migrated output with the same float_format
    as the original, then compare bit-for-bit (atol=0)."""
    if not GOLDEN_PERPOINT.exists():
        pytest.skip(f"Golden file not found at {GOLDEN_PERPOINT}")
    golden = pd.read_csv(GOLDEN_PERPOINT)

    # Save migrated with same format as original, then reload
    tmp_csv = tmp_path / "perpoint_4f.csv"
    perpoint_df.to_csv(tmp_csv, index=False, float_format="%.4f")
    reloaded = pd.read_csv(tmp_csv)

    for col in reloaded.columns:
        if reloaded[col].dtype.kind in "fc":
            max_abs = float(np.nanmax(np.abs(
                reloaded[col].values - golden[col].values
            )))
            assert max_abs == 0.0, (
                f"Column {col!r} differs by {max_abs:.3e} after 4f roundtrip "
                f"(should be exact)"
            )
        else:
            assert (reloaded[col].values == golden[col].values).all(), (
                f"Column {col!r} differs after roundtrip"
            )


def test_zones_csv_matches_golden(zones_df):
    """The migrated zones CSV must equal the golden one within 1e-9."""
    if not GOLDEN_ZONES.exists():
        pytest.skip(f"Golden file not found at {GOLDEN_ZONES}")
    golden = pd.read_csv(GOLDEN_ZONES)

    assert list(zones_df.columns) == list(golden.columns), (
        f"Column order differs.\nresult: {list(zones_df.columns)}\n"
        f"golden: {list(golden.columns)}"
    )
    assert len(zones_df) == len(golden), (
        f"Row count differs: result={len(zones_df)} golden={len(golden)}"
    )

    for col in zones_df.columns:
        if zones_df[col].dtype.kind in "fc":
            max_abs = float(np.nanmax(np.abs(
                zones_df[col].values - golden[col].values
            )))
            assert max_abs <= ATOL, (
                f"Column {col!r} differs by up to {max_abs:.3e} (atol={ATOL})"
            )
        else:
            assert (zones_df[col].values == golden[col].values).all(), (
                f"Column {col!r} (string/object) differs"
            )


def test_severity_counts_match_golden(perpoint_df):
    """Sanity check: per-well severity histograms reproduce the
    documented counts in the migration plan."""
    if not GOLDEN_PERPOINT.exists():
        pytest.skip(f"Golden file not found at {GOLDEN_PERPOINT}")
    expected = {
        "AW5D":   {"none": 846, "mild": 54,  "moderate": 0,   "severe": 0},
        "AW6D":   {"none": 793, "mild": 90,  "moderate": 15,  "severe": 0},
        "BW3D":   {"none": 606, "mild": 314, "moderate": 71,  "severe": 11},
        "LRS69D": {"none": 618, "mild": 221, "moderate": 26,  "severe": 0},
        "LRS70D": {"none": 449, "mild": 458, "moderate": 183, "severe": 70},
    }
    counts = perpoint_df.groupby(["well", "severity_per_sample"]).size().unstack(fill_value=0)
    for well, expected_counts in expected.items():
        for sev, n in expected_counts.items():
            actual = int(counts.loc[well].get(sev, 0))
            assert actual == n, (
                f"{well} severity={sev}: expected {n}, got {actual}"
            )
