"""Tests for ``karst_analysis.corrections.datum``.

Focus: the ``extract_vadose_from_ysi_csv`` function that was promoted
from ``scripts/extract_vadose_from_raw.py`` to the corrections package
in v10. The script's CLI is tested separately (it is a thin wrapper
around this function).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


PROJECT_ROOT = Path(__file__).parent.parent
HAS_DATA = (PROJECT_ROOT / "data" / "raw" / "sec" / "2022_02").exists()


# ──────────────────────────────────────────────────────────────────────
#  extract_vadose_from_ysi_csv  —  contract tests
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.skipif(not HAS_DATA, reason="raw SEC data not present")
def test_extract_vadose_matches_wells_csv():
    """The function reproduces wells.csv to 4 decimal places.

    This is the exact operation that originally populated wells.csv, so
    re-running it against the same raw files must yield the same values.
    """
    from karst_analysis.corrections import extract_vadose_from_ysi_csv

    wells_csv = PROJECT_ROOT / "data" / "metadata" / "wells.csv"
    metadata = pd.read_csv(wells_csv).set_index("site")

    raw_dir = PROJECT_ROOT / "data" / "raw" / "sec" / "2022_02"
    for f in raw_dir.glob("*_D_YSI_*.csv"):
        # Filename is like "AW6_D_YSI_20220219.csv" -> site = "AW6"
        site = f.name.split("_D_")[0]
        expected = float(metadata.loc[site, "vadose_thickness_m"])
        offset, status = extract_vadose_from_ysi_csv(f)
        assert offset is not None, f"{f.name}: offset returned None"
        assert status == "ok", f"{f.name}: status was {status!r}, expected 'ok'"
        assert abs(offset - expected) < 1e-4, (
            f"{f.name}: expected vadose {expected}, got {offset}"
        )


def test_extract_vadose_returns_no_gl_column_status(tmp_path):
    """Missing Depth-from-GL → (None, 'no_gl_column'), no exception."""
    from karst_analysis.corrections import extract_vadose_from_ysi_csv

    p = tmp_path / "no_gl.csv"
    pd.DataFrame({
        "Vertical Position m": [0.0, 0.5, 1.0],
        "SpCond µS/cm": [1000, 5000, 20000],
        # No "Depth from GL" column.
    }).to_csv(p, index=False)

    offset, status = extract_vadose_from_ysi_csv(p)
    assert offset is None
    assert status == "no_gl_column"


def test_extract_vadose_returns_no_vp_column_status(tmp_path):
    """Missing Vertical-Position → (None, 'no_vp_column'), no exception."""
    from karst_analysis.corrections import extract_vadose_from_ysi_csv

    p = tmp_path / "no_vp.csv"
    pd.DataFrame({
        "Depth from GL (m)": [0.5, 1.0, 1.5],
        "SpCond µS/cm": [1000, 5000, 20000],
        # No "Vertical Position m" column.
    }).to_csv(p, index=False)

    offset, status = extract_vadose_from_ysi_csv(p)
    assert offset is None
    assert status == "no_vp_column"


def test_extract_vadose_flags_inconsistent_offset(tmp_path):
    """If the row-wise offset is not constant (std > tolerance), the
    function returns the median but flags it via the status string.
    A constant 1.0 m offset would have std = 0, so we craft a CSV where
    the offset varies."""
    from karst_analysis.corrections import extract_vadose_from_ysi_csv

    p = tmp_path / "inconsistent.csv"
    # Vertical Position grows linearly; Depth from GL grows non-linearly,
    # so the offset (GL - VP) drifts substantially across the cast.
    n = 100
    vp = np.linspace(0.0, 10.0, n)
    drift = np.linspace(0.0, 0.5, n)  # half-metre drift over the cast
    gl = vp + 1.0 + drift             # nominal vadose 1.0 m, plus drift
    pd.DataFrame({
        "Vertical Position m": vp,
        "Depth from GL (m)": gl,
        "SpCond µS/cm": np.full(n, 5000),
    }).to_csv(p, index=False)

    offset, status = extract_vadose_from_ysi_csv(p)
    assert offset is not None, "should still return median even when inconsistent"
    assert status.startswith("inconsistent"), (
        f"expected 'inconsistent (...)' status, got {status!r}"
    )
    # Median of (1.0 + drift) where drift goes 0 → 0.5 is 1.25.
    assert abs(offset - 1.25) < 0.01


def test_extract_vadose_raises_on_missing_file(tmp_path):
    """Non-existent path must raise FileNotFoundError, not return None."""
    from karst_analysis.corrections import extract_vadose_from_ysi_csv

    with pytest.raises(FileNotFoundError):
        extract_vadose_from_ysi_csv(tmp_path / "does_not_exist.csv")


def test_extract_vadose_handles_column_aliases(tmp_path):
    """The function recognises the documented header variants."""
    from karst_analysis.corrections import extract_vadose_from_ysi_csv

    p = tmp_path / "aliases.csv"
    # Use "Vertical Position [m]" and "depth_bgl_m" — both in the variants list.
    pd.DataFrame({
        "Vertical Position [m]": [0.0, 0.5, 1.0],
        "depth_bgl_m": [0.8, 1.3, 1.8],   # constant offset = 0.8
    }).to_csv(p, index=False)

    offset, status = extract_vadose_from_ysi_csv(p)
    assert status == "ok"
    assert abs(offset - 0.8) < 1e-6


# ──────────────────────────────────────────────────────────────────────
#  Smoke test that the symbol is wired through the package __init__
# ──────────────────────────────────────────────────────────────────────

def test_extract_vadose_is_exported():
    """The function is reachable from the corrections sub-package."""
    from karst_analysis.corrections import extract_vadose_from_ysi_csv  # noqa: F401
    from karst_analysis import corrections
    assert "extract_vadose_from_ysi_csv" in corrections.__all__
