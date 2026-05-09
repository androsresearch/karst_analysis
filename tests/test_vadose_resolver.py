"""Tests for ``karst_analysis.sec.io.vadose_resolver``.

Exercises the three levels of the resolution policy independently and
together, plus the error and edge cases.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


# ──────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def wells_csv_with_campaign(tmp_path: Path) -> Path:
    """A wells.csv that uses the v11 schema: one row per (site, well_type, campaign)."""
    p = tmp_path / "wells.csv"
    pd.DataFrame({
        "site":               ["AW5", "AW6", "BW3", "LRS69", "LRS70", "AW6"],
        "well_type":          ["D",   "D",   "D",   "D",     "D",     "D"],
        "vadose_thickness_m": [0.64,  1.265, 3.28,  1.97,    0.83,    1.40],
        "reference_date":     ["2022-02-13"] * 5 + ["2023-06-15"],
        "campaign":           ["2022_02"] * 5 + ["2023_06"],
        "source":             ["extracted"] * 5 + ["manual"],
        "notes":              [""] * 6,
    }).to_csv(p, index=False)
    return p


@pytest.fixture
def wells_csv_v10_schema(tmp_path: Path) -> Path:
    """A wells.csv that mimics the v10 schema (no 'campaign' column).

    Used to verify backwards compatibility — the resolver should treat
    every row as belonging to the default fallback campaign.
    """
    p = tmp_path / "wells_v10.csv"
    pd.DataFrame({
        "site":               ["AW5", "AW6", "BW3", "LRS69", "LRS70"],
        "well_type":          ["D",   "D",   "D",   "D",     "D"],
        "vadose_thickness_m": [0.64,  1.265, 3.28,  1.97,    0.83],
        "reference_date":     ["2022-02-13"] * 5,
        "source":             ["extracted"] * 5,
        "notes":              [""] * 5,
    }).to_csv(p, index=False)
    return p


def _make_ysi_csv(path: Path, vp_values: np.ndarray, vadose: float) -> None:
    """Build a synthetic YSI CSV with both Vertical Position m and
    Depth from GL (m) so that extract_vadose_from_ysi_csv can recover
    the supplied vadose offset."""
    df = pd.DataFrame({
        "Vertical Position m": vp_values,
        "Depth from GL (m)":   vp_values + vadose,
        "SpCond µS/cm":        np.full_like(vp_values, 5000.0),
    })
    df.to_csv(path, index=False)


# ──────────────────────────────────────────────────────────────────────
#  Construction & schema validation
# ──────────────────────────────────────────────────────────────────────

def test_resolver_loads_wells_csv(wells_csv_with_campaign):
    """The resolver loads the metadata table and exposes it."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    r = VadoseResolver(metadata_csv_path=wells_csv_with_campaign)
    assert len(r.metadata) == 6
    assert "well_id" in r.metadata.columns
    assert "campaign" in r.metadata.columns


def test_resolver_raises_on_missing_file(tmp_path):
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver
    with pytest.raises(FileNotFoundError):
        VadoseResolver(metadata_csv_path=tmp_path / "does_not_exist.csv")


def test_resolver_raises_on_missing_required_column(tmp_path):
    """A wells.csv lacking 'vadose_thickness_m' must be rejected."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    p = tmp_path / "bad.csv"
    pd.DataFrame({"site": ["AW5"], "well_type": ["D"]}).to_csv(p, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        VadoseResolver(metadata_csv_path=p)


def test_resolver_handles_v10_schema_without_campaign_column(wells_csv_v10_schema):
    """If the loaded wells.csv has no 'campaign' column, every row is
    treated as belonging to the fallback campaign. This preserves
    backwards compatibility with the v10 schema."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    r = VadoseResolver(
        metadata_csv_path=wells_csv_v10_schema,
        fallback_campaign="2022_02",
    )
    res = r.resolve("AW6D", "2022_02")
    assert res.source == "explicit"
    assert res.thickness_m == pytest.approx(1.265)


# ──────────────────────────────────────────────────────────────────────
#  Level 1 — explicit lookup
# ──────────────────────────────────────────────────────────────────────

def test_resolve_explicit_hit_default_campaign(wells_csv_with_campaign):
    """A row that exists for the exact (well, campaign) is returned verbatim."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    r = VadoseResolver(metadata_csv_path=wells_csv_with_campaign)
    res = r.resolve("AW5D", "2022_02")

    assert res.source == "explicit"
    assert res.thickness_m == pytest.approx(0.64)
    assert res.well_id == "AW5D"
    assert res.campaign == "2022_02"
    assert res.fallback_campaign is None
    assert res.is_fallback is False
    assert "wells.csv" in res.note


def test_resolve_explicit_hit_non_default_campaign(wells_csv_with_campaign):
    """An explicit row for a NON-default campaign is preferred over the
    fallback. This is the key test for the v11 multi-campaign case."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    r = VadoseResolver(metadata_csv_path=wells_csv_with_campaign)
    res = r.resolve("AW6D", "2023_06")

    assert res.source == "explicit"
    # 1.40 from the explicit 2023_06 row, NOT 1.265 from the 2022_02 row.
    assert res.thickness_m == pytest.approx(1.40)
    assert res.is_fallback is False


# ──────────────────────────────────────────────────────────────────────
#  Level 2 — compute from CSV
# ──────────────────────────────────────────────────────────────────────

def test_resolve_computes_from_csv_when_no_explicit_row(
    wells_csv_with_campaign, tmp_path,
):
    """If the (well, campaign) is missing from wells.csv but the CSV
    itself has both depth columns, the resolver computes the offset."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    csv_path = tmp_path / "AW5_D_YSI_20240131.csv"
    _make_ysi_csv(csv_path, vp_values=np.linspace(0, 10, 100), vadose=1.10)

    r = VadoseResolver(metadata_csv_path=wells_csv_with_campaign)
    res = r.resolve("AW5D", "2024_01", csv_path=csv_path)

    assert res.source == "computed_from_csv"
    assert res.thickness_m == pytest.approx(1.10)
    assert res.is_fallback is False
    assert csv_path.name in res.note


def test_resolve_skips_csv_computation_when_csv_is_missing_columns(
    wells_csv_with_campaign, tmp_path,
):
    """If the CSV exists but lacks the GL column, the resolver moves on
    to the fallback level rather than failing."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    csv_path = tmp_path / "no_gl.csv"
    pd.DataFrame({
        "Vertical Position m": [0.0, 0.5, 1.0],
        "SpCond µS/cm":        [1000, 5000, 20000],
        # No "Depth from GL (m)" column.
    }).to_csv(csv_path, index=False)

    r = VadoseResolver(metadata_csv_path=wells_csv_with_campaign)
    res = r.resolve("AW5D", "2024_01", csv_path=csv_path)

    # Falls through to fallback (AW5D in 2022_02 = 0.64).
    assert res.source == "fallback"
    assert res.thickness_m == pytest.approx(0.64)


# ──────────────────────────────────────────────────────────────────────
#  Level 3 — fallback to reference campaign
# ──────────────────────────────────────────────────────────────────────

def test_resolve_falls_back_when_no_explicit_and_no_csv(wells_csv_with_campaign):
    """When neither an explicit row nor a CSV are available, the
    resolver falls back to the well's value in the reference campaign."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    r = VadoseResolver(metadata_csv_path=wells_csv_with_campaign)
    res = r.resolve("LRS70D", "2024_01")

    assert res.source == "fallback"
    assert res.thickness_m == pytest.approx(0.83)  # LRS70D in 2022_02
    assert res.fallback_campaign == "2022_02"
    assert res.is_fallback is True
    assert "fallback" in res.note.lower()


def test_resolve_fallback_uses_configured_campaign(wells_csv_with_campaign):
    """If the resolver was constructed with a different fallback campaign,
    that one is used as the safety net."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    # Use 2023_06 as fallback. AW6D has a row in 2023_06 (1.40), so a
    # query for ('AW6D', '2024_xx') should fall back to 1.40, not 1.265.
    r = VadoseResolver(
        metadata_csv_path=wells_csv_with_campaign,
        fallback_campaign="2023_06",
    )
    res = r.resolve("AW6D", "2024_xx")
    assert res.source == "fallback"
    assert res.thickness_m == pytest.approx(1.40)
    assert res.fallback_campaign == "2023_06"


# ──────────────────────────────────────────────────────────────────────
#  Error cases
# ──────────────────────────────────────────────────────────────────────

def test_resolve_raises_when_well_unknown_in_fallback(wells_csv_with_campaign):
    """If even the fallback campaign has no row for the requested well,
    the resolver raises rather than silently returning a wrong value."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    r = VadoseResolver(metadata_csv_path=wells_csv_with_campaign)
    with pytest.raises(KeyError, match="Cannot resolve vadose"):
        r.resolve("XX99D", "2024_01")


def test_resolve_treats_nan_as_missing(tmp_path):
    """A row with NaN in vadose_thickness_m must NOT count as an
    explicit hit — the resolver should fall through to the next level."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolver

    p = tmp_path / "wells_with_nan.csv"
    pd.DataFrame({
        "site":               ["AW5",   "AW5"],
        "well_type":          ["D",     "D"],
        "vadose_thickness_m": [0.64,    np.nan],   # second row has NaN
        "campaign":           ["2022_02", "2024_01"],
    }).to_csv(p, index=False)

    r = VadoseResolver(metadata_csv_path=p)
    res = r.resolve("AW5D", "2024_01")
    # The NaN row is ignored; falls back to 2022_02.
    assert res.source == "fallback"
    assert res.thickness_m == pytest.approx(0.64)


# ──────────────────────────────────────────────────────────────────────
#  Public API surface
# ──────────────────────────────────────────────────────────────────────

def test_resolution_dataclass_is_immutable():
    """VadoseResolution is frozen, so callers cannot mutate it by accident."""
    from karst_analysis.sec.io.vadose_resolver import VadoseResolution

    res = VadoseResolution(
        thickness_m=1.0, source="explicit",
        well_id="AW5D", campaign="2022_02",
        fallback_campaign=None, note="test",
    )
    with pytest.raises((AttributeError, Exception)):
        res.thickness_m = 2.0  # type: ignore[misc]
