"""End-to-end tests for the ERT 1D loader, well mapping, and filename
parsing.

Uses ``data/raw/ert/T16/1D/viz_sharp_x_160.csv`` as the live fixture
when present. When the file is absent (e.g. fresh checkout without
data), the live-fixture tests are skipped — the filename-parsing and
edge-case tests still run on synthetic inputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from karst_analysis.ert.io import (
    ErtTrace1D,
    ErtWellAssoc,
    ErtWellMap,
    REQUIRED_TRACE_COLUMNS,
    load_ert_1d_csv,
    load_ert_1d_traces,
    parse_ert_filename,
)


ROOT = Path(__file__).resolve().parent.parent
ERT_T16_DIR = ROOT / "data" / "raw" / "ert" / "T16" / "1D"
VIZ_SHARP_CSV = ERT_T16_DIR / "viz_sharp_x_160.csv"


# ════════════════════════════════════════════════════════════════════
# Pure-string filename parsing — no data needed
# ════════════════════════════════════════════════════════════════════
class TestParseErtFilename:
    @pytest.mark.parametrize("name,variant,x", [
        ("viz_sharp_x_160.csv", "viz_sharp", 160.0),
        ("tier3_robust_blocky_x_160.csv", "tier3_robust_blocky", 160.0),
        ("error_weighted_x_160.csv", "error_weighted", 160.0),
        ("fine_mesh_x_172.5.csv", "fine_mesh", 172.5),
        ("a_x_b_x_42.csv", "a_x_b", 42.0),  # multiple "_x_": last one wins
    ])
    def test_valid_names(self, name, variant, x):
        v, xv = parse_ert_filename(name)
        assert v == variant
        assert xv == x

    @pytest.mark.parametrize("bad", [
        "no_underscore_x.csv",
        "viz_sharp_160.csv",        # missing "_x_"
        "viz_sharp_x_abc.csv",      # x not numeric
        "viz_sharp_x_.csv",         # x missing
    ])
    def test_invalid_names_raise(self, bad):
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_ert_filename(bad)


# ════════════════════════════════════════════════════════════════════
# Live-fixture tests — require the viz_sharp CSV to be present
# ════════════════════════════════════════════════════════════════════
@pytest.mark.skipif(
    not VIZ_SHARP_CSV.is_file(),
    reason=f"ERT fixture not present at {VIZ_SHARP_CSV}",
)
class TestLoadErtSingleCsv:
    def test_returns_ert_trace1d(self):
        t = load_ert_1d_csv(VIZ_SHARP_CSV)
        assert isinstance(t, ErtTrace1D)

    def test_metadata_inferred_from_path(self):
        t = load_ert_1d_csv(VIZ_SHARP_CSV)
        assert t.transect == "T16"
        assert t.variant == "viz_sharp"
        assert t.x_requested == 160.0

    def test_x_extracted_close_to_requested(self):
        t = load_ert_1d_csv(VIZ_SHARP_CSV)
        assert abs(t.x_extracted - t.x_requested) < 1.0  # mesh discretisation

    def test_required_columns_present(self):
        t = load_ert_1d_csv(VIZ_SHARP_CSV)
        for col in REQUIRED_TRACE_COLUMNS:
            assert col in t.df.columns

    def test_depth_is_positive_and_sorted(self):
        t = load_ert_1d_csv(VIZ_SHARP_CSV)
        assert (t.df["depth_bgl_m"] >= 0).all()
        assert t.df["depth_bgl_m"].is_monotonic_increasing

    def test_resistlog10_consistent_with_resist(self):
        t = load_ert_1d_csv(VIZ_SHARP_CSV)
        np.testing.assert_allclose(
            t.df["resistlog10"],
            np.log10(t.df["resist_ohm_m"]),
            atol=1e-9,
        )

    def test_explicit_transect_overrides_inference(self, tmp_path):
        # Copy the file outside the canonical layout to verify the
        # explicit transect= kwarg.
        import shutil
        flat_path = tmp_path / "viz_sharp_x_160.csv"
        shutil.copy(VIZ_SHARP_CSV, flat_path)
        t = load_ert_1d_csv(flat_path, transect="T_custom")
        assert t.transect == "T_custom"

    def test_path_outside_layout_without_transect_raises(self, tmp_path):
        import shutil
        flat_path = tmp_path / "viz_sharp_x_160.csv"
        shutil.copy(VIZ_SHARP_CSV, flat_path)
        with pytest.raises(ValueError, match="Cannot infer transect"):
            load_ert_1d_csv(flat_path)


@pytest.mark.skipif(
    not ERT_T16_DIR.is_dir() or not VIZ_SHARP_CSV.is_file(),
    reason=f"ERT T16 directory not present at {ERT_T16_DIR}",
)
class TestLoadErtTracesPerTransect:
    def test_finds_all_variants(self):
        traces = load_ert_1d_traces("T16", project_root=ROOT)
        # We expect 5 variants if all uploaded fixtures are present.
        assert len(traces) >= 1
        variants = {t.variant for t in traces}
        assert "viz_sharp" in variants

    def test_sorted_by_x_then_variant(self):
        traces = load_ert_1d_traces("T16", project_root=ROOT)
        keys = [(t.x_requested, t.variant) for t in traces]
        assert keys == sorted(keys)

    def test_x_filter_match(self):
        traces = load_ert_1d_traces("T16", project_root=ROOT, x_filter=160.0)
        assert len(traces) >= 1
        assert all(t.x_requested == 160.0 for t in traces)

    def test_x_filter_no_match_raises_with_available(self):
        with pytest.raises(ValueError, match="Available x values"):
            load_ert_1d_traces("T16", project_root=ROOT, x_filter=999.0)

    def test_variant_filter(self):
        traces = load_ert_1d_traces("T16", project_root=ROOT,
                                     variant_filter=["viz_sharp"])
        assert all(t.variant == "viz_sharp" for t in traces)

    def test_missing_transect_raises(self, tmp_path):
        # Build an empty project_root so we trigger the FileNotFoundError.
        with pytest.raises(FileNotFoundError, match="ERT transect"):
            load_ert_1d_traces("T_does_not_exist", project_root=tmp_path)


# ════════════════════════════════════════════════════════════════════
# ErtWellMap — pure CSV roundtrip, no live fixture needed
# ════════════════════════════════════════════════════════════════════
class TestErtWellMap:
    def _write_csv(self, tmp_path, content: str) -> Path:
        p = tmp_path / "ert_wells.csv"
        p.write_text(content, encoding="utf-8")
        return p

    def test_roundtrip(self, tmp_path):
        path = self._write_csv(tmp_path,
            "well_id,transect,x,notes\n"
            "LRS70D,T16,160.0,\n"
            "AW6D,T8,42.5,near drilling\n"
        )
        m = ErtWellMap.from_csv(path)
        assert len(m) == 2

    def test_transects_for_well(self, tmp_path):
        path = self._write_csv(tmp_path,
            "well_id,transect,x,notes\n"
            "LRS70D,T16,160.0,\n"
            "LRS70D,T17,80.0,second line\n"
            "AW6D,T8,42.5,\n"
        )
        m = ErtWellMap.from_csv(path)
        results = m.transects_for_well("LRS70D")
        assert len(results) == 2
        transects = {a.transect for a in results}
        assert transects == {"T16", "T17"}

    def test_wells_for_transect(self, tmp_path):
        path = self._write_csv(tmp_path,
            "well_id,transect,x,notes\n"
            "LRS70D,T16,160.0,\n"
            "AW6D,T16,200.0,\n"
        )
        m = ErtWellMap.from_csv(path)
        results = m.wells_for_transect("T16")
        assert len(results) == 2
        wells = {a.well_id for a in results}
        assert wells == {"LRS70D", "AW6D"}

    def test_unknown_well_returns_empty(self, tmp_path):
        path = self._write_csv(tmp_path,
            "well_id,transect,x,notes\n"
            "LRS70D,T16,160.0,\n"
        )
        m = ErtWellMap.from_csv(path)
        assert m.transects_for_well("XYZ") == []

    def test_notes_column_optional(self, tmp_path):
        # No 'notes' column at all — should default to empty string.
        path = self._write_csv(tmp_path,
            "well_id,transect,x\n"
            "LRS70D,T16,160.0\n"
        )
        m = ErtWellMap.from_csv(path)
        assert m.transects_for_well("LRS70D")[0].notes == ""

    def test_missing_required_column_raises(self, tmp_path):
        path = self._write_csv(tmp_path,
            "well_id,transect,notes\n"
            "LRS70D,T16,\n"
        )
        with pytest.raises(ValueError, match="missing required columns"):
            ErtWellMap.from_csv(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ErtWellMap.from_csv(tmp_path / "nope.csv")
