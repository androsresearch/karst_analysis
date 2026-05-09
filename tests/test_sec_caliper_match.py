"""Tests for sec_caliper_match (Idea 3, SEC ↔ caliper convergence).

Covers: geometric primitives; matching rules; best-match strategies;
scoring; full pipeline orchestration; config-driven behaviour.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

# Import the module directly. We avoid `from karst_analysis.convergence
# import ...` because that triggers loading of unrelated submodules
# (sec_caliper_video) that pull plotly / ipywidgets via the sec.viz
# package — those are present in the project's dev env but irrelevant
# to these tests, and we don't want them to gate the suite.
from karst_analysis.convergence.sec_caliper_match import (
    compute_overlap_m,
    compute_center_distance_m,
    find_matches,
    select_best_match,
    score_cluster,
    compute_convergence,
)


# ──────────────────────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────────────────────
DEFAULT_CFG = {
    "sec_agreement_min": 3,
    "caliper_severity_filter": None,
    "matching_rule": "hybrid",
    "tolerance_m": 0.5,
    "best_match_priority": "overlap_then_distance",
    "severity_weights": {"mild": 1, "moderate": 2, "severe": 3},
    "unmatched_zones_min_severity": "moderate",
    "run_tag": "test_v1",
}


@pytest.fixture
def cfg():
    return deepcopy(DEFAULT_CFG)


@pytest.fixture
def cluster_row():
    """Synthetic cluster: well X, centre 12 m, span 10–14 m, agreement 9."""
    return pd.Series({
        "well_id": "X",
        "cluster_id": 0,
        "depth_median": 12.0,
        "depth_min": 10.0,
        "depth_max": 14.0,
        "agreement": 9,
        "persistence": 18,
        "cluster_diameter_m": 4.0,
        "wide_flag": True,
    })


@pytest.fixture
def zones_df():
    """
    Three caliper zones for well X:
      0 — severe, 13–16 m (overlaps cluster by 1 m).
      1 — mild, 12.1–12.4 m (fully inside cluster, tiny zone).
      2 — moderate, 29–30 m (far from cluster).
    Plus zone 3 for well Y, irrelevant when filtering by well.
    """
    return pd.DataFrame([
        {"well_id": "X", "zone_id": 0, "z_top": 16.0, "z_bot": 13.0,
         "z_centre": 14.5, "thickness_m": 3.0, "severity": "severe",
         "peak_excess_cm": 10.0},
        {"well_id": "X", "zone_id": 1, "z_top": 12.4, "z_bot": 12.1,
         "z_centre": 12.25, "thickness_m": 0.3, "severity": "mild",
         "peak_excess_cm": 2.0},
        {"well_id": "X", "zone_id": 2, "z_top": 30.0, "z_bot": 29.0,
         "z_centre": 29.5, "thickness_m": 1.0, "severity": "moderate",
         "peak_excess_cm": 5.0},
        {"well_id": "Y", "zone_id": 3, "z_top": 12.5, "z_bot": 11.5,
         "z_centre": 12.0, "thickness_m": 1.0, "severity": "severe",
         "peak_excess_cm": 8.0},
    ])


# ──────────────────────────────────────────────────────────────────────
#  Geometric primitives
# ──────────────────────────────────────────────────────────────────────
class TestOverlap:
    def test_partial_overlap(self):
        assert compute_overlap_m(10, 14, 13, 16) == 1.0

    def test_disjoint_returns_zero(self):
        assert compute_overlap_m(10, 14, 18, 20) == 0.0

    def test_touching_returns_zero(self):
        assert compute_overlap_m(10, 14, 14, 18) == 0.0

    def test_zone_inside_cluster(self):
        assert compute_overlap_m(10, 14, 11, 13) == 2.0

    def test_cluster_inside_zone(self):
        assert compute_overlap_m(11, 13, 10, 14) == 2.0

    def test_robust_to_swapped_endpoints(self):
        # Same intervals, opposite order
        a = compute_overlap_m(14, 10, 16, 13)
        b = compute_overlap_m(10, 14, 13, 16)
        assert a == b == 1.0


class TestCenterDistance:
    def test_basic(self):
        assert compute_center_distance_m(12.0, 15.0) == 3.0

    def test_symmetric(self):
        assert compute_center_distance_m(15.0, 12.0) == 3.0

    def test_zero(self):
        assert compute_center_distance_m(10.0, 10.0) == 0.0


# ──────────────────────────────────────────────────────────────────────
#  find_matches
# ──────────────────────────────────────────────────────────────────────
class TestFindMatches:
    def test_hybrid_rule_picks_overlapping_and_close(self, cluster_row, zones_df):
        # Hybrid with tol=0.5: zones 0 (overlap=1) and 1 (overlap=0.3) match;
        # zone 2 is far away.
        m = find_matches(cluster_row, zones_df[zones_df.well_id == "X"],
                         matching_rule="hybrid", tolerance_m=0.5)
        assert sorted(m["zone_id"].tolist()) == [0, 1]

    def test_overlap_only_excludes_distance_only_neighbours(self, cluster_row, zones_df):
        # Imagine a zone with 0 overlap but centre 0.4 m away.
        # In overlap_only mode, it should NOT match.
        zone_extra = pd.DataFrame([{
            "well_id": "X", "zone_id": 99, "z_top": 14.5, "z_bot": 14.2,
            "z_centre": 14.35, "thickness_m": 0.3, "severity": "mild",
            "peak_excess_cm": 1.5,
        }])
        zw = pd.concat([zones_df[zones_df.well_id == "X"], zone_extra],
                       ignore_index=True)
        m = find_matches(cluster_row, zw,
                         matching_rule="overlap_only", tolerance_m=0.5)
        assert 99 not in m["zone_id"].tolist()

    def test_center_distance_only_excludes_overlap_with_far_centre(
        self, cluster_row, zones_df,
    ):
        # The severe zone has overlap=1 but its centre is 14.5, distance=2.5.
        # In center_distance_only with tol=0.5, it should NOT match.
        m = find_matches(
            cluster_row, zones_df[zones_df.well_id == "X"],
            matching_rule="center_distance_only", tolerance_m=0.5,
        )
        assert 0 not in m["zone_id"].tolist()
        # The mild zone (centre 12.25) is within 0.5 m → matches.
        assert 1 in m["zone_id"].tolist()

    def test_unknown_rule_raises(self, cluster_row, zones_df):
        with pytest.raises(ValueError, match="Unknown matching_rule"):
            find_matches(cluster_row, zones_df[zones_df.well_id == "X"],
                         matching_rule="bogus", tolerance_m=0.5)

    def test_empty_zones_returns_empty(self, cluster_row, zones_df):
        empty = zones_df.iloc[0:0]
        m = find_matches(cluster_row, empty,
                         matching_rule="hybrid", tolerance_m=0.5)
        assert m.empty
        # Schema preservation
        for col in ("overlap_m", "center_distance_m"):
            assert col in m.columns

    def test_does_not_mutate_inputs(self, cluster_row, zones_df):
        z_copy = zones_df.copy(deep=True)
        _ = find_matches(cluster_row, zones_df,
                         matching_rule="hybrid", tolerance_m=0.5)
        pd.testing.assert_frame_equal(zones_df, z_copy)


# ──────────────────────────────────────────────────────────────────────
#  select_best_match
# ──────────────────────────────────────────────────────────────────────
class TestSelectBestMatch:
    def test_overlap_then_distance(self, cfg, cluster_row, zones_df):
        m = find_matches(cluster_row, zones_df[zones_df.well_id == "X"],
                         matching_rule="hybrid", tolerance_m=0.5)
        best = select_best_match(
            m, priority="overlap_then_distance",
            severity_weights=cfg["severity_weights"],
        )
        # severe zone has overlap=1 vs mild zone overlap=0.3 → severe wins.
        assert best["zone_id"] == 0

    def test_max_severity(self, cfg, cluster_row, zones_df):
        m = find_matches(cluster_row, zones_df[zones_df.well_id == "X"],
                         matching_rule="hybrid", tolerance_m=0.5)
        best = select_best_match(
            m, priority="max_severity",
            severity_weights=cfg["severity_weights"],
        )
        assert best["zone_id"] == 0   # severe outranks mild

    def test_max_peak_excess(self, cfg, cluster_row, zones_df):
        m = find_matches(cluster_row, zones_df[zones_df.well_id == "X"],
                         matching_rule="hybrid", tolerance_m=0.5)
        best = select_best_match(
            m, priority="max_peak_excess",
            severity_weights=cfg["severity_weights"],
        )
        # severe zone peak=10.0 vs mild peak=2.0 → severe wins.
        assert best["zone_id"] == 0

    def test_returns_none_on_empty(self, cfg, zones_df):
        empty = zones_df.iloc[0:0]
        # Need overlap/center_distance columns to be present in the schema.
        empty = empty.assign(overlap_m=np.array([], dtype=float),
                             center_distance_m=np.array([], dtype=float))
        out = select_best_match(
            empty, priority="overlap_then_distance",
            severity_weights=cfg["severity_weights"],
        )
        assert out is None

    def test_unknown_priority_raises(self, cfg, cluster_row, zones_df):
        m = find_matches(cluster_row, zones_df[zones_df.well_id == "X"],
                         matching_rule="hybrid", tolerance_m=0.5)
        with pytest.raises(ValueError, match="Unknown best_match_priority"):
            select_best_match(m, priority="bogus",
                              severity_weights=cfg["severity_weights"])


# ──────────────────────────────────────────────────────────────────────
#  score_cluster
# ──────────────────────────────────────────────────────────────────────
class TestScoreCluster:
    def test_score_with_severe_match(self, cfg, cluster_row, zones_df):
        m = find_matches(cluster_row, zones_df[zones_df.well_id == "X"],
                         matching_rule="hybrid", tolerance_m=0.5)
        s = score_cluster(cluster_row, m,
                          severity_weights=cfg["severity_weights"])
        assert s["has_caliper_match"] is True
        assert s["n_caliper_zones_matched"] == 2
        assert s["multi_match_flag"] is True
        assert s["max_matched_severity"] == "severe"
        # convergence_score = agreement (9) × max_weight (severe=3) = 27
        assert s["convergence_score"] == 27.0
        assert s["all_matched_severities"] == "mild,severe"

    def test_score_with_no_match(self, cfg, cluster_row, zones_df):
        empty = zones_df.iloc[0:0].assign(
            overlap_m=np.array([], dtype=float),
            center_distance_m=np.array([], dtype=float),
        )
        s = score_cluster(cluster_row, empty,
                          severity_weights=cfg["severity_weights"])
        assert s["has_caliper_match"] is False
        assert s["n_caliper_zones_matched"] == 0
        assert s["multi_match_flag"] is False
        assert s["max_matched_severity"] is None
        assert s["convergence_score"] == 0.0
        assert s["all_matched_severities"] == ""


# ──────────────────────────────────────────────────────────────────────
#  compute_convergence (full orchestration)
# ──────────────────────────────────────────────────────────────────────
class TestComputeConvergence:
    def test_filters_low_agreement_clusters(self, cfg, zones_df):
        clusters = pd.DataFrame([
            # agreement 9 → kept
            {"well_id": "X", "cluster_id": 0, "depth_median": 12.0,
             "depth_min": 10.0, "depth_max": 14.0, "agreement": 9,
             "persistence": 18, "cluster_diameter_m": 4.0, "wide_flag": True},
            # agreement 1 → dropped
            {"well_id": "X", "cluster_id": 1, "depth_median": 25.0,
             "depth_min": 25.0, "depth_max": 25.0, "agreement": 1,
             "persistence": 1, "cluster_diameter_m": 0.0, "wide_flag": False},
            # agreement 3 → kept
            {"well_id": "X", "cluster_id": 2, "depth_median": 29.5,
             "depth_min": 29.0, "depth_max": 30.0, "agreement": 3,
             "persistence": 6, "cluster_diameter_m": 1.0, "wide_flag": False},
        ])
        res = compute_convergence(clusters, zones_df, config=cfg)
        # Only clusters 0 and 2 should survive the filter.
        assert sorted(res.cluster_matches["cluster_id"].tolist()) == [0, 2]

    def test_well_partitioning(self, cfg, zones_df):
        # A cluster in well X must NOT match zones from well Y, even if
        # depth-wise they are close.
        clusters = pd.DataFrame([{
            "well_id": "X", "cluster_id": 0, "depth_median": 12.0,
            "depth_min": 11.5, "depth_max": 12.5, "agreement": 9,
            "persistence": 18, "cluster_diameter_m": 1.0, "wide_flag": False,
        }])
        # Zone 3 is in well Y at centre 12.0 — would match if well filter failed.
        res = compute_convergence(clusters, zones_df, config=cfg)
        # The X cluster matches zone 1 (mild, centre 12.25, distance 0.25)
        # and possibly zone 0 (severe, severe centre 14.5 → distance 2.5,
        # overlap=0, NOT a match in hybrid). Should not include Y zones.
        ids = set()
        for s in res.cluster_matches["all_matched_severities"]:
            for x in s.split(","):
                if x:
                    ids.add(x)
        # Y zone is severe — but if it matched we would also see it. The
        # X cluster has only zone 1 (mild) within range. Confirm exactly:
        assert res.cluster_matches.iloc[0]["n_caliper_zones_matched"] == 1
        assert res.cluster_matches.iloc[0]["max_matched_severity"] == "mild"

    def test_well_summary_aggregation(self, cfg, cluster_row, zones_df):
        clusters = pd.DataFrame([cluster_row])
        res = compute_convergence(clusters, zones_df, config=cfg)
        s = res.well_summary
        assert len(s) == 1
        assert s.iloc[0]["well_id"] == "X"
        assert s.iloc[0]["n_clusters_analyzed"] == 1
        assert s.iloc[0]["n_converging"] == 1
        assert s.iloc[0]["fraction_converging"] == 1.0
        assert s.iloc[0]["max_convergence_score"] == 27.0

    def test_unmatched_zones_filter(self, cfg, cluster_row, zones_df):
        clusters = pd.DataFrame([cluster_row])
        # With min_severity="moderate", only the moderate zone (id=2,
        # well X, far from cluster) and the severe zone (id=3, well Y,
        # never even considered) should appear.
        res = compute_convergence(clusters, zones_df, config=cfg)
        u = res.unmatched_zones
        # Zone 2 (moderate) and zone 3 (severe, well Y) — both unmatched.
        assert sorted(u["severity"].tolist()) == ["moderate", "severe"]
        assert "mild" not in set(u["severity"])

    def test_unmatched_min_severity_null_keeps_all(
        self, cfg, cluster_row, zones_df,
    ):
        cfg2 = deepcopy(cfg)
        cfg2["unmatched_zones_min_severity"] = None
        clusters = pd.DataFrame([cluster_row])
        res = compute_convergence(clusters, zones_df, config=cfg2)
        # Now mild zones that weren't matched should also appear. In our
        # fixture every mild zone IS matched, so this just confirms the
        # filter doesn't crash and includes lower-severity rows when
        # they are unmatched.
        assert "moderate" in set(res.unmatched_zones["severity"])

    def test_caliper_severity_filter_excludes_mild(
        self, cfg, cluster_row, zones_df,
    ):
        cfg2 = deepcopy(cfg)
        cfg2["caliper_severity_filter"] = ["moderate", "severe"]
        clusters = pd.DataFrame([cluster_row])
        res = compute_convergence(clusters, zones_df, config=cfg2)
        # Mild zone (id=1) should not be in match set.
        row = res.cluster_matches.iloc[0]
        assert "mild" not in row["all_matched_severities"]
        # Now only severe zone matches → score = 9 × 3 = 27 still
        assert row["convergence_score"] == 27.0
        assert row["n_caliper_zones_matched"] == 1

    def test_does_not_mutate_inputs(self, cfg, cluster_row, zones_df):
        clusters = pd.DataFrame([cluster_row])
        zc = zones_df.copy(deep=True)
        cc = clusters.copy(deep=True)
        _ = compute_convergence(clusters, zones_df, config=cfg)
        pd.testing.assert_frame_equal(zones_df, zc)
        pd.testing.assert_frame_equal(clusters, cc)
