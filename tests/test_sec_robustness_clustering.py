"""Unit tests for the 1-D single-linkage clustering of breakpoint depths."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from karst_analysis.sec.robustness.clustering import (
    cluster_depths_single_linkage,
    summarize_clusters,
)


class TestClusterDepthsSingleLinkage:
    def test_two_separated_become_two_clusters(self):
        out = cluster_depths_single_linkage(np.array([5.0, 9.0]), 0.5)
        assert list(out) == [0, 1]

    def test_single_input(self):
        out = cluster_depths_single_linkage(np.array([7.5]), 0.5)
        assert list(out) == [0]

    def test_empty_input(self):
        out = cluster_depths_single_linkage(np.array([]), 0.5)
        assert len(out) == 0

    def test_chaining_below_delta(self):
        """Single-linkage chains: gaps < delta all merge into one cluster."""
        out = cluster_depths_single_linkage(
            np.array([5.0, 5.4, 5.8, 6.2, 6.6]), 0.5,
        )
        assert all(c == 0 for c in out)

    def test_exact_delta_gap_splits(self):
        """Gap exactly equal to delta is the split boundary (>= delta splits)."""
        out = cluster_depths_single_linkage(np.array([5.0, 5.5]), 0.5)
        assert list(out) == [0, 1]

    def test_unsorted_input_preserves_order(self):
        out = cluster_depths_single_linkage(
            np.array([13.4, 5.07, 9.06, 13.5]), 0.5,
        )
        # 5.07 → cluster 0, 9.06 → cluster 1, 13.4 and 13.5 → cluster 2
        assert list(out) == [2, 0, 1, 2]

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="delta_m must be > 0"):
            cluster_depths_single_linkage(np.array([1.0, 2.0]), 0.0)
        with pytest.raises(ValueError, match="delta_m must be > 0"):
            cluster_depths_single_linkage(np.array([1.0, 2.0]), -0.5)

    def test_three_clusters_with_one_chain(self):
        depths = np.array([1.0, 1.4, 5.0, 5.3, 9.0])
        out = cluster_depths_single_linkage(depths, 0.5)
        assert list(out) == [0, 0, 1, 1, 2]

    def test_cluster_ids_ordered_by_depth(self):
        """Cluster 0 must be the shallowest, regardless of input order."""
        depths = np.array([15.0, 2.0, 9.0])
        out = cluster_depths_single_linkage(depths, 0.5)
        # 2.0 → 0, 9.0 → 1, 15.0 → 2
        assert list(out) == [2, 0, 1]


class TestSummarizeClusters:
    def test_basic_summary(self):
        bp_records = pd.DataFrame({
            "smoothing": ["savgol", "savgol", "lowess", "lowess"],
            "N": [3, 4, 3, 4],
            "depth_bgl_m": [5.0, 5.1, 5.05, 9.0],
        })
        cluster_ids = np.array([0, 0, 0, 1])
        out = summarize_clusters(
            bp_records, cluster_ids,
            n_max_smoothing={"savgol": 4, "lowess": 4},
        )
        assert len(out) == 2
        # Cluster 0: 3 BPs in [5.0, 5.1], savgol N={3,4}, lowess N={3} → n_savgol=2, n_lowess=1
        c0 = out[out["cluster_id"] == 0].iloc[0]
        assert c0["n_savgol"] == 2
        assert c0["n_lowess"] == 1
        assert c0["persistence"] == 3
        assert c0["agreement"] == 1
        assert c0["n_bp_total"] == 3

    def test_wide_flag_when_diameter_exceeds_2m(self):
        bp_records = pd.DataFrame({
            "smoothing": ["savgol"] * 3,
            "N": [3, 4, 5],
            "depth_bgl_m": [11.0, 12.0, 13.5],   # diameter 2.5 m
        })
        cluster_ids = np.array([0, 0, 0])
        out = summarize_clusters(
            bp_records, cluster_ids,
            n_max_smoothing={"savgol": 5},
        )
        assert out.iloc[0]["wide_flag"]

    def test_sorted_by_agreement_then_persistence(self):
        bp_records = pd.DataFrame({
            "smoothing": ["savgol", "lowess", "savgol", "savgol"],
            "N": [1, 1, 2, 3],
            "depth_bgl_m": [5.0, 5.1, 9.0, 9.1],
        })
        cluster_ids = np.array([0, 0, 1, 1])
        out = summarize_clusters(
            bp_records, cluster_ids,
            n_max_smoothing={"savgol": 3, "lowess": 1},
        )
        # Cluster 0: agreement = min(1, 1) = 1
        # Cluster 1: agreement = min(2, 0) = 0
        # → 0 first (higher agreement)
        assert out.iloc[0]["cluster_id"] == 0

    def test_distinct_n_per_smoothing(self):
        """If the same (smoothing, N) puts MULTIPLE BPs in the same cluster,
        it should still count as just one 'detection' for that cluster."""
        bp_records = pd.DataFrame({
            "smoothing": ["savgol", "savgol", "savgol"],
            "N": [3, 3, 3],   # all same N
            "depth_bgl_m": [5.0, 5.1, 5.2],
        })
        cluster_ids = np.array([0, 0, 0])
        out = summarize_clusters(
            bp_records, cluster_ids,
            n_max_smoothing={"savgol": 3},
        )
        # Only ONE distinct N for savgol, even though 3 BPs are in this cluster.
        assert out.iloc[0]["n_savgol"] == 1
        assert out.iloc[0]["n_bp_total"] == 3
