"""Tests for moonpiercer.null_model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moonpiercer.null_model import (
    benjamini_hochberg,
    compute_significance,
    empirical_p_value,
    null_model_best_scores,
    percentile_score,
)


class TestBenjaminiHochberg:
    def test_all_significant(self):
        """All p-values well below threshold → all rejected."""
        p = np.array([0.001, 0.002, 0.003, 0.004, 0.005])
        reject = benjamini_hochberg(p, alpha=0.05)
        assert reject.all()

    def test_none_significant(self):
        """All p-values above threshold → none rejected."""
        p = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        reject = benjamini_hochberg(p, alpha=0.05)
        assert not reject.any()

    def test_partial(self):
        """Mixed p-values: some significant, some not."""
        p = np.array([0.001, 0.01, 0.03, 0.5, 0.9])
        reject = benjamini_hochberg(p, alpha=0.05)
        # The first few should be significant
        assert reject[0]
        assert reject[1]
        # The last should not
        assert not reject[3]
        assert not reject[4]

    def test_empty(self):
        p = np.array([])
        reject = benjamini_hochberg(p)
        assert len(reject) == 0

    def test_single_significant(self):
        p = np.array([0.01])
        reject = benjamini_hochberg(p, alpha=0.05)
        assert reject[0]

    def test_single_not_significant(self):
        p = np.array([0.1])
        reject = benjamini_hochberg(p, alpha=0.05)
        assert not reject[0]

    def test_known_example(self):
        """BH procedure on a known example.

        With 5 tests, alpha=0.05:
        Sorted p:    0.005, 0.011, 0.020, 0.040, 0.500
        Thresholds:  0.010, 0.020, 0.030, 0.040, 0.050

        p[0]=0.005 <= 0.010 → reject
        p[1]=0.011 <= 0.020 → reject
        p[2]=0.020 <= 0.030 → reject
        p[3]=0.040 <= 0.040 → reject
        p[4]=0.500 > 0.050  → not rejected

        max_k = 4, so reject first 4 in sorted order.
        """
        p = np.array([0.040, 0.005, 0.500, 0.011, 0.020])
        reject = benjamini_hochberg(p, alpha=0.05)
        # Original indices: 0→0.040, 1→0.005, 2→0.500, 3→0.011, 4→0.020
        assert reject[0]   # 0.040 ≤ 4/5 * 0.05 = 0.040 → yes
        assert reject[1]   # 0.005
        assert not reject[2]  # 0.500
        assert reject[3]   # 0.011
        assert reject[4]   # 0.020


class TestEmpiricalPValue:
    def test_highest_score(self):
        """Score above all null → p ≈ 1/(n+1)."""
        null = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        p = empirical_p_value(0.6, null)
        assert p == pytest.approx(1.0 / 6.0)

    def test_lowest_score(self):
        """Score below all null → p ≈ 1.0."""
        null = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        p = empirical_p_value(0.1, null)
        assert p == pytest.approx(6.0 / 6.0)

    def test_empty_null(self):
        p = empirical_p_value(0.5, np.array([]))
        assert p == 1.0


class TestNullModelBestScores:
    def test_produces_correct_length(self):
        """Output should have exactly n_trials entries."""
        # Minimal crater catalogue (2 craters near-antipodal)
        df = pd.DataFrame({
            "lon_deg": [0.0, -179.0],
            "lat_deg": [0.0, 0.0],
            "radius_m": [4.0, 4.0],
            "freshness_index": [0.5, 0.5],
            "ellipticity": [1.0, 1.0],
            "orientation_deg": [0.0, 0.0],
            "shape_reliable": [False, False],
            "depth_proxy": [0.5, 0.5],
        })
        from moonpiercer.config import ChordConfig
        config = ChordConfig(
            min_chord_sep_deg=30.0,
            min_freshness=0.1,
            min_depth_proxy=0.1,
            search_cone_half_deg_unreliable=15.0,
        )
        scores = null_model_best_scores(df, config, n_trials=5, seed=42)
        assert len(scores) == 5

    def test_scores_are_nonnegative(self):
        df = pd.DataFrame({
            "lon_deg": [0.0, 180.0],
            "lat_deg": [0.0, 0.0],
            "radius_m": [4.0, 4.0],
            "freshness_index": [0.5, 0.5],
            "ellipticity": [1.0, 1.0],
            "orientation_deg": [0.0, 0.0],
            "shape_reliable": [False, False],
            "depth_proxy": [0.5, 0.5],
        })
        from moonpiercer.config import ChordConfig
        config = ChordConfig(
            min_chord_sep_deg=30.0,
            min_freshness=0.1,
            min_depth_proxy=0.1,
            search_cone_half_deg_unreliable=15.0,
        )
        scores = null_model_best_scores(df, config, n_trials=3, seed=7)
        assert (scores >= 0).all()


    def test_deduplication_prevents_perfect_null_scores(self):
        """Duplicate craters (identical attributes) should be deduplicated.

        Without deduplication, two identical craters randomised onto the
        sphere trivially score 1.0 on every attribute-match term (because
        T_radius, T_freshness = 1 when attributes match exactly, and
        T_ellipticity, T_orientation, T_position = 1 when shape is
        unreliable).  With deduplication, only one copy survives → no
        pair can be formed → null scores should all be 0.
        """
        # 10 clones of the same crater — all identical attributes
        n = 10
        df = pd.DataFrame({
            "lon_deg": np.linspace(0, 5, n),
            "lat_deg": np.zeros(n),
            "radius_m": np.full(n, 5.0),
            "freshness_index": np.full(n, 0.5),
            "ellipticity": np.full(n, 1.0),
            "orientation_deg": np.zeros(n),
            "shape_reliable": np.full(n, False),
            "depth_proxy": np.full(n, 0.5),
        })
        from moonpiercer.config import ChordConfig
        config = ChordConfig(
            min_chord_sep_deg=30.0,
            min_freshness=0.1,
            min_depth_proxy=0.1,
            search_cone_half_deg_unreliable=15.0,
        )
        scores = null_model_best_scores(df, config, n_trials=5, seed=99)
        # After dedup, only 1 crater remains → no pairs → all scores = 0
        assert (scores == 0.0).all()

    def test_distinct_craters_not_removed(self):
        """Craters with different attributes should NOT be deduplicated."""
        df = pd.DataFrame({
            "lon_deg": [0.0, 5.0],
            "lat_deg": [0.0, 0.0],
            "radius_m": [4.0, 8.0],        # different radii
            "freshness_index": [0.5, 0.6],  # different FI
            "ellipticity": [1.0, 1.0],
            "orientation_deg": [0.0, 0.0],
            "shape_reliable": [False, False],
            "depth_proxy": [0.5, 0.5],
        })
        from moonpiercer.config import ChordConfig
        config = ChordConfig(
            min_chord_sep_deg=30.0,
            min_freshness=0.1,
            min_depth_proxy=0.1,
            search_cone_half_deg_unreliable=15.0,
        )
        scores = null_model_best_scores(df, config, n_trials=5, seed=42)
        # Both craters survive dedup → pairs can form → some scores may be >0
        assert len(scores) == 5


class TestPercentileScore:
    def test_highest_score(self):
        """Score above all null → percentile ≈ 1 - 1/(n+1)."""
        null = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        ps = percentile_score(0.6, null)
        assert ps == pytest.approx(1.0 - 1.0 / 6.0)

    def test_lowest_score(self):
        """Score below all null → percentile ≈ 0."""
        null = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
        ps = percentile_score(0.1, null)
        assert ps == pytest.approx(0.0)

    def test_inverse_of_p_value(self):
        null = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        score = 0.35
        p = empirical_p_value(score, null)
        ps = percentile_score(score, null)
        assert ps == pytest.approx(1.0 - p)

    def test_empty_null(self):
        ps = percentile_score(0.5, np.array([]))
        assert ps == 0.0


class TestComputeSignificance:
    def test_adds_columns(self):
        pairs = pd.DataFrame({
            "score": [0.9, 0.5, 0.3],
            "idx_a": [0, 2, 4],
            "idx_b": [1, 3, 5],
        })
        null_scores = np.array([0.4, 0.5, 0.6, 0.45, 0.55] * 10)
        result = compute_significance(pairs, null_scores, alpha=0.05)
        assert "p_value" in result.columns
        assert "percentile_score" in result.columns
        assert "bh_significant" in result.columns
        assert len(result) == 3

    def test_percentile_score_consistency(self):
        """percentile_score should equal 1 - p_value."""
        pairs = pd.DataFrame({
            "score": [0.9, 0.5, 0.3],
            "idx_a": [0, 2, 4],
            "idx_b": [1, 3, 5],
        })
        null_scores = np.array([0.4, 0.5, 0.6, 0.45, 0.55] * 10)
        result = compute_significance(pairs, null_scores, alpha=0.05)
        np.testing.assert_allclose(
            result["percentile_score"].values,
            1.0 - result["p_value"].values,
        )

    def test_empty_pairs(self):
        pairs = pd.DataFrame(columns=["score", "idx_a", "idx_b"])
        result = compute_significance(pairs, np.array([0.1, 0.2]), alpha=0.05)
        assert len(result) == 0
        assert "p_value" in result.columns
        assert "percentile_score" in result.columns
