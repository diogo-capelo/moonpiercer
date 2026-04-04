"""Tests for moonpiercer.pairing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moonpiercer.config import ChordConfig
from moonpiercer.geometry import (
    angular_separation_deg,
    lonlat_to_unit_vectors,
    predict_exit_point,
    unit_vectors_to_lonlat,
)
from moonpiercer.pairing import (
    _gaussian_score,
    build_chord_pairs,
    score_pair,
    select_top_nonoverlapping_pairs,
)


# ======================================================================
# Helpers
# ======================================================================

def _make_crater_catalogue(craters_spec: list[dict]) -> pd.DataFrame:
    """Build a minimal crater DataFrame from a list of dicts.

    Each dict should have: lon_deg, lat_deg, radius_m, freshness_index,
    ellipticity, orientation_deg, shape_reliable, depth_proxy.
    """
    return pd.DataFrame(craters_spec)


def _make_perfect_antipodal_pair(
    lon_a=10.0, lat_a=20.0, radius_m=4.0, nls=1.5, rcr=3.0,
) -> pd.DataFrame:
    """Create two craters at nearly antipodal positions with matching properties.

    The two craters have *slightly* different ellipticity so that
    ``deduplicate_craters`` recognises them as distinct physical craters
    (identical attributes would be treated as NAC CCD clones).
    """
    return _make_crater_catalogue([
        {
            "lon_deg": lon_a, "lat_deg": lat_a,
            "radius_m": radius_m, "freshness_index": 0.6,
            "nls": nls, "rcr": rcr,
            "ellipticity": 1.0, "orientation_deg": 0.0,
            "shape_reliable": False, "depth_proxy": 0.5,
        },
        {
            "lon_deg": lon_a - 180.0, "lat_deg": -lat_a,
            "radius_m": radius_m, "freshness_index": 0.6,
            "nls": nls, "rcr": rcr,
            "ellipticity": 1.001, "orientation_deg": 0.0,
            "shape_reliable": False, "depth_proxy": 0.5,
        },
    ])


# ======================================================================
# Score component tests
# ======================================================================

class TestGaussianScore:
    def test_zero_diff(self):
        assert _gaussian_score(0.0, 1.0) == pytest.approx(1.0)

    def test_one_sigma(self):
        assert _gaussian_score(1.0, 1.0) == pytest.approx(np.exp(-1.0))

    def test_vectorised(self):
        diffs = np.array([0.0, 1.0, 2.0])
        scores = _gaussian_score(diffs, 1.0)
        np.testing.assert_allclose(scores, np.exp(-diffs ** 2))


class TestScorePair:
    def test_perfect_antipodal(self):
        """Two identical craters at 180deg separation -> high score."""
        v_a = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_b = lonlat_to_unit_vectors(np.array([180.0]), np.array([0.0])).ravel()
        config = ChordConfig()
        result = score_pair(
            sep_deg=180.0,
            radius_a_m=4.0, radius_b_m=4.0,
            nls_a=1.5, nls_b=1.5,
            rcr_a=3.0, rcr_b=3.0,
            ellipticity_a=1.0, ellipticity_b=1.0,
            orientation_a_deg=0.0, orientation_b_deg=0.0,
            shape_reliable_a=False, shape_reliable_b=False,
            v_a=v_a, v_b=v_b,
            config=config,
        )
        assert result["score"] > 0.9
        assert result["T_radius"] == pytest.approx(1.0)
        assert result["T_nls"] == pytest.approx(1.0)
        assert result["T_rcr"] == pytest.approx(1.0)

    def test_radius_mismatch_penalised(self):
        """Different radii -> lower T_radius."""
        v_a = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_b = -v_a
        config = ChordConfig()
        result = score_pair(
            sep_deg=180.0,
            radius_a_m=3.0, radius_b_m=5.0,
            nls_a=1.5, nls_b=1.5,
            rcr_a=3.0, rcr_b=3.0,
            ellipticity_a=1.0, ellipticity_b=1.0,
            orientation_a_deg=0.0, orientation_b_deg=0.0,
            shape_reliable_a=False, shape_reliable_b=False,
            v_a=v_a, v_b=v_b,
            config=config,
        )
        assert result["T_radius"] < 1.0
        assert result["score"] < 1.0

    def test_nls_mismatch_penalised(self):
        """Different NLS -> lower T_nls."""
        v_a = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_b = -v_a
        config = ChordConfig()
        result = score_pair(
            sep_deg=180.0,
            radius_a_m=4.0, radius_b_m=4.0,
            nls_a=1.0, nls_b=3.0,
            rcr_a=3.0, rcr_b=3.0,
            ellipticity_a=1.0, ellipticity_b=1.0,
            orientation_a_deg=0.0, orientation_b_deg=0.0,
            shape_reliable_a=False, shape_reliable_b=False,
            v_a=v_a, v_b=v_b,
            config=config,
        )
        assert result["T_nls"] < 0.01
        assert result["score"] < 0.02

    def test_rcr_mismatch_penalised(self):
        """Different RCR -> lower T_rcr."""
        v_a = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_b = -v_a
        config = ChordConfig()
        result = score_pair(
            sep_deg=180.0,
            radius_a_m=4.0, radius_b_m=4.0,
            nls_a=1.5, nls_b=1.5,
            rcr_a=1.0, rcr_b=8.0,
            ellipticity_a=1.0, ellipticity_b=1.0,
            orientation_a_deg=0.0, orientation_b_deg=0.0,
            shape_reliable_a=False, shape_reliable_b=False,
            v_a=v_a, v_b=v_b,
            config=config,
        )
        assert result["T_rcr"] < 0.01
        assert result["score"] < 0.02



# ======================================================================
# Build pairs tests
# ======================================================================

class TestBuildChordPairs:
    def test_finds_antipodal_pair(self):
        """Two matching craters at antipodal positions should be paired."""
        df = _make_perfect_antipodal_pair()
        config = ChordConfig(
            min_depth_proxy=0.1,
            max_radius_diff_m=3.0,
            min_chord_sep_deg=30.0,
            search_cone_half_deg_unreliable=15.0,
        )
        pairs = build_chord_pairs(df, config)
        assert len(pairs) >= 1
        assert pairs.iloc[0]["score"] > 0.5

    def test_no_pair_for_close_craters(self):
        """Two craters close together (< min_chord_sep) should not pair."""
        df = _make_crater_catalogue([
            {
                "lon_deg": 0.0, "lat_deg": 0.0,
                "radius_m": 4.0, "freshness_index": 0.6,
                "nls": 1.5, "rcr": 3.0,
                "ellipticity": 1.0, "orientation_deg": 0.0,
                "shape_reliable": False, "depth_proxy": 0.5,
            },
            {
                "lon_deg": 5.0, "lat_deg": 0.0,
                "radius_m": 4.0, "freshness_index": 0.6,
                "nls": 1.5, "rcr": 3.001,
                "ellipticity": 1.001, "orientation_deg": 0.0,
                "shape_reliable": False, "depth_proxy": 0.5,
            },
        ])
        config = ChordConfig(min_chord_sep_deg=30.0)
        pairs = build_chord_pairs(df, config)
        assert len(pairs) == 0

    def test_radius_hard_cut(self):
        """Pairs with radius difference > max_radius_diff_m are rejected."""
        df = _make_crater_catalogue([
            {
                "lon_deg": 10.0, "lat_deg": 20.0,
                "radius_m": 4.0, "freshness_index": 0.6,
                "nls": 1.5, "rcr": 3.0,
                "ellipticity": 1.0, "orientation_deg": 0.0,
                "shape_reliable": False, "depth_proxy": 0.5,
            },
            {
                "lon_deg": -170.0, "lat_deg": -20.0,
                "radius_m": 10.0, "freshness_index": 0.6,
                "nls": 1.5, "rcr": 3.001,
                "ellipticity": 1.001, "orientation_deg": 0.0,
                "shape_reliable": False, "depth_proxy": 0.5,
            },
        ])
        config = ChordConfig(
            max_radius_diff_m=2.0,
            min_depth_proxy=0.1,
            search_cone_half_deg_unreliable=15.0,
        )
        pairs = build_chord_pairs(df, config)
        assert len(pairs) == 0

    def test_nls_hard_cut(self):
        """Pairs with NLS difference > max_nls_diff are rejected."""
        df = _make_crater_catalogue([
            {
                "lon_deg": 10.0, "lat_deg": 20.0,
                "radius_m": 4.0, "freshness_index": 0.6,
                "nls": 0.5, "rcr": 3.0,
                "ellipticity": 1.0, "orientation_deg": 0.0,
                "shape_reliable": False, "depth_proxy": 0.5,
            },
            {
                "lon_deg": -170.0, "lat_deg": -20.0,
                "radius_m": 4.0, "freshness_index": 0.6,
                "nls": 3.0, "rcr": 3.001,
                "ellipticity": 1.001, "orientation_deg": 0.0,
                "shape_reliable": False, "depth_proxy": 0.5,
            },
        ])
        config = ChordConfig(
            max_nls_diff=1.5,
            min_depth_proxy=0.1,
            search_cone_half_deg_unreliable=15.0,
        )
        pairs = build_chord_pairs(df, config)
        assert len(pairs) == 0

    def test_rcr_hard_cut(self):
        """Pairs with RCR difference > max_rcr_diff are rejected."""
        df = _make_crater_catalogue([
            {
                "lon_deg": 10.0, "lat_deg": 20.0,
                "radius_m": 4.0, "freshness_index": 0.6,
                "nls": 1.5, "rcr": 1.0,
                "ellipticity": 1.0, "orientation_deg": 0.0,
                "shape_reliable": False, "depth_proxy": 0.5,
            },
            {
                "lon_deg": -170.0, "lat_deg": -20.0,
                "radius_m": 4.0, "freshness_index": 0.6,
                "nls": 1.5, "rcr": 8.0,
                "ellipticity": 1.001, "orientation_deg": 0.0,
                "shape_reliable": False, "depth_proxy": 0.5,
            },
        ])
        config = ChordConfig(
            max_rcr_diff=4.0,
            min_depth_proxy=0.1,
            search_cone_half_deg_unreliable=15.0,
        )
        pairs = build_chord_pairs(df, config)
        assert len(pairs) == 0

    def test_empty_input(self):
        df = pd.DataFrame(columns=[
            "lon_deg", "lat_deg", "radius_m", "freshness_index",
            "nls", "rcr",
            "ellipticity", "orientation_deg", "shape_reliable", "depth_proxy",
        ])
        pairs = build_chord_pairs(df)
        assert len(pairs) == 0

    def test_single_crater(self):
        df = _make_crater_catalogue([{
            "lon_deg": 0.0, "lat_deg": 0.0,
            "radius_m": 4.0, "freshness_index": 0.6,
            "nls": 1.5, "rcr": 3.0,
            "ellipticity": 1.0, "orientation_deg": 0.0,
            "shape_reliable": False, "depth_proxy": 0.5,
        }])
        pairs = build_chord_pairs(df)
        assert len(pairs) == 0


class TestSelectTopPairs:
    def test_non_overlapping(self):
        """Selected pairs should not share craters."""
        pairs = pd.DataFrame({
            "idx_a": [0, 0, 2],
            "idx_b": [1, 3, 3],
            "score": [0.9, 0.8, 0.7],
        })
        selected = select_top_nonoverlapping_pairs(pairs, top_k=10)
        # Pair (0, 1) is best; (0, 3) shares crater 0; (2, 3) should be kept
        assert len(selected) == 2
        assert set(selected.iloc[0][["idx_a", "idx_b"]].tolist()) == {0, 1}
        assert set(selected.iloc[1][["idx_a", "idx_b"]].tolist()) == {2, 3}

    def test_top_k_limit(self):
        pairs = pd.DataFrame({
            "idx_a": [0, 2, 4, 6],
            "idx_b": [1, 3, 5, 7],
            "score": [0.9, 0.8, 0.7, 0.6],
        })
        selected = select_top_nonoverlapping_pairs(pairs, top_k=2)
        assert len(selected) == 2

    def test_empty(self):
        pairs = pd.DataFrame()
        selected = select_top_nonoverlapping_pairs(pairs)
        assert len(selected) == 0
