"""Tests for moonpiercer.freshness."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from moonpiercer.config import ChordConfig
from moonpiercer.freshness import (
    compute_freshness_for_chip,
    freshness_index,
    normalised_log_strength,
    rim_contrast_ratio,
)


class TestNLS:
    def test_at_threshold(self):
        """Strength == threshold → NLS == 1.0."""
        assert normalised_log_strength(0.5, 0.5) == pytest.approx(1.0)

    def test_double_threshold(self):
        """Strength == 2*threshold → NLS == 2.0."""
        assert normalised_log_strength(1.0, 0.5) == pytest.approx(2.0)

    def test_vectorised(self):
        s = np.array([0.5, 1.0, 1.5])
        nls = normalised_log_strength(s, 0.5)
        np.testing.assert_allclose(nls, [1.0, 2.0, 3.0])


class TestRCR:
    def _make_crater_image(self, size=128, cx=64, cy=64, r=15, depth=0.4):
        """Create synthetic image: dark floor, bright rim, neutral background."""
        img = np.full((size, size), 0.5, dtype=np.float32)
        yy, xx = np.indices((size, size), dtype=float)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        # Floor: dark
        img[dist <= 0.5 * r] = 0.5 - depth
        # Rim: bright
        rim_mask = (dist >= 0.8 * r) & (dist <= 1.3 * r)
        img[rim_mask] = 0.5 + 0.15
        # Add small noise to background
        rng = np.random.default_rng(42)
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        return np.clip(img, 0.0, 1.0)

    def test_positive_for_crater(self):
        """A crater with distinct rim should have positive RCR."""
        img = self._make_crater_image()
        rcr = rim_contrast_ratio(img, 64.0, 64.0, 15.0)
        assert rcr > 1.0  # rim clearly above floor, normalised by bg noise

    def test_flat_image(self):
        """Flat image → RCR ≈ 0."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        rng = np.random.default_rng(7)
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        rcr = rim_contrast_ratio(img, 32.0, 32.0, 8.0)
        assert abs(rcr) < 1.0  # small or near zero

    def test_edge_crater(self):
        """Crater near edge should still return a value (may be 0 if zones are too small)."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        rcr = rim_contrast_ratio(img, 2.0, 2.0, 10.0)
        assert np.isfinite(rcr)


class TestFreshnessIndex:
    def test_zero_at_baseline(self):
        """NLS=1.0 and RCR=0 → FI=0."""
        fi = freshness_index(1.0, 0.0)
        assert fi == pytest.approx(0.0)

    def test_max(self):
        """NLS=nls_scale, RCR=rcr_scale → FI=1.0."""
        cfg = ChordConfig()
        fi = freshness_index(cfg.freshness_nls_scale, cfg.freshness_rcr_scale, cfg)
        assert fi == pytest.approx(1.0, abs=0.01)

    def test_intermediate(self):
        """FI should be between 0 and 1 for reasonable inputs."""
        fi = freshness_index(2.0, 4.0)
        assert 0.0 < float(fi) < 1.0

    def test_vectorised(self):
        nls = np.array([1.0, 2.0, 3.0])
        rcr = np.array([0.0, 4.0, 8.0])
        fi = freshness_index(nls, rcr)
        assert fi.shape == (3,)
        assert fi[0] == pytest.approx(0.0, abs=0.01)
        assert fi[-1] == pytest.approx(1.0, abs=0.05)


class TestComputeFreshnessForChip:
    def test_adds_columns(self):
        """compute_freshness_for_chip should add nls, rcr, freshness_index columns."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        df = pd.DataFrame({
            "x": [32.0],
            "y": [32.0],
            "radius_px": [8.0],
            "strength": [0.01],
        })
        result = compute_freshness_for_chip(img, df, threshold=0.005)
        assert "nls" in result.columns
        assert "rcr" in result.columns
        assert "freshness_index" in result.columns
        assert len(result) == 1

    def test_empty_input(self):
        """Empty detections should return empty DataFrame with correct columns."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        df = pd.DataFrame(columns=["x", "y", "radius_px", "strength"])
        result = compute_freshness_for_chip(img, df, threshold=0.01)
        assert len(result) == 0
        assert "nls" in result.columns
        assert "rcr" in result.columns
        assert "freshness_index" in result.columns

    def test_does_not_modify_input(self):
        """Input DataFrame should not be modified."""
        img = np.full((64, 64), 0.5, dtype=np.float32)
        df = pd.DataFrame({
            "x": [32.0],
            "y": [32.0],
            "radius_px": [8.0],
            "strength": [0.01],
        })
        original_cols = set(df.columns)
        _ = compute_freshness_for_chip(img, df, threshold=0.005)
        assert set(df.columns) == original_cols
