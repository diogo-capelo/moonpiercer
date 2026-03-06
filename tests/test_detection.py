"""Tests for moonpiercer.detection."""

from __future__ import annotations

import numpy as np
import pytest

from moonpiercer.config import ChordConfig
from moonpiercer.detection import (
    _characterise_shape,
    _log_blob_detect,
    _nms_no_cap,
    detect_craters_on_chip,
)


# ======================================================================
# Helpers: synthetic crater images
# ======================================================================

def _make_circular_crater(
    size: int = 128,
    cx: float = 64.0,
    cy: float = 64.0,
    radius_px: float = 12.0,
    depth: float = 0.6,
    noise_std: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic circular crater (dark Gaussian bowl on bright background)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size), dtype=float)
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    sigma = radius_px / np.sqrt(2.0)
    profile = depth * np.exp(-r2 / (2.0 * sigma ** 2))
    image = 0.7 - profile + rng.normal(0, noise_std, (size, size))
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def _make_elliptical_crater(
    size: int = 128,
    cx: float = 64.0,
    cy: float = 64.0,
    semi_major: float = 16.0,
    semi_minor: float = 8.0,
    angle_deg: float = 30.0,
    depth: float = 0.6,
    noise_std: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Create a synthetic elliptical crater."""
    rng = np.random.default_rng(seed)
    yy, xx = np.indices((size, size), dtype=float)
    dx = xx - cx
    dy = yy - cy
    theta = np.deg2rad(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    # Rotate to ellipse frame
    dx_rot = cos_t * dx + sin_t * dy
    dy_rot = -sin_t * dx + cos_t * dy
    r2 = (dx_rot / semi_major) ** 2 + (dy_rot / semi_minor) ** 2
    profile = depth * np.exp(-r2 / 2.0)
    image = 0.7 - profile + rng.normal(0, noise_std, (size, size))
    return np.clip(image, 0.0, 1.0).astype(np.float32)


# ======================================================================
# Tests
# ======================================================================

class TestLogBlobDetect:
    def test_detects_single_crater(self):
        img = _make_circular_crater(radius_px=10.0)
        y, x, r, s, thr = _log_blob_detect(
            img, mpp_mean=1.0, target_radius_m=(3.0, 20.0), peak_quantile=0.99
        )
        assert len(y) >= 1
        # Strongest detection near centre
        best = np.argmax(s)
        assert abs(x[best] - 64.0) < 8.0
        assert abs(y[best] - 64.0) < 8.0
        assert thr > 0

    def test_no_detection_on_flat(self):
        img = np.full((64, 64), 0.5, dtype=np.float32)
        y, x, r, s, thr = _log_blob_detect(
            img, mpp_mean=1.0, target_radius_m=(3.0, 20.0), peak_quantile=0.9999
        )
        # Flat image should produce few or no detections
        # (can't guarantee zero due to numerical noise, but strength should be very low)
        if len(y) > 0:
            assert s.max() < 0.01


class TestNMS:
    def test_removes_duplicates(self):
        y = np.array([10.0, 10.5, 50.0])
        x = np.array([10.0, 10.5, 50.0])
        r = np.array([5.0, 5.0, 5.0])
        s = np.array([1.0, 0.8, 0.9])
        keep = _nms_no_cap(y, x, r, s)
        # Should keep 2 (the two at 10 and 50, removing the duplicate at 10.5)
        assert len(keep) == 2

    def test_keeps_all_separated(self):
        y = np.array([10.0, 50.0, 90.0])
        x = np.array([10.0, 50.0, 90.0])
        r = np.array([5.0, 5.0, 5.0])
        s = np.array([1.0, 0.9, 0.8])
        keep = _nms_no_cap(y, x, r, s)
        assert len(keep) == 3


class TestCharacteriseShape:
    def test_circular_crater(self):
        img = _make_circular_crater(radius_px=12.0, noise_std=0.01)
        result = _characterise_shape(img, 64.0, 64.0, 12.0, confidence_min_radius_px=5.0)
        assert result["shape_reliable"] is True
        # Circular: circularity should be close to 1
        assert result["circularity"] > 0.85
        assert result["ellipticity"] < 1.2

    def test_elliptical_crater(self):
        img = _make_elliptical_crater(
            semi_major=16.0, semi_minor=8.0, angle_deg=45.0, noise_std=0.01
        )
        result = _characterise_shape(img, 64.0, 64.0, 12.0, confidence_min_radius_px=5.0)
        assert result["shape_reliable"] is True
        # Elliptical: circularity should be noticeably below 1
        assert result["circularity"] < 0.75
        assert result["ellipticity"] > 1.3

    def test_orientation_recovery(self):
        """Check that orientation angle is roughly recovered for an elongated crater."""
        for angle in [0.0, 30.0, 60.0, 90.0, -45.0]:
            img = _make_elliptical_crater(
                semi_major=18.0, semi_minor=7.0, angle_deg=angle, noise_std=0.005, seed=7
            )
            result = _characterise_shape(img, 64.0, 64.0, 12.0, confidence_min_radius_px=5.0)
            # Orientation has 180° ambiguity, so compare modulo 180
            measured = result["orientation_deg"] % 180.0
            expected = angle % 180.0
            diff = min(abs(measured - expected), 180.0 - abs(measured - expected))
            assert diff < 20.0, (
                f"angle={angle}, measured={result['orientation_deg']:.1f}, diff={diff:.1f}"
            )

    def test_small_crater_unreliable(self):
        img = _make_circular_crater(radius_px=3.0, noise_std=0.01)
        result = _characterise_shape(img, 64.0, 64.0, 3.0, confidence_min_radius_px=5.0)
        assert result["shape_reliable"] is False

    def test_empty_patch(self):
        img = np.full((32, 32), 0.5, dtype=np.float32)
        result = _characterise_shape(img, 16.0, 16.0, 5.0)
        assert np.isnan(result["circularity"])


class TestDetectCratersOnChip:
    def test_finds_crater(self):
        img = _make_circular_crater(size=128, radius_px=10.0, depth=0.7, noise_std=0.01)
        config = ChordConfig(
            target_crater_radius_range_m=(3.0, 20.0),
            chip_peak_quantile=0.99,
            min_circularity=0.3,
            min_crater_radius_px=2.0,
        )
        df, threshold = detect_craters_on_chip(img, mpp_mean=1.0, config=config)
        assert len(df) >= 1
        assert threshold > 0
        # Best detection near centre
        best = df.iloc[0]
        assert abs(best["x"] - 64.0) < 10.0
        assert abs(best["y"] - 64.0) < 10.0
        # Columns present
        for col in ["x", "y", "radius_px", "radius_m", "strength",
                     "depth_proxy", "circularity", "ellipticity",
                     "orientation_deg", "shape_reliable"]:
            assert col in df.columns

    def test_radius_recovery(self):
        """Detected radius should be roughly correct."""
        true_radius = 10.0
        img = _make_circular_crater(size=128, radius_px=true_radius, depth=0.7, noise_std=0.01)
        config = ChordConfig(
            target_crater_radius_range_m=(3.0, 20.0),
            chip_peak_quantile=0.99,
            min_circularity=0.3,
            min_crater_radius_px=2.0,
        )
        df, _ = detect_craters_on_chip(img, mpp_mean=1.0, config=config)
        assert len(df) >= 1
        best_radius = df.iloc[0]["radius_px"]
        assert abs(best_radius - true_radius) < 5.0  # within 50%

    def test_multiple_craters(self):
        """Detect multiple well-separated craters."""
        rng = np.random.default_rng(99)
        img = np.full((256, 256), 0.7, dtype=np.float32)
        img += rng.normal(0, 0.02, img.shape).astype(np.float32)
        # Add 3 craters
        for cx, cy in [(50, 50), (150, 50), (128, 200)]:
            yy, xx = np.indices(img.shape, dtype=float)
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            sigma = 8.0 / np.sqrt(2.0)
            img -= 0.5 * np.exp(-r2 / (2.0 * sigma ** 2)).astype(np.float32)
        img = np.clip(img, 0.0, 1.0)

        config = ChordConfig(
            target_crater_radius_range_m=(3.0, 20.0),
            chip_peak_quantile=0.99,
            min_circularity=0.3,
            min_crater_radius_px=2.0,
        )
        df, _ = detect_craters_on_chip(img, mpp_mean=1.0, config=config)
        assert len(df) >= 3

    def test_result_has_correct_columns(self):
        """Any detection result should have all expected columns."""
        img = _make_circular_crater(size=64, radius_px=8.0, depth=0.5, noise_std=0.01)
        config = ChordConfig(
            target_crater_radius_range_m=(3.0, 20.0),
            chip_peak_quantile=0.99,
            min_crater_radius_px=2.0,
            min_circularity=0.3,
        )
        df, threshold = detect_craters_on_chip(img, mpp_mean=1.0, config=config)
        for col in ["x", "y", "radius_px", "radius_m", "strength",
                     "depth_proxy", "circularity", "ellipticity",
                     "orientation_deg", "shape_reliable"]:
            assert col in df.columns, f"Missing column: {col}"
