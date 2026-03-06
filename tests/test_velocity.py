"""Tests for moonpiercer.velocity."""

from __future__ import annotations

import numpy as np
import pytest

from moonpiercer.constants import LUNAR_OMEGA_RAD_S, LUNAR_RADIUS_M
from moonpiercer.velocity import (
    max_physical_angular_offset_deg,
    max_rotation_offset_deg,
    maxwell_boltzmann_speed_pdf,
    offset_probability_factor,
    rotation_offset_deg,
    transit_time_s,
    velocity_cdf,
    velocity_from_offset,
)


class TestTransitTime:
    def test_diametral_at_220(self):
        """Diametral chord at 220 km/s → ~15.8 s."""
        L = 2.0 * LUNAR_RADIUS_M
        t = transit_time_s(L, 220.0)
        assert t == pytest.approx(L / (220.0 * 1e3), rel=1e-10)
        assert 15.0 < t < 17.0

    def test_diametral_at_50(self):
        """Diametral chord at 50 km/s → ~69.5 s."""
        L = 2.0 * LUNAR_RADIUS_M
        t = transit_time_s(L, 50.0)
        assert 68.0 < t < 71.0


class TestRotationOffset:
    def test_diametral_slow(self):
        """Diametral chord at 50 km/s → rotation ≈ 0.0106°."""
        L = 2.0 * LUNAR_RADIUS_M
        offset = rotation_offset_deg(L, 50.0)
        assert 0.005 < float(offset) < 0.015

    def test_diametral_fast(self):
        """Diametral chord at 220 km/s → rotation ≈ 0.0024°."""
        L = 2.0 * LUNAR_RADIUS_M
        offset = rotation_offset_deg(L, 220.0)
        assert 0.001 < float(offset) < 0.004

    def test_short_chord(self):
        """Shorter chord → smaller rotation."""
        L_short = 0.5 * 2.0 * LUNAR_RADIUS_M  # half-diameter
        L_full = 2.0 * LUNAR_RADIUS_M
        assert float(rotation_offset_deg(L_short, 50.0)) < float(
            rotation_offset_deg(L_full, 50.0)
        )

    def test_max_offset(self):
        """Max rotation for diametral chord at 50 km/s."""
        L = 2.0 * LUNAR_RADIUS_M
        offset = max_rotation_offset_deg(L, v_min_km_s=50.0)
        assert float(offset) > 0
        assert float(offset) < 0.02  # well under 1°


class TestVelocityFromOffset:
    def test_roundtrip(self):
        """Infer velocity from offset, then recompute offset — should match."""
        L = 2.0 * LUNAR_RADIUS_M
        v_true = 100.0  # km/s
        offset_rad = LUNAR_OMEGA_RAD_S * L / (v_true * 1e3)
        v_inferred = velocity_from_offset(offset_rad, L)
        assert float(v_inferred) == pytest.approx(v_true, rel=1e-6)

    def test_zero_offset(self):
        """Zero offset → infinite velocity."""
        L = 2.0 * LUNAR_RADIUS_M
        v = velocity_from_offset(0.0, L)
        assert np.isinf(float(v))


class TestSHM:
    def test_pdf_normalised(self):
        """The truncated MB PDF should integrate to ~1."""
        v = np.linspace(0.01, 544.0, 5000)
        pdf = maxwell_boltzmann_speed_pdf(v)
        integral = np.trapezoid(pdf, v)
        assert integral == pytest.approx(1.0, abs=0.01)

    def test_pdf_peak_near_v0(self):
        """Peak of f(v) ∝ v² exp(-v²/v0²) is at v0 ≈ 220 km/s."""
        v = np.linspace(0.01, 544.0, 5000)
        pdf = maxwell_boltzmann_speed_pdf(v)
        v_peak = v[np.argmax(pdf)]
        assert 180.0 < v_peak < 260.0

    def test_pdf_zero_above_escape(self):
        """PDF should be 0 above escape speed."""
        v = np.array([545.0, 600.0, 1000.0])
        pdf = maxwell_boltzmann_speed_pdf(v)
        np.testing.assert_allclose(pdf, 0.0)

    def test_cdf_bounds(self):
        """CDF should go from ~0 to ~1."""
        v = np.linspace(0.01, 544.0, 5000)
        cdf = velocity_cdf(v)
        assert cdf[0] < 0.01
        assert cdf[-1] == pytest.approx(1.0, abs=0.02)


class TestOffsetProbability:
    def test_zero_offset(self):
        """Zero offset is consistent with any speed → probability 1."""
        p = offset_probability_factor(0.0, 2.0 * LUNAR_RADIUS_M)
        assert p == pytest.approx(1.0)

    def test_small_offset_high(self):
        """Small offset → fast PBH → high probability."""
        p = offset_probability_factor(0.001, 2.0 * LUNAR_RADIUS_M)
        assert p > 0.5

    def test_large_offset_low(self):
        """Large offset (requiring very slow PBH) → low probability."""
        p = offset_probability_factor(0.01, 2.0 * LUNAR_RADIUS_M)
        assert p < 0.3

    def test_impossible_offset(self):
        """Offset beyond what even the slowest PBH could produce → ~0."""
        # 1° offset would require v < 1 km/s — physically impossible
        p = offset_probability_factor(1.0, 2.0 * LUNAR_RADIUS_M)
        assert p < 1e-4


class TestMaxPhysicalOffset:
    def test_order_of_magnitude(self):
        """Maximum offset should be ~0.01° (order of magnitude)."""
        max_off = max_physical_angular_offset_deg(v_min_km_s=50.0)
        assert 0.005 < max_off < 0.02

    def test_consistent_with_rotation(self):
        """Should equal rotation_offset_deg for diametral chord at v_min."""
        L = 2.0 * LUNAR_RADIUS_M
        expected = float(rotation_offset_deg(L, 50.0))
        actual = max_physical_angular_offset_deg(v_min_km_s=50.0)
        assert actual == pytest.approx(expected, rel=1e-10)
