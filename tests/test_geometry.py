"""Tests for moonpiercer.geometry."""

from __future__ import annotations

import numpy as np
import pytest

from moonpiercer.constants import LUNAR_RADIUS_M
from moonpiercer.geometry import (
    angular_separation_deg,
    angular_separation_deg_batch,
    bbox_mpp,
    chip_pixel_to_lonlat,
    chord_impact_parameter_from_separation,
    chord_incidence_angle_deg,
    chord_length,
    chord_length_from_ellipticity,
    chord_length_from_separation,
    expected_ellipticity_from_separation,
    incidence_angle_from_ellipticity,
    local_bearing_deg,
    local_bearing_rad,
    lonlat_to_unit_vectors,
    make_bbox_around_point,
    normalize_lon,
    normalize_percentile,
    predict_exit_point,
    separation_from_ellipticity,
    slerp_arc,
    to_float,
    unit_vectors_to_lonlat,
)


# ======================================================================
# normalize_lon
# ======================================================================

class TestNormalizeLon:
    def test_identity(self):
        assert normalize_lon(0.0) == pytest.approx(0.0)
        assert normalize_lon(90.0) == pytest.approx(90.0)
        assert normalize_lon(-90.0) == pytest.approx(-90.0)

    def test_wrap_positive(self):
        assert normalize_lon(181.0) == pytest.approx(-179.0)
        assert normalize_lon(360.0) == pytest.approx(0.0)
        assert normalize_lon(540.0) == pytest.approx(-180.0)

    def test_wrap_negative(self):
        assert normalize_lon(-181.0) == pytest.approx(179.0)
        assert normalize_lon(-360.0) == pytest.approx(0.0)

    def test_boundary(self):
        # -180 maps to -180
        assert normalize_lon(-180.0) == pytest.approx(-180.0)
        # 180 wraps to -180
        assert normalize_lon(180.0) == pytest.approx(-180.0)

    def test_array(self):
        arr = np.array([0.0, 181.0, -181.0, 360.0])
        result = normalize_lon(arr)
        expected = np.array([0.0, -179.0, 179.0, 0.0])
        np.testing.assert_allclose(result, expected)


# ======================================================================
# Coordinate transforms
# ======================================================================

class TestCoordinateTransforms:
    def test_origin(self):
        """(0°, 0°) → (1, 0, 0)."""
        v = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0]))
        np.testing.assert_allclose(v, [[1.0, 0.0, 0.0]], atol=1e-15)

    def test_north_pole(self):
        """(*, 90°) → (0, 0, 1)."""
        v = lonlat_to_unit_vectors(np.array([0.0]), np.array([90.0]))
        np.testing.assert_allclose(v, [[0.0, 0.0, 1.0]], atol=1e-15)

    def test_south_pole(self):
        v = lonlat_to_unit_vectors(np.array([0.0]), np.array([-90.0]))
        np.testing.assert_allclose(v, [[0.0, 0.0, -1.0]], atol=1e-15)

    def test_east(self):
        """(90°, 0°) → (0, 1, 0)."""
        v = lonlat_to_unit_vectors(np.array([90.0]), np.array([0.0]))
        np.testing.assert_allclose(v, [[0.0, 1.0, 0.0]], atol=1e-15)

    def test_roundtrip_single(self):
        lon_in, lat_in = 45.0, -30.0
        v = lonlat_to_unit_vectors(np.array([lon_in]), np.array([lat_in]))
        lon_out, lat_out = unit_vectors_to_lonlat(v)
        assert float(lon_out) == pytest.approx(lon_in, abs=1e-12)
        assert float(lat_out) == pytest.approx(lat_in, abs=1e-12)

    def test_roundtrip_batch(self):
        rng = np.random.default_rng(42)
        lons = rng.uniform(-180, 180, 100)
        lats = rng.uniform(-90, 90, 100)
        v = lonlat_to_unit_vectors(lons, lats)
        lons_out, lats_out = unit_vectors_to_lonlat(v)
        np.testing.assert_allclose(lons_out, lons, atol=1e-10)
        np.testing.assert_allclose(lats_out, lats, atol=1e-10)

    def test_unit_norm(self):
        rng = np.random.default_rng(7)
        lons = rng.uniform(-180, 180, 50)
        lats = rng.uniform(-90, 90, 50)
        v = lonlat_to_unit_vectors(lons, lats)
        norms = np.linalg.norm(v, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-15)


# ======================================================================
# Angular separation
# ======================================================================

class TestAngularSeparation:
    def test_identical_points(self):
        v = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        assert angular_separation_deg(v, v) == pytest.approx(0.0, abs=1e-12)

    def test_antipodal(self):
        v1 = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v2 = lonlat_to_unit_vectors(np.array([180.0]), np.array([0.0])).ravel()
        assert angular_separation_deg(v1, v2) == pytest.approx(180.0, abs=1e-10)

    def test_poles(self):
        vn = lonlat_to_unit_vectors(np.array([0.0]), np.array([90.0])).ravel()
        vs = lonlat_to_unit_vectors(np.array([0.0]), np.array([-90.0])).ravel()
        assert angular_separation_deg(vn, vs) == pytest.approx(180.0, abs=1e-10)

    def test_quarter_sphere(self):
        v1 = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v2 = lonlat_to_unit_vectors(np.array([90.0]), np.array([0.0])).ravel()
        assert angular_separation_deg(v1, v2) == pytest.approx(90.0, abs=1e-10)

    def test_symmetry(self):
        v1 = lonlat_to_unit_vectors(np.array([10.0]), np.array([20.0])).ravel()
        v2 = lonlat_to_unit_vectors(np.array([50.0]), np.array([-30.0])).ravel()
        assert angular_separation_deg(v1, v2) == pytest.approx(
            angular_separation_deg(v2, v1), abs=1e-12
        )

    def test_batch(self):
        v1 = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v2 = lonlat_to_unit_vectors(np.array([180.0]), np.array([0.0])).ravel()
        result = angular_separation_deg_batch(
            np.array([v1, v1]), np.array([v1, v2])
        )
        np.testing.assert_allclose(result, [0.0, 180.0], atol=1e-10)


# ======================================================================
# Chord geometry
# ======================================================================

class TestChordLength:
    def test_diametral(self):
        """Impact parameter 0 → chord = 2R (diameter)."""
        assert chord_length(0.0) == pytest.approx(2.0 * LUNAR_RADIUS_M)

    def test_grazing(self):
        """Impact parameter = R → chord = 0."""
        assert chord_length(LUNAR_RADIUS_M) == pytest.approx(0.0)

    def test_beyond_radius(self):
        """Impact parameter > R → chord = 0 (miss)."""
        assert chord_length(LUNAR_RADIUS_M * 1.1) == pytest.approx(0.0)

    def test_half_radius(self):
        """b = R/2 → L = 2*sqrt(R² - R²/4) = R*sqrt(3)."""
        R = LUNAR_RADIUS_M
        expected = R * np.sqrt(3.0)
        assert chord_length(R / 2.0) == pytest.approx(expected, rel=1e-12)

    def test_custom_radius(self):
        assert chord_length(0.0, R=100.0) == pytest.approx(200.0)
        assert chord_length(100.0, R=100.0) == pytest.approx(0.0)


class TestChordFromSeparation:
    def test_antipodal(self):
        """180° separation → chord = 2R (diameter)."""
        L = chord_length_from_separation(180.0)
        assert L == pytest.approx(2.0 * LUNAR_RADIUS_M, rel=1e-12)

    def test_zero_separation(self):
        """0° separation → chord = 0."""
        L = chord_length_from_separation(0.0)
        assert L == pytest.approx(0.0, abs=1e-10)

    def test_quarter_sphere(self):
        """90° → chord = R*sqrt(2)."""
        L = chord_length_from_separation(90.0)
        expected = LUNAR_RADIUS_M * np.sqrt(2.0)
        assert L == pytest.approx(expected, rel=1e-12)

    def test_impact_parameter_antipodal(self):
        """180° → b = R*cos(90°) = 0 (diametral)."""
        b = chord_impact_parameter_from_separation(180.0)
        assert b == pytest.approx(0.0, abs=1e-6)

    def test_impact_parameter_zero(self):
        """0° → b = R*cos(0°) = R (grazing)."""
        b = chord_impact_parameter_from_separation(0.0)
        assert b == pytest.approx(LUNAR_RADIUS_M, rel=1e-12)


class TestChordIncidenceAngle:
    def test_diametral(self):
        """b=0 → theta=0 (normal incidence)."""
        assert chord_incidence_angle_deg(0.0) == pytest.approx(0.0)

    def test_grazing(self):
        """b=R → theta=90° (grazing)."""
        assert chord_incidence_angle_deg(LUNAR_RADIUS_M) == pytest.approx(90.0, abs=1e-10)

    def test_half(self):
        """b=R/2 → theta=30°."""
        assert chord_incidence_angle_deg(LUNAR_RADIUS_M / 2.0) == pytest.approx(30.0, abs=1e-10)


class TestEllipticity:
    def test_antipodal_circular(self):
        """180° separation (diametral chord) → ellipticity = 1 (circular)."""
        e = expected_ellipticity_from_separation(180.0)
        assert e == pytest.approx(1.0, rel=1e-12)

    def test_quarter_sphere(self):
        """90° separation → e = 1/sin(45°) = sqrt(2)."""
        e = expected_ellipticity_from_separation(90.0)
        assert e == pytest.approx(np.sqrt(2.0), rel=1e-12)

    def test_small_separation_large_ellipticity(self):
        """Small separation → large ellipticity."""
        e = expected_ellipticity_from_separation(10.0)
        assert e > 5.0

    def test_zero_separation_inf(self):
        """0° separation → infinite ellipticity."""
        e = expected_ellipticity_from_separation(0.0)
        assert np.isinf(e)

    def test_vectorised(self):
        seps = np.array([180.0, 90.0, 60.0])
        e = expected_ellipticity_from_separation(seps)
        assert e[0] == pytest.approx(1.0, rel=1e-12)
        assert e[1] == pytest.approx(np.sqrt(2.0), rel=1e-12)
        assert e[2] == pytest.approx(1.0 / np.sin(np.deg2rad(30.0)), rel=1e-12)


class TestInverseEllipticity:
    def test_circular(self):
        """e=1 → theta=0 (normal incidence)."""
        assert incidence_angle_from_ellipticity(1.0) == pytest.approx(0.0)

    def test_below_one(self):
        """e<1 → clamp to 0."""
        assert incidence_angle_from_ellipticity(0.5) == pytest.approx(0.0)

    def test_sqrt2(self):
        """e=sqrt(2) → theta=45°."""
        assert incidence_angle_from_ellipticity(np.sqrt(2.0)) == pytest.approx(45.0, abs=1e-10)

    def test_chord_from_ellipticity(self):
        """e=1 → L=2R, e=2 → L=R."""
        R = LUNAR_RADIUS_M
        assert chord_length_from_ellipticity(1.0) == pytest.approx(2.0 * R, rel=1e-12)
        assert chord_length_from_ellipticity(2.0) == pytest.approx(R, rel=1e-12)

    def test_separation_from_ellipticity_roundtrip(self):
        """Check separation_from_ellipticity inverts expected_ellipticity_from_separation."""
        for sep in [180.0, 120.0, 90.0, 60.0, 30.0]:
            e = float(expected_ellipticity_from_separation(sep))
            sep_recovered = separation_from_ellipticity(e)
            assert sep_recovered == pytest.approx(sep, abs=1e-8)


# ======================================================================
# Bearing and exit-point prediction
# ======================================================================

class TestBearing:
    def test_north(self):
        """From equator, bearing to north pole is 0° (North)."""
        v_from = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_to = lonlat_to_unit_vectors(np.array([0.0]), np.array([45.0])).ravel()
        brg = local_bearing_deg(v_from, v_to)
        assert brg == pytest.approx(0.0, abs=1e-8)

    def test_east(self):
        """From equator-prime-meridian, bearing to (90°E, 0°) is 90°."""
        v_from = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_to = lonlat_to_unit_vectors(np.array([90.0]), np.array([0.0])).ravel()
        brg = local_bearing_deg(v_from, v_to)
        assert brg == pytest.approx(90.0, abs=1e-8)

    def test_south(self):
        """From equator, bearing to south pole is 180° (or -180°)."""
        v_from = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_to = lonlat_to_unit_vectors(np.array([0.0]), np.array([-45.0])).ravel()
        brg = local_bearing_deg(v_from, v_to)
        assert abs(brg) == pytest.approx(180.0, abs=1e-8)

    def test_west(self):
        """Bearing to (90°W, 0°) from origin is -90°."""
        v_from = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_to = lonlat_to_unit_vectors(np.array([-90.0]), np.array([0.0])).ravel()
        brg = local_bearing_deg(v_from, v_to)
        assert brg == pytest.approx(-90.0, abs=1e-8)


class TestPredictExitPoint:
    def test_antipodal_from_equator(self):
        """Entry at (0°, 0°), bearing south, sep 180° → exit at (0°±180°, 0°)."""
        v_entry = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_exit = predict_exit_point(v_entry, np.pi, np.pi)  # bearing=south, sep=180°
        lon, lat = unit_vectors_to_lonlat(v_exit)
        # Should be (180° or -180°, 0°)
        assert abs(float(lat)) == pytest.approx(0.0, abs=1e-8)
        assert abs(float(lon)) == pytest.approx(180.0, abs=1e-6)

    def test_north_pole_to_south(self):
        """Entry at north pole, bearing south (any), sep=180° → south pole."""
        v_entry = lonlat_to_unit_vectors(np.array([0.0]), np.array([90.0])).ravel()
        v_exit = predict_exit_point(v_entry, 0.0, np.pi)
        lon, lat = unit_vectors_to_lonlat(v_exit)
        assert float(lat) == pytest.approx(-90.0, abs=1e-6)

    def test_short_chord_along_equator(self):
        """Entry at (0°, 0°), bearing east, sep=90° → (90°E, 0°)."""
        v_entry = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v_exit = predict_exit_point(v_entry, np.pi / 2, np.pi / 2)  # east, 90°
        lon, lat = unit_vectors_to_lonlat(v_exit)
        assert float(lon) == pytest.approx(90.0, abs=1e-6)
        assert float(lat) == pytest.approx(0.0, abs=1e-6)

    def test_roundtrip_bearing_separation(self):
        """Predict exit, then check separation and back-bearing are consistent."""
        rng = np.random.default_rng(123)
        for _ in range(20):
            lon0 = rng.uniform(-170, 170)
            lat0 = rng.uniform(-80, 80)
            brg = rng.uniform(-np.pi, np.pi)
            sep = rng.uniform(np.deg2rad(30), np.deg2rad(170))
            v0 = lonlat_to_unit_vectors(np.array([lon0]), np.array([lat0])).ravel()
            v1 = predict_exit_point(v0, brg, sep)
            # Check angular separation
            sep_check = np.deg2rad(angular_separation_deg(v0, v1))
            assert sep_check == pytest.approx(sep, abs=1e-6)


# ======================================================================
# Bounding box helpers
# ======================================================================

class TestBBox:
    def test_equator_valid(self):
        bbox = make_bbox_around_point(0.0, 0.0, 10_000.0)
        assert bbox is not None
        lon_min, lat_min, lon_max, lat_max = bbox
        assert lon_min < 0.0 < lon_max
        assert lat_min < 0.0 < lat_max

    def test_pole_returns_none(self):
        """Near-pole chips should return None."""
        assert make_bbox_around_point(0.0, 89.999, 10_000.0) is None

    def test_meridian_wrap_returns_none(self):
        """Chips straddling ±180° meridian return None."""
        assert make_bbox_around_point(179.99, 0.0, 100_000.0) is None

    def test_bbox_mpp_reasonable(self):
        bbox = make_bbox_around_point(0.0, 0.0, 10_000.0)
        assert bbox is not None
        mpp_x, mpp_y = bbox_mpp(bbox, 1000, 1000)
        # 10 km chip at 1000 px → ~10 m/px
        assert 5.0 < mpp_x < 15.0
        assert 5.0 < mpp_y < 15.0

    def test_chip_pixel_to_lonlat_corners(self):
        bbox = (10.0, -5.0, 12.0, -3.0)
        w, h = 100, 100
        lon_tl, lat_tl = chip_pixel_to_lonlat(
            np.array([0.0]), np.array([0.0]), bbox, w, h
        )
        np.testing.assert_allclose(lon_tl, 10.0)
        np.testing.assert_allclose(lat_tl, -3.0)  # top of image = lat_max

        lon_br, lat_br = chip_pixel_to_lonlat(
            np.array([99.0]), np.array([99.0]), bbox, w, h
        )
        np.testing.assert_allclose(lon_br, 12.0)
        np.testing.assert_allclose(lat_br, -5.0)


# ======================================================================
# Slerp arc
# ======================================================================

class TestSlerpArc:
    def test_arc_length(self):
        """Arc between two equatorial points should return correct longitudes."""
        v1 = lonlat_to_unit_vectors(np.array([0.0]), np.array([0.0])).ravel()
        v2 = lonlat_to_unit_vectors(np.array([90.0]), np.array([0.0])).ravel()
        lons, lats = slerp_arc(v1, v2, n_points=91)
        assert len(lons) == 91
        # All latitudes should be ~0
        np.testing.assert_allclose(lats, 0.0, atol=1e-10)
        # Longitudes should span 0 to 90
        assert float(lons[0]) == pytest.approx(0.0, abs=1e-10)
        assert float(lons[-1]) == pytest.approx(90.0, abs=1e-10)

    def test_same_point(self):
        v = lonlat_to_unit_vectors(np.array([45.0]), np.array([30.0])).ravel()
        lons, lats = slerp_arc(v, v, n_points=10)
        np.testing.assert_allclose(lons, 45.0, atol=1e-10)
        np.testing.assert_allclose(lats, 30.0, atol=1e-10)


# ======================================================================
# Utility functions
# ======================================================================

class TestUtility:
    def test_normalize_percentile(self):
        arr = np.arange(100, dtype=float)
        result = normalize_percentile(arr, p_lo=0.0, p_hi=100.0)
        assert result.min() >= 0.0
        assert result.max() <= 1.0
        assert result.min() == pytest.approx(0.0, abs=0.02)
        assert result.max() == pytest.approx(1.0, abs=0.02)

    def test_to_float_valid(self):
        assert to_float(3.14) == pytest.approx(3.14)
        assert to_float("2.5") == pytest.approx(2.5)
        assert to_float(42) == pytest.approx(42.0)

    def test_to_float_invalid(self):
        assert np.isnan(to_float("abc"))
        assert np.isnan(to_float(None))
        assert to_float("abc", default=-1.0) == pytest.approx(-1.0)


# ======================================================================
# Consistency checks (integrated)
# ======================================================================

class TestConsistency:
    def test_chord_ellipticity_separation_chain(self):
        """Verify the full chain: separation → ellipticity → separation."""
        for sep in [170.0, 150.0, 120.0, 90.0, 60.0, 45.0]:
            e = float(expected_ellipticity_from_separation(sep))
            L = chord_length_from_separation(sep)
            L_from_e = chord_length_from_ellipticity(e)
            assert float(L) == pytest.approx(float(L_from_e), rel=1e-10)

            sep_back = separation_from_ellipticity(e)
            assert sep_back == pytest.approx(sep, abs=1e-8)

    def test_chord_length_vs_direct(self):
        """chord_length(b) should equal chord_length_from_separation for
        the same chord, where b = chord_impact_parameter_from_separation(sep)."""
        for sep in [180.0, 120.0, 90.0, 60.0, 30.0]:
            b = float(chord_impact_parameter_from_separation(sep))
            L_via_b = chord_length(b)
            L_via_sep = float(chord_length_from_separation(sep))
            assert L_via_b == pytest.approx(L_via_sep, rel=1e-10)

    def test_incidence_angle_consistency(self):
        """Incidence angle from impact parameter should be consistent with
        the angle implied by ellipticity."""
        for sep in [170.0, 120.0, 90.0, 60.0]:
            b = float(chord_impact_parameter_from_separation(sep))
            theta_from_b = chord_incidence_angle_deg(b)
            e = float(expected_ellipticity_from_separation(sep))
            theta_from_e = incidence_angle_from_ellipticity(e)
            assert theta_from_b == pytest.approx(theta_from_e, abs=1e-8)

    def test_exit_point_matches_separation(self):
        """predict_exit_point should produce a point at the correct angular distance."""
        rng = np.random.default_rng(77)
        for _ in range(50):
            lon0 = rng.uniform(-170, 170)
            lat0 = rng.uniform(-80, 80)
            brg = rng.uniform(-np.pi, np.pi)
            sep_deg = rng.uniform(30, 170)
            sep_rad = np.deg2rad(sep_deg)

            v0 = lonlat_to_unit_vectors(np.array([lon0]), np.array([lat0])).ravel()
            v1 = predict_exit_point(v0, brg, sep_rad)
            sep_check = angular_separation_deg(v0, v1)
            assert sep_check == pytest.approx(sep_deg, abs=1e-4)
