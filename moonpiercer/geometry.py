"""Spherical geometry, chord math, and coordinate transforms.

All angles are in **degrees** at the public API boundary unless a
function name explicitly says ``_rad``.  Internal helpers may work in
radians for performance.
"""

from __future__ import annotations

import numpy as np

from moonpiercer.constants import LUNAR_RADIUS_M

# ======================================================================
# Coordinate transforms
# ======================================================================

def normalize_lon(lon_deg: float | np.ndarray) -> float | np.ndarray:
    """Wrap longitude to [-180, 180)."""
    return ((np.asarray(lon_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0


def lonlat_to_unit_vectors(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
) -> np.ndarray:
    """Convert selenographic (lon, lat) in degrees to unit vectors (N, 3)."""
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    cl = np.cos(lat)
    return np.column_stack((cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)))


def unit_vectors_to_lonlat(
    vectors: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert unit vectors (N, 3) → (lon_deg, lat_deg) arrays."""
    v = np.asarray(vectors, dtype=np.float64)
    if v.ndim == 1:
        v = v[np.newaxis, :]
    lon = np.rad2deg(np.arctan2(v[:, 1], v[:, 0]))
    lat = np.rad2deg(np.arcsin(np.clip(v[:, 2], -1.0, 1.0)))
    return lon.squeeze(), lat.squeeze()


# ======================================================================
# Angular distances
# ======================================================================

def angular_separation_deg(
    v1: np.ndarray,
    v2: np.ndarray,
) -> float | np.ndarray:
    """Great-circle angular separation in degrees between unit vectors.

    *v1* and *v2* can each be a single (3,) vector or arrays of shape
    (N, 3).  When both are (N, 3), computes element-wise separations.
    When one is (3,) and the other is (N, 3), broadcasts.
    """
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    dot = np.sum(v1 * v2, axis=-1)
    return float(np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))))


def angular_separation_deg_batch(
    v1: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Like :func:`angular_separation_deg` but always returns an ndarray."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    dot = np.sum(v1 * v2, axis=-1)
    return np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))


# ======================================================================
# Chord geometry
# ======================================================================

def chord_length(impact_parameter_m: float, R: float = LUNAR_RADIUS_M) -> float:
    """Chord length [m] for a chord with impact parameter *b* [m]."""
    b = float(impact_parameter_m)
    if b >= R:
        return 0.0
    return 2.0 * np.sqrt(R * R - b * b)


def chord_impact_parameter_from_separation(
    sep_deg: float | np.ndarray,
    R: float = LUNAR_RADIUS_M,
) -> float | np.ndarray:
    """Impact parameter [m] of the chord connecting two surface points.

    For two points at angular separation *sep_deg*, the chord through the
    sphere has impact parameter ``b = R * cos(sep/2)``.
    """
    half = np.deg2rad(np.asarray(sep_deg, dtype=np.float64) / 2.0)
    return R * np.cos(half)


def chord_length_from_separation(
    sep_deg: float | np.ndarray,
    R: float = LUNAR_RADIUS_M,
) -> float | np.ndarray:
    """Chord length [m] between two surface points at angular separation *sep_deg*."""
    half = np.deg2rad(np.asarray(sep_deg, dtype=np.float64) / 2.0)
    return 2.0 * R * np.sin(half)


def chord_incidence_angle_deg(
    impact_parameter_m: float,
    R: float = LUNAR_RADIUS_M,
) -> float:
    """Angle from surface normal at entry/exit [degrees].

    ``theta = arcsin(b / R)``.  A diametral chord (b=0) gives theta=0
    (normal incidence); a grazing chord (b→R) gives theta→90°.
    """
    return float(np.degrees(np.arcsin(np.clip(impact_parameter_m / R, 0.0, 1.0))))


def expected_ellipticity_from_separation(
    sep_deg: float | np.ndarray,
) -> float | np.ndarray:
    """Expected PBH crater ellipticity from the angular separation.

    For a chord connecting two surface points at angular separation *sep*,
    the incidence angle is ``theta = 90° - sep/2``, and the geometric
    footprint ellipticity of the cylindrical blast column is
    ``e = 1 / sin(sep/2)``.

    Near-antipodal (sep ≈ 180°) → e ≈ 1 (circular).
    Short chords (small sep) → large e (elongated).
    """
    half = np.deg2rad(np.asarray(sep_deg, dtype=np.float64) / 2.0)
    sin_half = np.sin(half)
    # Avoid division by zero for sep ≈ 0
    return np.where(sin_half > 1e-12, 1.0 / sin_half, np.inf)


def incidence_angle_from_ellipticity(ellipticity: float) -> float:
    """Infer surface incidence angle [degrees] from measured ellipticity.

    ``theta = arccos(1 / e)``

    Circular crater (e=1) → theta=0 (normal incidence).
    """
    if ellipticity <= 1.0:
        return 0.0
    return float(np.degrees(np.arccos(np.clip(1.0 / ellipticity, 0.0, 1.0))))


def chord_length_from_ellipticity(
    ellipticity: float,
    R: float = LUNAR_RADIUS_M,
) -> float:
    """Infer chord length [m] from measured crater ellipticity.

    ``L = 2R / e``.  A perfectly circular crater (e=1) implies a
    diametral chord (L=2R).
    """
    if ellipticity <= 0.0:
        return 0.0
    return 2.0 * R / ellipticity


def separation_from_ellipticity(ellipticity: float) -> float:
    """Infer angular separation [degrees] from measured crater ellipticity.

    ``sep = 2 * arcsin(1/e)``.
    """
    if ellipticity < 1.0:
        return 180.0
    return float(np.degrees(2.0 * np.arcsin(np.clip(1.0 / ellipticity, 0.0, 1.0))))


# ======================================================================
# Bearing / azimuth on the sphere
# ======================================================================

def local_bearing_rad(v_from: np.ndarray, v_to: np.ndarray) -> float:
    """Bearing angle [radians, 0=North, pi/2=East] at *v_from* towards *v_to*.

    Uses the spherical forward-azimuth formula.  Both inputs are unit
    vectors (3,).
    """
    lon1, lat1 = np.deg2rad(unit_vectors_to_lonlat(v_from))
    lon2, lat2 = np.deg2rad(unit_vectors_to_lonlat(v_to))
    dlon = lon2 - lon1
    x = np.cos(lat2) * np.sin(dlon)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    return float(np.arctan2(x, y))


def local_bearing_deg(v_from: np.ndarray, v_to: np.ndarray) -> float:
    """Bearing angle [degrees, 0=North, 90=East] at *v_from* towards *v_to*."""
    return float(np.degrees(local_bearing_rad(v_from, v_to)))


def predict_exit_point(
    v_entry: np.ndarray,
    bearing_rad: float,
    angular_sep_rad: float,
) -> np.ndarray:
    """Predict the exit-crater position on the sphere.

    Given an entry point (*v_entry*, unit vector), a local bearing
    (radians, 0=North, π/2=East), and an angular separation (radians),
    compute the destination point along the great circle.

    Uses the spherical destination-point formula.

    Returns a unit vector (3,).
    """
    v = np.asarray(v_entry, dtype=np.float64).ravel()
    lon1, lat1 = np.deg2rad(unit_vectors_to_lonlat(v))
    lon1 = float(lon1)
    lat1 = float(lat1)
    d = float(angular_sep_rad)
    brg = float(bearing_rad)

    lat2 = np.arcsin(
        np.sin(lat1) * np.cos(d) + np.cos(lat1) * np.sin(d) * np.cos(brg)
    )
    lon2 = lon1 + np.arctan2(
        np.sin(brg) * np.sin(d) * np.cos(lat1),
        np.cos(d) - np.sin(lat1) * np.sin(lat2),
    )
    return lonlat_to_unit_vectors(
        np.degrees(lon2), np.degrees(lat2)
    ).ravel()


# ======================================================================
# Bounding-box helpers (for WMS chip queries)
# ======================================================================

def make_bbox_around_point(
    lon_deg: float,
    lat_deg: float,
    span_m: float,
    lunar_radius_m: float = LUNAR_RADIUS_M,
) -> tuple[float, float, float, float] | None:
    """Return (lon_min, lat_min, lon_max, lat_max) for a square chip.

    Returns *None* if the chip straddles the ±180° meridian or the pole
    is too close for a valid bounding box.
    """
    lat_rad = np.deg2rad(lat_deg)
    cos_lat = np.cos(lat_rad)
    if cos_lat < 1e-4:
        return None
    dlat_deg = np.degrees(span_m / lunar_radius_m)
    dlon_deg = np.degrees(span_m / (lunar_radius_m * cos_lat))
    lon_min = lon_deg - dlon_deg / 2.0
    lon_max = lon_deg + dlon_deg / 2.0
    if lon_min < -180.0 or lon_max > 180.0:
        return None
    lat_min = max(lat_deg - dlat_deg / 2.0, -90.0)
    lat_max = min(lat_deg + dlat_deg / 2.0, 90.0)
    return lon_min, lat_min, lon_max, lat_max


def bbox_mpp(
    bbox: tuple[float, float, float, float],
    width_px: int,
    height_px: int,
    lunar_radius_m: float = LUNAR_RADIUS_M,
) -> tuple[float, float]:
    """Metres-per-pixel in (x, y) for a bounding box."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lat_mid = np.deg2rad((lat_min + lat_max) / 2.0)
    width_m = lunar_radius_m * np.cos(lat_mid) * np.deg2rad(lon_max - lon_min)
    height_m = lunar_radius_m * np.deg2rad(lat_max - lat_min)
    return abs(width_m) / width_px, abs(height_m) / height_px


def chip_pixel_to_lonlat(
    x: np.ndarray,
    y: np.ndarray,
    bbox: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert chip pixel coordinates to selenographic (lon, lat) [degrees]."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lon = lon_min + (np.asarray(x, dtype=np.float64) / (width - 1)) * (lon_max - lon_min)
    lat = lat_max - (np.asarray(y, dtype=np.float64) / (height - 1)) * (lat_max - lat_min)
    return lon, lat


# ======================================================================
# Great-circle arc for plotting
# ======================================================================

def slerp_arc(
    v1: np.ndarray,
    v2: np.ndarray,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (lon_deg, lat_deg) arrays along the great-circle arc v1→v2."""
    v1 = np.asarray(v1, dtype=np.float64)
    v2 = np.asarray(v2, dtype=np.float64)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    omega = np.arccos(dot)
    if omega < 1e-10:
        pts = np.repeat(v1[np.newaxis, :], n_points, axis=0)
        return unit_vectors_to_lonlat(pts)

    t = np.linspace(0.0, 1.0, n_points)
    sin_omega = np.sin(omega)
    pts = (
        (np.sin((1.0 - t) * omega)[:, np.newaxis] / sin_omega) * v1
        + (np.sin(t * omega)[:, np.newaxis] / sin_omega) * v2
    )
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return unit_vectors_to_lonlat(pts)


# ======================================================================
# Utility
# ======================================================================

def normalize_percentile(
    image: np.ndarray,
    p_lo: float = 1.0,
    p_hi: float = 99.0,
) -> np.ndarray:
    """Clip and rescale *image* to [0, 1] using percentile bounds."""
    lo, hi = np.percentile(image, [p_lo, p_hi])
    return np.clip((image - lo) / (hi - lo + 1e-6), 0.0, 1.0)


def to_float(value, default: float = np.nan) -> float:
    """Safely cast *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except Exception:
        return default
