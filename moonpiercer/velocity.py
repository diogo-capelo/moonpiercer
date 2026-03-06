"""PBH velocity model, transit physics, and angular-separation constraints.

Key physics: a PBH traversing a chord of length *L* through the Moon at
speed *v* takes time *t = L / v*.  During that transit the Moon rotates
by an angle ``delta_theta = omega * t``, displacing the exit crater
relative to the geometric chord endpoint.

For a diametral chord (L = 2R ≈ 3475 km) and the slowest plausible
encounter speed (v ≈ 50 km/s), the maximum rotation is ~0.011° (~580 m
at the surface).  This is the physical upper bound on the angular
offset between a crater pair and the ideal chord geometry.
"""

from __future__ import annotations

import numpy as np

from moonpiercer.constants import (
    LUNAR_OMEGA_RAD_S,
    LUNAR_RADIUS_M,
    V_CIRCULAR_KM_S,
    V_ESCAPE_KM_S,
)


# ======================================================================
# Transit time and rotation offset
# ======================================================================

def transit_time_s(
    chord_length_m: float | np.ndarray,
    v_km_s: float | np.ndarray,
) -> float | np.ndarray:
    """Transit time [s] for a PBH traversing a chord at speed *v*."""
    return np.asarray(chord_length_m) / (np.asarray(v_km_s) * 1.0e3)


def rotation_offset_rad(
    chord_length_m: float | np.ndarray,
    v_km_s: float | np.ndarray,
    omega: float = LUNAR_OMEGA_RAD_S,
) -> float | np.ndarray:
    """Angular rotation offset [radians] of the exit crater due to lunar rotation."""
    return omega * transit_time_s(chord_length_m, v_km_s)


def rotation_offset_deg(
    chord_length_m: float | np.ndarray,
    v_km_s: float | np.ndarray,
    omega: float = LUNAR_OMEGA_RAD_S,
) -> float | np.ndarray:
    """Angular rotation offset [degrees]."""
    return np.degrees(rotation_offset_rad(chord_length_m, v_km_s, omega))


def max_rotation_offset_deg(
    chord_length_m: float | np.ndarray,
    v_min_km_s: float = 50.0,
    omega: float = LUNAR_OMEGA_RAD_S,
) -> float | np.ndarray:
    """Maximum rotation offset [degrees] for the slowest PBH speed."""
    return rotation_offset_deg(chord_length_m, v_min_km_s, omega)


# ======================================================================
# Velocity inference
# ======================================================================

def velocity_from_offset(
    angular_offset_rad: float | np.ndarray,
    chord_length_m: float | np.ndarray,
    omega: float = LUNAR_OMEGA_RAD_S,
) -> float | np.ndarray:
    """Infer PBH speed [km/s] from the observed angular offset.

    v = omega * L / delta_theta
    """
    offset = np.asarray(angular_offset_rad, dtype=np.float64)
    L = np.asarray(chord_length_m, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        v_m_s = np.where(offset > 1e-15, omega * L / offset, np.inf)
    return v_m_s / 1.0e3


# ======================================================================
# Standard Halo Model velocity distribution
# ======================================================================

def maxwell_boltzmann_speed_pdf(
    v: np.ndarray,
    v0: float = V_CIRCULAR_KM_S,
    v_esc: float = V_ESCAPE_KM_S,
) -> np.ndarray:
    """Truncated Maxwell-Boltzmann speed distribution (Standard Halo Model).

    f(v) ∝ v² exp(-v² / v0²)  for v < v_esc, 0 otherwise.

    Normalised numerically.
    """
    v = np.asarray(v, dtype=np.float64)
    raw = v ** 2 * np.exp(-(v ** 2) / (v0 ** 2))
    raw[v > v_esc] = 0.0
    raw[v < 0.0] = 0.0
    norm = np.trapezoid(raw, v)
    if norm > 0:
        raw /= norm
    return raw


def velocity_cdf(
    v_grid: np.ndarray,
    v0: float = V_CIRCULAR_KM_S,
    v_esc: float = V_ESCAPE_KM_S,
) -> np.ndarray:
    """Cumulative distribution function of the SHM speed distribution."""
    pdf = maxwell_boltzmann_speed_pdf(v_grid, v0, v_esc)
    cdf = np.cumsum(pdf) * np.gradient(v_grid)
    cdf /= cdf[-1] if cdf[-1] > 0 else 1.0
    return cdf


# ======================================================================
# Angular offset probability
# ======================================================================

def offset_probability_factor(
    angular_offset_deg: float,
    chord_length_m: float,
    v0_km_s: float = V_CIRCULAR_KM_S,
    v_esc_km_s: float = V_ESCAPE_KM_S,
    omega: float = LUNAR_OMEGA_RAD_S,
) -> float:
    """Probability weight for an observed angular offset given the velocity distribution.

    Returns a value in [0, 1] representing how likely this offset is
    under the SHM velocity distribution.  An offset of 0 is consistent
    with any fast PBH (high probability); larger offsets require slower
    PBHs (lower probability).
    """
    offset_rad = np.deg2rad(float(angular_offset_deg))
    if offset_rad < 1e-15:
        return 1.0

    v_inferred = velocity_from_offset(offset_rad, chord_length_m, omega)
    v_inf = float(v_inferred)

    if not np.isfinite(v_inf) or v_inf > v_esc_km_s or v_inf < 0:
        return 0.0

    # Evaluate the survival function P(v <= v_inferred)
    # Larger offset → slower PBH → P(v <= v_slow) is smaller → less likely
    v_grid = np.linspace(0.01, v_esc_km_s, 2000)
    cdf = velocity_cdf(v_grid, v0_km_s, v_esc_km_s)
    prob = float(np.interp(v_inf, v_grid, cdf))

    return prob


# ======================================================================
# Maximum physically allowed angular separation
# ======================================================================

def max_physical_angular_offset_deg(
    v_min_km_s: float = 50.0,
    R: float = LUNAR_RADIUS_M,
    omega: float = LUNAR_OMEGA_RAD_S,
) -> float:
    """Maximum angular offset [degrees] from lunar rotation.

    Uses the diametral chord (longest possible) and slowest plausible speed.
    """
    L_max = 2.0 * R
    return float(rotation_offset_deg(L_max, v_min_km_s, omega))
