"""Physical constants for the MOONPIERCER pipeline.

All values are in SI units unless otherwise noted.  Sources are cited inline.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Lunar parameters
# ---------------------------------------------------------------------------
LUNAR_RADIUS_M: float = 1_737_400.0
"""Mean lunar radius [m].  IAU 2015 value."""

LUNAR_SIDEREAL_PERIOD_S: float = 27.321661 * 86_400.0
"""Sidereal rotation period of the Moon [s]."""

LUNAR_OMEGA_RAD_S: float = 2.0 * math.pi / LUNAR_SIDEREAL_PERIOD_S
"""Lunar sidereal angular velocity [rad s^-1].  ~2.6617e-6."""

LUNAR_SURFACE_AREA_KM2: float = 4.0 * math.pi * (LUNAR_RADIUS_M / 1.0e3) ** 2
"""Total lunar surface area [km^2].  ~3.793e7."""

LUNAR_DIAMETER_M: float = 2.0 * LUNAR_RADIUS_M
"""Lunar diameter [m]."""

# ---------------------------------------------------------------------------
# Dark-matter flux through the Moon  (Santarelli, Caplan & Smith 2025,
# MNRAS 538, 108, Table 2)
# ---------------------------------------------------------------------------
DM_FLUX_HALO_G_PER_S: float = 2.277
"""Dark-matter mass flux through the Moon from the Galactic halo [g s^-1]."""

DM_FLUX_HALO_DISC_G_PER_S: float = 3.19
"""Combined halo + thick dark disc mass flux [g s^-1]."""

# ---------------------------------------------------------------------------
# PBH velocity distribution  (Standard Halo Model; Caplan+ 2023, Santarelli+
# 2025)
# ---------------------------------------------------------------------------
V_CIRCULAR_KM_S: float = 220.0
"""Local circular speed of the Milky Way [km s^-1].  SHM peak."""

V_ESCAPE_KM_S: float = 544.0
"""Galactic escape speed at the solar radius [km s^-1]."""

V_DARK_DISC_KM_S: float = 50.0
"""Solar velocity relative to the thick dark disc [km s^-1].

Santarelli+ 2025, velocity dispersion sigma = (63, 39, 39) km/s in
cylindrical (R, phi, z) components.
"""

V_MIN_PLAUSIBLE_KM_S: float = 20.0
"""Conservative lower bound on PBH encounter speed [km s^-1].

Below this, gravitational focusing would dominate and the encounter
geometry changes qualitatively."""

V_MAX_PLAUSIBLE_KM_S: float = 600.0
"""Conservative upper bound, above Galactic escape speed [km s^-1]."""

# ---------------------------------------------------------------------------
# Topographic diffusion (Fassett & Thomson 2014; Fassett 2022)
# ---------------------------------------------------------------------------
KAPPA_CLASSICAL_LOW_M2_PER_MYR: float = 0.5
"""Lower bound of topographic diffusivity [m^2 Myr^-1]."""

KAPPA_CLASSICAL_HIGH_M2_PER_MYR: float = 5.5
"""Upper bound (Fassett & Thomson 2014 global average) [m^2 Myr^-1]."""

KAPPA_ADOPTED_M2_PER_MYR: float = 1.6
"""Geometric mean adopted as point estimate [m^2 Myr^-1]."""

# ---------------------------------------------------------------------------
# WMS endpoints
# ---------------------------------------------------------------------------
WMS_BASE_URL: str = "https://wms.im-ldi.com/"
"""Lunar Mapping and Modeling Portal WMS endpoint."""

NAC_STAMP_LAYER: str = "luna_pds_nac_stamp"
"""LROC NAC stamp imagery layer."""

NAC_OBSERVATION_LAYER: str = "luna_pds_nac_observation"
"""LROC NAC observation metadata layer."""

LOLA_DTM_LAYER: str = "luna_wac_dtm_numeric_meters"
"""LOLA digital terrain model (elevation in metres) layer."""
