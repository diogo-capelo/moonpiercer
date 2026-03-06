"""Chord-based crater pair matching and scoring.

The pairing engine uses a **shape-directed chord search** strategy:

1. For each crater with reliable shape measurements, infer the most
   likely chord direction from the crater's ellipticity and orientation.
2. Predict the exit-crater position on the sphere.
3. Search a narrow cone around the predicted exit point using a kd-tree.
4. Apply hard cuts (radius match, freshness match, angular separation).
5. Score surviving pairs with a multi-component metric.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from moonpiercer.config import ChordConfig
from moonpiercer.geometry import (
    angular_separation_deg,
    angular_separation_deg_batch,
    chord_length_from_separation,
    expected_ellipticity_from_separation,
    local_bearing_rad,
    lonlat_to_unit_vectors,
    predict_exit_point,
    separation_from_ellipticity,
)
from moonpiercer.velocity import offset_probability_factor


# ======================================================================
# Score components (Gaussian terms)
# ======================================================================

def _gaussian_score(diff: float | np.ndarray, sigma: float) -> float | np.ndarray:
    """exp(-(diff/sigma)²)."""
    return np.exp(-((np.asarray(diff) / sigma) ** 2))


# ======================================================================
# Pair scoring
# ======================================================================

def score_pair(
    sep_deg: float,
    radius_a_m: float,
    radius_b_m: float,
    fi_a: float,
    fi_b: float,
    ellipticity_a: float,
    ellipticity_b: float,
    orientation_a_deg: float,
    orientation_b_deg: float,
    shape_reliable_a: bool,
    shape_reliable_b: bool,
    v_a: np.ndarray,
    v_b: np.ndarray,
    config: ChordConfig,
) -> dict:
    """Compute the full pair score and its component terms.

    Returns a dict with keys: score, T_diametrality, T_radius,
    T_freshness, T_ellipticity, T_orientation, T_velocity.
    """
    # T_diametrality: sin(sep/2) ∈ [0, 1], weights longer chords more
    T_diam = float(np.sin(np.deg2rad(sep_deg / 2.0))) if config.prefer_diametrality else 1.0

    # T_radius
    dr = abs(radius_a_m - radius_b_m)
    T_radius = float(_gaussian_score(dr, config.sigma_radius))

    # T_freshness
    dfi = abs(fi_a - fi_b)
    T_freshness = float(_gaussian_score(dfi, config.sigma_freshness))

    # T_ellipticity: compare each crater's measured ellipticity to
    # the predicted ellipticity from the chord geometry
    e_expected = float(expected_ellipticity_from_separation(sep_deg))
    T_ellip = 1.0
    if shape_reliable_a and np.isfinite(ellipticity_a):
        T_ellip *= float(_gaussian_score(ellipticity_a - e_expected, config.sigma_ellipticity))
    if shape_reliable_b and np.isfinite(ellipticity_b):
        T_ellip *= float(_gaussian_score(ellipticity_b - e_expected, config.sigma_ellipticity))

    # T_orientation: both craters' major axes should align with the
    # great circle connecting them
    T_orient = 1.0
    if (shape_reliable_a and shape_reliable_b
            and np.isfinite(orientation_a_deg) and np.isfinite(orientation_b_deg)
            and ellipticity_a > 1.05 and ellipticity_b > 1.05):
        # Bearing from A→B and B→A
        brg_ab = float(np.degrees(local_bearing_rad(v_a, v_b)))
        brg_ba = float(np.degrees(local_bearing_rad(v_b, v_a)))

        # Compare orientation of each crater with the chord bearing at that point
        # Orientation has 180° ambiguity, so use min angle mod 180
        def _angle_diff_mod180(a, b):
            d = (a - b) % 180.0
            return min(d, 180.0 - d)

        diff_a = _angle_diff_mod180(orientation_a_deg, brg_ab)
        diff_b = _angle_diff_mod180(orientation_b_deg, brg_ba)
        T_orient = float(
            _gaussian_score(diff_a, config.sigma_orientation_deg)
            * _gaussian_score(diff_b, config.sigma_orientation_deg)
        )

    # T_velocity: based on angular offset consistency with velocity distribution
    # The "offset" is how far the pair deviates from perfect chord geometry
    # For now, compute as the actual angular separation vs the prediction from
    # the chord (if available). The main constraint is via the search cone.
    chord_length = float(chord_length_from_separation(sep_deg))
    T_velocity = 1.0  # default — the search cone already enforces the hard cut

    score = T_diam * T_radius * T_freshness * T_ellip * T_orient * T_velocity

    return {
        "score": score,
        "T_diametrality": T_diam,
        "T_radius": T_radius,
        "T_freshness": T_freshness,
        "T_ellipticity": T_ellip,
        "T_orientation": T_orient,
        "T_velocity": T_velocity,
        "chord_length_m": chord_length,
        "expected_ellipticity": e_expected,
    }


# ======================================================================
# Shape-directed search: predict exit point from crater shape
# ======================================================================

def _predict_search_center(
    v_entry: np.ndarray,
    ellipticity: float,
    orientation_deg: float,
    shape_reliable: bool,
) -> tuple[np.ndarray, float]:
    """Predict exit-point unit vector and angular separation from shape.

    Returns (v_exit_predicted, angular_sep_rad).
    If shape is unreliable, returns the antipodal point with sep = pi.
    """
    if not shape_reliable or not np.isfinite(ellipticity) or ellipticity < 1.0:
        # Default: search near the antipode
        return -v_entry, np.pi

    sep_deg = separation_from_ellipticity(ellipticity)
    sep_rad = np.deg2rad(sep_deg)

    # Bearing from orientation: the major axis of the crater ellipse
    # points along the chord direction projected on the surface.
    # The orientation_deg is measured from the pixel x-axis, which in our
    # coordinate system aligns with East. For bearing (0=North, 90=East):
    bearing_rad = np.deg2rad(90.0 - orientation_deg)  # convert from x-axis to bearing

    v_exit = predict_exit_point(v_entry, bearing_rad, sep_rad)
    return v_exit, sep_rad


# ======================================================================
# Main pairing function
# ======================================================================

def build_chord_pairs(
    craters: pd.DataFrame,
    config: ChordConfig | None = None,
) -> pd.DataFrame:
    """Find and score crater pairs using shape-directed chord search.

    Parameters
    ----------
    craters : DataFrame
        Global crater catalogue with columns: lon_deg, lat_deg, radius_m,
        freshness_index, ellipticity, orientation_deg, shape_reliable,
        plus any metadata (product_id, etc.).
    config : ChordConfig, optional

    Returns
    -------
    pairs : DataFrame
        Scored pairs with columns: idx_a, idx_b, score, and all component
        terms, plus crater metadata for both members.
    """
    if config is None:
        config = ChordConfig()

    n = len(craters)
    if n < 2:
        return pd.DataFrame()

    # Build unit vectors
    lons = craters["lon_deg"].to_numpy(dtype=np.float64)
    lats = craters["lat_deg"].to_numpy(dtype=np.float64)
    vectors = lonlat_to_unit_vectors(lons, lats)

    # Build kd-tree over unit vectors
    tree = cKDTree(vectors)

    # Extract columns
    radii = craters["radius_m"].to_numpy(dtype=np.float64)
    fi = craters["freshness_index"].to_numpy(dtype=np.float64)
    ellip = craters["ellipticity"].to_numpy(dtype=np.float64)
    orient = craters["orientation_deg"].to_numpy(dtype=np.float64)
    reliable = craters["shape_reliable"].to_numpy(dtype=bool)

    # Hard cut pre-filtering
    min_fi = config.min_freshness
    min_depth = config.min_depth_proxy
    depth_proxy = craters["depth_proxy"].to_numpy(dtype=np.float64) if "depth_proxy" in craters.columns else np.ones(n)

    # For each crater, predict the exit point and search
    pairs_list: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()

    min_chord_sep_rad = np.deg2rad(config.min_chord_sep_deg)

    for i in range(n):
        # Skip craters that don't meet freshness/depth threshold
        if fi[i] < min_fi or depth_proxy[i] < min_depth:
            continue

        v_i = vectors[i]

        # Predict exit point
        v_pred, pred_sep_rad = _predict_search_center(
            v_i, float(ellip[i]), float(orient[i]), bool(reliable[i])
        )

        # Search cone radius (in Euclidean chord distance for kd-tree)
        if reliable[i] and np.isfinite(ellip[i]) and ellip[i] > 1.0:
            cone_deg = config.search_cone_half_deg_reliable
            # Also search the opposite bearing (180° ambiguity in orientation)
            v_pred_opp = predict_exit_point(
                v_i,
                np.deg2rad(90.0 - orient[i]) + np.pi,  # opposite bearing
                pred_sep_rad,
            )
            search_centers = [v_pred, v_pred_opp]
        else:
            cone_deg = config.search_cone_half_deg_unreliable
            search_centers = [v_pred]

        # Convert angular cone to Euclidean distance for kd-tree
        # For small angles: d_euclidean ≈ 2*sin(theta/2) ≈ theta (radians)
        cone_rad = np.deg2rad(cone_deg)
        euclidean_radius = 2.0 * np.sin(cone_rad / 2.0)

        for v_center in search_centers:
            # Query kd-tree
            candidates = tree.query_ball_point(v_center, euclidean_radius)

            for j in candidates:
                if j <= i:  # avoid self-pair and duplicates (i < j)
                    continue
                pair_key = (i, j)
                if pair_key in seen_pairs:
                    continue

                # Hard cuts
                # 1. Freshness
                if fi[j] < min_fi or depth_proxy[j] < min_depth:
                    continue
                if abs(fi[i] - fi[j]) > config.max_freshness_diff:
                    continue

                # 2. Radius match
                if abs(radii[i] - radii[j]) > config.max_radius_diff_m:
                    continue

                # 3. Minimum angular separation
                sep = angular_separation_deg(v_i, vectors[j])
                if sep < config.min_chord_sep_deg:
                    continue

                # Score the pair
                result = score_pair(
                    sep_deg=sep,
                    radius_a_m=float(radii[i]),
                    radius_b_m=float(radii[j]),
                    fi_a=float(fi[i]),
                    fi_b=float(fi[j]),
                    ellipticity_a=float(ellip[i]),
                    ellipticity_b=float(ellip[j]),
                    orientation_a_deg=float(orient[i]),
                    orientation_b_deg=float(orient[j]),
                    shape_reliable_a=bool(reliable[i]),
                    shape_reliable_b=bool(reliable[j]),
                    v_a=v_i,
                    v_b=vectors[j],
                    config=config,
                )

                seen_pairs.add(pair_key)
                pairs_list.append({
                    "idx_a": i,
                    "idx_b": j,
                    "lon_a": float(lons[i]),
                    "lat_a": float(lats[i]),
                    "lon_b": float(lons[j]),
                    "lat_b": float(lats[j]),
                    "separation_deg": sep,
                    "radius_a_m": float(radii[i]),
                    "radius_b_m": float(radii[j]),
                    "fi_a": float(fi[i]),
                    "fi_b": float(fi[j]),
                    "ellipticity_a": float(ellip[i]),
                    "ellipticity_b": float(ellip[j]),
                    **result,
                })

    if not pairs_list:
        return pd.DataFrame()

    pairs = pd.DataFrame(pairs_list)
    pairs = pairs.sort_values("score", ascending=False).reset_index(drop=True)
    return pairs


def select_top_nonoverlapping_pairs(
    pairs: pd.DataFrame,
    top_k: int = 50,
) -> pd.DataFrame:
    """Select the top-k non-overlapping pairs (no crater appears twice)."""
    if pairs.empty:
        return pairs

    pairs = pairs.sort_values("score", ascending=False).reset_index(drop=True)
    used: set[int] = set()
    keep: list[int] = []

    for row_idx in range(len(pairs)):
        ia = int(pairs.iloc[row_idx]["idx_a"])
        ib = int(pairs.iloc[row_idx]["idx_b"])
        if ia not in used and ib not in used:
            keep.append(row_idx)
            used.add(ia)
            used.add(ib)
            if len(keep) >= top_k:
                break

    return pairs.iloc[keep].reset_index(drop=True)
