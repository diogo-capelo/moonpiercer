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

import heapq
import sys
import time

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
# Exit-position prediction uncertainty
# ======================================================================

def _exit_position_sigma_deg(
    sep_deg: float,
    ellipticity: float,
    orientation_unc_deg: float,
    ellipticity_unc: float,
) -> float:
    """Estimate the 1-sigma angular uncertainty [degrees] of the predicted
    exit point, propagated from shape measurement uncertainties.

    Two error sources:
    - **Cross-track** (from orientation error):
      ``delta_cross = sep_rad * sin(delta_bearing)``
    - **Along-track** (from ellipticity → separation error):
      ``delta_along = |d(sep)/d(e)| * delta_e``

    Returns the combined sigma in degrees.
    """
    sep_rad = np.deg2rad(sep_deg)

    # Cross-track: bearing error → positional offset on the sphere
    orient_unc_rad = np.deg2rad(orientation_unc_deg)
    cross_track_rad = sep_rad * np.sin(min(orient_unc_rad, np.pi / 2.0))

    # Along-track: d(sep)/d(e) for sep = 2*arcsin(1/e)
    #   d(sep)/d(e) = -2 / (e^2 * sqrt(1 - 1/e^2))  [in radians]
    e = max(ellipticity, 1.001)
    denom = e ** 2 * np.sqrt(max(1.0 - 1.0 / e ** 2, 1e-12))
    dsep_de = 2.0 / denom  # magnitude, radians per unit e
    along_track_rad = dsep_de * ellipticity_unc

    sigma_rad = np.sqrt(cross_track_rad ** 2 + along_track_rad ** 2)
    return float(np.degrees(max(sigma_rad, 1e-6)))


def _compute_t_position(
    v_a: np.ndarray,
    v_b: np.ndarray,
    sep_deg: float,
    ellipticity_a: float,
    orientation_a_deg: float,
    shape_reliable_a: bool,
    orientation_unc_a_deg: float,
    ellipticity_unc_a: float,
    ellipticity_b: float,
    orientation_b_deg: float,
    shape_reliable_b: bool,
    orientation_unc_b_deg: float,
    ellipticity_unc_b: float,
    config: ChordConfig,
) -> float:
    """Compute T_position: how close each actual exit crater is to the
    shape-predicted exit, normalised by prediction uncertainty.

    For each crater with a reliable shape, predicts where the partner
    should be, measures the offset, and scores it as a Gaussian.
    Returns the geometric mean of available predictions, or 1.0 if
    neither shape is reliable.
    """
    scores: list[float] = []

    for v_entry, v_exit, ellip, orient, reliable, o_unc, e_unc in [
        (v_a, v_b, ellipticity_a, orientation_a_deg, shape_reliable_a,
         orientation_unc_a_deg, ellipticity_unc_a),
        (v_b, v_a, ellipticity_b, orientation_b_deg, shape_reliable_b,
         orientation_unc_b_deg, ellipticity_unc_b),
    ]:
        if not reliable or not np.isfinite(ellip) or ellip < 1.0:
            continue

        # Predict exit point from this crater's shape
        v_pred, _ = _predict_search_center(v_entry, ellip, orient, reliable)

        # Also predict from the opposite bearing (180° ambiguity)
        pred_sep_rad = np.deg2rad(separation_from_ellipticity(ellip))
        v_pred_opp = predict_exit_point(
            v_entry,
            np.deg2rad(90.0 - orient) + np.pi,
            pred_sep_rad,
        )

        # Take the closer of the two predictions
        offset_deg = min(
            angular_separation_deg(v_pred, v_exit),
            angular_separation_deg(v_pred_opp, v_exit),
        )

        # Compute per-crater sigma from uncertainties if available,
        # otherwise fall back to config default.
        if np.isfinite(o_unc) and np.isfinite(e_unc):
            sigma = _exit_position_sigma_deg(
                sep_deg, ellip, o_unc, e_unc,
            )
        else:
            sigma = config.sigma_position_deg

        scores.append(float(_gaussian_score(offset_deg, sigma)))

    if not scores:
        return 1.0  # no reliable shapes — cannot constrain position
    # Geometric mean of available predictions
    return float(np.prod(scores) ** (1.0 / len(scores)))


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
    orientation_unc_a_deg: float = np.nan,
    orientation_unc_b_deg: float = np.nan,
    ellipticity_unc_a: float = np.nan,
    ellipticity_unc_b: float = np.nan,
) -> dict:
    """Compute the full pair score and its component terms.

    Returns a dict with keys: score, T_diametrality, T_radius,
    T_freshness, T_ellipticity, T_orientation, T_position, T_velocity.
    """
    # T_diametrality: sin(sep/2)^n ∈ [0, 1], weights longer chords more
    if config.prefer_diametrality:
        T_diam = float(np.sin(np.deg2rad(sep_deg / 2.0)) ** config.diametrality_exponent)
    else:
        T_diam = 1.0

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

    # T_position: how close is the actual exit crater to the shape-predicted
    # exit point?  Computed from each crater's shape prediction, with sigma
    # derived from shape measurement uncertainties (or a config fallback).
    chord_length = float(chord_length_from_separation(sep_deg))

    T_position = _compute_t_position(
        v_a, v_b, sep_deg,
        ellipticity_a, orientation_a_deg, shape_reliable_a,
        orientation_unc_a_deg, ellipticity_unc_a,
        ellipticity_b, orientation_b_deg, shape_reliable_b,
        orientation_unc_b_deg, ellipticity_unc_b,
        config,
    )

    score = T_diam * T_radius * T_freshness * T_ellip * T_orient * T_position

    return {
        "score": score,
        "T_diametrality": T_diam,
        "T_radius": T_radius,
        "T_freshness": T_freshness,
        "T_ellipticity": T_ellip,
        "T_orientation": T_orient,
        "T_position": T_position,
        "chord_length_m": chord_length,
        "expected_ellipticity": e_expected,
    }


# ======================================================================
# Rescoring utility
# ======================================================================

def rescore_pairs(
    pairs: pd.DataFrame,
    config: ChordConfig | None = None,
) -> pd.DataFrame:
    """Rescore an existing pairs DataFrame with a (possibly different) config.

    The input *pairs* must contain the columns saved by
    ``build_chord_pairs``: separation_deg, radius_a_m, radius_b_m,
    fi_a, fi_b, ellipticity_a, ellipticity_b, lon_a, lat_a, lon_b,
    lat_b, and (optionally) orientation_a_deg, orientation_b_deg,
    shape_reliable_a, shape_reliable_b.

    Returns a copy with updated score and T_* columns, re-sorted by
    descending score.
    """
    if config is None:
        config = ChordConfig()
    if pairs.empty:
        return pairs.copy()

    df = pairs.copy()

    # Reconstruct unit vectors for orientation scoring
    lons_a = df["lon_a"].to_numpy(dtype=np.float64)
    lats_a = df["lat_a"].to_numpy(dtype=np.float64)
    lons_b = df["lon_b"].to_numpy(dtype=np.float64)
    lats_b = df["lat_b"].to_numpy(dtype=np.float64)
    vecs_a = lonlat_to_unit_vectors(lons_a, lats_a)
    vecs_b = lonlat_to_unit_vectors(lons_b, lats_b)

    has_orient = ("orientation_a_deg" in df.columns
                  and "orientation_b_deg" in df.columns)
    has_reliable = ("shape_reliable_a" in df.columns
                    and "shape_reliable_b" in df.columns)
    has_orient_unc = ("orientation_unc_a_deg" in df.columns
                      and "orientation_unc_b_deg" in df.columns)
    has_ellip_unc = ("ellipticity_unc_a" in df.columns
                     and "ellipticity_unc_b" in df.columns)

    new_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        result = score_pair(
            sep_deg=float(row["separation_deg"]),
            radius_a_m=float(row["radius_a_m"]),
            radius_b_m=float(row["radius_b_m"]),
            fi_a=float(row["fi_a"]),
            fi_b=float(row["fi_b"]),
            ellipticity_a=float(row["ellipticity_a"]),
            ellipticity_b=float(row["ellipticity_b"]),
            orientation_a_deg=float(row["orientation_a_deg"]) if has_orient else 0.0,
            orientation_b_deg=float(row["orientation_b_deg"]) if has_orient else 0.0,
            shape_reliable_a=bool(row["shape_reliable_a"]) if has_reliable else False,
            shape_reliable_b=bool(row["shape_reliable_b"]) if has_reliable else False,
            v_a=vecs_a[i],
            v_b=vecs_b[i],
            config=config,
            orientation_unc_a_deg=float(row["orientation_unc_a_deg"]) if has_orient_unc else np.nan,
            orientation_unc_b_deg=float(row["orientation_unc_b_deg"]) if has_orient_unc else np.nan,
            ellipticity_unc_a=float(row["ellipticity_unc_a"]) if has_ellip_unc else np.nan,
            ellipticity_unc_b=float(row["ellipticity_unc_b"]) if has_ellip_unc else np.nan,
        )
        new_rows.append(result)

    result_df = pd.DataFrame(new_rows)
    for col in result_df.columns:
        df[col] = result_df[col].values

    return df.sort_values("score", ascending=False).reset_index(drop=True)


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
    progress_interval_sec: float | None = None,
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

    # Extract columns
    radii = craters["radius_m"].to_numpy(dtype=np.float64)
    fi = craters["freshness_index"].to_numpy(dtype=np.float64)
    ellip = craters["ellipticity"].to_numpy(dtype=np.float64)
    orient = craters["orientation_deg"].to_numpy(dtype=np.float64)
    reliable = craters["shape_reliable"].to_numpy(dtype=bool)

    # Shape uncertainty columns (optional — absent for older catalogues)
    has_orient_unc = "orientation_unc_deg" in craters.columns
    has_ellip_unc = "ellipticity_unc" in craters.columns
    orient_unc = craters["orientation_unc_deg"].to_numpy(dtype=np.float64) if has_orient_unc else np.full(n, np.nan)
    ellip_unc = craters["ellipticity_unc"].to_numpy(dtype=np.float64) if has_ellip_unc else np.full(n, np.nan)

    # Pixel radii (optional — saved for downstream analysis)
    has_radius_px = "radius_px" in craters.columns
    radius_px = craters["radius_px"].to_numpy(dtype=np.float64) if has_radius_px else np.full(n, np.nan)

    # Hard cut pre-filtering: only qualifying craters enter the kd-tree.
    # This is the key optimisation for large catalogues — a 10° search
    # cone on 1 M craters returns ~7 700 candidates, but 95 %+ fail the
    # freshness/depth check.  Pre-filtering shrinks the tree and ensures
    # every kd-tree hit is already a valid candidate.
    min_fi = config.min_freshness
    min_depth = config.min_depth_proxy
    depth_proxy = craters["depth_proxy"].to_numpy(dtype=np.float64) if "depth_proxy" in craters.columns else np.ones(n)

    qual_mask = (fi >= min_fi) & (depth_proxy >= min_depth)
    qual_idx = np.where(qual_mask)[0]
    n_qual = len(qual_idx)

    if n_qual < 2:
        return pd.DataFrame()

    # Build kd-tree over qualifying craters only
    qual_vectors = vectors[qual_idx]
    tree = cKDTree(qual_vectors)
    # Reverse mapping: kd-tree index → original catalogue index
    # (qual_idx[k] gives the original index for kd-tree entry k)

    print(
        f"[pairing] Pre-filter: {n_qual:,d} of {n:,d} craters qualify "
        f"({n_qual / n:.1%})",
        file=sys.stderr, flush=True,
    )

    # Bounded min-heap: keeps only the top max_pairs_in_memory pairs by score.
    # This replaces an unbounded list (which OOM-kills at ~400M+ pairs).
    # The global seen_pairs set is replaced by a per-crater local set that
    # only deduplicates hits from the forward/backward search of the SAME
    # crater — the `j > i` guard already prevents cross-crater duplicates.
    _max_pairs = config.max_pairs_in_memory
    pairs_heap: list[tuple[float, int, dict]] = []  # min-heap (score, ctr, row)
    _heap_ctr = 0
    _total_candidates = 0  # pairs that passed all hard cuts (for progress log)

    min_chord_sep_rad = np.deg2rad(config.min_chord_sep_deg)

    use_progress = progress_interval_sec is not None and progress_interval_sec > 0
    t_start = time.monotonic()
    next_report = t_start + progress_interval_sec if use_progress else None

    for qi in range(n_qual):
        i = int(qual_idx[qi])

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
        cone_rad = np.deg2rad(cone_deg)
        euclidean_radius = 2.0 * np.sin(cone_rad / 2.0)

        # Local deduplication: prevents the same j being added twice when
        # both forward and backward search centres find it.  Much cheaper
        # than a global set (cleared every outer iteration).
        seen_j: set[int] = set()

        for v_center in search_centers:
            cand_qi_list = tree.query_ball_point(v_center, euclidean_radius)

            for cand_qi in cand_qi_list:
                j = int(qual_idx[cand_qi])
                if j <= i:  # self-pair and cross-crater duplicates (i always < j)
                    continue
                if j in seen_j:  # duplicate from forward/backward cone of same i
                    continue

                # Hard cuts
                if abs(fi[i] - fi[j]) > config.max_freshness_diff:
                    continue
                if abs(radii[i] - radii[j]) > config.max_radius_diff_m:
                    continue
                sep = angular_separation_deg(v_i, vectors[j])
                if sep < config.min_chord_sep_deg:
                    continue

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
                    orientation_unc_a_deg=float(orient_unc[i]),
                    orientation_unc_b_deg=float(orient_unc[j]),
                    ellipticity_unc_a=float(ellip_unc[i]),
                    ellipticity_unc_b=float(ellip_unc[j]),
                )

                seen_j.add(j)
                _total_candidates += 1

                pair_row = {
                    "idx_a": i,
                    "idx_b": j,
                    "lon_a": float(lons[i]),
                    "lat_a": float(lats[i]),
                    "lon_b": float(lons[j]),
                    "lat_b": float(lats[j]),
                    "separation_deg": sep,
                    "radius_a_m": float(radii[i]),
                    "radius_b_m": float(radii[j]),
                    "radius_px_a": float(radius_px[i]),
                    "radius_px_b": float(radius_px[j]),
                    "fi_a": float(fi[i]),
                    "fi_b": float(fi[j]),
                    "ellipticity_a": float(ellip[i]),
                    "ellipticity_b": float(ellip[j]),
                    "orientation_a_deg": float(orient[i]),
                    "orientation_b_deg": float(orient[j]),
                    "orientation_unc_a_deg": float(orient_unc[i]),
                    "orientation_unc_b_deg": float(orient_unc[j]),
                    "ellipticity_unc_a": float(ellip_unc[i]),
                    "ellipticity_unc_b": float(ellip_unc[j]),
                    "shape_reliable_a": bool(reliable[i]),
                    "shape_reliable_b": bool(reliable[j]),
                    **result,
                }

                # Bounded min-heap: only keep the top _max_pairs by score
                s = result["score"]
                if len(pairs_heap) < _max_pairs:
                    heapq.heappush(pairs_heap, (s, _heap_ctr, pair_row))
                elif s > pairs_heap[0][0]:
                    heapq.heapreplace(pairs_heap, (s, _heap_ctr, pair_row))
                _heap_ctr += 1

        if next_report is not None and time.monotonic() >= next_report:
            done = qi + 1
            elapsed = time.monotonic() - t_start
            rate = done / elapsed if elapsed > 0 else 0.0
            heap_min = pairs_heap[0][0] if pairs_heap else 0.0
            print(
                f"[pairing] {done}/{n_qual} qualifying craters ({done / n_qual:.1%}) "
                f"elapsed={elapsed:.1f}s rate={rate:.2f}/s "
                f"pairs_found={_total_candidates:,d} "
                f"heap={len(pairs_heap):,d}/{_max_pairs:,d} "
                f"heap_min={heap_min:.6f}",
                file=sys.stderr,
                flush=True,
            )
            next_report = time.monotonic() + progress_interval_sec

    if not pairs_heap:
        return pd.DataFrame()

    pairs = pd.DataFrame([item[2] for item in pairs_heap])
    pairs = pairs.sort_values("score", ascending=False).reset_index(drop=True)
    return pairs


def max_pair_score(
    craters: pd.DataFrame,
    config: ChordConfig | None = None,
    return_details: bool = False,
) -> float | tuple[float, dict]:
    """Compute the maximum pair score without storing all pairs.

    Functionally equivalent to::

        pairs = build_chord_pairs(craters, config)
        return float(pairs["score"].max()) if len(pairs) > 0 else 0.0

    but uses O(1) memory instead of O(N_pairs), making it suitable for
    Monte Carlo null-model trials where only the best score is needed.

    If *return_details* is True, returns ``(score, details_dict)`` where
    *details_dict* contains the characteristics of the best-scoring pair
    (separation, radii, freshness, ellipticities, orientations, score
    components).  This enables rescoring null trials with a different
    config without rerunning the Monte Carlo.
    """
    if config is None:
        config = ChordConfig()

    n = len(craters)
    if n < 2:
        if return_details:
            return 0.0, {}
        return 0.0

    lons = craters["lon_deg"].to_numpy(dtype=np.float64)
    lats = craters["lat_deg"].to_numpy(dtype=np.float64)
    vectors = lonlat_to_unit_vectors(lons, lats)

    radii = craters["radius_m"].to_numpy(dtype=np.float64)
    fi = craters["freshness_index"].to_numpy(dtype=np.float64)
    ellip = craters["ellipticity"].to_numpy(dtype=np.float64)
    orient = craters["orientation_deg"].to_numpy(dtype=np.float64)
    reliable = craters["shape_reliable"].to_numpy(dtype=bool)

    has_orient_unc = "orientation_unc_deg" in craters.columns
    has_ellip_unc = "ellipticity_unc" in craters.columns
    orient_unc = craters["orientation_unc_deg"].to_numpy(dtype=np.float64) if has_orient_unc else np.full(n, np.nan)
    ellip_unc = craters["ellipticity_unc"].to_numpy(dtype=np.float64) if has_ellip_unc else np.full(n, np.nan)

    min_fi = config.min_freshness
    min_depth = config.min_depth_proxy
    depth_proxy = (
        craters["depth_proxy"].to_numpy(dtype=np.float64)
        if "depth_proxy" in craters.columns
        else np.ones(n)
    )

    # Pre-filter: kd-tree from qualifying craters only (same as build_chord_pairs)
    qual_mask = (fi >= min_fi) & (depth_proxy >= min_depth)
    qual_idx = np.where(qual_mask)[0]
    n_qual = len(qual_idx)

    if n_qual < 2:
        if return_details:
            return 0.0, {}
        return 0.0

    qual_vectors = vectors[qual_idx]
    tree = cKDTree(qual_vectors)

    best = 0.0
    best_details: dict = {}

    for qi in range(n_qual):
        i = int(qual_idx[qi])

        v_i = vectors[i]
        v_pred, pred_sep_rad = _predict_search_center(
            v_i, float(ellip[i]), float(orient[i]), bool(reliable[i])
        )

        if reliable[i] and np.isfinite(ellip[i]) and ellip[i] > 1.0:
            cone_deg = config.search_cone_half_deg_reliable
            v_pred_opp = predict_exit_point(
                v_i,
                np.deg2rad(90.0 - orient[i]) + np.pi,
                pred_sep_rad,
            )
            search_centers = [v_pred, v_pred_opp]
        else:
            cone_deg = config.search_cone_half_deg_unreliable
            search_centers = [v_pred]

        cone_rad = np.deg2rad(cone_deg)
        euclidean_radius = 2.0 * np.sin(cone_rad / 2.0)

        for v_center in search_centers:
            cand_qi_list = tree.query_ball_point(v_center, euclidean_radius)

            for cand_qi in cand_qi_list:
                j = int(qual_idx[cand_qi])
                if j <= i:
                    continue

                if abs(fi[i] - fi[j]) > config.max_freshness_diff:
                    continue
                if abs(radii[i] - radii[j]) > config.max_radius_diff_m:
                    continue

                sep = angular_separation_deg(v_i, vectors[j])
                if sep < config.min_chord_sep_deg:
                    continue

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
                    orientation_unc_a_deg=float(orient_unc[i]),
                    orientation_unc_b_deg=float(orient_unc[j]),
                    ellipticity_unc_a=float(ellip_unc[i]),
                    ellipticity_unc_b=float(ellip_unc[j]),
                )

                s = result["score"]
                if s > best:
                    best = s
                    if return_details:
                        best_details = {
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
                            "orientation_a_deg": float(orient[i]),
                            "orientation_b_deg": float(orient[j]),
                            "orientation_unc_a_deg": float(orient_unc[i]),
                            "orientation_unc_b_deg": float(orient_unc[j]),
                            "ellipticity_unc_a": float(ellip_unc[i]),
                            "ellipticity_unc_b": float(ellip_unc[j]),
                            "shape_reliable_a": bool(reliable[i]),
                            "shape_reliable_b": bool(reliable[j]),
                            **result,
                        }

    if return_details:
        return best, best_details
    return best


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
