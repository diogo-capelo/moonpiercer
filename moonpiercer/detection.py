"""Crater detection via scale-normalised LoG + elliptical shape characterisation.

The pipeline:
1. Scale-normalised Laplacian-of-Gaussian (LoG) blob detection on inverted,
   percentile-normalised imagery.
2. Non-maximum suppression in (x, y, scale) space.
3. No artificial cap on candidate count — all detections above the adaptive
   threshold are retained.
4. Elliptical shape characterisation via weighted intensity moments (full
   eigen-decomposition for orientation).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from moonpiercer.config import ChordConfig
from moonpiercer.geometry import normalize_percentile


# ======================================================================
# Core LoG detection
# ======================================================================

def _log_blob_detect(
    gray: np.ndarray,
    mpp_mean: float,
    target_radius_m: tuple[float, float],
    peak_quantile: float,
    n_scales: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Scale-normalised LoG blob detection.

    Returns (y, x, radius_px, strength, threshold) where y/x are
    pixel coordinates, radius_px is the estimated radius in pixels,
    strength is the LoG response at each detection, and threshold
    is the adaptive quantile threshold used.
    """
    min_radius_px = max(target_radius_m[0] / mpp_mean, 1.2)
    max_radius_px = max(target_radius_m[1] / mpp_mean, min_radius_px * 1.2)

    work = 1.0 - normalize_percentile(gray, 1.0, 99.0)

    sigma_min = max(min_radius_px / np.sqrt(2.0), 0.55)
    sigma_max = max(max_radius_px / np.sqrt(2.0), sigma_min * 1.01)
    sigmas = np.geomspace(sigma_min, sigma_max, n_scales)

    cube = np.stack(
        [(s ** 2) * (-ndi.gaussian_laplace(work, sigma=s)) for s in sigmas],
        axis=-1,
    )

    threshold = float(np.quantile(cube, peak_quantile))

    local_max = cube == ndi.maximum_filter(cube, size=(5, 5, 3), mode="nearest")
    y, x, k = np.where(local_max & (cube >= threshold))

    if len(y) == 0:
        return (
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            np.array([], dtype=float),
            threshold,
        )

    strength = cube[y, x, k]
    radius_px = np.sqrt(2.0) * sigmas[k]

    return y.astype(float), x.astype(float), radius_px, strength, threshold


# ======================================================================
# Non-maximum suppression (no candidate cap)
# ======================================================================

def _nms_no_cap(
    y: np.ndarray,
    x: np.ndarray,
    radius_px: np.ndarray,
    strength: np.ndarray,
    min_separation_factor: float = 0.75,
    min_separation_px: float = 3.0,
) -> np.ndarray:
    """Non-maximum suppression without an artificial cap on detections.

    Returns indices into the input arrays for kept detections.
    """
    order = np.argsort(strength)[::-1]
    keep: list[int] = []

    for idx in order:
        if not keep:
            keep.append(idx)
            continue
        kept_arr = np.array(keep, dtype=int)
        dy = y[kept_arr] - y[idx]
        dx = x[kept_arr] - x[idx]
        min_sep = max(min_separation_px, min_separation_factor * radius_px[idx])
        if np.min(dx * dx + dy * dy) > min_sep ** 2:
            keep.append(idx)

    return np.asarray(keep, dtype=int)


# ======================================================================
# Elliptical shape characterisation
# ======================================================================

def _characterise_shape(
    gray: np.ndarray,
    x: float,
    y: float,
    radius_px: float,
    confidence_min_radius_px: float = 5.0,
) -> dict:
    """Compute elliptical shape parameters from weighted intensity moments.

    Returns a dict with keys:
        circularity, ellipticity, orientation_deg, shape_reliable
    """
    r = max(3, int(np.ceil(2.5 * radius_px)))
    yc, xc = int(round(y)), int(round(x))

    y0, y1 = max(0, yc - r), min(gray.shape[0], yc + r + 1)
    x0, x1 = max(0, xc - r), min(gray.shape[1], xc + r + 1)
    patch = gray[y0:y1, x0:x1]

    fail = {
        "circularity": np.nan,
        "ellipticity": np.nan,
        "orientation_deg": np.nan,
        "shape_reliable": False,
    }
    if patch.size < 25:
        return fail

    # Weight = inverted intensity above median (crater interior is dark)
    inv = 1.0 - patch
    inv = inv - np.percentile(inv, 50)
    inv[inv < 0] = 0.0
    wsum = float(inv.sum())
    if wsum <= 1e-8:
        return fail

    yy, xx = np.indices(inv.shape, dtype=float)
    mx = float((xx * inv).sum() / wsum)
    my = float((yy * inv).sum() / wsum)
    dx = xx - mx
    dy = yy - my

    cxx = float((inv * dx * dx).sum() / wsum)
    cyy = float((inv * dy * dy).sum() / wsum)
    cxy = float((inv * dx * dy).sum() / wsum)
    cov = np.array([[cxx, cxy], [cxy, cyy]], dtype=float)

    # Full eigen-decomposition: eigh returns sorted eigenvalues (ascending)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-12)

    major = float(np.sqrt(evals[1]))
    minor = float(np.sqrt(evals[0]))
    circularity = float(np.clip(minor / (major + 1e-12), 0.0, 1.0))
    ellipticity = 1.0 / circularity if circularity > 1e-6 else np.inf

    # Orientation: angle of the major axis (eigenvector for largest eigenvalue)
    # evecs[:, 1] is the eigenvector for evals[1] (the larger eigenvalue)
    orientation_rad = float(np.arctan2(evecs[1, 1], evecs[0, 1]))
    orientation_deg = float(np.degrees(orientation_rad))

    shape_reliable = radius_px >= confidence_min_radius_px

    return {
        "circularity": circularity,
        "ellipticity": ellipticity,
        "orientation_deg": orientation_deg,
        "shape_reliable": bool(shape_reliable),
    }


# ======================================================================
# Public API
# ======================================================================

def detect_craters_on_chip(
    gray: np.ndarray,
    mpp_mean: float,
    config: ChordConfig | None = None,
) -> tuple[pd.DataFrame, float]:
    """Detect craters on a grayscale chip image.

    Parameters
    ----------
    gray : ndarray
        Grayscale image in [0, 1], shape (H, W).
    mpp_mean : float
        Metres per pixel (average of x/y).
    config : ChordConfig, optional
        Pipeline configuration.  Defaults to ``ChordConfig()``.

    Returns
    -------
    detections : DataFrame
        Columns: x, y, radius_px, radius_m, strength, depth_proxy,
        circularity, ellipticity, orientation_deg, shape_reliable
    threshold : float
        The adaptive LoG threshold used (for freshness normalisation).
    """
    if config is None:
        config = ChordConfig()

    # 1. LoG blob detection
    y, x, radius_px, strength, threshold = _log_blob_detect(
        gray,
        mpp_mean,
        target_radius_m=config.target_crater_radius_range_m,
        peak_quantile=config.chip_peak_quantile,
    )

    if len(y) == 0:
        cols = [
            "x", "y", "radius_px", "radius_m", "strength", "depth_proxy",
            "circularity", "ellipticity", "orientation_deg", "shape_reliable",
        ]
        return pd.DataFrame(columns=cols), threshold

    # 2. NMS (no cap)
    keep = _nms_no_cap(y, x, radius_px, strength)
    y, x, radius_px, strength = y[keep], x[keep], radius_px[keep], strength[keep]

    # 3. Filter by minimum radius
    min_r = config.min_crater_radius_px
    mask = radius_px >= min_r
    y, x, radius_px, strength = y[mask], x[mask], radius_px[mask], strength[mask]

    if len(y) == 0:
        cols = [
            "x", "y", "radius_px", "radius_m", "strength", "depth_proxy",
            "circularity", "ellipticity", "orientation_deg", "shape_reliable",
        ]
        return pd.DataFrame(columns=cols), threshold

    # 4. Shape characterisation
    shapes = [
        _characterise_shape(
            gray, float(xi), float(yi), float(ri),
            confidence_min_radius_px=config.ellipticity_confidence_min_radius_px,
        )
        for xi, yi, ri in zip(x, y, radius_px)
    ]

    # 5. Depth proxy (normalised strength)
    s = strength
    s_range = s.max() - s.min()
    depth_proxy = (s - s.min()) / (s_range + 1e-6) if s_range > 0 else np.zeros_like(s)

    # 6. Filter by minimum circularity
    circ_arr = np.array([sh["circularity"] for sh in shapes])
    circ_mask = np.isnan(circ_arr) | (circ_arr >= config.min_circularity)
    y, x = y[circ_mask], x[circ_mask]
    radius_px, strength = radius_px[circ_mask], strength[circ_mask]
    depth_proxy = depth_proxy[circ_mask]
    shapes = [sh for sh, m in zip(shapes, circ_mask) if m]

    # Build output DataFrame
    df = pd.DataFrame({
        "x": x,
        "y": y,
        "radius_px": radius_px,
        "radius_m": radius_px * mpp_mean,
        "strength": strength,
        "depth_proxy": depth_proxy,
        "circularity": [sh["circularity"] for sh in shapes],
        "ellipticity": [sh["ellipticity"] for sh in shapes],
        "orientation_deg": [sh["orientation_deg"] for sh in shapes],
        "shape_reliable": [sh["shape_reliable"] for sh in shapes],
    })

    return df.sort_values("strength", ascending=False).reset_index(drop=True), threshold
