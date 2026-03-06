"""Freshness Index (FI) computation for crater age estimation.

The Freshness Index is a composite proxy for relative crater age,
combining two cross-chip comparable metrics:

1. **Normalised LoG Strength (NLS)** — strength / threshold.  Since the
   threshold is set at the same quantile (0.9994) everywhere, this ratio
   is comparable across chips.  Fresh craters have higher NLS.

2. **Rim Contrast Ratio (RCR)** — (mean_rim - mean_floor) / std_background.
   Measures how sharply the crater rim stands out.  Fresh craters have
   higher RCR.

The composite Freshness Index is:
    FI = w_s * clip01(NLS / NLS_scale) + w_c * clip01(RCR / RCR_scale)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from moonpiercer.config import ChordConfig


# ======================================================================
# Normalised LoG Strength
# ======================================================================

def normalised_log_strength(
    strength: np.ndarray | float,
    threshold: float,
) -> np.ndarray | float:
    """Compute NLS = strength / threshold.

    All detections satisfy strength >= threshold, so NLS >= 1.0.
    """
    return np.asarray(strength, dtype=np.float64) / max(float(threshold), 1e-30)


# ======================================================================
# Rim Contrast Ratio
# ======================================================================

def rim_contrast_ratio(
    gray: np.ndarray,
    x: float,
    y: float,
    radius_px: float,
) -> float:
    """Compute Rim Contrast Ratio (RCR) for a single crater.

    Zones (as fractions of crater radius):
        - Floor:      [0, 0.5r]
        - Rim:        [0.8r, 1.3r]
        - Background: [2r, 3r]

    RCR = (mean_rim - mean_floor) / std_background
    """
    h, w = gray.shape
    yy, xx = np.ogrid[0:h, 0:w]
    r2 = (xx - x) ** 2 + (yy - y) ** 2
    r = radius_px

    floor_mask = r2 <= (0.5 * r) ** 2
    rim_mask = (r2 >= (0.8 * r) ** 2) & (r2 <= (1.3 * r) ** 2)
    bg_mask = (r2 >= (2.0 * r) ** 2) & (r2 <= (3.0 * r) ** 2)

    # Bounds check: only keep pixels within image
    n_floor = int(floor_mask.sum())
    n_rim = int(rim_mask.sum())
    n_bg = int(bg_mask.sum())

    if n_floor < 3 or n_rim < 3 or n_bg < 5:
        return 0.0

    mean_floor = float(gray[floor_mask].mean())
    mean_rim = float(gray[rim_mask].mean())
    std_bg = float(gray[bg_mask].std())

    if std_bg < 1e-8:
        return 0.0

    return (mean_rim - mean_floor) / std_bg


# ======================================================================
# Composite Freshness Index
# ======================================================================

def freshness_index(
    nls: np.ndarray | float,
    rcr: np.ndarray | float,
    config: ChordConfig | None = None,
) -> np.ndarray | float:
    """Compute the composite Freshness Index.

    FI = w_s * clip01((NLS - 1) / (NLS_scale - 1)) + w_c * clip01(RCR / RCR_scale)

    We subtract 1 from NLS because all detections have NLS >= 1, so
    the "zero freshness" baseline corresponds to NLS=1.
    """
    if config is None:
        config = ChordConfig()

    w_s = config.freshness_weight_strength
    w_c = config.freshness_weight_contrast
    nls_scale = config.freshness_nls_scale
    rcr_scale = config.freshness_rcr_scale

    nls_arr = np.asarray(nls, dtype=np.float64)
    rcr_arr = np.asarray(rcr, dtype=np.float64)

    # Map NLS: 1.0 → 0.0, nls_scale → 1.0
    nls_norm = np.clip((nls_arr - 1.0) / max(nls_scale - 1.0, 1e-6), 0.0, 1.0)
    rcr_norm = np.clip(rcr_arr / max(rcr_scale, 1e-6), 0.0, 1.0)

    fi = w_s * nls_norm + w_c * rcr_norm
    return fi


# ======================================================================
# Batch computation for a chip
# ======================================================================

def compute_freshness_for_chip(
    gray: np.ndarray,
    detections: pd.DataFrame,
    threshold: float,
    config: ChordConfig | None = None,
) -> pd.DataFrame:
    """Add freshness columns to a detections DataFrame.

    Adds columns: ``nls``, ``rcr``, ``freshness_index``.
    Returns a copy of the DataFrame with the new columns.
    """
    if config is None:
        config = ChordConfig()

    df = detections.copy()

    if df.empty:
        df["nls"] = pd.Series(dtype=float)
        df["rcr"] = pd.Series(dtype=float)
        df["freshness_index"] = pd.Series(dtype=float)
        return df

    # NLS: vectorised
    df["nls"] = normalised_log_strength(df["strength"].to_numpy(), threshold)

    # RCR: per-crater (needs spatial context)
    rcr_vals = np.array([
        rim_contrast_ratio(gray, float(row["x"]), float(row["y"]), float(row["radius_px"]))
        for _, row in df.iterrows()
    ])
    df["rcr"] = rcr_vals

    # Composite FI
    df["freshness_index"] = freshness_index(
        df["nls"].to_numpy(), df["rcr"].to_numpy(), config
    )

    return df
