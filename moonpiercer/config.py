"""Pipeline configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from moonpiercer.constants import (
    KAPPA_ADOPTED_M2_PER_MYR,
    LOLA_DTM_LAYER,
    LUNAR_RADIUS_M,
    NAC_OBSERVATION_LAYER,
    NAC_STAMP_LAYER,
    V_MAX_PLAUSIBLE_KM_S,
    V_MIN_PLAUSIBLE_KM_S,
    WMS_BASE_URL,
)


@dataclass
class ChordConfig:
    """Central configuration for the MOONPIERCER chord-search pipeline.

    Sections mirror the pipeline stages: data access, detection,
    shape characterisation, freshness estimation, chord pairing, null
    model, and output control.
    """

    # ------------------------------------------------------------------
    # Lunar parameters
    # ------------------------------------------------------------------
    lunar_radius_m: float = LUNAR_RADIUS_M

    # ------------------------------------------------------------------
    # WMS / network
    # ------------------------------------------------------------------
    wms_base_url: str = WMS_BASE_URL
    cache_dir: Path = field(default_factory=lambda: Path("cache") / "wms")
    use_http_cache: bool = True
    request_timeout_s: int = 60

    nac_layer: str = NAC_STAMP_LAYER
    nac_query_layer: str = NAC_OBSERVATION_LAYER
    lola_layer: str = LOLA_DTM_LAYER

    # ------------------------------------------------------------------
    # Global sweep (manifest building)
    # ------------------------------------------------------------------
    sweep_grid_step_deg: float = 2.0
    max_grid_queries: int = 0  # 0 = unlimited
    feature_count: int = 40
    max_nac_resolution_mpp: float = 2.0
    sweep_max_workers: int = 6
    sweep_max_requests_per_second: float = 2.0

    # ------------------------------------------------------------------
    # Chip download
    # ------------------------------------------------------------------
    chip_size_px: int = 1024
    chip_span_m: float = 1200.0
    lola_tile_size_px: int = 256
    use_lola_topography: bool = True

    # ------------------------------------------------------------------
    # Crater detection
    # ------------------------------------------------------------------
    target_crater_radius_range_m: tuple[float, float] = (1.0, 10.0)
    chip_peak_quantile: float = 0.9994
    # NOTE: no max_craters_per_chip — detect ALL passing quality cuts
    min_depth_proxy: float = 0.22
    min_circularity: float = 0.55  # relaxed from 0.65 to allow mildly elliptical craters

    # ------------------------------------------------------------------
    # Shape characterisation
    # ------------------------------------------------------------------
    min_crater_radius_px: float = 3.0
    """Minimum radius in pixels for a detection to be retained at all."""

    ellipticity_confidence_min_radius_px: float = 5.0
    """Minimum radius in pixels for ellipticity/orientation to be trusted.

    Below this threshold, ``shape_reliable`` is set to False, and the
    crater's ellipticity and orientation are not used in pairing."""

    # ------------------------------------------------------------------
    # Freshness index
    # ------------------------------------------------------------------
    freshness_weight_strength: float = 0.7
    """Weight of Normalised LoG Strength (NLS) in the Freshness Index."""

    freshness_weight_contrast: float = 0.3
    """Weight of Rim Contrast Ratio (RCR) in the Freshness Index."""

    freshness_nls_scale: float = 3.0
    """Scaling constant for NLS → [0, 1] mapping.

    NLS values are clipped to [0, freshness_nls_scale] then divided.
    The default of 3.0 means a crater with strength 3× the detection
    threshold maps to FI_strength = 1.0."""

    freshness_rcr_scale: float = 8.0
    """Scaling constant for RCR → [0, 1] mapping."""

    min_freshness: float = 0.15
    """Minimum Freshness Index for a crater to participate in pairing."""

    max_freshness_diff: float = 0.30
    """Hard cut: maximum |ΔFI| between paired craters."""

    sigma_freshness: float = 0.12
    """Gaussian width for the freshness-match scoring term."""

    # ------------------------------------------------------------------
    # Chord pairing
    # ------------------------------------------------------------------
    min_chord_sep_deg: float = 30.0
    """Minimum angular separation to consider a chord (degrees).

    Chords shorter than this do not pass through enough of the Moon's
    interior to be physically meaningful for PBH transits."""

    max_radius_diff_m: float = 2.0
    """Hard cut: maximum radius difference between paired craters [m]."""

    sigma_radius: float = 1.0
    """Gaussian width for the radius-match scoring term [m]."""

    max_chord_deviation_deg: float = 0.05
    """Hard cut: max angular offset from predicted chord endpoint [degrees].

    Based on physics (lunar rotation ≤ 0.011°), measurement uncertainty,
    and a safety margin."""

    search_cone_half_deg_reliable: float = 2.0
    """Search cone half-angle for craters with reliable shapes [degrees].

    The predicted exit point is uncertain due to shape measurement noise;
    this defines the kd-tree search radius around the prediction."""

    search_cone_half_deg_unreliable: float = 10.0
    """Search cone half-angle for craters without reliable shapes [degrees].

    When shape is unreliable, we search a wider cone near the antipode."""

    sigma_ellipticity: float = 0.15
    """Gaussian width for the ellipticity-match scoring term."""

    sigma_orientation_deg: float = 15.0
    """Gaussian width for the orientation-match scoring term [degrees]."""

    prefer_diametrality: bool = True
    """If True, the scoring function includes T_diametrality = sin(sep/2)."""

    pair_k_neighbors: int = 32
    """kd-tree query size for the nearest-neighbour search."""

    top_pairs_to_report: int = 50
    """Number of top non-overlapping pairs to report."""

    # ------------------------------------------------------------------
    # Velocity constraints
    # ------------------------------------------------------------------
    v_min_km_s: float = V_MIN_PLAUSIBLE_KM_S
    v_max_km_s: float = V_MAX_PLAUSIBLE_KM_S

    # ------------------------------------------------------------------
    # Null model / statistics
    # ------------------------------------------------------------------
    random_trials: int = 2000
    random_seed: int = 42
    fdr_alpha: float = 0.05
    """Benjamini-Hochberg FDR significance level."""

    null_quantile: float = 0.95
    """Quantile of the null best-score distribution for threshold."""

    null_model_workers: int = 1
    """Number of parallel processes for null-model trials.

    Each trial is independent (embarrassingly parallel).  Set to the
    number of available CPU cores for maximum throughput.  Default 1
    runs trials sequentially (original behaviour)."""

    # ------------------------------------------------------------------
    # Topographic diffusion
    # ------------------------------------------------------------------
    kappa_m2_per_myr: float = KAPPA_ADOPTED_M2_PER_MYR

    # ------------------------------------------------------------------
    # Output controls
    # ------------------------------------------------------------------
    save_annotated_chips_every_n: int = 50
    """Save an annotated detection image every N chips for visual QA."""

    max_significant_images: int = 60
    """Maximum number of significant-pair images to save."""
