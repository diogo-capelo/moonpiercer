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
    n_scales: int = 30
    """Number of LoG scale-space octaves for crater detection.

    More scales give finer radius resolution (at slightly higher compute
    cost).  With 30 scales over the default 1–10 m range, adjacent
    scales are ~8% apart, giving sub-metre radius resolution."""

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
    # Position prediction scoring
    # ------------------------------------------------------------------
    sigma_position_deg: float = 0.5
    """Fallback Gaussian width [degrees] for T_position when per-crater
    shape uncertainties are unavailable (e.g. rescoring old data).

    Used as ``T_position = exp(-(offset / sigma)²)``.  Tighter values
    penalise pairs whose shape-predicted exit is far from the actual
    partner position.  Per-crater propagated uncertainties take
    precedence when available."""

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

    min_freshness: float = 0.05
    """Minimum Freshness Index for a crater to participate in pairing.

    Set just above the noise floor — crater quality is ensured by
    min_depth_proxy and min_circularity.  Freshness is used as a
    correlation metric (sigma_freshness), not an absolute age gate."""

    max_freshness_diff: float = 0.30
    """Hard cut: maximum |ΔFI| between paired craters."""

    sigma_freshness: float = 0.01
    """Gaussian width for the freshness-match scoring term.

    Freshness Index is continuous and well-resolved, so a tight sigma
    (strong penalty for mismatches) is appropriate.  Set to 0.01 so that
    pairs must match in age to within ~1 FI unit to score well."""

    # ------------------------------------------------------------------
    # Chord pairing
    # ------------------------------------------------------------------
    min_chord_sep_deg: float = 30.0
    """Minimum angular separation to consider a chord (degrees).

    Chords shorter than this do not pass through enough of the Moon's
    interior to be physically meaningful for PBH transits."""

    max_radius_diff_m: float = 2.0
    """Hard cut: maximum radius difference between paired craters [m]."""

    sigma_radius: float = 5.0
    """Gaussian width for the radius-match scoring term [m].

    Crater radii are quantised by the LoG scale space (~30% spacing
    between adjacent scales at 10 scales), giving ~1 m resolution for
    typical detections.  A wide sigma down-weights this poorly-resolved
    measurement relative to better-constrained quantities."""

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

    sigma_ellipticity: float = 0.04
    """Gaussian width for the ellipticity-match scoring term.

    Ellipticity encodes the chord incidence angle — a strong physical
    signal.  Set to 0.04, roughly half the typical measurement noise
    floor for shape-reliable craters, for strong discrimination."""

    sigma_orientation_deg: float = 3.0
    """Gaussian width for the orientation-match scoring term [degrees].

    Both craters' major axes should align with the great circle
    connecting them.  Set to 3° — tight but within the measurement
    noise for shape-reliable craters (radius ≥ 5 px)."""

    prefer_diametrality: bool = False
    """If True, the scoring function includes T_diametrality = sin(sep/2)^n.

    Default False: for an isotropic PBH flux, off-centre trajectories
    vastly outnumber diametral ones, so preferring diametrality would
    bias against the most likely chord geometries.  The hard minimum
    separation cut (min_chord_sep_deg) already excludes trivially short
    chords; highly eccentric craters from short chords are penalised by
    T_ellipticity."""

    diametrality_exponent: float = 2.0
    """Exponent for the diametrality term: sin(sep/2)^n.

    Higher values more aggressively penalise non-diametral chords.
    n=1 is linear, n=2 (default) is quadratic."""

    pair_k_neighbors: int = 32
    """kd-tree query size for the nearest-neighbour search."""

    top_pairs_to_report: int = 200
    """Number of top non-overlapping pairs to report."""

    max_pairs_in_memory: int = 500_000
    """Maximum number of crater pairs held in memory during pairing.

    A bounded min-heap of this size replaces an unbounded list, keeping
    memory usage constant regardless of how many pairs pass the hard cuts.
    Only pairs with a score above the current heap minimum are retained;
    lower-scoring pairs are discarded immediately.  Set large enough that
    the true top-N pairs are never evicted (500k >> top_pairs_to_report)."""

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
