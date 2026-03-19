"""Monte Carlo null model and Benjamini-Hochberg FDR correction.

The null hypothesis is that crater positions are random on the sphere.
We test this by:
1. Generating random crater positions (uniform on the sphere) while
   preserving measured attributes (radius, FI, ellipticity, etc.).
2. Running the same pairing algorithm on the randomised catalogue.
3. Recording the best score from each trial.
4. Computing empirical p-values for the real pairs.
5. Applying Benjamini-Hochberg FDR correction.
"""

from __future__ import annotations

import sys
import time

import numpy as np
import pandas as pd

from moonpiercer.config import ChordConfig
from moonpiercer.pairing import max_pair_score


# ======================================================================
# Benjamini-Hochberg FDR correction
# ======================================================================

def benjamini_hochberg(
    p_values: np.ndarray,
    alpha: float = 0.05,
) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : array of shape (n,)
        Raw p-values for each test.
    alpha : float
        FDR significance level.

    Returns
    -------
    reject : bool array of shape (n,)
        True for discoveries (significant at the FDR-corrected level).
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = len(p)
    if n == 0:
        return np.array([], dtype=bool)

    sorted_idx = np.argsort(p)
    sorted_p = p[sorted_idx]
    thresholds = alpha * np.arange(1, n + 1) / n

    # Find largest k where p_(k) <= k * alpha / n
    max_k = 0
    for k in range(n):
        if sorted_p[k] <= thresholds[k]:
            max_k = k + 1

    reject = np.zeros(n, dtype=bool)
    reject[sorted_idx[:max_k]] = True
    return reject


# ======================================================================
# Random sphere sampling
# ======================================================================

def _random_positions_on_sphere(
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate *n* uniformly distributed points on the unit sphere.

    Returns (lon_deg, lat_deg) arrays.
    """
    # Uniform on sphere: z uniform in [-1, 1], phi uniform in [0, 2π)
    z = rng.uniform(-1.0, 1.0, n)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    lat = np.degrees(np.arcsin(z))
    lon = np.degrees(phi) - 180.0  # → [-180, 180)
    return lon, lat


# ======================================================================
# Monte Carlo null model — sequential entry point
# ======================================================================

def null_model_best_scores(
    craters: pd.DataFrame,
    config: ChordConfig | None = None,
    n_trials: int | None = None,
    seed: int | None = None,
    trial_offset: int = 0,
    trial_count: int | None = None,
    progress_interval_sec: float | None = None,
    save_pair_details: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    """Run the null model and return the best score from each trial.

    Each trial randomises crater positions on the sphere (preserving all
    other attributes), then computes the maximum pair score using the
    memory-efficient ``max_pair_score`` function (O(1) pair storage).

    Before the trial loop, the catalogue is pre-filtered to craters that
    meet the pairing thresholds (FI >= min_freshness, depth >= min_depth).
    Non-qualifying craters never participate in any pair, so excluding
    them is mathematically equivalent but dramatically reduces the
    kd-tree size and query time.

    Parameters
    ----------
    craters : DataFrame
        Real crater catalogue (attributes are preserved, positions randomised).
    config : ChordConfig, optional
    n_trials : int, optional
        Total number of Monte Carlo trials (default: config.random_trials).
    seed : int, optional
        Random seed (default: config.random_seed).
    trial_offset : int, optional
        Starting trial index (for chunked runs).
    trial_count : int, optional
        Number of trials to run from trial_offset.
    progress_interval_sec : float, optional
        Emit progress logs every N seconds (None/<=0 disables).
    save_pair_details : bool, optional
        If True, also return the characteristics of the best-scoring pair
        from each trial, enabling rescoring with different configs without
        rerunning the Monte Carlo.

    Returns
    -------
    best_scores : array of shape (trial_count,)
        Maximum pair score from each randomised trial.
    pair_details : list[dict]  *(only if save_pair_details is True)*
        Characteristics of the best pair from each trial.
    """
    if config is None:
        config = ChordConfig()
    if n_trials is None:
        n_trials = config.random_trials
    if seed is None:
        seed = config.random_seed
    if trial_offset < 0:
        raise ValueError("trial_offset must be non-negative.")

    # Derive independent per-trial RNG streams from a single parent seed.
    child_seeds = np.random.SeedSequence(seed).spawn(n_trials)
    if trial_count is None:
        trial_count = n_trials - trial_offset
    if trial_count < 0:
        raise ValueError("trial_count must be >= 0.")
    child_seeds = child_seeds[trial_offset:trial_offset + trial_count]

    if trial_count == 0:
        return np.array([], dtype=np.float64)

    # ------------------------------------------------------------------
    # Pre-filter to craters eligible for pairing.
    # build_chord_pairs / max_pair_score skip craters with FI < min_fi
    # or depth < min_depth anyway, so excluding them upfront only
    # removes dead weight from the kd-tree and query loops.
    # ------------------------------------------------------------------
    mask = craters["freshness_index"] >= config.min_freshness
    if "depth_proxy" in craters.columns:
        mask &= craters["depth_proxy"] >= config.min_depth_proxy
    qualifying = craters[mask].reset_index(drop=True)
    n_qualifying = len(qualifying)

    print(
        f"[null_model] Pre-filtered to {n_qualifying:,d} qualifying craters "
        f"(from {len(craters):,d})",
        file=sys.stderr,
        flush=True,
    )

    if n_qualifying < 2:
        print(
            "[null_model] Fewer than 2 qualifying craters — all null scores will be 0.",
            file=sys.stderr,
            flush=True,
        )
        zeros = np.zeros(len(child_seeds), dtype=np.float64)
        if save_pair_details:
            return zeros, [{} for _ in child_seeds]
        return zeros

    total_trials = len(child_seeds)
    use_progress = progress_interval_sec is not None and progress_interval_sec > 0
    best_scores = np.zeros(total_trials, dtype=np.float64)
    all_details: list[dict] = [] if save_pair_details else []
    t_start = time.monotonic()
    next_report = t_start + progress_interval_sec if use_progress else None

    for trial, cs in enumerate(child_seeds):
        rng = np.random.default_rng(cs)
        rand_lon, rand_lat = _random_positions_on_sphere(n_qualifying, rng)
        random_cat = qualifying.copy()
        random_cat["lon_deg"] = rand_lon
        random_cat["lat_deg"] = rand_lat

        if save_pair_details:
            score, details = max_pair_score(
                random_cat, config, return_details=True,
            )
            best_scores[trial] = score
            all_details.append(details)
        else:
            best_scores[trial] = max_pair_score(random_cat, config)

        # Explicitly free large objects before the next trial.
        del random_cat, rand_lon, rand_lat

        if next_report is not None and time.monotonic() >= next_report:
            done = trial + 1
            elapsed = time.monotonic() - t_start
            if trial_offset == 0 and total_trials == n_trials:
                msg = f"[null_model] {done}/{total_trials} trials ({elapsed:.1f}s)"
            else:
                msg = (
                    f"[null_model] chunk {trial_offset}-{trial_offset + total_trials - 1}: "
                    f"{done}/{total_trials} trials ({elapsed:.1f}s)"
                )
            print(msg, file=sys.stderr, flush=True)
            next_report = time.monotonic() + progress_interval_sec

    if save_pair_details:
        return best_scores, all_details
    return best_scores


def empirical_p_value(
    score: float,
    null_scores: np.ndarray,
) -> float:
    """Compute empirical p-value: fraction of null scores >= observed score."""
    n = len(null_scores)
    if n == 0:
        return 1.0
    return float((np.sum(null_scores >= score) + 1) / (n + 1))


def compute_significance(
    real_pairs: pd.DataFrame,
    null_best_scores: np.ndarray,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Add p-values and BH-FDR significance to the real pairs.

    Adds columns: ``p_value``, ``bh_significant``.
    Returns a copy.
    """
    if real_pairs.empty:
        df = real_pairs.copy()
        df["p_value"] = pd.Series(dtype=float)
        df["bh_significant"] = pd.Series(dtype=bool)
        return df

    df = real_pairs.copy()
    scores = df["score"].to_numpy()
    p_values = np.array([empirical_p_value(s, null_best_scores) for s in scores])
    df["p_value"] = p_values
    df["bh_significant"] = benjamini_hochberg(p_values, alpha=alpha)
    return df
