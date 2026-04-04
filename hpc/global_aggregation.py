"""HPC global aggregation script for MOONPIERCER.

Combines all per-chip crater CSVs produced by the HPC per-chip pipeline,
runs the global chord-based pairing, null model, and significance assessment,
and writes consolidated output files ready for analysis.

Expected chip result layout on disk::

    <chip-results-dir>/
        chip_0000/
            craters.csv
            metadata.json        (optional — terrain / acquisition info)
        chip_0001/
            craters.csv
            metadata.json
        ...

Usage::

    python global_aggregation.py \\
        --chip-results-dir /scratch/results/chips \\
        --output-dir /scratch/results/global \\
        --random-trials 2000 \\
        --random-seed 42 \\
        --fdr-alpha 0.05 \\
        --top-pairs 50
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from moonpiercer.config import ChordConfig
from moonpiercer.io_utils import load_dataframe, load_json, save_dataframe, save_json
from moonpiercer.null_model import compute_significance, null_model_best_scores
from moonpiercer.pairing import build_chord_pairs, select_top_nonoverlapping_pairs


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "MOONPIERCER global aggregation: combine all chip results, run "
            "global chord pairing, null model, and write output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--chip-results-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory containing chip_XXXX/ subdirectories with craters.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        metavar="DIR",
        help="Directory where global output files will be written.",
    )
    parser.add_argument(
        "--random-trials",
        type=int,
        default=2000,
        metavar="N",
        help="Number of Monte Carlo null-model trials.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        metavar="SEED",
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.05,
        metavar="ALPHA",
        help="Benjamini-Hochberg FDR significance level.",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=50,
        metavar="K",
        help="Number of top non-overlapping pairs to report.",
    )
    parser.add_argument(
        "--save-pair-images",
        action="store_true",
        default=False,
        help=(
            "If set, save annotated chip images for significant pairs "
            "(requires chip imagery to be present in chip-results-dir)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("full", "prep", "null", "final"),
        default="full",
        help=(
            "Execution mode: full runs everything in one job; prep/null/final "
            "enable checkpointed, array-parallel null model."
        ),
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="Directory to store intermediate checkpoints (default: output-dir/checkpoints).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from existing checkpoints if present.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Ignore checkpoint config mismatches when resuming.",
    )
    parser.add_argument(
        "--progress-interval-sec",
        type=float,
        default=300.0,
        metavar="SEC",
        help="Emit progress logs every N seconds (0 disables).",
    )
    parser.add_argument(
        "--null-chunk-index",
        type=int,
        default=-1,
        metavar="IDX",
        help="Chunk index for null-model array jobs (0-based). Defaults to SLURM_ARRAY_TASK_ID.",
    )
    parser.add_argument(
        "--null-chunk-count",
        type=int,
        default=1,
        metavar="N",
        help="Total number of null-model chunks (array size).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Chip discovery and loading
# ---------------------------------------------------------------------------

def _discover_chip_dirs(chip_results_dir: Path) -> list[Path]:
    """Return sorted list of chip subdirectories that contain a craters.csv."""
    chip_dirs = sorted(
        p for p in chip_results_dir.iterdir()
        if p.is_dir() and p.name.startswith("chip_") and (p / "craters.csv").exists()
    )
    return chip_dirs


def _load_craters(chip_dirs: list[Path]) -> pd.DataFrame:
    """Load all craters.csv files into a single DataFrame.

    Adds a ``chip_index`` column (integer, zero-based, matching the
    chip directory sort order) and a ``chip_dir`` column (string path).

    Skips empty or missing files with a warning.
    """
    frames: list[pd.DataFrame] = []
    for chip_index, chip_dir in enumerate(chip_dirs):
        csv_path = chip_dir / "craters.csv"
        try:
            df = load_dataframe(csv_path)
        except Exception as exc:  # noqa: BLE001
            print(
                f"  WARNING: could not load {csv_path}: {exc}",
                file=sys.stderr,
            )
            continue

        if df.empty:
            continue

        df["chip_index"] = chip_index
        df["chip_dir"] = str(chip_dir)
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _load_chip_metadata(chip_dirs: list[Path]) -> list[dict]:
    """Load metadata.json from each chip directory (if present).

    Returns a list of dicts (one per chip directory, in the same order).
    Missing files result in an empty dict for that entry.
    """
    metadata: list[dict] = []
    for chip_dir in chip_dirs:
        meta_path = chip_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = load_json(meta_path)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"  WARNING: could not load {meta_path}: {exc}",
                    file=sys.stderr,
                )
                meta = {}
        else:
            meta = {}
        metadata.append(meta)
    return metadata


# ---------------------------------------------------------------------------
# Chip health reporting
# ---------------------------------------------------------------------------

def _report_chip_health(chip_dirs: list[Path]) -> None:
    """Print a breakdown of chip statuses for accountability."""
    if not chip_dirs:
        return

    status_counts: dict[str, int] = {}
    n_with_craters = 0
    total_craters = 0

    for chip_dir in chip_dirs:
        meta_path = chip_dir / "metadata.json"
        if meta_path.exists():
            try:
                meta = load_json(meta_path)
                status = meta.get("status", "unknown")
                n_craters = int(meta.get("n_craters", 0))
            except Exception:
                status = "unreadable_metadata"
                n_craters = 0
        else:
            status = "missing_metadata"
            n_craters = 0

        status_counts[status] = status_counts.get(status, 0) + 1
        if n_craters > 0:
            n_with_craters += 1
            total_craters += n_craters

    print(f"  Chip status breakdown:")
    for status, count in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"    {status:<25s} : {count:>6,d}")
    print(f"  Chips with craters: {n_with_craters:,d} ({total_craters:,d} total craters)")


# ---------------------------------------------------------------------------
# Coverage estimation
# ---------------------------------------------------------------------------

def _estimate_coverage_km2(
    chip_metadata: list[dict],
    chip_dirs: list[Path],
) -> float:
    """Estimate total surveyed area in km².

    Uses ``chip_span_m`` from each chip's metadata.json where available.
    Falls back to the ChordConfig default (1200 m chip span) for chips
    that lack metadata.  Returns 0.0 if no chips are present.
    """
    default_span_m = ChordConfig().chip_span_m
    total_m2 = 0.0
    for meta in chip_metadata:
        span_m = float(meta.get("chip_span_m") or default_span_m)
        total_m2 += span_m * span_m
    return total_m2 / 1.0e6  # m² → km²


# ---------------------------------------------------------------------------
# Summary printing helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    width = 70
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def _print_summary_stats(
    n_chips: int,
    n_craters: int,
    coverage_km2: float,
) -> None:
    _print_section("Global catalogue summary")
    print(f"  Chips with craters : {n_chips:>8,d}")
    print(f"  Total craters      : {n_craters:>8,d}")
    print(f"  Estimated coverage : {coverage_km2:>10,.1f} km²")
    if n_chips > 0:
        print(f"  Craters per chip   : {n_craters / n_chips:>10.1f}")


def _print_pairing_summary(n_pairs: int, top_k: int) -> None:
    _print_section("Chord pairing")
    print(f"  Raw pairs found    : {n_pairs:>8,d}")
    print(f"  Top non-overlapping: {top_k:>8,d}  (requested)")


def _print_null_model_summary(n_trials: int, null_scores: np.ndarray) -> None:
    _print_section("Null model")
    print(f"  Trials             : {n_trials:>8,d}")
    if len(null_scores) > 0:
        print(f"  Null 50th pctile   : {float(np.percentile(null_scores, 50)):>10.4f}")
        print(f"  Null 95th pctile   : {float(np.percentile(null_scores, 95)):>10.4f}")
        print(f"  Null 99th pctile   : {float(np.percentile(null_scores, 99)):>10.4f}")
        print(f"  Null max           : {float(null_scores.max()):>10.4f}")


def _print_results_summary(pairs_scored: pd.DataFrame, n_significant: int) -> None:
    _print_section("Results")
    if pairs_scored.empty:
        print("  No pairs found.")
        return

    best_row = pairs_scored.iloc[0]
    best_score = float(best_row["score"])
    best_p = float(best_row.get("p_value", float("nan")))
    best_sep = float(best_row.get("separation_deg", float("nan")))
    best_lon_a = float(best_row.get("lon_a", float("nan")))
    best_lat_a = float(best_row.get("lat_a", float("nan")))
    best_lon_b = float(best_row.get("lon_b", float("nan")))
    best_lat_b = float(best_row.get("lat_b", float("nan")))

    print(f"  Best pair score    : {best_score:>10.6f}")
    print(f"  Best pair p-value  : {best_p:>10.4f}")
    print(f"  Best pair sep (deg): {best_sep:>10.2f}")
    print(
        f"  Best pair A (lon,lat): ({best_lon_a:+.3f}, {best_lat_a:+.3f})"
    )
    print(
        f"  Best pair B (lon,lat): ({best_lon_b:+.3f}, {best_lat_b:+.3f})"
    )
    print(f"  Significant pairs  : {n_significant:>8,d}")
    print()


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _resolve_checkpoint_dir(output_dir: Path, checkpoint_dir: Path | None) -> Path:
    return checkpoint_dir if checkpoint_dir is not None else output_dir / "checkpoints"


def _craters_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "craters.pkl"


def _top_pairs_checkpoint_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "top_pairs.csv"


def _checkpoint_meta_path(checkpoint_dir: Path) -> Path:
    return checkpoint_dir / "checkpoint_meta.json"


def _null_part_path(checkpoint_dir: Path, chunk_index: int) -> Path:
    return checkpoint_dir / f"null_scores_part_{chunk_index:05d}.npy"


def _null_part_meta_path(checkpoint_dir: Path, chunk_index: int) -> Path:
    return checkpoint_dir / f"null_scores_part_{chunk_index:05d}.json"


def _null_trial_score_path(checkpoint_dir: Path, trial_index: int) -> Path:
    """Per-trial checkpoint: single float score."""
    return checkpoint_dir / f"null_trial_{trial_index:05d}.npy"


def _null_trial_details_path(checkpoint_dir: Path, trial_index: int) -> Path:
    """Per-trial checkpoint: best-pair details JSON."""
    return checkpoint_dir / f"null_trial_{trial_index:05d}_details.json"


def _load_checkpoint_meta(checkpoint_dir: Path) -> dict:
    meta_path = _checkpoint_meta_path(checkpoint_dir)
    if not meta_path.exists():
        raise FileNotFoundError(f"Checkpoint metadata not found: {meta_path}")
    return load_json(meta_path)


def _write_checkpoint_meta(checkpoint_dir: Path, meta: dict) -> None:
    meta_path = _checkpoint_meta_path(checkpoint_dir)
    save_json(meta, meta_path)


def _validate_checkpoint_meta(meta: dict, args: argparse.Namespace) -> None:
    expected = {
        "random_trials": args.random_trials,
        "random_seed": args.random_seed,
        "fdr_alpha": args.fdr_alpha,
        "top_pairs": args.top_pairs,
        "chip_results_dir": str(args.chip_results_dir.resolve()),
    }
    mismatches = []
    for key, value in expected.items():
        if key in meta and meta[key] != value:
            mismatches.append(f"{key} (checkpoint={meta[key]} vs current={value})")
    if "null_chunk_count" in meta and meta["null_chunk_count"] != args.null_chunk_count:
        mismatches.append(
            f"null_chunk_count (checkpoint={meta['null_chunk_count']} vs current={args.null_chunk_count})"
        )
    if mismatches and not args.force:
        details = "; ".join(mismatches)
        raise ValueError(f"Checkpoint config mismatch: {details}")


def _compute_chunk_bounds(
    total_trials: int,
    chunk_count: int,
    chunk_index: int,
) -> tuple[int, int]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be >= 1.")
    if not (0 <= chunk_index < chunk_count):
        raise ValueError("chunk_index out of range.")
    base = total_trials // chunk_count
    remainder = total_trials % chunk_count
    if chunk_index < remainder:
        count = base + 1
        start = chunk_index * (base + 1)
    else:
        count = base
        start = remainder * (base + 1) + (chunk_index - remainder) * base
    return start, count


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _run_full(args: argparse.Namespace) -> int:
    """Execute the global aggregation pipeline.

    Returns 0 on success, non-zero on failure.
    """
    t_start = time.monotonic()

    chip_results_dir: Path = args.chip_results_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    random_trials: int = args.random_trials
    random_seed: int = args.random_seed
    fdr_alpha: float = args.fdr_alpha
    top_pairs: int = args.top_pairs
    progress_interval_sec: float | None = (
        args.progress_interval_sec if args.progress_interval_sec and args.progress_interval_sec > 0 else None
    )

    # ------------------------------------------------------------------
    # Validate inputs
    # ------------------------------------------------------------------
    if not chip_results_dir.exists():
        print(
            f"ERROR: chip-results-dir does not exist: {chip_results_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1: discover chip directories
    # ------------------------------------------------------------------
    _print_section("Discovering chip results")
    chip_dirs = _discover_chip_dirs(chip_results_dir)
    print(f"  Found {len(chip_dirs):,d} chip directories with craters.csv")

    if not chip_dirs:
        print(
            "ERROR: no chip_XXXX/craters.csv files found under "
            f"{chip_results_dir}",
            file=sys.stderr,
        )
        return 1

    # ------------------------------------------------------------------
    # Step 2: load craters and metadata
    # ------------------------------------------------------------------
    _print_section("Loading crater catalogues")
    craters = _load_craters(chip_dirs)
    chip_metadata = _load_chip_metadata(chip_dirs)

    n_chips_with_data = int(craters["chip_index"].nunique()) if not craters.empty else 0
    n_craters = len(craters)
    coverage_km2 = _estimate_coverage_km2(chip_metadata, chip_dirs)

    _print_summary_stats(n_chips_with_data, n_craters, coverage_km2)

    if craters.empty:
        print("ERROR: no craters loaded — cannot proceed.", file=sys.stderr)
        return 1

    # ------------------------------------------------------------------
    # Step 3: build ChordConfig from CLI args
    # ------------------------------------------------------------------
    config = ChordConfig(
        random_trials=random_trials,
        random_seed=random_seed,
        fdr_alpha=fdr_alpha,
        top_pairs_to_report=top_pairs,
    )

    # ------------------------------------------------------------------
    # Step 4: global chord pairing
    # ------------------------------------------------------------------
    _print_section("Running global chord-based pairing")
    print(f"  Building pairs from {n_craters:,d} craters …")

    t_pair_start = time.monotonic()
    all_pairs = build_chord_pairs(craters, config, progress_interval_sec=progress_interval_sec)
    t_pair_elapsed = time.monotonic() - t_pair_start

    n_raw_pairs = len(all_pairs)
    print(f"  Raw pairs found    : {n_raw_pairs:,d}  ({t_pair_elapsed:.1f} s)")

    # ------------------------------------------------------------------
    # Step 5: select top non-overlapping pairs
    # ------------------------------------------------------------------
    if not all_pairs.empty:
        top_nonoverlap = select_top_nonoverlapping_pairs(all_pairs, top_k=top_pairs)
    else:
        top_nonoverlap = pd.DataFrame()

    n_top = len(top_nonoverlap)
    _print_pairing_summary(n_raw_pairs, n_top)

    # ------------------------------------------------------------------
    # Step 6: null model
    # ------------------------------------------------------------------
    _print_section(
        f"Running null model ({random_trials:,d} trials, seed={random_seed})"
    )
    t_null_start = time.monotonic()
    null_scores, null_pair_details = null_model_best_scores(
        craters,
        config=config,
        n_trials=random_trials,
        seed=random_seed,
        progress_interval_sec=progress_interval_sec,
        save_pair_details=True,
    )
    t_null_elapsed = time.monotonic() - t_null_start
    print(f"  Null model completed in {t_null_elapsed:.1f} s")

    _print_null_model_summary(random_trials, null_scores)

    # ------------------------------------------------------------------
    # Step 7: compute significance on top non-overlapping pairs
    # ------------------------------------------------------------------
    _print_section("Computing significance (Benjamini-Hochberg FDR)")
    if not top_nonoverlap.empty:
        pairs_scored = compute_significance(top_nonoverlap, null_scores, alpha=fdr_alpha)
    else:
        pairs_scored = pd.DataFrame()

    n_significant = (
        int(pairs_scored["bh_significant"].sum())
        if not pairs_scored.empty and "bh_significant" in pairs_scored.columns
        else 0
    )

    # Determine headline statistics for the summary JSON
    if not pairs_scored.empty:
        best_score = float(pairs_scored["score"].iloc[0])
        best_p_value = float(pairs_scored["p_value"].iloc[0]) if "p_value" in pairs_scored.columns else None
    else:
        best_score = 0.0
        best_p_value = None

    # ------------------------------------------------------------------
    # Step 8: significant pairs subset
    # ------------------------------------------------------------------
    if not pairs_scored.empty and "bh_significant" in pairs_scored.columns:
        significant_pairs = pairs_scored[pairs_scored["bh_significant"]].reset_index(drop=True)
    else:
        significant_pairs = pd.DataFrame()

    _print_results_summary(pairs_scored, n_significant)

    # ------------------------------------------------------------------
    # Step 9: save outputs
    # ------------------------------------------------------------------
    _print_section("Saving outputs")

    # 9a: all top pairs with scores and p-values
    all_pairs_path = output_dir / "all_pairs_scored.csv"
    if not pairs_scored.empty:
        save_dataframe(pairs_scored, all_pairs_path)
    else:
        # Write an empty CSV so downstream scripts always find the file
        pd.DataFrame(columns=[
            "idx_a", "idx_b", "lon_a", "lat_a", "lon_b", "lat_b",
            "separation_deg", "radius_a_m", "radius_b_m",
            "radius_px_a", "radius_px_b",
            "nls_a", "nls_b", "rcr_a", "rcr_b",
            "ellipticity_a", "ellipticity_b",
            "orientation_a_deg", "orientation_b_deg",
            "orientation_unc_a_deg", "orientation_unc_b_deg",
            "ellipticity_unc_a", "ellipticity_unc_b",
            "shape_reliable_a", "shape_reliable_b",
            "score",
            "T_radius", "T_nls", "T_rcr", "T_ellipticity", "T_orientation",
            "T_position", "chord_length_m", "expected_ellipticity",
            "p_value", "percentile_score", "bh_significant",
        ]).to_csv(all_pairs_path, index=False)
    print(f"  Saved: {all_pairs_path}")

    # 9b: significant pairs only
    sig_pairs_path = output_dir / "significant_pairs.csv"
    if not significant_pairs.empty:
        save_dataframe(significant_pairs, sig_pairs_path)
    else:
        pd.DataFrame(columns=[
            "idx_a", "idx_b", "lon_a", "lat_a", "lon_b", "lat_b",
            "separation_deg", "radius_a_m", "radius_b_m",
            "radius_px_a", "radius_px_b",
            "nls_a", "nls_b", "rcr_a", "rcr_b",
            "ellipticity_a", "ellipticity_b",
            "orientation_a_deg", "orientation_b_deg",
            "orientation_unc_a_deg", "orientation_unc_b_deg",
            "ellipticity_unc_a", "ellipticity_unc_b",
            "shape_reliable_a", "shape_reliable_b",
            "score",
            "T_radius", "T_nls", "T_rcr", "T_ellipticity", "T_orientation",
            "T_position", "chord_length_m", "expected_ellipticity",
            "p_value", "bh_significant",
        ]).to_csv(sig_pairs_path, index=False)
    print(f"  Saved: {sig_pairs_path}")

    # 9c: null best scores (binary NumPy array) + pair details
    null_path = output_dir / "null_best_scores.npy"
    np.save(str(null_path), null_scores)
    print(f"  Saved: {null_path}")

    null_details_path = output_dir / "null_pair_details.json"
    save_json(null_pair_details, null_details_path)
    print(f"  Saved: {null_details_path}")

    # 9d: global summary JSON
    t_total = time.monotonic() - t_start
    summary = {
        "n_chips_discovered": len(chip_dirs),
        "n_chips_with_data": n_chips_with_data,
        "n_craters": n_craters,
        "coverage_km2": round(coverage_km2, 2),
        "n_raw_pairs": n_raw_pairs,
        "n_top_pairs": n_top,
        "n_significant": n_significant,
        "best_score": best_score,
        "best_p_value": best_p_value,
        "fdr_alpha": fdr_alpha,
        "random_trials": random_trials,
        "random_seed": random_seed,
        "top_pairs_requested": top_pairs,
        "pairing_time_s": round(t_pair_elapsed, 2),
        "null_model_time_s": round(t_null_elapsed, 2),
        "total_runtime_s": round(t_total, 2),
        "chip_results_dir": str(chip_results_dir),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "global_summary.json"
    save_json(summary, summary_path)
    print(f"  Saved: {summary_path}")

    # ------------------------------------------------------------------
    # Step 10 (optional): save pair images for significant pairs
    # ------------------------------------------------------------------
    if args.save_pair_images and not significant_pairs.empty:
        _print_section("Saving significant pair images")
        _save_pair_images(
            significant_pairs=significant_pairs,
            craters=craters,
            chip_results_dir=chip_results_dir,
            output_dir=output_dir,
            config=config,
        )

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    _print_section("Complete")
    print(f"  Total runtime      : {t_total:.1f} s")
    print(f"  Output directory   : {output_dir}")
    print()

    return 0


# ---------------------------------------------------------------------------
# Checkpointed / array-friendly modes
# ---------------------------------------------------------------------------

def _run_prep(args: argparse.Namespace) -> int:
    """Prepare checkpoints for array-parallel null-model runs."""
    t_start = time.monotonic()
    chip_results_dir: Path = args.chip_results_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    random_trials: int = args.random_trials
    random_seed: int = args.random_seed
    fdr_alpha: float = args.fdr_alpha
    top_pairs: int = args.top_pairs
    progress_interval_sec: float | None = (
        args.progress_interval_sec if args.progress_interval_sec and args.progress_interval_sec > 0 else None
    )

    checkpoint_dir = _resolve_checkpoint_dir(output_dir, args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    craters_path = _craters_checkpoint_path(checkpoint_dir)
    top_pairs_path = _top_pairs_checkpoint_path(checkpoint_dir)
    meta_path = _checkpoint_meta_path(checkpoint_dir)
    if args.resume and craters_path.exists() and top_pairs_path.exists() and meta_path.exists():
        meta = _load_checkpoint_meta(checkpoint_dir)
        _validate_checkpoint_meta(meta, args)
        print(f"Checkpoint found; skipping prep. ({checkpoint_dir})")
        return 0

    if not chip_results_dir.exists():
        print(
            f"ERROR: chip-results-dir does not exist: {chip_results_dir}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    _print_section("Discovering chip results")
    chip_dirs = _discover_chip_dirs(chip_results_dir)
    print(f"  Found {len(chip_dirs):,d} chip directories with craters.csv")

    # Report chip health: scan metadata for status breakdown
    _report_chip_health(chip_dirs)

    if not chip_dirs:
        print(
            "ERROR: no chip_XXXX/craters.csv files found under "
            f"{chip_results_dir}",
            file=sys.stderr,
        )
        return 1

    _print_section("Loading crater catalogues")
    craters = _load_craters(chip_dirs)
    chip_metadata = _load_chip_metadata(chip_dirs)
    n_chips_with_data = int(craters["chip_index"].nunique()) if not craters.empty else 0
    n_craters = len(craters)
    coverage_km2 = _estimate_coverage_km2(chip_metadata, chip_dirs)
    _print_summary_stats(n_chips_with_data, n_craters, coverage_km2)

    if craters.empty:
        print("ERROR: no craters loaded — cannot proceed.", file=sys.stderr)
        print(
            "DETAIL: All chip directories existed but contained 0 craters. "
            "This can happen when all chips are blank/no-data, the WMS "
            "was unreachable, or the detection threshold was too stringent.",
            file=sys.stderr,
        )
        return 1

    config = ChordConfig(
        random_trials=random_trials,
        random_seed=random_seed,
        fdr_alpha=fdr_alpha,
        top_pairs_to_report=top_pairs,
    )

    _print_section("Running global chord-based pairing")
    print(f"  Building pairs from {n_craters:,d} craters …")
    t_pair_start = time.monotonic()
    all_pairs = build_chord_pairs(craters, config, progress_interval_sec=progress_interval_sec)
    t_pair_elapsed = time.monotonic() - t_pair_start
    n_raw_pairs = len(all_pairs)
    print(f"  Raw pairs found    : {n_raw_pairs:,d}  ({t_pair_elapsed:.1f} s)")

    if not all_pairs.empty:
        top_nonoverlap = select_top_nonoverlapping_pairs(all_pairs, top_k=top_pairs)
    else:
        top_nonoverlap = pd.DataFrame()

    n_top = len(top_nonoverlap)
    _print_pairing_summary(n_raw_pairs, n_top)

    _print_section("Saving checkpoints")
    craters.to_pickle(craters_path)
    save_dataframe(top_nonoverlap, top_pairs_path)
    meta = {
        "random_trials": random_trials,
        "random_seed": random_seed,
        "fdr_alpha": fdr_alpha,
        "top_pairs": top_pairs,
        "chip_results_dir": str(chip_results_dir),
        "output_dir": str(output_dir),
        "n_chips_discovered": len(chip_dirs),
        "n_chips_with_data": n_chips_with_data,
        "n_craters": n_craters,
        "coverage_km2": round(coverage_km2, 2),
        "n_raw_pairs": n_raw_pairs,
        "n_top_pairs": n_top,
        "pairing_time_s": round(t_pair_elapsed, 2),
        "null_chunk_count": args.null_chunk_count,
        "created_at_epoch_s": time.time(),
    }
    _write_checkpoint_meta(checkpoint_dir, meta)
    print(f"  Saved: {craters_path}")
    print(f"  Saved: {top_pairs_path}")
    print(f"  Saved: {meta_path}")

    t_total = time.monotonic() - t_start
    _print_section("Prep complete")
    print(f"  Total runtime      : {t_total:.1f} s")
    print(f"  Checkpoint dir     : {checkpoint_dir}")
    print()

    return 0


def _run_null(args: argparse.Namespace) -> int:
    """Run one null-model chunk with per-trial checkpointing.

    Each trial is saved individually as soon as it completes, so that if
    the job is killed (timeout, OOM, preemption) the completed trials are
    preserved.  On restart the chunk skips already-checkpointed trials.

    At the end, a legacy chunk file (``null_scores_part_XXXXX.npy``) is
    written so that ``_run_final`` and the resume sbatch can detect chunk
    completion without scanning individual trial files.
    """
    t_start = time.monotonic()
    output_dir: Path = args.output_dir.resolve()
    checkpoint_dir = _resolve_checkpoint_dir(output_dir, args.checkpoint_dir)
    meta = _load_checkpoint_meta(checkpoint_dir)
    _validate_checkpoint_meta(meta, args)

    craters_path = _craters_checkpoint_path(checkpoint_dir)
    if not craters_path.exists():
        print(f"ERROR: missing craters checkpoint: {craters_path}", file=sys.stderr)
        return 1
    craters = pd.read_pickle(craters_path)

    total_trials = args.random_trials
    chunk_count = args.null_chunk_count
    chunk_index = args.null_chunk_index
    if chunk_index < 0:
        env_idx = os.environ.get("SLURM_ARRAY_TASK_ID")
        if env_idx is None:
            print(
                "ERROR: null-chunk-index not provided and SLURM_ARRAY_TASK_ID not set.",
                file=sys.stderr,
            )
            return 1
        chunk_index = int(env_idx)
    start, count = _compute_chunk_bounds(total_trials, chunk_count, chunk_index)

    part_path = _null_part_path(checkpoint_dir, chunk_index)
    part_meta_path = _null_part_meta_path(checkpoint_dir, chunk_index)
    if args.resume and part_path.exists():
        print(f"Chunk checkpoint exists; skipping. ({part_path})")
        return 0

    # Determine which trials in this chunk are already done (per-trial files)
    trial_indices = list(range(start, start + count))
    remaining = [
        t for t in trial_indices
        if not _null_trial_score_path(checkpoint_dir, t).exists()
    ]

    _print_section("Running null model chunk")
    print(f"  Total trials       : {total_trials:,d}")
    print(f"  Chunk index        : {chunk_index} / {chunk_count - 1}")
    print(f"  Chunk trial range  : {start}–{start + count - 1}")
    print(f"  Already done       : {count - len(remaining)}")
    print(f"  Remaining          : {len(remaining)}")

    if not remaining:
        print("  All trials in this chunk already checkpointed.")
    else:
        config = ChordConfig(
            random_trials=total_trials,
            random_seed=args.random_seed,
            fdr_alpha=args.fdr_alpha,
            top_pairs_to_report=args.top_pairs,
        )
        use_progress = (
            args.progress_interval_sec
            and args.progress_interval_sec > 0
        )

        # Pre-filter craters once for all trials in this chunk
        from moonpiercer.null_model import (
            prefilter_qualifying_craters,
            run_single_null_trial,
        )
        qualifying = prefilter_qualifying_craters(craters, config)
        n_qualifying = len(qualifying)
        print(
            f"  Qualifying craters : {n_qualifying:,d} / {len(craters):,d}",
        )

        if n_qualifying < 2:
            print("  Fewer than 2 qualifying craters — all null scores will be 0.")
            for trial_idx in remaining:
                np.save(
                    str(_null_trial_score_path(checkpoint_dir, trial_idx)),
                    np.array([0.0]),
                )
                save_json({}, _null_trial_details_path(checkpoint_dir, trial_idx))
        else:
            # Derive all seed sequences upfront (indexed by global trial number)
            all_seeds = np.random.SeedSequence(args.random_seed).spawn(total_trials)

            next_report = time.monotonic() + (args.progress_interval_sec or 60)
            for i, trial_idx in enumerate(remaining):
                t_trial = time.monotonic()
                score, details = run_single_null_trial(
                    qualifying, all_seeds[trial_idx], config,
                    save_pair_details=True,
                )

                # Save immediately — this trial is now crash-safe
                np.save(
                    str(_null_trial_score_path(checkpoint_dir, trial_idx)),
                    np.array([score]),
                )
                save_json(
                    details,
                    _null_trial_details_path(checkpoint_dir, trial_idx),
                )

                trial_elapsed = time.monotonic() - t_trial
                if use_progress and time.monotonic() >= next_report:
                    done_in_chunk = i + 1
                    overall_done = sum(
                        1 for t in trial_indices
                        if _null_trial_score_path(checkpoint_dir, t).exists()
                    )
                    elapsed = time.monotonic() - t_start
                    print(
                        f"  [chunk {chunk_index}] trial {trial_idx} done "
                        f"({trial_elapsed:.1f}s)  |  "
                        f"chunk {done_in_chunk}/{len(remaining)}  |  "
                        f"elapsed {elapsed:.0f}s",
                        flush=True,
                    )
                    next_report = time.monotonic() + args.progress_interval_sec

    # Assemble chunk file from individual trial files (for _run_final compat)
    chunk_scores = []
    chunk_details = []
    for trial_idx in trial_indices:
        trial_path = _null_trial_score_path(checkpoint_dir, trial_idx)
        if trial_path.exists():
            chunk_scores.append(float(np.load(str(trial_path))[0]))
        else:
            chunk_scores.append(0.0)
        trial_det_path = _null_trial_details_path(checkpoint_dir, trial_idx)
        if trial_det_path.exists():
            chunk_details.append(load_json(trial_det_path))
        else:
            chunk_details.append({})

    null_runtime = round(time.monotonic() - t_start, 2)

    np.save(str(part_path), np.array(chunk_scores, dtype=np.float64))
    details_path = part_path.parent / part_path.name.replace(".npy", "_details.json")
    save_json(chunk_details, details_path)

    part_meta = {
        "chunk_index": chunk_index,
        "chunk_count": chunk_count,
        "trial_offset": start,
        "trial_count": count,
        "runtime_s": null_runtime,
    }
    save_json(part_meta, part_meta_path)
    print(f"  Saved: {part_path}")
    print(f"  Saved: {part_meta_path}")

    # Report overall null-model progress
    completed_chunks = sum(
        1 for i in range(chunk_count)
        if _null_part_path(checkpoint_dir, i).exists()
    )
    completed_trials = sum(
        1 for t in range(total_trials)
        if _null_trial_score_path(checkpoint_dir, t).exists()
    )
    print()
    print(f"  ══════════════════════════════════════════════")
    print(f"  Null chunk {chunk_index} COMPLETE in {null_runtime:.1f}s")
    print(f"  Overall: {completed_trials}/{total_trials} trials done"
          f"  ({completed_chunks}/{chunk_count} chunks complete)")
    if completed_chunks < chunk_count:
        missing_chunks = [
            i for i in range(chunk_count)
            if not _null_part_path(checkpoint_dir, i).exists()
        ]
        if len(missing_chunks) <= 10:
            print(f"  Incomplete chunks: {missing_chunks}")
        else:
            print(f"  Incomplete chunks: {len(missing_chunks)} remaining")
    print(f"  ══════════════════════════════════════════════")

    return 0


def _run_final(args: argparse.Namespace) -> int:
    """Merge null-model parts, compute significance, and write outputs."""
    t_start = time.monotonic()
    chip_results_dir: Path = args.chip_results_dir.resolve()
    output_dir: Path = args.output_dir.resolve()
    random_trials: int = args.random_trials
    random_seed: int = args.random_seed
    fdr_alpha: float = args.fdr_alpha
    top_pairs: int = args.top_pairs

    checkpoint_dir = _resolve_checkpoint_dir(output_dir, args.checkpoint_dir)
    meta = _load_checkpoint_meta(checkpoint_dir)
    _validate_checkpoint_meta(meta, args)

    top_pairs_path = _top_pairs_checkpoint_path(checkpoint_dir)
    if not top_pairs_path.exists():
        print(f"ERROR: missing top pairs checkpoint: {top_pairs_path}", file=sys.stderr)
        return 1

    top_nonoverlap = load_dataframe(top_pairs_path)
    if top_nonoverlap.empty:
        print("ERROR: top pairs checkpoint is empty — cannot proceed.", file=sys.stderr)
        return 1

    chunk_count = args.null_chunk_count
    part_paths = [_null_part_path(checkpoint_dir, i) for i in range(chunk_count)]
    missing = [p for p in part_paths if not p.exists()]
    if missing:
        print("ERROR: missing null-model chunk files:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 1

    null_scores = np.concatenate([np.load(str(p)) for p in part_paths])
    if len(null_scores) != random_trials:
        print(
            f"WARNING: expected {random_trials} null scores, got {len(null_scores)}",
            file=sys.stderr,
        )

    # Assemble null pair details (if available)
    null_pair_details: list[dict] = []
    for p in part_paths:
        details_file = p.parent / p.name.replace(".npy", "_details.json")
        if details_file.exists():
            null_pair_details.extend(load_json(details_file))
    has_null_details = len(null_pair_details) > 0

    _print_section("Computing significance (Benjamini-Hochberg FDR)")
    pairs_scored = compute_significance(top_nonoverlap, null_scores, alpha=fdr_alpha)
    n_significant = (
        int(pairs_scored["bh_significant"].sum())
        if not pairs_scored.empty and "bh_significant" in pairs_scored.columns
        else 0
    )

    if not pairs_scored.empty:
        best_score = float(pairs_scored["score"].iloc[0])
        best_p_value = float(pairs_scored["p_value"].iloc[0]) if "p_value" in pairs_scored.columns else None
    else:
        best_score = 0.0
        best_p_value = None

    _print_results_summary(pairs_scored, n_significant)

    _print_section("Saving outputs")
    all_pairs_path = output_dir / "all_pairs_scored.csv"
    if not pairs_scored.empty:
        save_dataframe(pairs_scored, all_pairs_path)
    else:
        pd.DataFrame(columns=[
            "idx_a", "idx_b", "lon_a", "lat_a", "lon_b", "lat_b",
            "separation_deg", "radius_a_m", "radius_b_m",
            "radius_px_a", "radius_px_b",
            "nls_a", "nls_b", "rcr_a", "rcr_b",
            "ellipticity_a", "ellipticity_b",
            "orientation_a_deg", "orientation_b_deg",
            "orientation_unc_a_deg", "orientation_unc_b_deg",
            "ellipticity_unc_a", "ellipticity_unc_b",
            "shape_reliable_a", "shape_reliable_b",
            "score",
            "T_radius", "T_nls", "T_rcr", "T_ellipticity", "T_orientation",
            "T_position", "chord_length_m", "expected_ellipticity",
            "p_value", "percentile_score", "bh_significant",
        ]).to_csv(all_pairs_path, index=False)
    print(f"  Saved: {all_pairs_path}")

    sig_pairs_path = output_dir / "significant_pairs.csv"
    if not pairs_scored.empty and "bh_significant" in pairs_scored.columns:
        significant_pairs = pairs_scored[pairs_scored["bh_significant"]].reset_index(drop=True)
        save_dataframe(significant_pairs, sig_pairs_path)
    else:
        significant_pairs = pd.DataFrame()
        pd.DataFrame(columns=[
            "idx_a", "idx_b", "lon_a", "lat_a", "lon_b", "lat_b",
            "separation_deg", "radius_a_m", "radius_b_m",
            "radius_px_a", "radius_px_b",
            "nls_a", "nls_b", "rcr_a", "rcr_b",
            "ellipticity_a", "ellipticity_b",
            "orientation_a_deg", "orientation_b_deg",
            "orientation_unc_a_deg", "orientation_unc_b_deg",
            "ellipticity_unc_a", "ellipticity_unc_b",
            "shape_reliable_a", "shape_reliable_b",
            "score",
            "T_radius", "T_nls", "T_rcr", "T_ellipticity", "T_orientation",
            "T_position", "chord_length_m", "expected_ellipticity",
            "p_value", "bh_significant",
        ]).to_csv(sig_pairs_path, index=False)
    print(f"  Saved: {sig_pairs_path}")

    null_path = output_dir / "null_best_scores.npy"
    np.save(str(null_path), null_scores)
    print(f"  Saved: {null_path}")

    if has_null_details:
        null_details_path = output_dir / "null_pair_details.json"
        save_json(null_pair_details, null_details_path)
        print(f"  Saved: {null_details_path}")

    null_time_total = 0.0
    for idx in range(chunk_count):
        meta_path = _null_part_meta_path(checkpoint_dir, idx)
        if meta_path.exists():
            part_meta = load_json(meta_path)
            null_time_total += float(part_meta.get("runtime_s", 0.0))

    t_total = time.monotonic() - t_start
    summary = {
        "n_chips_discovered": int(meta.get("n_chips_discovered", 0)),
        "n_chips_with_data": int(meta.get("n_chips_with_data", 0)),
        "n_craters": int(meta.get("n_craters", 0)),
        "coverage_km2": float(meta.get("coverage_km2", 0.0)),
        "n_raw_pairs": int(meta.get("n_raw_pairs", 0)),
        "n_top_pairs": int(meta.get("n_top_pairs", 0)),
        "n_significant": n_significant,
        "best_score": best_score,
        "best_p_value": best_p_value,
        "fdr_alpha": fdr_alpha,
        "random_trials": random_trials,
        "random_seed": random_seed,
        "top_pairs_requested": top_pairs,
        "pairing_time_s": float(meta.get("pairing_time_s", 0.0)),
        "null_model_time_s": round(null_time_total, 2),
        "total_runtime_s": round(t_total, 2),
        "chip_results_dir": str(chip_results_dir),
        "output_dir": str(output_dir),
    }
    summary_path = output_dir / "global_summary.json"
    save_json(summary, summary_path)
    print(f"  Saved: {summary_path}")

    if args.save_pair_images and not significant_pairs.empty:
        _print_section("Saving significant pair images")
        craters_path = _craters_checkpoint_path(checkpoint_dir)
        if craters_path.exists():
            craters = pd.read_pickle(craters_path)
            config = ChordConfig(
                random_trials=random_trials,
                random_seed=random_seed,
                fdr_alpha=fdr_alpha,
                top_pairs_to_report=top_pairs,
            )
            _save_pair_images(
                significant_pairs=significant_pairs,
                craters=craters,
                chip_results_dir=chip_results_dir,
                output_dir=output_dir,
                config=config,
            )
        else:
            print("  WARNING: craters checkpoint missing; skipping pair images.", file=sys.stderr)

    _print_section("Final stage complete")
    print(f"  Total runtime      : {t_total:.1f} s")
    print(f"  Output directory   : {output_dir}")
    print()
    return 0


def run(args: argparse.Namespace) -> int:
    if args.mode == "prep":
        return _run_prep(args)
    if args.mode == "null":
        return _run_null(args)
    if args.mode == "final":
        return _run_final(args)
    return _run_full(args)


# ---------------------------------------------------------------------------
# Optional: pair image saving
# ---------------------------------------------------------------------------

def _save_pair_images(
    significant_pairs: pd.DataFrame,
    craters: pd.DataFrame,
    chip_results_dir: Path,
    output_dir: Path,
    config: ChordConfig,
) -> None:
    """Attempt to save annotated chip images for each significant pair.

    This is a best-effort operation: if imagery is not available for a
    chip, that pair image is silently skipped.  Requires matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "  WARNING: matplotlib not available — skipping pair images.",
            file=sys.stderr,
        )
        return

    images_dir = output_dir / "pair_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    n_saved = 0
    max_images = config.max_significant_images

    for rank, row in significant_pairs.iterrows():
        if n_saved >= max_images:
            break

        idx_a = int(row["idx_a"])
        idx_b = int(row["idx_b"])

        # Retrieve crater records
        crater_a = craters.iloc[idx_a] if idx_a < len(craters) else None
        crater_b = craters.iloc[idx_b] if idx_b < len(craters) else None

        if crater_a is None or crater_b is None:
            continue

        # Try to locate chip imagery for each crater
        chip_a_dir = Path(str(crater_a.get("chip_dir", "")))
        chip_b_dir = Path(str(crater_b.get("chip_dir", "")))

        nac_a = chip_a_dir / "nac.png"
        nac_b = chip_b_dir / "nac.png"

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle(
            f"Pair rank {rank + 1}  |  score={row['score']:.4f}  "
            f"p={row.get('p_value', float('nan')):.4f}  "
            f"sep={row.get('separation_deg', float('nan')):.1f}°",
            fontsize=11,
        )

        for ax, nac_path, crater, label in [
            (axes[0], nac_a, crater_a, "Crater A"),
            (axes[1], nac_b, crater_b, "Crater B"),
        ]:
            ax.set_title(
                f"{label}  ({float(crater.get('lon_deg', 0)):+.3f}°, "
                f"{float(crater.get('lat_deg', 0)):+.3f}°)\n"
                f"r={float(crater.get('radius_m', 0)):.2f} m  "
                f"FI={float(crater.get('freshness_index', 0)):.3f}",
                fontsize=8,
            )
            if nac_path.exists():
                try:
                    from PIL import Image as _PILImage
                    img = np.asarray(_PILImage.open(nac_path).convert("L"))
                    ax.imshow(img, cmap="gray", origin="upper")
                except Exception:  # noqa: BLE001
                    ax.text(0.5, 0.5, "Image unavailable", transform=ax.transAxes,
                            ha="center", va="center")
            else:
                ax.text(0.5, 0.5, "Image unavailable", transform=ax.transAxes,
                        ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])

        out_path = images_dir / f"pair_{rank + 1:04d}.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        n_saved += 1

    print(f"  Saved {n_saved} pair image(s) to {images_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    sys.exit(run(args))


if __name__ == "__main__":
    main()
