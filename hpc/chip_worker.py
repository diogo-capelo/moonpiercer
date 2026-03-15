"""HPC chip worker — processes one or more chips from the manifest.

Called as a SLURM array task:

    python hpc/chip_worker.py \
        --manifest-path results/manifest.csv \
        --output-dir   results/chips \
        --chips-per-task 16 \
        --total-chips 15000

When --chips-per-task > 1, the SLURM_ARRAY_TASK_ID is a *task* index and
each task processes chips [task_id * chips_per_task, ...) up to --total-chips.

For each chip the script:
  1. Reads the manifest row matching --chip-index (manifest_index column).
  2. Builds a WMSClient and fetches the NAC grayscale chip.
  3. Optionally fetches LOLA elevation and computes terrain statistics.
  4. Runs crater detection (LoG + shape characterisation).
  5. Computes the Freshness Index for each detection.
  6. Adds selenographic (lon, lat) to each detection row.
  7. Saves craters.csv and metadata.json under output_dir/chip_{index:04d}/.
  8. Optionally saves an annotated PNG with viridis-coloured crater circles.

Exit codes
----------
0  success (including empty-chip cases where n_craters == 0)
1  fatal error (bad arguments, manifest row not found, etc.)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for HPC nodes
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

from moonpiercer.config import ChordConfig
from moonpiercer.wms import WMSClient
from moonpiercer.geometry import make_bbox_around_point, bbox_mpp, chip_pixel_to_lonlat
from moonpiercer.detection import detect_craters_on_chip
from moonpiercer.freshness import compute_freshness_for_chip
from moonpiercer.io_utils import save_dataframe, save_json


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process a single manifest chip as a SLURM array task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    parser.add_argument(
        "--manifest-path",
        type=Path,
        required=True,
        help="Path to the manifest CSV produced by the manifest builder.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Root directory for per-chip output subdirectories.",
    )

    # Chip identity — may also be taken from SLURM_ARRAY_TASK_ID
    parser.add_argument(
        "--chip-index",
        type=int,
        default=None,
        help=(
            "manifest_index of the chip to process.  "
            "If omitted, falls back to the SLURM_ARRAY_TASK_ID environment variable."
        ),
    )

    # Chip geometry
    parser.add_argument(
        "--chip-size-px",
        type=int,
        default=1024,
        help="NAC chip width and height in pixels.",
    )
    parser.add_argument(
        "--chip-span-m",
        type=float,
        default=1200.0,
        help="Physical half-width of the chip on the lunar surface [m].",
    )
    parser.add_argument(
        "--lola-tile-size-px",
        type=int,
        default=256,
        help="Width/height of the LOLA elevation tile in pixels.",
    )

    # Network / caching
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache") / "wms",
        help="Directory used for WMS disk cache.",
    )
    parser.add_argument(
        "--no-http-cache",
        action="store_true",
        help="Disable the WMS disk cache (always re-fetch).",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=int,
        default=60,
        help="Per-request HTTP timeout in seconds.",
    )

    # Multi-chip mode (used when MAX_CHIPS > ARRAY_SIZE)
    parser.add_argument(
        "--chips-per-task",
        type=int,
        default=1,
        help=(
            "Number of chips each array task processes.  "
            "chip indices = [task_id * chips_per_task, ..., (task_id+1) * chips_per_task - 1]."
        ),
    )
    parser.add_argument(
        "--total-chips",
        type=int,
        default=None,
        help="Total number of chips in the manifest.  Prevents tasks from exceeding bounds.",
    )

    # Output control
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save an annotated PNG showing detected craters colour-coded by freshness_index.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Chip-index resolution
# ---------------------------------------------------------------------------

def resolve_chip_index(requested: int | None) -> int:
    """Return the chip index from the CLI argument or SLURM_ARRAY_TASK_ID."""
    if requested is not None:
        return int(requested)
    slurm_id = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
    if slurm_id:
        return int(slurm_id)
    raise ValueError(
        "--chip-index was not provided and SLURM_ARRAY_TASK_ID is not set."
    )


# ---------------------------------------------------------------------------
# Annotated image
# ---------------------------------------------------------------------------

def save_annotated_image(
    gray: np.ndarray,
    detections: pd.DataFrame,
    out_path: Path,
) -> None:
    """Save a PNG of the chip with viridis circles for each detected crater.

    Circle radius matches the detected crater radius in pixels.
    Colour encodes freshness_index (viridis, 0 = purple, 1 = yellow).
    Chips with no detections are saved as plain grayscale images.
    """
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    ax.imshow(gray, cmap="gray", origin="upper", vmin=0.0, vmax=1.0)
    ax.set_title(f"Detections: {len(detections)}")
    ax.set_xlabel("x [px]")
    ax.set_ylabel("y [px]")

    if not detections.empty and "freshness_index" in detections.columns:
        cmap = cm.viridis
        fi_vals = detections["freshness_index"].to_numpy(dtype=float)
        fi_min = float(np.nanmin(fi_vals)) if np.isfinite(fi_vals).any() else 0.0
        fi_max = float(np.nanmax(fi_vals)) if np.isfinite(fi_vals).any() else 1.0
        fi_range = fi_max - fi_min if fi_max > fi_min else 1.0

        for _, row in detections.iterrows():
            fi = float(row["freshness_index"]) if np.isfinite(row["freshness_index"]) else 0.0
            colour = cmap((fi - fi_min) / fi_range)
            circle = plt.Circle(
                (float(row["x"]), float(row["y"])),
                radius=float(row["radius_px"]),
                edgecolor=colour,
                facecolor="none",
                linewidth=1.0,
                alpha=0.85,
            )
            ax.add_patch(circle)

        # Colourbar
        sm = cm.ScalarMappable(
            cmap=cmap,
            norm=plt.Normalize(vmin=fi_min, vmax=fi_max),
        )
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label="Freshness Index", fraction=0.046, pad=0.04)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process_chip(
    args: argparse.Namespace,
    chip_index: int,
    manifest: pd.DataFrame,
    config: "ChordConfig",
    client: "WMSClient",
) -> int:
    """Process a single chip identified by *chip_index* (manifest_index).

    Returns 0 on success (including empty-chip cases), 1 on fatal error.
    """
    t_start = time.perf_counter()

    # ------------------------------------------------------------ manifest lookup
    matching = manifest[manifest["manifest_index"] == chip_index]
    if matching.empty:
        print(
            f"[chip_worker] ERROR: no manifest row with manifest_index={chip_index}.",
            file=sys.stderr,
        )
        return 1

    row = matching.iloc[0]
    product_id  = str(row["product_id"])
    center_lon  = float(row["center_lon"])
    center_lat  = float(row["center_lat"])

    print(
        f"[chip_worker] chip_index={chip_index}  product_id={product_id} "
        f"center=({center_lon:.4f}, {center_lat:.4f})",
        flush=True,
    )

    # ------------------------------------------------- output directory
    chip_dir = Path(args.output_dir) / f"chip_{chip_index:04d}"
    chip_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------- bounding box
    bbox = make_bbox_around_point(
        lon_deg=center_lon,
        lat_deg=center_lat,
        span_m=config.chip_span_m,
        lunar_radius_m=config.lunar_radius_m,
    )
    if bbox is None:
        print(
            f"[chip_worker] WARNING: could not compute bbox for chip {chip_index} "
            f"(near pole or antimeridian). Saving empty outputs.",
            file=sys.stderr,
        )
        _save_empty(chip_dir, chip_index, product_id, center_lon, center_lat)
        _print_timing(t_start, chip_index, n_craters=0, status="invalid_bbox")
        return 0

    lon_min, lat_min, lon_max, lat_max = bbox

    # --------------------------------------------------------- fetch NAC chip
    t_fetch = time.perf_counter()
    gray = client.fetch_nac_chip(
        bbox=bbox,
        width_px=config.chip_size_px,
        height_px=config.chip_size_px,
    )
    t_fetch_done = time.perf_counter()
    print(
        f"[chip_worker] NAC fetch: {t_fetch_done - t_fetch:.1f}s "
        f"({'ok' if gray is not None else 'FAILED'})",
        flush=True,
    )

    if gray is None:
        print(
            f"[chip_worker] WARNING: NAC chip download failed for chip {chip_index}.",
            file=sys.stderr,
        )
        _save_empty(chip_dir, chip_index, product_id, center_lon, center_lat, bbox=bbox)
        _print_timing(t_start, chip_index, n_craters=0, status="chip_download_failed")
        return 0

    # Guard: blank / uniform NAC images produce only noise-floor LoG
    # detections (identical strength, radius, zero depth_proxy & freshness).
    chip_std = float(np.std(gray))
    if chip_std < 1e-4:
        print(
            f"[chip_worker] WARNING: NAC chip is blank/uniform for chip {chip_index} "
            f"(std={chip_std:.6g}). No real craters possible. Saving empty outputs.",
            file=sys.stderr,
        )
        _save_empty(
            chip_dir, chip_index, product_id, center_lon, center_lat,
            bbox=bbox, status="blank_chip",
        )
        _print_timing(t_start, chip_index, n_craters=0, status="blank_chip")
        return 0

    # ---------------------------------------------------- metres per pixel
    mpp_x, mpp_y = bbox_mpp(
        bbox=bbox,
        width_px=config.chip_size_px,
        height_px=config.chip_size_px,
        lunar_radius_m=config.lunar_radius_m,
    )
    mpp_mean = 0.5 * (float(mpp_x) + float(mpp_y))

    # ----------------------------------------------------- LOLA elevation
    terrain_mean_m   = float("nan")
    terrain_std_m    = float("nan")
    terrain_slope_proxy = float("nan")

    t_lola = time.perf_counter()
    lola = client.fetch_lola_elevation(
        bbox=bbox,
        width_px=config.lola_tile_size_px,
        height_px=config.lola_tile_size_px,
    )
    t_lola_done = time.perf_counter()

    if lola is not None and np.isfinite(lola).any():
        terrain_mean_m      = float(np.nanmean(lola))
        terrain_std_m       = float(np.nanstd(lola))
        gy, gx              = np.gradient(lola.astype(np.float64))
        terrain_slope_proxy = float(np.nanmedian(np.hypot(gx, gy)))
        print(
            f"[chip_worker] LOLA fetch: {t_lola_done - t_lola:.1f}s  "
            f"terrain_mean={terrain_mean_m:.1f}m "
            f"terrain_std={terrain_std_m:.1f}m "
            f"slope_proxy={terrain_slope_proxy:.4f}",
            flush=True,
        )
    else:
        print(
            f"[chip_worker] LOLA fetch: {t_lola_done - t_lola:.1f}s  (no data)",
            flush=True,
        )

    # --------------------------------------------------- crater detection
    t_detect = time.perf_counter()
    detections, threshold = detect_craters_on_chip(
        gray=gray,
        mpp_mean=mpp_mean,
        config=config,
    )
    t_detect_done = time.perf_counter()
    print(
        f"[chip_worker] detection: {t_detect_done - t_detect:.1f}s  "
        f"n_craters={len(detections)}  threshold={threshold:.6g}",
        flush=True,
    )

    # Guard: noise-floor detections.
    MIN_LOG_THRESHOLD = 0.01
    if threshold < MIN_LOG_THRESHOLD:
        print(
            f"[chip_worker] WARNING: LoG threshold ({threshold:.6g}) below floor "
            f"({MIN_LOG_THRESHOLD}) for chip {chip_index}. "
            f"Image is likely featureless. Saving empty outputs.",
            file=sys.stderr,
        )
        _save_empty(
            chip_dir, chip_index, product_id, center_lon, center_lat,
            bbox=bbox, status="noise_floor",
        )
        _print_timing(t_start, chip_index, n_craters=0, status="noise_floor")
        return 0

    # ---------------------------------------------------- freshness index
    if not detections.empty:
        t_fresh = time.perf_counter()
        detections = compute_freshness_for_chip(
            gray=gray,
            detections=detections,
            threshold=threshold,
            config=config,
        )
        t_fresh_done = time.perf_counter()
        print(
            f"[chip_worker] freshness: {t_fresh_done - t_fresh:.1f}s",
            flush=True,
        )

        # ----------------------------------------- selenographic coordinates
        crater_lon, crater_lat = chip_pixel_to_lonlat(
            x=detections["x"].to_numpy(),
            y=detections["y"].to_numpy(),
            bbox=bbox,
            width=config.chip_size_px,
            height=config.chip_size_px,
        )
        detections = detections.copy()
        detections["lon_deg"] = crater_lon
        detections["lat_deg"] = crater_lat
        detections["product_id"] = product_id
        detections["chip_index"] = chip_index
    else:
        # Ensure freshness columns exist even when detections are empty
        for col in ("nls", "rcr", "freshness_index", "lon_deg", "lat_deg",
                    "product_id", "chip_index"):
            detections[col] = pd.Series(dtype=float if col not in ("product_id",) else object)

    # ------------------------------------------------------- save outputs
    craters_path = chip_dir / "craters.csv"
    save_dataframe(detections, craters_path)
    print(f"[chip_worker] craters saved → {craters_path}", flush=True)

    # Save raw grayscale chip for pair image visualisation
    nac_path = chip_dir / "nac.png"
    try:
        from PIL import Image as _PILImage
        nac_uint8 = np.clip(gray * 255.0, 0, 255).astype(np.uint8)
        _PILImage.fromarray(nac_uint8, mode="L").save(str(nac_path))
    except Exception as exc:
        print(
            f"[chip_worker] WARNING: could not save nac.png for chip {chip_index}: {exc}",
            file=sys.stderr,
        )

    metadata = {
        "chip_index":          chip_index,
        "product_id":          product_id,
        "center_lon":          center_lon,
        "center_lat":          center_lat,
        "bbox": {
            "lon_min": lon_min,
            "lat_min": lat_min,
            "lon_max": lon_max,
            "lat_max": lat_max,
        },
        "mpp_x":               float(mpp_x),
        "mpp_y":               float(mpp_y),
        "mpp_mean":            float(mpp_mean),
        "chip_size_px":        config.chip_size_px,
        "chip_span_m":         config.chip_span_m,
        "n_craters":           int(len(detections)),
        "threshold":           float(threshold),
        "terrain_mean_m":      terrain_mean_m   if np.isfinite(terrain_mean_m)   else None,
        "terrain_std_m":       terrain_std_m    if np.isfinite(terrain_std_m)    else None,
        "terrain_slope_proxy": terrain_slope_proxy if np.isfinite(terrain_slope_proxy) else None,
        "status":              "ok",
    }
    metadata_path = chip_dir / "metadata.json"
    save_json(metadata, metadata_path)
    print(f"[chip_worker] metadata saved → {metadata_path}", flush=True)

    # ------------------------------------------------- annotated image
    if args.save_annotated:
        annotated_path = chip_dir / "annotated.png"
        save_annotated_image(gray, detections, annotated_path)
        print(f"[chip_worker] annotated image saved → {annotated_path}", flush=True)

    _print_timing(t_start, chip_index, n_craters=len(detections), status="ok")
    return 0


def main() -> int:
    t_main = time.perf_counter()

    # ------------------------------------------------------------------ args
    args = parse_args()

    try:
        task_id = resolve_chip_index(args.chip_index)
    except ValueError as exc:
        print(f"[chip_worker] ERROR: {exc}", file=sys.stderr)
        return 1

    # ----------------------------------------- compute chip range for this task
    chips_per_task = args.chips_per_task
    total_chips = args.total_chips

    start_chip = task_id * chips_per_task
    end_chip = start_chip + chips_per_task
    if total_chips is not None:
        end_chip = min(end_chip, total_chips)

    if total_chips is not None and start_chip >= total_chips:
        print(
            f"[chip_worker] Task {task_id}: no chips in range "
            f"(start={start_chip} >= total={total_chips}). Nothing to do.",
            flush=True,
        )
        return 0

    chip_indices = list(range(start_chip, end_chip))
    print(
        f"[chip_worker] Task {task_id}: processing {len(chip_indices)} chip(s) "
        f"[{chip_indices[0]}..{chip_indices[-1]}]",
        flush=True,
    )

    # --------------------------------------------------------- load manifest once
    manifest_path = Path(args.manifest_path)
    if not manifest_path.exists():
        print(f"[chip_worker] ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = pd.read_csv(manifest_path)

    if "manifest_index" not in manifest.columns:
        print(
            "[chip_worker] ERROR: manifest CSV must contain a 'manifest_index' column.",
            file=sys.stderr,
        )
        return 1

    # ----------------------------------------- build config + WMS client once
    config = ChordConfig(
        cache_dir=Path(args.cache_dir),
        use_http_cache=not bool(args.no_http_cache),
        request_timeout_s=int(args.request_timeout_s),
        chip_size_px=int(args.chip_size_px),
        chip_span_m=float(args.chip_span_m),
        lola_tile_size_px=int(args.lola_tile_size_px),
    )

    client = WMSClient(
        base_url=config.wms_base_url,
        cache_dir=config.cache_dir,
        use_cache=config.use_http_cache,
        timeout_s=config.request_timeout_s,
    )

    # ---------------------------------------------------- process each chip
    n_ok = 0
    n_fail = 0
    failed_chips: list[int] = []
    for idx_in_batch, chip_index in enumerate(chip_indices):
        print(
            f"[chip_worker] Task {task_id}: chip {idx_in_batch + 1}/{len(chip_indices)} "
            f"(manifest_index={chip_index})",
            flush=True,
        )
        rc = process_chip(args, chip_index, manifest, config, client)
        if rc == 0:
            n_ok += 1
        else:
            n_fail += 1
            failed_chips.append(chip_index)

    elapsed = time.perf_counter() - t_main
    print(flush=True)
    print(
        f"[chip_worker] ══════════════════════════════════════════════",
        flush=True,
    )
    print(
        f"[chip_worker] Task {task_id} COMPLETE: {n_ok} ok, {n_fail} failed "
        f"out of {len(chip_indices)} chips  elapsed={elapsed:.1f}s",
        flush=True,
    )
    if failed_chips:
        print(
            f"[chip_worker] FAILED chip indices: {failed_chips}",
            flush=True,
        )
    print(
        f"[chip_worker] ══════════════════════════════════════════════",
        flush=True,
    )
    # Only fail the task if every chip failed
    return 1 if n_ok == 0 and n_fail > 0 else 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_empty(
    chip_dir: Path,
    chip_index: int,
    product_id: str,
    center_lon: float,
    center_lat: float,
    bbox: tuple[float, float, float, float] | None = None,
    status: str = "no_data",
) -> None:
    """Write empty craters.csv and failure metadata.json for an unprocessable chip."""
    empty_cols = [
        "x", "y", "radius_px", "radius_m", "strength", "depth_proxy",
        "circularity", "ellipticity", "orientation_deg", "shape_reliable",
        "nls", "rcr", "freshness_index", "lon_deg", "lat_deg",
        "product_id", "chip_index",
    ]
    empty_df = pd.DataFrame(columns=empty_cols)
    save_dataframe(empty_df, chip_dir / "craters.csv")

    bbox_dict: dict = {}
    if bbox is not None:
        lon_min, lat_min, lon_max, lat_max = bbox
        bbox_dict = {
            "lon_min": lon_min,
            "lat_min": lat_min,
            "lon_max": lon_max,
            "lat_max": lat_max,
        }

    metadata = {
        "chip_index":          chip_index,
        "product_id":          product_id,
        "center_lon":          center_lon,
        "center_lat":          center_lat,
        "bbox":                bbox_dict,
        "mpp_x":               None,
        "mpp_y":               None,
        "mpp_mean":            None,
        "chip_size_px":        None,
        "chip_span_m":         None,
        "n_craters":           0,
        "threshold":           None,
        "terrain_mean_m":      None,
        "terrain_std_m":       None,
        "terrain_slope_proxy": None,
        "status":              status,
    }
    save_json(metadata, chip_dir / "metadata.json")


def _print_timing(
    t_start: float,
    chip_index: int,
    n_craters: int,
    status: str,
) -> None:
    elapsed = time.perf_counter() - t_start
    print(
        f"[chip_worker] DONE  chip_index={chip_index}  "
        f"n_craters={n_craters}  status={status}  "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    raise SystemExit(main())
