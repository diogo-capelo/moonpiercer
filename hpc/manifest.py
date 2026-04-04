"""Build a full-sphere NAC chip manifest for MOONPIERCER HPC chord-search jobs.

Sweeps the entire lunar sphere on a lon/lat grid, queries NAC observation
metadata via WMS, deduplicates by product_id (keeping best resolution),
then selects chips with good spatial coverage via stratified binning.

Unlike the antipodal half-sphere manifest builder this sweeps lon [-180, 180)
because MOONPIERCER matches any chord through the Moon, not just antipodal pairs.

Usage
-----
python hpc/manifest.py \\
    --manifest-path results/fullres_hpc/manifest.csv \\
    --max-chips 448 \\
    --sweep-grid-step-deg 2.0 \\
    --max-workers 6 \\
    --max-requests-per-second 2.0
"""
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from moonpiercer.config import ChordConfig
from moonpiercer.geometry import normalize_lon, to_float
from moonpiercer.io_utils import save_json
from moonpiercer.wms import WMSClient


# ---------------------------------------------------------------------------
# Longitude range helper (full-sphere aware)
# ---------------------------------------------------------------------------

def lon_in_range(lon_deg: float, lon_min_deg: float, lon_max_deg: float) -> bool:
    """Return True if *lon_deg* falls within [lon_min_deg, lon_max_deg).

    Handles wrap-around correctly (e.g. lon_min=170, lon_max=-170 wraps
    across the antimeridian).  For the full-sphere default (-180 to 180)
    every longitude is included.
    """
    lon = float(normalize_lon(float(lon_deg)))
    lon_min = float(normalize_lon(float(lon_min_deg)))
    lon_max = float(normalize_lon(float(lon_max_deg)))

    # Full-sphere shortcut: lon_min == lon_max after normalisation means
    # the caller specified a 360° range (e.g. -180 / 180 → both become -180).
    if lon_min == lon_max:
        return True

    if lon_min < lon_max:
        return lon_min <= lon < lon_max
    # Wrap-around case: range crosses the antimeridian.
    return lon >= lon_min or lon < lon_max


# ---------------------------------------------------------------------------
# Progress logger
# ---------------------------------------------------------------------------

class ManifestProgressLogger:
    """Emit manifest-build progress every fixed percentage or time interval."""

    def __init__(
        self,
        total: int,
        percent_step: float = 5.0,
        time_step_s: float = 600.0,
    ) -> None:
        self.total = max(int(total), 0)
        self.percent_step = max(float(percent_step), 0.0)
        self.time_step_s = max(float(time_step_s), 0.0)
        self.start = time.monotonic()
        self.last_log = self.start
        self.next_percent = self.percent_step if self.percent_step > 0 else float("inf")

    def maybe_log(self, completed: int, *, force: bool = False) -> None:
        if self.total <= 0:
            return

        done = min(max(int(completed), 0), self.total)
        now = time.monotonic()
        pct = 100.0 * float(done) / float(self.total)
        hit_percent = pct >= self.next_percent
        hit_time = self.time_step_s > 0 and (now - self.last_log) >= self.time_step_s
        finished = done >= self.total

        if not (force or hit_percent or hit_time or finished):
            return

        elapsed_s = max(now - self.start, 1e-9)
        rate = float(done) / elapsed_s
        remaining = (self.total - done) / rate if rate > 0 else np.inf
        eta_str = f", eta={remaining / 60.0:.1f}m" if np.isfinite(remaining) else ""
        print(
            f"[manifest] progress {done:,}/{self.total:,} ({pct:5.1f}%), "
            f"elapsed={elapsed_s / 60.0:.1f}m{eta_str}",
            flush=True,
        )
        self.last_log = now

        if self.percent_step > 0:
            while pct >= self.next_percent:
                self.next_percent += self.percent_step


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RequestRateLimiter:
    """Simple global request rate limiter shared across worker threads."""

    def __init__(self, max_requests_per_second: float) -> None:
        self.max_requests_per_second = max(float(max_requests_per_second), 0.0)
        self.min_interval = (
            0.0
            if self.max_requests_per_second <= 0
            else 1.0 / self.max_requests_per_second
        )
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def wait(self) -> None:
        if self.min_interval <= 0:
            return
        while True:
            now = time.monotonic()
            with self._lock:
                if now >= self._next_allowed:
                    self._next_allowed = now + self.min_interval
                    return
                sleep_s = self._next_allowed - now
            if sleep_s > 0:
                time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# Grid generation
# ---------------------------------------------------------------------------

def global_lonlat_grid(
    step_deg: float,
    min_lat_deg: float = -85.0,
    max_lat_deg: float = 85.0,
) -> list[tuple[float, float]]:
    """Return a regular (lon, lat) grid covering the full sphere.

    Longitude always spans [-180, 180).  Latitude spans
    [*min_lat_deg*, *max_lat_deg*] inclusive.
    """
    step = float(step_deg)
    lats = np.arange(float(min_lat_deg), float(max_lat_deg) + 1e-9, step)
    lons = np.arange(-180.0, 180.0, step)
    return [(float(lon), float(lat)) for lat in lats for lon in lons]


# ---------------------------------------------------------------------------
# Single-grid-point query
# ---------------------------------------------------------------------------

def query_grid_point_products(
    client: WMSClient,
    config: ChordConfig,
    lon: float,
    lat: float,
    search_radius: float,
    lon_min_deg: float,
    lon_max_deg: float,
    limiter: RequestRateLimiter | None,
) -> list[dict]:
    """Query NAC observation metadata for one grid point and return valid rows."""
    if limiter is not None:
        limiter.wait()

    observations = client.get_featureinfo_yaml(
        layer=config.nac_query_layer,
        lon_deg=float(lon),
        lat_deg=float(lat),
        search_radius_deg=float(search_radius),
        feature_count=int(config.feature_count),
    )

    rows: list[dict] = []
    for obs in observations:
        product_id = str(obs.get("product_id", "")).strip()
        if not product_id:
            continue

        resolution = to_float(obs.get("resolution"), default=np.nan)
        if not np.isfinite(resolution) or float(resolution) > float(config.max_nac_resolution_mpp):
            continue

        center_lon = normalize_lon(to_float(obs.get("center_longitude"), default=lon))
        center_lat = to_float(obs.get("center_latitude"), default=lat)

        if not lon_in_range(float(center_lon), lon_min_deg, lon_max_deg):
            continue
        if not np.isfinite(center_lat):
            continue

        rows.append(
            {
                "product_id": product_id,
                "resolution_mpp": float(resolution),
                "center_lon": float(center_lon),
                "center_lat": float(center_lat),
                "incidence_angle_deg": to_float(obs.get("incidence_angle"), default=np.nan),
                "emission_angle_deg": to_float(obs.get("emission_angle"), default=np.nan),
                "phase_angle_deg": to_float(obs.get("phase_angle"), default=np.nan),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Full-sphere collection
# ---------------------------------------------------------------------------

def collect_nac_products_full_sphere(
    config: ChordConfig,
    lon_min_deg: float = -180.0,
    lon_max_deg: float = 180.0,
    min_lat_deg: float = -85.0,
    max_lat_deg: float = 85.0,
    max_workers: int = 6,
    max_requests_per_second: float = 2.0,
) -> pd.DataFrame:
    """Sweep the full sphere on a grid and collect unique high-res NAC products.

    Parameters
    ----------
    config:
        ChordConfig holding WMS/sweep settings.
    lon_min_deg, lon_max_deg:
        Longitude bounds [degrees].  Defaults cover the full sphere.
    min_lat_deg, max_lat_deg:
        Latitude bounds [degrees].  Defaults exclude poles.
    max_workers:
        Thread pool size.  Set 1 for sequential (easier debugging).
    max_requests_per_second:
        Global rate cap across all worker threads (<=0 disables limiter).

    Returns
    -------
    pd.DataFrame
        One row per unique product, sorted by resolution_mpp ascending.
        Columns: product_id, resolution_mpp, center_lon, center_lat,
        incidence_angle_deg, emission_angle_deg, phase_angle_deg.
    """
    client = WMSClient(
        base_url=config.wms_base_url,
        cache_dir=config.cache_dir,
        use_cache=config.use_http_cache,
        timeout_s=config.request_timeout_s,
    )

    grid = global_lonlat_grid(
        step_deg=config.sweep_grid_step_deg,
        min_lat_deg=float(min_lat_deg),
        max_lat_deg=float(max_lat_deg),
    )
    grid = [
        (lon, lat)
        for lon, lat in grid
        if lon_in_range(lon, lon_min_deg, lon_max_deg)
    ]
    if config.max_grid_queries > 0:
        grid = grid[: config.max_grid_queries]

    n_grid = len(grid)
    print(
        f"[manifest] full-sphere sweep: {n_grid:,} grid points "
        f"(step={config.sweep_grid_step_deg}°, "
        f"lon=[{lon_min_deg}, {lon_max_deg}], "
        f"lat=[{min_lat_deg}, {max_lat_deg}])",
        flush=True,
    )
    if n_grid > 0:
        print(
            f"[manifest] logging progress every 5% or 600s",
            flush=True,
        )

    progress_logger = ManifestProgressLogger(
        total=n_grid, percent_step=5.0, time_step_s=600.0
    )
    product_map: dict[str, dict] = {}
    search_radius = max(0.6 * float(config.sweep_grid_step_deg), 0.15)
    limiter = RequestRateLimiter(max_requests_per_second=float(max_requests_per_second))

    def _merge(rows: list[dict]) -> None:
        for row in rows:
            prev = product_map.get(row["product_id"])
            if prev is None or row["resolution_mpp"] < prev["resolution_mpp"]:
                product_map[row["product_id"]] = row

    if int(max_workers) <= 1:
        for idx, (lon, lat) in enumerate(grid, start=1):
            rows = query_grid_point_products(
                client=client,
                config=config,
                lon=float(lon),
                lat=float(lat),
                search_radius=float(search_radius),
                lon_min_deg=float(lon_min_deg),
                lon_max_deg=float(lon_max_deg),
                limiter=limiter,
            )
            _merge(rows)
            progress_logger.maybe_log(idx)
    else:
        with ThreadPoolExecutor(max_workers=int(max_workers)) as executor:
            futures = [
                executor.submit(
                    query_grid_point_products,
                    client,
                    config,
                    float(lon),
                    float(lat),
                    float(search_radius),
                    float(lon_min_deg),
                    float(lon_max_deg),
                    limiter,
                )
                for lon, lat in grid
            ]
            for idx, future in enumerate(as_completed(futures), start=1):
                try:
                    rows = future.result()
                except Exception as exc:
                    print(
                        f"[manifest] WARNING: grid query raised {type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    rows = []
                _merge(rows)
                progress_logger.maybe_log(idx)

    progress_logger.maybe_log(n_grid, force=True)

    if not product_map:
        print("[manifest] WARNING: no products discovered.", flush=True)
        return pd.DataFrame()

    products = pd.DataFrame(product_map.values())
    products = products.dropna(subset=["center_lon", "center_lat"]).copy()
    products["center_lon"] = products["center_lon"].astype(float)
    products["center_lat"] = products["center_lat"].astype(float)
    products = products.sort_values("resolution_mpp", ascending=True).reset_index(drop=True)
    n_before = len(products)

    products = _dedup_calibration_levels(products)
    print(
        f"[manifest] discovered {n_before:,} unique products, "
        f"{len(products):,} after calibration-level dedup.",
        flush=True,
    )
    return products


# ---------------------------------------------------------------------------
# Calibration-level deduplication
# ---------------------------------------------------------------------------

def _dedup_calibration_levels(products: pd.DataFrame) -> pd.DataFrame:
    """Keep one calibration level per observation stem.

    NAC product IDs end with ``<L|R><E|C>`` where L/R identifies the
    left/right camera and E/C is the EDR/CDR calibration level.  E and C
    versions of the same image produce identical crater detections, so
    processing both wastes HPC time and creates duplicate craters.

    Group by observation stem (``product_id[:-1]``, e.g.
    ``M175462328R``) and keep the row with best (lowest) resolution.
    """
    if products.empty or "product_id" not in products.columns:
        return products.copy()

    df = products.copy()
    df["_obs_stem"] = df["product_id"].str[:-1]
    df = df.sort_values("resolution_mpp")
    df = df.drop_duplicates(subset="_obs_stem", keep="first")
    return df.drop(columns="_obs_stem").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Stratified spatial selection
# ---------------------------------------------------------------------------

def select_products_stratified(
    products: pd.DataFrame,
    max_chips: int,
    bin_size_deg: float = 15.0,
) -> pd.DataFrame:
    """Pick up to *max_chips* products with good spatial coverage.

    The sphere is divided into lat/lon bins of *bin_size_deg*.  The
    best-resolution product from each occupied bin is selected first.
    Remaining slots are filled by overall best resolution.

    Parameters
    ----------
    products:
        DataFrame with columns center_lon, center_lat, resolution_mpp.
    max_chips:
        Maximum number of chips to return.
    bin_size_deg:
        Angular size of each spatial bin in degrees.

    Returns
    -------
    pd.DataFrame
        Subset of *products*, sorted by resolution_mpp ascending.
    """
    if products.empty:
        return products.copy()

    n_max = int(max_chips)
    if n_max <= 0 or n_max >= len(products):
        return products.sort_values("resolution_mpp", ascending=True).reset_index(drop=True)

    bs = float(bin_size_deg)
    df = products.copy()
    df["_lat_bin"] = np.floor((df["center_lat"] + 90.0) / bs).astype(int)
    df["_lon_bin"] = np.floor((df["center_lon"] + 180.0) / bs).astype(int)
    df["_geo_bin"] = df["_lat_bin"].astype(str) + "_" + df["_lon_bin"].astype(str)

    chosen_indices: list[int] = []
    used_idx: set[int] = set()

    # Pass 1: best product per spatial bin.
    for _, grp in df.groupby("_geo_bin"):
        if len(chosen_indices) >= n_max:
            break
        best_loc = grp.sort_values("resolution_mpp").index[0]
        chosen_indices.append(int(best_loc))
        used_idx.add(int(best_loc))

    # Pass 2: fill remaining slots by resolution.
    if len(chosen_indices) < n_max:
        for idx in df.sort_values("resolution_mpp").index:
            if int(idx) in used_idx:
                continue
            chosen_indices.append(int(idx))
            used_idx.add(int(idx))
            if len(chosen_indices) >= n_max:
                break

    out = (
        df.loc[chosen_indices]
        .drop(columns=["_lat_bin", "_lon_bin", "_geo_bin"])
        .sort_values("resolution_mpp", ascending=True)
        .reset_index(drop=True)
    )
    return out


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a full-sphere NAC chip manifest for MOONPIERCER HPC chord-search jobs."
        )
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("results") / "fullres_hpc" / "manifest.csv",
        help="Output CSV path for the chip manifest.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=None,
        help=(
            "Output JSON summary path "
            "(default: <manifest parent>/manifest_summary.json)."
        ),
    )
    parser.add_argument(
        "--max-chips",
        type=int,
        default=448,
        help="Maximum number of chips to keep in the manifest.",
    )
    parser.add_argument(
        "--sweep-grid-step-deg",
        type=float,
        default=2.0,
        help="Grid spacing for the lon/lat sweep [degrees].",
    )
    parser.add_argument(
        "--max-grid-queries",
        type=int,
        default=0,
        help="Cap on total grid queries (0 = unlimited).",
    )
    parser.add_argument(
        "--feature-count",
        type=int,
        default=40,
        help="WMS GetFeatureInfo FEATURE_COUNT per query.",
    )
    parser.add_argument(
        "--max-nac-resolution-mpp",
        type=float,
        default=2.0,
        help="Reject NAC products coarser than this resolution [m/px].",
    )
    # Longitude / latitude bounds.
    parser.add_argument(
        "--lon-min-deg",
        type=float,
        default=-180.0,
        help="Minimum longitude of sweep region [degrees] (default: -180).",
    )
    parser.add_argument(
        "--lon-max-deg",
        type=float,
        default=180.0,
        help="Maximum longitude of sweep region [degrees] (default: 180).",
    )
    parser.add_argument(
        "--min-lat-deg",
        type=float,
        default=-85.0,
        help="Minimum latitude of sweep region [degrees] (default: -85).",
    )
    parser.add_argument(
        "--max-lat-deg",
        type=float,
        default=85.0,
        help="Maximum latitude of sweep region [degrees] (default: 85).",
    )
    # Network / caching.
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("cache") / "wms",
        help="Directory for WMS HTTP response cache.",
    )
    parser.add_argument(
        "--no-http-cache",
        action="store_true",
        help="Disable the on-disk WMS response cache.",
    )
    parser.add_argument(
        "--request-timeout-s",
        type=int,
        default=60,
        help="HTTP request timeout in seconds.",
    )
    # Parallelism / rate.
    parser.add_argument(
        "--max-workers",
        type=int,
        default=6,
        help=(
            "Thread pool size for concurrent metadata queries "
            "(set 1 for sequential mode)."
        ),
    )
    parser.add_argument(
        "--max-requests-per-second",
        type=float,
        default=2.0,
        help=(
            "Global request-rate cap across all worker threads "
            "(set <=0 to disable)."
        ),
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    t_start = time.time()

    print("[manifest] MOONPIERCER full-sphere manifest builder starting.", flush=True)

    # Build ChordConfig from CLI args, overriding only the sweep-relevant fields.
    config = ChordConfig(
        sweep_grid_step_deg=float(args.sweep_grid_step_deg),
        max_grid_queries=int(args.max_grid_queries),
        feature_count=int(args.feature_count),
        max_nac_resolution_mpp=float(args.max_nac_resolution_mpp),
        cache_dir=Path(args.cache_dir),
        use_http_cache=not bool(args.no_http_cache),
        request_timeout_s=int(args.request_timeout_s),
    )

    # --- Sweep ---
    products = collect_nac_products_full_sphere(
        config=config,
        lon_min_deg=float(args.lon_min_deg),
        lon_max_deg=float(args.lon_max_deg),
        min_lat_deg=float(args.min_lat_deg),
        max_lat_deg=float(args.max_lat_deg),
        max_workers=int(args.max_workers),
        max_requests_per_second=float(args.max_requests_per_second),
    )

    # --- Stratified selection ---
    selected = select_products_stratified(products, max_chips=int(args.max_chips))
    selected = selected.sort_values("resolution_mpp", ascending=True).reset_index(drop=True)

    # Insert manifest_index as the first column (no hemisphere column — full sphere).
    selected.insert(0, "manifest_index", np.arange(len(selected), dtype=int))

    # Ensure output columns are in the documented order.
    _desired_cols = [
        "manifest_index",
        "product_id",
        "resolution_mpp",
        "center_lon",
        "center_lat",
        "incidence_angle_deg",
        "emission_angle_deg",
        "phase_angle_deg",
    ]
    # Keep only columns that actually exist (guard against empty DataFrame).
    output_cols = [c for c in _desired_cols if c in selected.columns]
    selected = selected[output_cols]

    # --- Write manifest CSV ---
    manifest_path = Path(args.manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(manifest_path, index=False)
    print(f"[manifest] wrote manifest → {manifest_path}", flush=True)

    # --- Write summary JSON ---
    summary_path = (
        Path(args.summary_path)
        if args.summary_path is not None
        else manifest_path.with_name("manifest_summary.json")
    )

    runtime_s = float(time.time() - t_start)
    summary = {
        "manifest_path": str(manifest_path),
        "n_products_discovered": int(len(products)),
        "n_products_selected": int(len(selected)),
        "max_chips": int(args.max_chips),
        "sweep_grid_step_deg": float(args.sweep_grid_step_deg),
        "max_grid_queries": int(args.max_grid_queries),
        "feature_count": int(args.feature_count),
        "max_nac_resolution_mpp": float(args.max_nac_resolution_mpp),
        "max_workers": int(args.max_workers),
        "max_requests_per_second": float(args.max_requests_per_second),
        "lon_range_deg": [float(args.lon_min_deg), float(args.lon_max_deg)],
        "lat_range_deg": [float(args.min_lat_deg), float(args.max_lat_deg)],
        "use_http_cache": not bool(args.no_http_cache),
        "request_timeout_s": int(args.request_timeout_s),
        "runtime_seconds": runtime_s,
    }

    save_json(summary, summary_path)
    print(f"[manifest] wrote summary  → {summary_path}", flush=True)

    # --- Final report ---
    print(f"[manifest] products discovered : {len(products):,}", flush=True)
    print(f"[manifest] chips selected      : {len(selected):,}", flush=True)
    print(f"[manifest] runtime             : {runtime_s / 60.0:.1f} min", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
