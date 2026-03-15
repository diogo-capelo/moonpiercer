"""MOONPIERCER — Pipeline progress monitor.

Reads per-chip and per-chunk status files to produce a clear progress
dashboard.  Can be run at any time while the pipeline is executing.

Usage::

    python hpc/progress.py results/moonpiercer_full_run
    python hpc/progress.py results/moonpiercer_full_run --watch 30   # refresh every 30s
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Status readers
# ---------------------------------------------------------------------------

def _read_json_safe(path: Path) -> dict:
    """Read a JSON file, returning {} on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _scan_chips(chip_dir: Path, total_chips: int) -> dict:
    """Scan chip results and classify each chip's status."""
    ok = 0
    failed = 0
    empty = 0
    missing = 0
    failures: list[dict] = []

    for i in range(total_chips):
        d = chip_dir / f"chip_{i:04d}"
        meta_path = d / "metadata.json"
        craters_path = d / "craters.csv"

        if not d.exists():
            missing += 1
            continue

        if not craters_path.exists():
            missing += 1
            continue

        meta = _read_json_safe(meta_path)
        status = meta.get("status", "unknown")

        if status == "ok":
            n = meta.get("n_craters", 0)
            if n == 0:
                empty += 1
            else:
                ok += 1
        elif status in ("no_data", "chip_download_failed", "blank_chip",
                        "noise_floor", "invalid_bbox"):
            empty += 1  # salvageable (empty output, not a failure)
        elif status == "unknown":
            # craters.csv exists but no metadata or unrecognised status
            ok += 1
        else:
            failed += 1
            failures.append({
                "chip_index": i,
                "status": status,
                "detail": meta.get("error", ""),
            })

    return {
        "total": total_chips,
        "ok": ok,
        "empty": empty,
        "failed": failed,
        "missing": missing,
        "completed": ok + empty + failed,
        "failures": failures,
    }


def _scan_null_chunks(checkpoint_dir: Path, chunk_count: int) -> dict:
    """Scan null-model chunk results."""
    ok = 0
    failed = 0
    missing = 0
    failures: list[dict] = []
    total_trials_done = 0
    total_runtime_s = 0.0

    for i in range(chunk_count):
        npy_path = checkpoint_dir / f"null_scores_part_{i:05d}.npy"
        meta_path = checkpoint_dir / f"null_scores_part_{i:05d}.json"

        if not npy_path.exists():
            missing += 1
            continue

        ok += 1
        meta = _read_json_safe(meta_path)
        total_trials_done += meta.get("trial_count", 0)
        total_runtime_s += meta.get("runtime_s", 0.0)

    return {
        "total": chunk_count,
        "ok": ok,
        "failed": failed,
        "missing": missing,
        "completed": ok,
        "total_trials_done": total_trials_done,
        "total_runtime_s": total_runtime_s,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# Dashboard printer
# ---------------------------------------------------------------------------

def _bar(done: int, total: int, width: int = 40) -> str:
    """ASCII progress bar."""
    if total <= 0:
        return "[" + "?" * width + "]"
    frac = min(done / total, 1.0)
    filled = int(frac * width)
    return "[" + "#" * filled + "-" * (width - filled) + f"] {done}/{total} ({frac:.1%})"


def _status_icon(done: int, total: int, failed: int) -> str:
    if done >= total and failed == 0:
        return "DONE"
    if done >= total and failed > 0:
        return "DONE (with failures)"
    if done > 0:
        return "IN PROGRESS"
    return "PENDING"


def print_dashboard(results_dir: Path) -> None:
    """Print a full progress dashboard for a pipeline run."""
    results_dir = results_dir.resolve()
    manifest_path = results_dir / "manifest.csv"
    chip_dir = results_dir / "chips"
    global_dir = results_dir / "global"
    checkpoint_dir = global_dir / "checkpoints"
    summary_path = global_dir / "global_summary.json"

    print()
    print("=" * 72)
    print("  MOONPIERCER Pipeline Progress")
    print("=" * 72)
    print(f"  Results directory: {results_dir}")
    print(f"  Timestamp:        {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # --- Stage 1: Manifest ---
    print("-" * 72)
    print("  Stage 1: Manifest")
    print("-" * 72)
    if manifest_path.exists():
        try:
            # Count lines (minus header)
            with open(manifest_path) as f:
                n_lines = sum(1 for _ in f) - 1
            print(f"  Status: DONE ({n_lines} chips in manifest)")
        except Exception as e:
            print(f"  Status: ERROR reading manifest ({e})")
    else:
        print("  Status: PENDING (manifest.csv not found)")
    print()

    # --- Stage 2: Chip Processing ---
    total_chips = 0
    if manifest_path.exists():
        try:
            with open(manifest_path) as f:
                total_chips = sum(1 for _ in f) - 1
        except Exception:
            pass

    print("-" * 72)
    print("  Stage 2: Chip Processing")
    print("-" * 72)
    if total_chips > 0 and chip_dir.exists():
        chip_info = _scan_chips(chip_dir, total_chips)
        print(f"  {_bar(chip_info['completed'], chip_info['total'])}")
        print(f"  Status:    {_status_icon(chip_info['completed'], chip_info['total'], chip_info['failed'])}")
        print(f"  OK (with craters): {chip_info['ok']:>6,d}")
        print(f"  Empty/no-data:     {chip_info['empty']:>6,d}")
        print(f"  Failed:            {chip_info['failed']:>6,d}")
        print(f"  Missing:           {chip_info['missing']:>6,d}")

        if chip_info["failures"]:
            print()
            print("  Failure details (first 20):")
            for f in chip_info["failures"][:20]:
                detail = f": {f['detail']}" if f.get("detail") else ""
                print(f"    chip_{f['chip_index']:04d} — {f['status']}{detail}")
            if len(chip_info["failures"]) > 20:
                print(f"    ... and {len(chip_info['failures']) - 20} more")
    elif total_chips == 0:
        print("  Status: PENDING (no manifest yet)")
    else:
        print("  Status: PENDING (no chip results directory)")
    print()

    # --- Stage 3a: Prep ---
    print("-" * 72)
    print("  Stage 3a: Global Aggregation — Prep")
    print("-" * 72)
    meta_path = checkpoint_dir / "checkpoint_meta.json"
    craters_pkl = checkpoint_dir / "craters.pkl"
    top_pairs_csv = checkpoint_dir / "top_pairs.csv"

    prep_done = all(p.exists() for p in (meta_path, craters_pkl, top_pairs_csv))
    if prep_done:
        meta = _read_json_safe(meta_path)
        print(f"  Status: DONE")
        print(f"  Craters loaded:    {meta.get('n_craters', '?'):>10,}")
        print(f"  Raw pairs found:   {meta.get('n_raw_pairs', '?'):>10,}")
        print(f"  Top pairs:         {meta.get('n_top_pairs', '?'):>10,}")
        print(f"  Null chunk count:  {meta.get('null_chunk_count', '?')}")
        print(f"  Pairing time:      {meta.get('pairing_time_s', '?')} s")
    else:
        missing_files = [
            p.name for p in (meta_path, craters_pkl, top_pairs_csv)
            if not p.exists()
        ]
        print(f"  Status: {'IN PROGRESS' if any(p.exists() for p in (meta_path, craters_pkl, top_pairs_csv)) else 'PENDING'}")
        print(f"  Missing: {', '.join(missing_files)}")
    print()

    # --- Stage 3b: Null Model ---
    print("-" * 72)
    print("  Stage 3b: Global Aggregation — Null Model")
    print("-" * 72)
    chunk_count = 0
    random_trials = 0
    if prep_done:
        meta = _read_json_safe(meta_path)
        chunk_count = int(meta.get("null_chunk_count", 0))
        random_trials = int(meta.get("random_trials", 0))

    if chunk_count > 0:
        null_info = _scan_null_chunks(checkpoint_dir, chunk_count)
        print(f"  {_bar(null_info['completed'], null_info['total'])}")
        print(f"  Status:        {_status_icon(null_info['completed'], null_info['total'], null_info['failed'])}")
        print(f"  Chunks done:   {null_info['ok']:>6,d} / {null_info['total']}")
        print(f"  Trials done:   {null_info['total_trials_done']:>6,d} / {random_trials}")
        print(f"  Missing:       {null_info['missing']:>6,d}")
        if null_info["total_runtime_s"] > 0:
            print(f"  Wall time sum: {null_info['total_runtime_s']:.1f} s ({null_info['total_runtime_s']/3600:.1f} h)")
    elif prep_done:
        print("  Status: PENDING (prep done, waiting for null model)")
    else:
        print("  Status: PENDING (waiting for prep)")
    print()

    # --- Stage 3c: Final ---
    print("-" * 72)
    print("  Stage 3c: Global Aggregation — Final")
    print("-" * 72)
    if summary_path.exists():
        summary = _read_json_safe(summary_path)
        print(f"  Status: DONE")
        print(f"  Significant pairs: {summary.get('n_significant', '?')}")
        print(f"  Best score:        {summary.get('best_score', '?')}")
        print(f"  Best p-value:      {summary.get('best_p_value', '?')}")
        print(f"  Total runtime:     {summary.get('total_runtime_s', '?')} s")
    else:
        # Check if outputs exist but summary doesn't
        outputs_exist = (global_dir / "all_pairs_scored.csv").exists()
        if outputs_exist:
            print("  Status: IN PROGRESS (outputs found, summary missing)")
        else:
            print("  Status: PENDING")
    print()

    # --- Overall ---
    print("=" * 72)
    all_done = (
        manifest_path.exists()
        and prep_done
        and chunk_count > 0
        and summary_path.exists()
    )
    if all_done:
        print("  PIPELINE COMPLETE")
    else:
        stages_done = sum([
            manifest_path.exists(),
            total_chips > 0 and chip_dir.exists() and _scan_chips(chip_dir, total_chips)["missing"] == 0,
            prep_done,
            chunk_count > 0 and _scan_null_chunks(checkpoint_dir, chunk_count)["missing"] == 0,
            summary_path.exists(),
        ]) if total_chips > 0 else 0
        print(f"  PIPELINE IN PROGRESS ({stages_done}/5 stages complete)")
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="MOONPIERCER pipeline progress monitor.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Path to the pipeline results directory (e.g., results/moonpiercer_full_run).",
    )
    parser.add_argument(
        "--watch",
        type=int,
        default=0,
        metavar="SEC",
        help="If >0, refresh every SEC seconds (Ctrl+C to stop).",
    )
    args = parser.parse_args()

    if not args.results_dir.exists():
        print(f"ERROR: results directory not found: {args.results_dir}", file=sys.stderr)
        return 1

    if args.watch > 0:
        try:
            while True:
                # Clear terminal (works on most systems)
                print("\033[2J\033[H", end="", flush=True)
                print_dashboard(args.results_dir)
                time.sleep(args.watch)
        except KeyboardInterrupt:
            print("\nStopped.")
            return 0
    else:
        print_dashboard(args.results_dir)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
