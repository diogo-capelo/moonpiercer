# MOONPIERCER

Full-chord Primordial Black Hole (PBH) lunar crater search pipeline.

MOONPIERCER searches for pairs of small, fresh craters on the Moon that lie on
opposite ends of a chord through the lunar interior — the geometric signature of
a PBH transit. It downloads high-resolution LROC NAC imagery via WMS, detects
metre-scale craters with a multiscale Laplacian-of-Gaussian (LoG) filter,
computes a Freshness Index for each detection, and matches crater pairs using a
shape-directed spherical geometry search. Significance is assessed against a
Monte Carlo null model with Benjamini-Hochberg FDR correction.

## Pipeline Workflow

The pipeline runs in three stages, orchestrated as dependent SLURM jobs:

### Stage 1 — Build Manifest

**Script:** `hpc/manifest.py`

Sweeps the full lunar sphere on a configurable lon/lat grid (default 2° step,
latitudes ±85°), querying the Lunar Mapping and Modeling Portal WMS for NAC
observation metadata. Deduplicates by `product_id`, keeping the
highest-resolution observation per location. Performs stratified spatial
selection (15° bins) to ensure uniform coverage, then outputs a manifest CSV
listing up to `max_chips` chips sorted by resolution.

**Outputs:** `results/moonpiercer_run/manifest.csv`,
`results/moonpiercer_run/manifest_summary.json`

### Stage 2 — Chip Processing

**Script:** `hpc/chip_worker.py` (one invocation per chip, via SLURM array jobs)

For each manifest row:

1. Computes a bounding box around the chip centre (default 1200 m span).
2. Fetches NAC grayscale imagery (1024×1024 px) and LOLA elevation (256×256 px)
   from WMS.
3. Runs scale-normalised LoG crater detection (target radius 1–10 m) with
   non-maximum suppression and shape characterisation via intensity moments.
4. Computes the Freshness Index (FI = 0.7 × NLS + 0.3 × RCR).
5. Maps pixel coordinates to selenographic (lon, lat).
6. Saves per-chip `craters.csv`, `metadata.json`, and optionally an annotated
   PNG.

Each chip is fully independent — this stage is embarrassingly parallel.

**Outputs per chip:** `results/moonpiercer_run/chips/chip_XXXX/{craters.csv,
metadata.json}`

### Stage 3 — Global Aggregation

**Script:** `hpc/global_aggregation.py`

1. Discovers and concatenates all per-chip crater catalogues.
2. Runs chord-based pairing: for each sufficiently fresh crater (FI ≥ 0.15),
   infers chord direction from ellipticity and orientation, predicts the exit
   point on the sphere, and searches a kd-tree around the prediction.
3. Scores pairs on diametrality, radius match, freshness match, ellipticity
   match, orientation alignment, and velocity plausibility.
4. Selects the top non-overlapping pairs (default 50).
5. Runs a Monte Carlo null model (default 2000 trials): randomises crater
   positions while preserving attributes, re-runs pairing, and records the best
   score per trial.
6. Computes empirical p-values and applies Benjamini-Hochberg FDR correction
   (default α = 0.05).

**Outputs:** `results/moonpiercer_run/global/{all_pairs_scored.csv,
significant_pairs.csv, null_best_scores.npy, global_summary.json}`

### Post-Run Analysis

The Jupyter notebook `notebooks/analysis.ipynb` loads HPC results and generates
all figures (methodology diagrams, spatial coverage maps, score distributions,
chord maps, sensitivity analysis). Figures are saved to `plots/pdf/` and
`plots/png/`.

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e .
pip install -e ".[dev]"   # includes pytest
```

## Running Tests

```bash
pytest tests/
```

## HPC Environment Setup

The sbatch scripts rely on the cluster's module system (`module load python`)
to provide a Python interpreter with the scientific stack (numpy, scipy, pandas,
matplotlib, Pillow, requests, PyYAML). This is the same approach used by the
predecessor pipeline and avoids architecture mismatches on heterogeneous
clusters.

### Prerequisites

The cluster's `python` module must provide Python ≥ 3.10 with the required
packages. If any are missing, install them once from a login node:

```bash
module load python
pip install --user numpy scipy pandas matplotlib Pillow requests PyYAML
pip install --user -e .   # moonpiercer itself
```

### Environment variable overrides

All optional:

| Variable | Description |
|---|---|
| `MOONPIERCER_PYTHON_BIN` | Skip detection; use this Python interpreter directly |
| `MOONPIERCER_PROJECT_DIR` | Project root (for PYTHONPATH) |

## HPC Submission

Two sbatch scripts are provided for SLURM clusters:

### Quick Test

Runs a minimal pipeline (20 chips, 100 null-model trials) to validate that the
full three-stage workflow completes and produces outputs suitable for the
analysis notebook. Takes roughly 1–2 hours end-to-end.

```bash
sbatch hpc/test_moonpiercer.sbatch
```

### Full Production Run

Runs the full pipeline (2000 chips, 2000 null-model trials) with maximum
parallelisation. Chip processing is split into SLURM array batches of up to 999
tasks each. Typical wall time: 12–24 hours depending on cluster load and WMS
response times.

# Usage:

```bash
sbatch sbatch hpc/run_moonpiercer.sbatch --max-chips 2000 --random-trials 2000
```

Override parameters via environment variables:
   MOONPIERCER_MAX_CHIPS      — max chips in manifest (default: 2000)
   MOONPIERCER_GRID_STEP      — sweep grid step in degrees (default: 2.0)
   MOONPIERCER_MAX_WORKERS    — manifest build thread count (default: 6)
   MOONPIERCER_RANDOM_TRIALS  — null model MC trials (default: 2000)
   MOONPIERCER_ARRAY_SIZE     — max SLURM array size (default: 999)

Example with overrides:

```bash
MOONPIERCER_MAX_CHIPS=500 MOONPIERCER_RANDOM_TRIALS=1000 sbatch hpc/full_pipeline.sbatch
```

### Monitoring

Check job status:

```bash
squeue -u $USER
```

View logs:

```bash
# Manifest stage
cat results/slurm_logs/manifest_<JOBID>.out

# Individual chip
cat results/slurm_logs/chip_<JOBID>_<TASKID>.out

# Global aggregation
cat results/slurm_logs/global_<JOBID>.out
```

### After a Run

Open `notebooks/analysis.ipynb` to generate figures and review results. The
notebook auto-detects whether HPC outputs are present and falls back to
methodology-only figures if not.

## Project Structure

```
moonpiercer/
├── hpc/
│   ├── setup_env.sh             # Environment bootstrap (spack + venv)
│   ├── test_moonpiercer.sbatch  # Quick test (20 chips)
│   ├── run_moonpiercer.sbatch   # Full production run (2000 chips)
│   ├── manifest.py              # Stage 1: build chip manifest
│   ├── chip_worker.py           # Stage 2: per-chip crater detection
│   └── global_aggregation.py    # Stage 3: pairing, null model, significance
├── moonpiercer/                 # Core library
│   ├── config.py                # ChordConfig dataclass
│   ├── constants.py             # Physical constants
│   ├── detection.py             # LoG crater detection + shape characterisation
│   ├── freshness.py             # Freshness Index computation
│   ├── geometry.py              # Spherical geometry and transforms
│   ├── io_utils.py              # CSV/JSON/figure I/O
│   ├── null_model.py            # Monte Carlo null model + BH-FDR
│   ├── pairing.py               # Chord-based pair matching and scoring
│   ├── velocity.py              # PBH velocity model
│   ├── plotting.py              # Visualisation utilities
│   └── wms.py                   # WMS client for LROC NAC and LOLA
├── tests/                       # Unit tests (pytest)
├── notebooks/
│   └── analysis.ipynb           # Post-run analysis and figure generation
├── plots/                       # Output figures (pdf/, png/)
├── results/                     # Pipeline outputs
└── pyproject.toml               # Package metadata and dependencies
```
