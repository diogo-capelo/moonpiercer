# MOONPIERCER

Full-chord Primordial Black Hole (PBH) lunar crater search pipeline.

MOONPIERCER searches for pairs of small, fresh craters on the Moon that lie on
opposite ends of a chord through the lunar interior — the geometric signature of
a PBH transit. It downloads high-resolution LROC NAC imagery via WMS, detects
metre-scale craters with a multiscale Laplacian-of-Gaussian (LoG) filter,
computes a Freshness Index for each detection, and matches crater pairs using a
shape-directed spherical geometry search. Significance is assessed against a
Monte Carlo null model with Benjamini-Hochberg FDR correction.

## How It Works

### 1 — Crater Detection

Each 1200 m × 1200 m image chip (1024 × 1024 px, NAC) is processed
independently:

- Scale-normalised LoG filtering across 6 octaves targets craters with
  apparent radii of 1–10 m.
- Non-maximum suppression removes duplicate detections.
- Intensity moments characterise each crater's ellipticity and orientation.
- **Freshness Index** `FI = 0.7 × NLS + 0.3 × RCR` quantifies crater age
  relative to the local terrain (higher = fresher).

### 2 — Chord Pairing

For each fresh crater (`FI ≥ 0.15`, depth proxy `≥ 0.22`):

- Its ellipticity and orientation encode the chord direction. A predicted
  exit point is computed on the unit sphere.
- A kd-tree query finds candidates within a search cone around the
  prediction (2° for shape-reliable craters, 10° otherwise).
- Candidate pairs are scored on six criteria: diametrality, radius match,
  freshness match, ellipticity match, orientation alignment, and velocity
  plausibility under the Standard Halo Model Maxwell-Boltzmann PBH
  velocity distribution.

### 3 — Statistical Significance

The null hypothesis is that any apparent pairing is due to chance alignment.
For each of N Monte Carlo trials:

- Crater positions are randomised uniformly on the sphere while preserving
  all physical attributes.
- Only craters satisfying the freshness and depth cuts are randomised
  (pre-filtering reduces the kd-tree from ~435 k to ~thousands of entries,
  making each trial take seconds rather than hours).
- The maximum pair score across the randomised catalog is recorded.

The real top-pair scores are compared against this null distribution to
compute empirical p-values. Benjamini-Hochberg FDR correction is applied at
α = 0.05 to control the false-discovery rate.

## Pipeline Stages

The full run is split into three dependent SLURM stages:

| Stage | Script | What it does |
|-------|--------|--------------|
| **prep** | `global_aggregation.py --mode prep` | Reads all chip catalogs, runs pairing, saves `craters.pkl` + `top_pairs.csv` |
| **null** | `global_aggregation.py --mode null` | One SLURM array job per null chunk; each writes `null_scores_part_XXXXX.npy` |
| **final** | `global_aggregation.py --mode final` | Assembles null chunks, computes p-values, writes final outputs |

All stages are checkpointed. A resume run detects which null chunks are missing
and resubmits only those.

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

## HPC Setup

The sbatch scripts use the cluster module system (`module load python`) for the
scientific stack. If any packages are missing, install them once from a login
node:

```bash
module load python
pip install --user numpy scipy pandas matplotlib Pillow requests PyYAML
pip install --user -e .
```

## HPC Submission

### Quick Test

20 chips, 100 null trials, equatorial band (±15°). Validates the full workflow
end-to-end in ~1–2 hours.

```bash
sbatch hpc/mp_test.sbatch
```

### Full Run

15,000 chips, 1.5° grid, null trials distributed across a 999-task SLURM array.

```bash
sbatch hpc/mp_run.sbatch
```

### Resume Run

Inspects an existing results directory and resubmits only what is missing
(chips, null chunks, final aggregation).

```bash
sbatch hpc/mp_resume.sbatch results/moonpiercer_full_run
```

### Environment Variable Overrides

All optional:

| Variable | Default | Description |
|---|---|---|
| `MOONPIERCER_MAX_CHIPS` | 15000 | Max chips in manifest |
| `MOONPIERCER_GRID_STEP` | 1.5 | Sweep grid step in degrees |
| `MOONPIERCER_MAX_WORKERS` | 8 | Manifest build thread count |
| `MOONPIERCER_RANDOM_TRIALS` | 2000 | Null model MC trials |
| `MOONPIERCER_ARRAY_SIZE` | 999 | Max SLURM array size (= null chunk count) |
| `MOONPIERCER_PROGRESS_INTERVAL_SEC` | 300 | Progress log interval |
| `MOONPIERCER_RESUME` | 0 | Use checkpoints in `mp_run`; `mp_resume` defaults to 1 |
| `MOONPIERCER_CACHE_DIR` | `./cache` | WMS disk cache directory |
| `MOONPIERCER_PYTHON_BIN` | *(auto)* | Skip detection; use this interpreter directly |

```bash
MOONPIERCER_MAX_CHIPS=500 MOONPIERCER_RANDOM_TRIALS=1000 sbatch hpc/mp_run.sbatch
```

### Monitoring

```bash
# Live progress (updates every 60 s)
python hpc/progress.py results/moonpiercer_full_run --watch 60

# SLURM queue
squeue -u $USER

# Stage logs
cat results/slurm_logs/stage_logs/global_prep_<JOBID>.out
cat results/slurm_logs/null_logs/null_<JOBID>_<TASKID>.out
cat results/slurm_logs/stage_logs/global_final_<JOBID>.out
```

### After a Run

Open `notebooks/analysis.ipynb` to generate figures and review results. The
notebook auto-detects whether HPC outputs are present and falls back to
methodology-only figures otherwise.

## Project Structure

```
moonpiercer/
├── hpc/
│   ├── setup_env.sh                 # Environment bootstrap (module + venv)
│   ├── mp_test.sbatch               # Quick test (equatorial, 20 chips)
│   ├── mp_run.sbatch                # Full-coverage run (15 000 chips)
│   ├── mp_resume.sbatch             # Resume from existing results
│   ├── manifest.py                  # Stage 1: build chip manifest
│   ├── chip_worker.py               # Stage 2: per-chip crater detection
│   ├── global_aggregation.py        # Stage 3: pairing, null model, significance
│   └── progress.py                  # Pipeline progress reporter
├── moonpiercer/                     # Core library
│   ├── config.py                    # ChordConfig dataclass
│   ├── constants.py                 # Physical constants
│   ├── detection.py                 # LoG crater detection + shape characterisation
│   ├── freshness.py                 # Freshness Index computation
│   ├── geometry.py                  # Spherical geometry and transforms
│   ├── io_utils.py                  # CSV/JSON/figure I/O
│   ├── null_model.py                # Monte Carlo null model + BH-FDR
│   ├── pairing.py                   # Chord-based pair matching and scoring
│   ├── velocity.py                  # PBH velocity model
│   ├── plotting.py                  # Visualisation utilities
│   └── wms.py                       # WMS client for LROC NAC and LOLA
├── tests/                           # Unit tests (pytest)
├── notebooks/
│   └── analysis.ipynb               # Post-run analysis and figure generation
├── plots/                           # Output figures (pdf/, png/)
├── results/                         # Pipeline outputs
│   └── slurm_logs/
│       ├── chip_logs/               # Chip array task logs
│       ├── null_logs/               # Null model array logs
│       └── stage_logs/              # Prep / final aggregation logs
└── pyproject.toml                   # Package metadata and dependencies
```
