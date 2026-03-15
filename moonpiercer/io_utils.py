"""I/O helpers: CSV/JSON persistence and plot export.

All output paths default to the ``plots/`` and ``results/`` directories
at the MOONPIERCER project root.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

PLOTS_PDF_DIR: Path = _PROJECT_ROOT / "plots" / "pdf"
PLOTS_PNG_DIR: Path = _PROJECT_ROOT / "plots" / "png"
RESULTS_DIR: Path = _PROJECT_ROOT / "results"


def ensure_dir(path: Path) -> Path:
    """Create *path* (and parents) if it doesn't exist; return *path*."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ------------------------------------------------------------------
# CSV / JSON persistence
# ------------------------------------------------------------------

def save_dataframe(df: pd.DataFrame, path: Path | str) -> Path:
    """Write *df* to CSV, creating parent dirs as needed."""
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=False)
    return p


def load_dataframe(path: Path | str) -> pd.DataFrame:
    """Read a CSV into a DataFrame.

    Returns an empty DataFrame if the file is missing, empty, or
    contains only whitespace (e.g. a 0-pair results CSV).
    """
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return pd.DataFrame()
    text = p.read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def save_json(obj: dict, path: Path | str, indent: int = 2) -> Path:
    """Serialise *obj* to JSON, creating parent dirs as needed."""
    p = Path(path)
    ensure_dir(p.parent)

    class _Encoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, Path):
                return str(o)
            return super().default(o)

    p.write_text(json.dumps(obj, indent=indent, cls=_Encoder), encoding="utf-8")
    return p


def load_json(path: Path | str) -> dict:
    """Read a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------
# Plot export
# ------------------------------------------------------------------

def save_figure(
    fig,
    stem: str,
    *,
    pdf_dir: Path = PLOTS_PDF_DIR,
    png_dir: Path = PLOTS_PNG_DIR,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Save a matplotlib figure to both PDF and PNG.

    Returns ``(pdf_path, png_path)``.
    """
    ensure_dir(pdf_dir)
    ensure_dir(png_dir)
    pdf_path = pdf_dir / f"{stem}.pdf"
    png_path = png_dir / f"{stem}.png"
    fig.savefig(str(pdf_path), bbox_inches="tight")
    fig.savefig(str(png_path), dpi=dpi, bbox_inches="tight")
    return pdf_path, png_path
