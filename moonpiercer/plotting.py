"""Figure generation for the MOONPIERCER pipeline.

All figures are generated with matplotlib and can be saved via
``io_utils.save_figure`` to both PDF and PNG.

Methodology figures:
- Transit cone diagram (2D orthographic projection)
- Annotated detection chip
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd

from moonpiercer.constants import LUNAR_RADIUS_M
from moonpiercer.geometry import (
    angular_separation_deg,
    expected_ellipticity_from_separation,
    lonlat_to_unit_vectors,
    predict_exit_point,
    separation_from_ellipticity,
    slerp_arc,
    unit_vectors_to_lonlat,
)


# ======================================================================
# Methodology Figure A: Transit Cone Diagram
# ======================================================================

def plot_transit_cone_diagram(
    entry_lon: float = 30.0,
    entry_lat: float = 15.0,
    ellipticity: float = 1.5,
    orientation_deg: float = 45.0,
    cone_half_deg: float = 2.0,
    figsize: tuple[float, float] = (10, 5),
) -> "matplotlib.figure.Figure":
    """Create a 2D orthographic/Mollweide transit cone diagram.

    Shows a detected crater on the sphere surface, the inferred chord
    direction, and the search region on the far side where the exit
    crater is expected.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse, FancyArrowPatch

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=figsize,
        subplot_kw={"projection": "mollweide"},
    )

    # --- Left panel: Near-diametral case (circular crater) ---
    _draw_transit_cone_panel(
        ax1,
        entry_lon=entry_lon,
        entry_lat=entry_lat,
        ellipticity=1.05,  # nearly circular
        orientation_deg=0.0,
        cone_half_deg=10.0,
        title="Near-diametral chord\n(circular crater, wide search)",
    )

    # --- Right panel: Oblique case (elongated crater) ---
    _draw_transit_cone_panel(
        ax2,
        entry_lon=entry_lon,
        entry_lat=entry_lat,
        ellipticity=ellipticity,
        orientation_deg=orientation_deg,
        cone_half_deg=cone_half_deg,
        title="Oblique chord\n(elongated crater, narrow search)",
    )

    fig.tight_layout()
    return fig


def _draw_transit_cone_panel(
    ax,
    entry_lon: float,
    entry_lat: float,
    ellipticity: float,
    orientation_deg: float,
    cone_half_deg: float,
    title: str = "",
) -> None:
    """Draw a single transit cone panel on a Mollweide axis."""
    import matplotlib.pyplot as plt

    # Entry point
    entry_lon_rad = np.deg2rad(entry_lon)
    entry_lat_rad = np.deg2rad(entry_lat)
    ax.plot(entry_lon_rad, entry_lat_rad, "o", color="red", ms=10, zorder=5)
    ax.annotate(
        "Entry",
        xy=(entry_lon_rad, entry_lat_rad),
        xytext=(entry_lon_rad + 0.15, entry_lat_rad + 0.15),
        fontsize=8, color="red",
    )

    # Compute predicted exit point
    v_entry = lonlat_to_unit_vectors(
        np.array([entry_lon]), np.array([entry_lat])
    ).ravel()
    sep_deg = separation_from_ellipticity(ellipticity)
    sep_rad = np.deg2rad(sep_deg)
    bearing_rad = np.deg2rad(90.0 - orientation_deg)

    v_exit = predict_exit_point(v_entry, bearing_rad, sep_rad)
    exit_lon, exit_lat = unit_vectors_to_lonlat(v_exit)
    exit_lon_rad = np.deg2rad(float(exit_lon))
    exit_lat_rad = np.deg2rad(float(exit_lat))

    # Plot predicted exit point
    ax.plot(exit_lon_rad, exit_lat_rad, "s", color="blue", ms=8, zorder=5)
    ax.annotate(
        "Predicted exit",
        xy=(exit_lon_rad, exit_lat_rad),
        xytext=(exit_lon_rad + 0.15, exit_lat_rad - 0.2),
        fontsize=8, color="blue",
    )

    # Draw great-circle arc (the chord path on the surface)
    arc_lons, arc_lats = slerp_arc(v_entry, v_exit, n_points=100)
    arc_lons_rad = np.deg2rad(np.asarray(arc_lons))
    arc_lats_rad = np.deg2rad(np.asarray(arc_lats))
    ax.plot(arc_lons_rad, arc_lats_rad, "-", color="gray", alpha=0.6, lw=1.5)

    # Draw search cone (circle around predicted exit)
    cone_rad = np.deg2rad(cone_half_deg)
    n_cone = 60
    cone_bearings = np.linspace(0, 2 * np.pi, n_cone)
    cone_lons = np.zeros(n_cone)
    cone_lats = np.zeros(n_cone)
    for k, brg in enumerate(cone_bearings):
        v_cone = predict_exit_point(v_exit, brg, cone_rad)
        cl, cla = unit_vectors_to_lonlat(v_cone)
        cone_lons[k] = float(cl)
        cone_lats[k] = float(cla)

    cone_lons_rad = np.deg2rad(cone_lons)
    cone_lats_rad = np.deg2rad(cone_lats)
    ax.fill(cone_lons_rad, cone_lats_rad, alpha=0.2, color="blue")
    ax.plot(cone_lons_rad, cone_lats_rad, "-", color="blue", alpha=0.5, lw=1)

    # Add grid and formatting
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=10, pad=10)

    # Add annotation with chord properties
    chord_len_km = float(
        2.0 * LUNAR_RADIUS_M * np.sin(sep_rad / 2.0) / 1e3
    )
    incidence = float(np.degrees(np.arccos(np.clip(1.0 / ellipticity, 0, 1))))
    ax.text(
        0.02, 0.02,
        f"e = {ellipticity:.2f}\n"
        f"$\\theta$ = {incidence:.0f}$^\\circ$\n"
        f"L = {chord_len_km:.0f} km\n"
        f"cone = {cone_half_deg:.0f}$^\\circ$",
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )


# ======================================================================
# Methodology Figure B: Annotated Detection Chip
# ======================================================================

def plot_annotated_chip(
    gray: np.ndarray,
    detections: "pd.DataFrame",
    mpp_mean: float = 1.0,
    highlight_idx: int | None = None,
    figsize: tuple[float, float] = (8, 8),
    title: str = "",
) -> "matplotlib.figure.Figure":
    """Plot a NAC chip with all detected craters annotated.

    Parameters
    ----------
    gray : ndarray
        Grayscale chip image.
    detections : DataFrame
        Must contain: x, y, radius_px, freshness_index, shape_reliable,
        ellipticity, orientation_deg.
    mpp_mean : float
        Metres per pixel for scale bar.
    highlight_idx : int, optional
        Index in *detections* to highlight as the paired crater.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from matplotlib import cm

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(gray, cmap="gray", origin="upper")

    if detections.empty:
        ax.set_title(title or "No craters detected")
        return fig

    fi = detections["freshness_index"].to_numpy()
    fi_norm = np.clip(fi, 0, 1)
    cmap = cm.viridis

    for i, row in detections.iterrows():
        x, y = float(row["x"]), float(row["y"])
        r = float(row["radius_px"])
        color = cmap(fi_norm[i])
        lw = 1.0

        if highlight_idx is not None and i == highlight_idx:
            color = "yellow"
            lw = 3.0

        # Draw ellipse if shape is reliable and crater is elongated
        if row.get("shape_reliable", False) and row.get("ellipticity", 1.0) > 1.05:
            e = float(row["ellipticity"])
            angle = float(row.get("orientation_deg", 0.0))
            semi_major = r * np.sqrt(e)
            semi_minor = r / np.sqrt(e)
            ellipse = Ellipse(
                (x, y), 2 * semi_major, 2 * semi_minor,
                angle=angle, fill=False, edgecolor=color, lw=lw,
            )
            ax.add_patch(ellipse)
        else:
            circle = plt.Circle(
                (x, y), r, fill=False, edgecolor=color, lw=lw,
            )
            ax.add_patch(circle)

    # Scale bar (bottom-right)
    h, w = gray.shape[:2]
    bar_len_px = 100.0 / mpp_mean  # 100 m bar
    if bar_len_px > w * 0.4:
        bar_len_px = 50.0 / mpp_mean
        bar_label = "50 m"
    else:
        bar_label = "100 m"
    bar_x = w - bar_len_px - 20
    bar_y = h - 30
    ax.plot([bar_x, bar_x + bar_len_px], [bar_y, bar_y], "w-", lw=3)
    ax.text(
        bar_x + bar_len_px / 2, bar_y - 10, bar_label,
        color="white", ha="center", fontsize=9,
    )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Freshness Index", fontsize=9)

    ax.set_title(title or f"{len(detections)} craters detected", fontsize=11)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    return fig


# ======================================================================
# Results Figures
# ======================================================================

def plot_score_distribution(
    real_scores: np.ndarray,
    null_scores: np.ndarray,
    figsize: tuple[float, float] = (8, 5),
) -> "matplotlib.figure.Figure":
    """Histogram of real pair scores vs null model best scores."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    bins = np.linspace(0, 1, 50)

    ax.hist(
        null_scores, bins=bins, alpha=0.5, color="gray",
        label=f"Null model ({len(null_scores)} trials)", density=True,
    )
    ax.axvline(
        np.percentile(null_scores, 95), color="gray", ls="--", lw=1,
        label="Null 95th percentile",
    )

    if len(real_scores) > 0:
        ax.hist(
            real_scores, bins=bins, alpha=0.7, color="steelblue",
            label=f"Real pairs (n={len(real_scores)})", density=True,
        )
        ax.axvline(
            real_scores.max(), color="red", ls="-", lw=2,
            label=f"Best real: {real_scores.max():.4f}",
        )

    ax.set_xlabel("Pair Score")
    ax.set_ylabel("Density")
    ax.set_title("Pair Score Distribution: Real vs Null Model")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_spatial_coverage(
    craters: "pd.DataFrame",
    figsize: tuple[float, float] = (10, 5),
) -> "matplotlib.figure.Figure":
    """Mollweide map of crater detections coloured by freshness."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "mollweide"})

    if craters.empty:
        ax.set_title("No craters to plot")
        ax.grid(True, alpha=0.3)
        return fig

    lons = np.deg2rad(craters["lon_deg"].to_numpy())
    lats = np.deg2rad(craters["lat_deg"].to_numpy())
    fi = craters["freshness_index"].to_numpy() if "freshness_index" in craters.columns else np.ones(len(craters))

    sc = ax.scatter(lons, lats, c=fi, s=1, cmap="viridis", alpha=0.5, vmin=0, vmax=1)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.04)
    cbar.set_label("Freshness Index")

    ax.grid(True, alpha=0.3)
    ax.set_title(f"Crater Detections (n = {len(craters):,})")
    fig.tight_layout()
    return fig


def plot_chord_map(
    pairs: "pd.DataFrame",
    n_best: int = 20,
    figsize: tuple[float, float] = (10, 5),
) -> "matplotlib.figure.Figure":
    """Mollweide map showing the top-scoring chord pairs as great-circle arcs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "mollweide"})

    if pairs.empty:
        ax.set_title("No pairs to plot")
        ax.grid(True, alpha=0.3)
        return fig

    top = pairs.head(n_best)
    cmap = plt.cm.hot_r
    scores = top["score"].to_numpy()
    s_min, s_max = scores.min(), scores.max()

    for _, row in top.iterrows():
        v1 = lonlat_to_unit_vectors(
            np.array([row["lon_a"]]), np.array([row["lat_a"]])
        ).ravel()
        v2 = lonlat_to_unit_vectors(
            np.array([row["lon_b"]]), np.array([row["lat_b"]])
        ).ravel()
        arc_lons, arc_lats = slerp_arc(v1, v2, n_points=80)

        norm_score = (row["score"] - s_min) / (s_max - s_min + 1e-10)
        color = cmap(norm_score)

        ax.plot(
            np.deg2rad(np.asarray(arc_lons)),
            np.deg2rad(np.asarray(arc_lats)),
            "-", color=color, alpha=0.7, lw=1.5,
        )
        ax.plot(
            np.deg2rad(row["lon_a"]), np.deg2rad(row["lat_a"]),
            "o", color=color, ms=4,
        )
        ax.plot(
            np.deg2rad(row["lon_b"]), np.deg2rad(row["lat_b"]),
            "s", color=color, ms=4,
        )

    ax.grid(True, alpha=0.3)
    ax.set_title(f"Top {n_best} Chord Pairs")
    fig.tight_layout()
    return fig
