"""Figure generation for the MOONPIERCER pipeline.

All figures are generated with matplotlib and can be saved via
``io_utils.save_figure`` to both PDF and PNG.

Methodology figures:
- Transit cone diagram (2D orthographic projection)
- Chord space diagram (allowed vs forbidden search space)
- Annotated detection chip

Results figures:
- Score component radar (star) plot
- Null-model best-score distribution
- Pair score histogram with null threshold
- Crater location heatmap with top pairs
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import matplotlib.figure
    import pandas as pd
    from moonpiercer.config import ChordConfig

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
# Methodology Figure B: Chord Search-Space Diagram
# ======================================================================

def plot_chord_space_diagram(
    config: "ChordConfig",
    entry_lon_deg: float = 30.0,
    entry_lat_deg: float = 15.0,
    orientation_deg: float = 45.0,
    ellipticity: float = 1.5,
    figsize: tuple[float, float] = (6, 6),
) -> "matplotlib.figure.Figure":
    """Methodology figure: allowed vs forbidden chord search space.

    Shows the Moon as a 2-D orthographic disc centred on the entry
    crater.  The **red** shaded region (forbidden zone) is excluded by
    the minimum angular separation cut (``config.min_chord_sep_deg``).
    The **blue** filled circles are the pair search cones around the two
    shape-predicted exit positions (forward and backward along the crater
    major axis, owing to the 180° orientation ambiguity).  Everything
    outside the forbidden zone is the *allowed* chord space.

    Parameters
    ----------
    config : ChordConfig
        Pipeline configuration (uses ``min_chord_sep_deg`` and
        ``search_cone_half_deg_reliable``).
    entry_lon_deg, entry_lat_deg : float
        Longitude and latitude of the example entry crater [degrees].
    orientation_deg : float
        Major-axis orientation of the entry crater [degrees from N].
    ellipticity : float
        Ellipticity of the entry crater (used to infer chord separation).
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Patch
    import matplotlib.lines as mlines

    # ---- Orthographic projection centred on the entry crater ----------
    v_entry = lonlat_to_unit_vectors(
        np.array([entry_lon_deg]), np.array([entry_lat_deg])
    ).ravel()

    z_hat = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(v_entry, z_hat)) > 0.999:
        east = np.array([1.0, 0.0, 0.0])
    else:
        east = np.cross(z_hat, v_entry)
        east /= np.linalg.norm(east)
    north = np.cross(v_entry, east)
    north /= np.linalg.norm(north)

    def _ortho(v: np.ndarray):
        """Return (x, y, visible) for a single unit vector."""
        x = float(np.dot(v, east))
        y = float(np.dot(v, north))
        vis = bool(np.dot(v, v_entry) >= -1e-4)
        return x, y, vis

    def _cap_xy(v_centre: np.ndarray, half_angle_deg: float, n: int = 360):
        """Orthographic (x, y) of a spherical cap boundary."""
        r_rad = np.deg2rad(half_angle_deg)
        bearings = np.linspace(0, 2 * np.pi, n, endpoint=True)
        pts = np.stack(
            [predict_exit_point(v_centre, float(b), r_rad) for b in bearings]
        )
        xs = pts @ east
        ys = pts @ north
        return xs, ys

    def _arc_xy(v1: np.ndarray, v2: np.ndarray, n: int = 120):
        """Orthographic (x, y, visible) arrays for a great-circle arc."""
        arc_lons, arc_lats = slerp_arc(v1, v2, n_points=n)
        vecs = lonlat_to_unit_vectors(
            np.asarray(arc_lons), np.asarray(arc_lats)
        )
        xs = vecs @ east
        ys = vecs @ north
        vis = (vecs @ v_entry) >= -1e-4
        return xs, ys, vis

    # ---- Geometry for this example crater ----------------------------
    sep_deg = float(separation_from_ellipticity(ellipticity))
    sep_rad = np.deg2rad(sep_deg)
    brg_fwd = np.deg2rad(90.0 - orientation_deg)
    brg_bwd = brg_fwd + np.pi

    v_exit_fwd = predict_exit_point(v_entry, brg_fwd, sep_rad)
    v_exit_bwd = predict_exit_point(v_entry, brg_bwd, sep_rad)

    ex_fwd, ey_fwd, vis_fwd = _ortho(v_exit_fwd)
    ex_bwd, ey_bwd, vis_bwd = _ortho(v_exit_bwd)

    cone_half = config.search_cone_half_deg_reliable

    # ---- Statistics --------------------------------------------------
    frac_forbidden = (1.0 - np.cos(np.deg2rad(config.min_chord_sep_deg))) / 2.0
    frac_allowed = 1.0 - frac_forbidden
    frac_cone = (1.0 - np.cos(np.deg2rad(cone_half))) / 2.0

    # ---- Publication rcParams ----------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.axis("off")

    # ---- Moon disc (back hemisphere first — slightly darker) ----------
    moon_back = Circle(
        (0, 0), 1.0, facecolor="0.82", edgecolor="none", zorder=1,
    )
    ax.add_patch(moon_back)

    # Allowed zone (front hemisphere, light cream)
    moon_front = Circle(
        (0, 0), 1.0, facecolor="#f5f0e8", edgecolor="none", zorder=2,
    )
    ax.add_patch(moon_front)

    # ---- Forbidden zone (min separation cap) -------------------------
    bx, by = _cap_xy(v_entry, config.min_chord_sep_deg)
    ax.fill(bx, by, color="#f2b4b4", alpha=0.90, zorder=3)
    ax.plot(bx, by, "-", color="#b03030", lw=0.9, zorder=4)

    # ---- Great-circle chord paths ------------------------------------
    for v_exit, ls, alpha in [
        (v_exit_fwd, "-",  0.65),
        (v_exit_bwd, "--", 0.45),
    ]:
        ax_xs, ax_ys, ax_vis = _arc_xy(v_entry, v_exit)
        # Mask invisible segments by inserting NaN
        ax_x_masked = np.where(ax_vis, ax_xs, np.nan)
        ax_y_masked = np.where(ax_vis, ax_ys, np.nan)
        ax.plot(
            ax_x_masked, ax_y_masked,
            ls, color="#3a6ab0", alpha=alpha, lw=1.0, zorder=5,
        )

    # ---- Moon disc edge (drawn after fill so it sits on top) ---------
    moon_edge = Circle(
        (0, 0), 1.0, fill=False, edgecolor="0.35", lw=1.2, zorder=8,
    )
    ax.add_patch(moon_edge)

    # ---- Search cones around predicted exit points -------------------
    cone_kwargs = [
        dict(v=v_exit_fwd, ex=ex_fwd, ey=ey_fwd, vis=vis_fwd,
             ls="-",  alpha_fill=0.55, label="forward"),
        dict(v=v_exit_bwd, ex=ex_bwd, ey=ey_bwd, vis=vis_bwd,
             ls="--", alpha_fill=0.35, label="backward"),
    ]
    for kw in cone_kwargs:
        cx, cy = _cap_xy(kw["v"], cone_half)
        ax.fill(cx, cy, color="#3a6ab0", alpha=kw["alpha_fill"], zorder=6)
        ax.plot(cx, cy, kw["ls"], color="#1a3a80", lw=0.7, zorder=7)
        if kw["vis"]:
            ax.plot(
                kw["ex"], kw["ey"],
                "x", color="#1a3a80", ms=7, mew=1.8, zorder=9,
            )
        else:
            # Behind the Moon — indicate with a dashed circle on the horizon
            ax.plot(
                kw["ex"], kw["ey"],
                "x", color="#1a3a80", ms=7, mew=1.8, zorder=9,
                alpha=0.35,
            )

    # ---- Entry crater marker -----------------------------------------
    ax.plot(
        0, 0,
        "o", color="#b03030", ms=9, zorder=10,
        markeredgecolor="white", markeredgewidth=1.2,
    )

    # ---- Zone labels -------------------------------------------------
    # Forbidden zone label (inside the cap)
    ax.text(
        0.0, -0.22,
        f"Forbidden\n({config.min_chord_sep_deg:.0f}° min sep)",
        ha="center", va="center", fontsize=7.5,
        color="#7a1010", zorder=11,
    )
    # Allowed zone label (top-right)
    ax.text(
        0.62, 0.72,
        f"Allowed\n({100 * frac_allowed:.1f}% of sphere)",
        ha="center", va="center", fontsize=7.5,
        color="#2a5a2a", zorder=11,
    )

    # ---- Annotate the search cone ------------------------------------
    if vis_fwd:
        ax.annotate(
            f"Search cone\n({cone_half:.0f}° half-angle,\n"
            f"{100 * frac_cone:.3f}% of sphere)",
            xy=(ex_fwd, ey_fwd),
            xytext=(ex_fwd + 0.35, ey_fwd + 0.25),
            fontsize=7.5, color="#1a3a80",
            arrowprops=dict(arrowstyle="->", color="#1a3a80", lw=0.8),
            zorder=12,
        )

    # ---- Title -------------------------------------------------------
    ax.set_title(
        rf"Chord search space  "
        rf"($e = {ellipticity:.2f}$,  "
        rf"$\theta_{{\rm sep}} = {sep_deg:.0f}^\circ$)",
        fontsize=10, pad=8,
    )

    # ---- Legend ------------------------------------------------------
    legend_elements = [
        Patch(facecolor="#f2b4b4", edgecolor="#b03030", lw=0.8,
              label=f"Forbidden (sep < {config.min_chord_sep_deg:.0f}°, "
                    f"{100 * frac_forbidden:.1f}%)"),
        Patch(facecolor="#f5f0e8", edgecolor="0.5", lw=0.6,
              label=f"Allowed ({100 * frac_allowed:.1f}% of sphere)"),
        Patch(facecolor="#3a6ab0", alpha=0.55,
              label=f"Search cone ×2 ({cone_half:.0f}° half-angle)"),
        mlines.Line2D(
            [], [], marker="o", color="w",
            markerfacecolor="#b03030", markeredgecolor="white",
            markeredgewidth=1.0, ms=7, label="Entry crater",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower left",
        fontsize=7,
        framealpha=0.92,
        edgecolor="0.6",
        handlelength=1.4,
    )

    fig.tight_layout()
    return fig


# ======================================================================
# Methodology Figure C: Annotated Detection Chip
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

    fi = detections["freshness_index"].to_numpy(dtype=float)
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

    lons = np.deg2rad(craters["lon_deg"].to_numpy(dtype=float))
    lats = np.deg2rad(craters["lat_deg"].to_numpy(dtype=float))
    fi = craters["freshness_index"].to_numpy(dtype=float) if "freshness_index" in craters.columns else np.ones(len(craters))

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
    scores = top["score"].to_numpy(dtype=float)
    s_min, s_max = scores.min(), scores.max()

    for _, row in top.iterrows():
        v1 = lonlat_to_unit_vectors(
            np.array([row["lon_a"]], dtype=float), np.array([row["lat_a"]], dtype=float)
        ).ravel()
        v2 = lonlat_to_unit_vectors(
            np.array([row["lon_b"]], dtype=float), np.array([row["lat_b"]], dtype=float)
        ).ravel()
        arc_lons, arc_lats = slerp_arc(v1, v2, n_points=80)

        norm_score = (float(row["score"]) - s_min) / (s_max - s_min + 1e-10)
        color = cmap(norm_score)

        ax.plot(
            np.deg2rad(np.asarray(arc_lons, dtype=float)),
            np.deg2rad(np.asarray(arc_lats, dtype=float)),
            "-", color=color, alpha=0.7, lw=1.5,
        )
        ax.plot(
            np.deg2rad(float(row["lon_a"])), np.deg2rad(float(row["lat_a"])),
            "o", color=color, ms=4,
        )
        ax.plot(
            np.deg2rad(float(row["lon_b"])), np.deg2rad(float(row["lat_b"])),
            "s", color=color, ms=4,
        )

    ax.grid(True, alpha=0.3)
    ax.set_title(f"Top {n_best} Chord Pairs")
    fig.tight_layout()
    return fig


# ======================================================================
# Score component radar (star) plot
# ======================================================================

def plot_score_component_star(
    pairs: "pd.DataFrame",
    n_top: int = 10,
    figsize: tuple[float, float] = (6, 6),
) -> "matplotlib.figure.Figure":
    """Radar plot of mean score-component contributions for the top pairs.

    Each axis represents one of the six scoring terms.  The solid line
    shows the mean across the top *n_top* pairs and the shaded region
    spans the mean +/- 1 standard deviation.
    """
    import matplotlib.pyplot as plt

    component_cols = [
        "T_diametrality", "T_radius", "T_freshness",
        "T_ellipticity", "T_orientation", "T_position",
    ]
    labels = [
        r"$T_{\rm diam}$", r"$T_{\rm radius}$", r"$T_{\rm fresh}$",
        r"$T_{\rm ellip}$", r"$T_{\rm orient}$", r"$T_{\rm pos}$",
    ]

    top = pairs.head(n_top)
    available = [c for c in component_cols if c in top.columns]
    avail_labels = [labels[component_cols.index(c)] for c in available]

    values = top[available].to_numpy(dtype=float)
    means = np.nanmean(values, axis=0)
    stds = np.nanstd(values, axis=0)

    n_axes = len(available)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    # Close the polygon
    angles += angles[:1]
    means_closed = np.append(means, means[0])
    lo = np.append(np.clip(means - stds, 0, 1), np.clip(means[0] - stds[0], 0, 1))
    hi = np.append(np.clip(means + stds, 0, 1), np.clip(means[0] + stds[0], 0, 1))

    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "polar"})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), avail_labels)

    ax.plot(angles, means_closed, "o-", color="steelblue", lw=1.8, ms=5, zorder=3)
    ax.fill_between(angles, lo, hi, alpha=0.20, color="steelblue", zorder=2)

    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Score components (top {min(n_top, len(top))} pairs)",
        y=1.10, fontsize=11,
    )
    fig.tight_layout()
    return fig


# ======================================================================
# Null-model best-score distribution
# ======================================================================

def plot_null_distribution(
    null_scores: np.ndarray,
    best_real_score: float | None = None,
    figsize: tuple[float, float] = (7, 4.5),
) -> "matplotlib.figure.Figure":
    """Histogram of null-model best scores with the best real score marked.

    Parameters
    ----------
    null_scores : array
        Best score from each Monte Carlo null trial.
    best_real_score : float, optional
        Best score from the real crater catalogue.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(
        null_scores, bins=50, color="0.65", edgecolor="0.45", linewidth=0.4,
        density=True, zorder=2, label=f"Null trials ($n={len(null_scores)}$)",
    )

    p95 = np.percentile(null_scores, 95)
    p99_87 = np.percentile(null_scores, 99.87)  # 3-sigma equivalent
    ax.axvline(p95, color="0.35", ls="--", lw=1.0,
               label=r"95th percentile", zorder=3)
    ax.axvline(p99_87, color="darkorange", ls="--", lw=1.0,
               label=r"$3\sigma$ threshold", zorder=3)

    if best_real_score is not None:
        ax.axvline(best_real_score, color="crimson", ls="-", lw=2.0,
                   label=f"Best real score ({best_real_score:.4f})", zorder=4)

    ax.set_xlabel("Best pair score per trial")
    ax.set_ylabel("Probability density")
    ax.set_title("Null-Model Best-Score Distribution")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# ======================================================================
# Pair score histogram with null threshold (Figure 2 style)
# ======================================================================

def plot_pair_scores_with_threshold(
    real_scores: np.ndarray,
    null_scores: np.ndarray,
    figsize: tuple[float, float] = (7, 4.5),
) -> "matplotlib.figure.Figure":
    """Distribution of all real pair scores with the 3-sigma null threshold.

    Mirrors the style of Figure 2 in Caplan et al. — the histogram of
    pair scores is shown with a vertical line at the 3-sigma (99.87th
    percentile) threshold derived from the null model.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    if len(real_scores) > 0:
        ax.hist(
            real_scores, bins=60, color="steelblue", edgecolor="none",
            alpha=0.8, density=True, zorder=2,
            label=f"Real pairs ($n={len(real_scores)}$)",
        )

    if len(null_scores) > 0:
        thresh_3sig = np.percentile(null_scores, 99.87)
        ax.axvline(
            thresh_3sig, color="darkorange", ls="-", lw=2.0, zorder=4,
            label=rf"Null $3\sigma$ threshold ({thresh_3sig:.4f})",
        )

    if len(real_scores) > 0:
        best = real_scores.max()
        ax.axvline(
            best, color="crimson", ls="--", lw=1.5, zorder=4,
            label=f"Best pair ({best:.4f})",
        )

    ax.set_xlabel("Pair score")
    ax.set_ylabel("Probability density")
    ax.set_title("Pair Score Distribution vs Null Threshold")
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig


# ======================================================================
# Crater heatmap with top-pair locations
# ======================================================================

def plot_crater_map_with_pairs(
    craters: "pd.DataFrame",
    pairs: "pd.DataFrame",
    n_best: int = 10,
    figsize: tuple[float, float] = (12, 6),
) -> "matplotlib.figure.Figure":
    """Mollweide map of all craters with top-pair locations highlighted.

    All craters are shown as small grey dots.  Entry and exit craters of
    the top *n_best* pairs are shown as larger markers colour-coded by
    rank (best = brightest).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=figsize, subplot_kw={"projection": "mollweide"},
    )

    # --- All craters (background) ---
    if not craters.empty and "lon_deg" in craters.columns:
        lons_all = np.deg2rad(craters["lon_deg"].to_numpy(dtype=float))
        lats_all = np.deg2rad(craters["lat_deg"].to_numpy(dtype=float))
        ax.scatter(
            lons_all, lats_all, s=0.3, c="0.78", alpha=0.25,
            rasterized=True, zorder=1,
        )

    # --- Top pairs ---
    if not pairs.empty:
        top = pairs.head(n_best)
        n_shown = len(top)
        cmap = plt.cm.plasma_r
        norm = plt.Normalize(0, max(n_shown - 1, 1))

        for rank, (_, row) in enumerate(top.iterrows()):
            color = cmap(norm(rank))
            size_entry = 70 - rank * 4  # best pair = largest
            size_exit = size_entry

            ax.scatter(
                np.deg2rad(float(row["lon_a"])), np.deg2rad(float(row["lat_a"])),
                s=size_entry, c=[color], marker="o", edgecolors="k",
                linewidths=0.5, zorder=3,
            )
            ax.scatter(
                np.deg2rad(float(row["lon_b"])), np.deg2rad(float(row["lat_b"])),
                s=size_exit, c=[color], marker="s", edgecolors="k",
                linewidths=0.5, zorder=3,
            )

            # Great-circle arc
            v1 = lonlat_to_unit_vectors(
                np.array([row["lon_a"]], dtype=float), np.array([row["lat_a"]], dtype=float)
            ).ravel()
            v2 = lonlat_to_unit_vectors(
                np.array([row["lon_b"]], dtype=float), np.array([row["lat_b"]], dtype=float)
            ).ravel()
            arc_lons, arc_lats = slerp_arc(v1, v2, n_points=80)
            ax.plot(
                np.deg2rad(np.asarray(arc_lons, dtype=float)),
                np.deg2rad(np.asarray(arc_lats, dtype=float)),
                "-", color=color, alpha=0.6, lw=1.2, zorder=2,
            )

        # Colourbar for rank
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.04, shrink=0.7)
        cbar.set_label("Pair rank", fontsize=9)
        cbar.set_ticks(np.arange(0, n_shown, max(1, n_shown // 5)))
        cbar.set_ticklabels(
            [f"#{i+1}" for i in range(0, n_shown, max(1, n_shown // 5))]
        )

    ax.grid(True, alpha=0.3)
    ax.set_title(
        f"Crater Detections & Top {n_best} Pairs",
        fontsize=11,
    )
    fig.tight_layout()
    return fig
