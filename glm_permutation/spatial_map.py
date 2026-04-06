"""
Spatial ROI map: color-code neurons by functional class on the FOV.

Requires ROI boundary polygons (see data_loader.load_rois).
Renders neurons as filled polygons on a black background, with color
indicating functional class (lick-only, taste-only, both, neither).
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.collections import PatchCollection
from glm_permutation.plot_style import CLASS_FACE, CLASS_EDGE, panel_label


def draw_spatial_map(ax, polygons, classes, n_neurons,
                     highlight_cells=None, highlight_colors=None):
    """
    Draw spatial ROI map color-coded by functional class.

    Parameters
    ----------
    ax : matplotlib Axes
    polygons : list of (n_vertices, 2) arrays
        ROI boundary polygons from data_loader.load_rois().
    classes : (n_neurons,) int array
        0=neither, 1=lick_only, 2=taste_only, 3=both
    n_neurons : int
        Number of neurons to plot (may be < len(polygons)).
    highlight_cells : list of int or None
        Indices of cells to mark with a star.
    highlight_colors : list of str or None
        Colors for highlighted cells.
    """
    # Draw in order: neither first, then classified on top
    for cls in [0, 1, 2, 3]:
        patches = []
        for ci in range(min(n_neurons, len(polygons))):
            if classes[ci] == cls:
                poly = polygons[ci]
                if len(poly) >= 3:
                    patches.append(Polygon(poly, closed=True))
        if patches:
            pc = PatchCollection(
                patches,
                facecolor=CLASS_FACE[cls],
                edgecolor=CLASS_EDGE[cls],
                linewidth=0.25 if cls == 0 else 0.5
            )
            ax.add_collection(pc)

    # Highlight specific cells with stars
    if highlight_cells and highlight_colors:
        for ci, color in zip(highlight_cells, highlight_colors):
            if ci < len(polygons):
                poly = polygons[ci]
                cx, cy = np.mean(poly[:, 0]), np.mean(poly[:, 1])
                ax.plot(cx, cy, '*', color='white', markersize=7,
                        markeredgecolor=color, markeredgewidth=0.8, zorder=10)

    # Auto-scale axes
    all_pts = np.vstack(polygons[:n_neurons])
    pad = 8
    ax.set_xlim(all_pts[:, 0].min() - pad, all_pts[:, 0].max() + pad)
    ax.set_ylim(all_pts[:, 1].max() + pad, all_pts[:, 1].min() - pad)
    ax.set_aspect('equal')
    ax.set_facecolor('black')
    ax.set_xticks([])
    ax.set_yticks([])

    # Count annotation
    n_l = np.sum(classes == 1)
    n_t = np.sum(classes == 2)
    n_b = np.sum(classes == 3)
    ax.text(0.03, 0.02,
            f'Lick: {n_l}   Taste: {n_t}   Both: {n_b}\nn = {n_neurons}',
            transform=ax.transAxes, fontsize=6, color='white', va='bottom',
            bbox=dict(boxstyle='round,pad=0.25', facecolor='black', alpha=0.7))


def make_legend_elements():
    """Return legend patch elements for the spatial map."""
    return [
        Patch(facecolor=CLASS_FACE[1][:3], edgecolor=CLASS_EDGE[1][:3],
              linewidth=0.7, label='Lick only (early)'),
        Patch(facecolor=CLASS_FACE[2][:3], edgecolor=CLASS_EDGE[2][:3],
              linewidth=0.7, label='Taste only (late)'),
        Patch(facecolor=CLASS_FACE[3][:3], edgecolor=CLASS_EDGE[3][:3],
              linewidth=0.7, label='Both'),
        Patch(facecolor=(0.85, 0.85, 0.85), edgecolor=(0.7, 0.7, 0.7),
              linewidth=0.7, label='Neither'),
    ]
