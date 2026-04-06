"""
Shared plotting style and utilities for publication-quality figures.

Import this module in any analysis script to apply consistent formatting.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d


# ====================================================================
# MATPLOTLIB CONFIGURATION
# ====================================================================

def set_composite_style():
    """Style for multi-panel composite figures (Nature double-column)."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Nimbus Sans', 'DejaVu Sans'],
        'font.size': 7.5,
        'axes.titlesize': 8.5,
        'axes.labelsize': 8,
        'xtick.labelsize': 6.5,
        'ytick.labelsize': 6.5,
        'legend.fontsize': 6.5,
        'axes.linewidth': 0.7,
        'xtick.major.width': 0.5,
        'ytick.major.width': 0.5,
        'lines.linewidth': 1.4,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def set_standalone_style():
    """Style for single standalone figures."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Liberation Sans', 'Nimbus Sans', 'DejaVu Sans'],
        'font.size': 14,
        'axes.titlesize': 15,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'axes.linewidth': 1.2,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'lines.linewidth': 2,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


# ====================================================================
# COLOR PALETTES
# ====================================================================

# Example experimental group colors (adapt to your groups)
GROUP_COLORS_EXAMPLE = {
    'CTA': '#d62728',
    'CONT': '#1f77b4',
    'ROSA CRISPR': '#2ca02c',
    'ChAT CRISPR': '#9467bd',
}

# Population class colors
POP_COLORS = {
    'lick_only': '#2196F3',
    'taste_only': '#FF5722',
    'both': '#9C27B0',
    'neither': '#BDBDBD',
}

# RGBA for spatial map overlays (on black background)
CLASS_FACE = {
    0: (0.85, 0.85, 0.85, 0.25),   # neither
    1: (0.129, 0.588, 0.953, 0.85), # lick only
    2: (1.0, 0.341, 0.133, 0.85),   # taste only
    3: (0.612, 0.153, 0.690, 0.90), # both
}

CLASS_EDGE = {
    0: (0.7, 0.7, 0.7, 0.15),
    1: (0.05, 0.35, 0.7, 0.9),
    2: (0.7, 0.15, 0.05, 0.9),
    3: (0.4, 0.05, 0.5, 0.95),
}


# ====================================================================
# PLOTTING UTILITIES
# ====================================================================

def panel_label(ax, label, x=-0.12, y=1.08):
    """Add a bold panel label (a, b, c, ...) to an axes."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top', ha='left')


def smooth_trace(trace, sigma=2.5):
    """Light Gaussian smoothing for display traces."""
    return gaussian_filter1d(trace, sigma=sigma)


def bar_with_points(ax, x_pos, values_per_group, bar_width, color, label,
                    alpha=0.8, hatch=None, point_jitter=0.06):
    """
    Plot a bar (mean +/- SEM) with individual mouse datapoints.

    Parameters
    ----------
    ax : matplotlib Axes
    x_pos : array-like
        X positions for each bar.
    values_per_group : list of arrays
        One array of mouse-level values per x position.
    bar_width : float
    color : str
    label : str
    alpha : float
    hatch : str or None
    point_jitter : float
        Horizontal jitter for individual points.

    Returns
    -------
    bars : BarContainer
    """
    means = [np.mean(v) for v in values_per_group]
    sems = [np.std(v) / np.sqrt(len(v)) if len(v) > 1 else 0
            for v in values_per_group]

    bars = ax.bar(x_pos, means, bar_width, yerr=sems,
                  color=color, alpha=alpha, edgecolor='k', linewidth=0.8,
                  capsize=4, error_kw={'linewidth': 1.5, 'capthick': 1.2},
                  hatch=hatch, label=label)

    rng = np.random.default_rng(42)
    for i, (xp, vals) in enumerate(zip(x_pos, values_per_group)):
        jitter = rng.uniform(-point_jitter, point_jitter, size=len(vals))
        ax.scatter(xp + jitter, vals, s=28, color='k', alpha=0.55,
                   edgecolors='white', linewidths=0.5, zorder=5)
    return bars


def shade_windows(ax, early=(0, 2), late=(3, 5)):
    """Add light background shading for early (lick) and late (taste) windows."""
    ax.axvspan(early[0], early[1], color='#BBDEFB', alpha=0.12)
    ax.axvspan(late[0], late[1], color='#FFF3E0', alpha=0.2)


def draw_combined_heatmap(ax_trace, ax_heat, taste_data, water_data,
                          time_axis, vmax, neuron_color='#333',
                          show_xlabel=True, global_ylim=None):
    """
    Combined heatmap: taste trials stacked above water trials with divider.
    Mean traces for both trial types overlaid above.

    Parameters
    ----------
    ax_trace : Axes for the mean trace
    ax_heat : Axes for the heatmap
    taste_data : (n_taste_trials, T) array
    water_data : (n_water_trials, T) array
    time_axis : (T,) array
    vmax : float, shared colorscale max
    neuron_color : str, base color for the neuron type
    show_xlabel : bool
    global_ylim : tuple or None, shared y-limits for trace
    """
    n_taste = taste_data.shape[0]
    n_water = water_data.shape[0]
    combined = np.vstack([taste_data, water_data])
    n_total = combined.shape[0]

    # Heatmap
    ax_heat.imshow(combined, aspect='auto', cmap='inferno',
                   extent=[time_axis[0], time_axis[-1], n_total, 0],
                   vmin=0, vmax=vmax, interpolation='none')
    ax_heat.axvline(0, color='white', linestyle=':', linewidth=0.3, alpha=0.4)
    ax_heat.axhline(n_taste, color='white', linestyle='-',
                    linewidth=0.8, alpha=0.8)
    ax_heat.set_ylabel('Trial', fontsize=6.5)
    if show_xlabel:
        ax_heat.set_xlabel('Time (s)', fontsize=6.5)
    else:
        ax_heat.set_xticklabels([])

    # Trial type labels on right
    ax_heat.text(1.02, (n_taste / 2) / n_total, 'Taste',
                 transform=ax_heat.transAxes, fontsize=5.5, color='#d62728',
                 va='center', ha='left', rotation=-90, fontweight='bold')
    ax_heat.text(1.02, (n_taste + n_water / 2) / n_total, 'Water',
                 transform=ax_heat.transAxes, fontsize=5.5, color='#1f77b4',
                 va='center', ha='left', rotation=-90, fontweight='bold')

    # Overlaid mean traces
    taste_color, water_color = '#d62728', '#1f77b4'
    for data, color_t, label in [(taste_data, taste_color, 'Taste'),
                                  (water_data, water_color, 'Water')]:
        n_t = data.shape[0]
        mean_t = smooth_trace(np.mean(data, axis=0), sigma=2.5)
        sem_t = smooth_trace(np.std(data, axis=0) / np.sqrt(n_t), sigma=2.5)
        ax_trace.plot(time_axis, mean_t, color=color_t,
                      linewidth=0.8, label=label)
        ax_trace.fill_between(time_axis, mean_t - sem_t, mean_t + sem_t,
                              color=color_t, alpha=0.15)

    ax_trace.axvline(0, color='gray', linestyle=':', linewidth=0.3, alpha=0.3)
    ax_trace.set_xlim(time_axis[0], time_axis[-1])
    ax_trace.legend(fontsize=5, frameon=False, loc='upper left',
                    handlelength=1.0, labelspacing=0.2, handletextpad=0.3)
    ax_trace.set_axis_off()

    if global_ylim is not None:
        ax_trace.set_ylim(global_ylim)
