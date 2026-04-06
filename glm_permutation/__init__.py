"""
glm_permutation: Circular-shift permutation GLM for calcium imaging.

A robust framework for identifying which neurons encode specific task
variables in 2-photon calcium imaging experiments. Uses leave-one-out
F-statistics with circular-shift permutation to generate non-parametric
null distributions, avoiding assumptions about noise structure in calcium data.

Modules
-------
data_loader        - HDF5/MATLAB v7.3 data loading utilities
glm_core           - Core GLM engine (regressors, SVD, circular shift, p-values)
glm_timeresolved   - Time-resolved windowed GLM
glm_magnitude      - 2-window magnitude analysis (early vs late)
population_classify - Neuron functional classification and statistics
kernel_sweep       - Kernel parameter sensitivity analysis
spatial_map        - ROI spatial visualization
plot_style         - Shared plotting style and utilities
"""

from .glm_core import (
    make_event_kernel,
    make_lick_rate_regressor,
    make_gcamp_lick_regressor,
    build_full_dm,
    build_windowed_dm,
    circular_shift_null,
    compute_delta_r2,
    compute_pvalues_circular_shift,
)
from .data_loader import H5SessionReader, load_rois
from .population_classify import classify_neurons
from .plot_style import (
    set_composite_style,
    set_standalone_style,
    bar_with_points,
    smooth_trace,
    panel_label,
    draw_combined_heatmap,
    shade_windows,
)

__version__ = "0.1.0"
