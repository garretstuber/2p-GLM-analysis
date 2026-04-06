# 2p-GLM-analysis

Circular-shift permutation GLM for identifying neural encoding in 2-photon calcium imaging data.

This package implements a non-parametric approach for testing which neurons significantly encode task variables (e.g., sensory stimuli, motor actions, interactions). It uses a leave-one-out F-statistic with circular-shift permutation to generate null distributions, avoiding the parametric assumptions that are violated by calcium imaging signals (temporal autocorrelation, non-Gaussian noise, nonstationarity).

## Method

Standard parametric significance tests inflate Type I error rates on calcium data because they assume i.i.d. Gaussian residuals. Circular shifting solves this by preserving the temporal autocorrelation structure of the signal while destroying alignment with task events.

The pipeline:

1. Fit a full GLM with all task regressors (stimulus identity, lick rate, interactions, nuisance variables)
2. For each predictor, fit a reduced model leaving that predictor out
3. Compute the leave-one-out F-statistic and delta-R2 (unique variance explained)
4. Generate a null distribution by circularly shifting each neuron's activity within session boundaries
5. Derive non-parametric p-values by comparing observed F-stats to the empirical null

SVD decomposition of the design matrix makes the computation efficient: RSS = ||y||^2 - ||U'y||^2, avoiding explicit least-squares per neuron.

## Installation

```bash
git clone https://github.com/garretstuber/2p-GLM-analysis.git
cd 2p-GLM-analysis
pip install -e .
```

Or install dependencies directly:

```bash
pip install numpy scipy h5py matplotlib jupyter
```

## Quick Start

```python
import h5py
from glm_permutation import (
    H5SessionReader,
    build_full_dm,
    compute_pvalues_circular_shift,
    classify_neurons,
)

# Load data
f = h5py.File('your_data.mat', 'r')
session_ds = f['your_struct']['subject']['session']

# Create session readers
readers = [H5SessionReader(f, session_ds, mouse_idx=0, session_idx=si)
           for si in [5, 6, 7]]  # your test session indices

# Build design matrix and run GLM
X_list, Y, boundaries = build_full_dm(readers, sigma=0.15, shift_ms=300)
pvals, delta_r2, sig_fracs = compute_pvalues_circular_shift(
    X_list, Y, boundaries, alpha=0.01
)

print(f'Taste encoding: {sig_fracs[0]:.1f}% of neurons')
print(f'Lick encoding:  {sig_fracs[1]:.1f}% of neurons')
```

See `notebooks/glm_tutorial.ipynb` for the full walkthrough.

## Package Structure

```
2p-GLM-analysis/
  glm_permutation/
    __init__.py              # Public API
    data_loader.py           # HDF5 (MATLAB v7.3) data loading
    glm_core.py              # Core GLM engine
    glm_timeresolved.py      # Sliding-window temporal analysis
    glm_magnitude.py         # 2-window magnitude comparison
    population_classify.py   # Functional neuron classification
    kernel_sweep.py          # Kernel parameter sensitivity
    spatial_map.py           # ROI spatial visualization
    plot_style.py            # Publication figure utilities
  notebooks/
    glm_tutorial.ipynb       # Step-by-step tutorial notebook
  setup.py
  requirements.txt
  LICENSE
```

## Modules

### `glm_core` -- Core engine

The main computational module. Key functions:

- `build_full_dm()` -- Concatenate sessions and build the design matrix with taste kernel, lick rate (Gaussian-smoothed with time shifts), lick-taste interaction, spout onset, trial drift, and session indicators
- `build_windowed_dm()` -- Build a design matrix for a specific time window (used by time-resolved and magnitude analyses)
- `compute_pvalues_circular_shift()` -- Run the full permutation test: generate null, compute F-stats, return p-values and delta-R2
- `compute_delta_r2()` -- Compute unique variance explained per predictor per neuron
- `make_lick_rate_regressor()` -- Gaussian-smoothed lick rate with time-shift expansion
- `make_gcamp_lick_regressor()` -- Alternative GCaMP impulse-response kernel

### `glm_timeresolved` -- Temporal encoding profiles

Slides a window (default 1s, 0.25s step) across the trial to reveal when each variable is encoded. Shows, for example, that motor signals peak early (0-2s) while sensory signals peak later (3-5s).

### `glm_magnitude` -- Early vs late window comparison

Independently tests encoding in two time windows to directly compare motor vs sensory effect sizes. Outputs per-neuron significance and delta-R2 in each window.

### `population_classify` -- Functional classification

Classifies neurons into four populations based on the 2-window results:
- **Lick-only**: significant motor encoding in the early window, no sensory in late
- **Taste-only**: significant sensory in late, no motor in early
- **Both**: significant in both windows
- **Neither**: not significant in either

Includes chi-squared tests for comparing proportions across experimental groups.

### `kernel_sweep` -- Robustness validation

Sweeps kernel parameters (Gaussian sigma, time-shift range) and tests an alternative GCaMP-shaped kernel. If results are stable across parameters, the findings are not artifacts of kernel choice.

### `spatial_map` -- ROI visualization

Renders neuron ROI boundaries color-coded by functional class on the imaging field of view. Requires polygon coordinates (typically from suite2p or similar).

### `plot_style` -- Figure utilities

Publication-ready plotting helpers:
- `bar_with_points()` -- Bar chart with mean +/- SEM and individual datapoints
- `smooth_trace()` -- Gaussian smoothing for display traces
- `draw_combined_heatmap()` -- Trial heatmap with overlaid mean traces
- `shade_windows()` -- Background shading for time windows
- `set_composite_style()` / `set_standalone_style()` -- matplotlib rcParams presets

## Adapting to Your Data

The GLM code is dataset-agnostic. To use it with your own data format:

1. Write a reader class with this interface:
   - `.time_axis`: 1D array of timepoints per trial
   - `.n_trials`, `.n_neurons`: integers
   - `.get_trial_spikes(trial_idx)` -> (T, N) array
   - `.get_trial_sol_id(trial_idx)` -> trial type integer
   - `.get_trial_lick_times(trial_idx)` -> 1D array of event times

2. Define your regressors in `build_full_dm()` or write a custom builder. Each predictor chunk is a (T, n_columns) array; you can have any number of chunks.

3. Set group definitions, session indices, and time windows for your task.

### HDF5 / MATLAB v7.3 Note

MATLAB's `-v7.3` save format stores data as HDF5 with nested object references. A critical pitfall: MATLAB empty cells `[]` are stored as `uint64` HDF5 object references, NOT empty `float64` arrays. The included `H5SessionReader` handles this by checking `raw.dtype != np.float64`.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.01 | Significance threshold for permutation test |
| `sigma` | 0.15 s | Gaussian kernel width for lick rate smoothing |
| `shift_ms` | 300 ms | +/- time shift range for lick regressors |
| `kernel_type` | 'gaussian' | Lick kernel shape ('gaussian' or 'gcamp') |
| `early_window` | (0, 2) s | Motor/lick encoding window |
| `late_window` | (3, 5) s | Sensory/taste encoding window |

## License

MIT License. See [LICENSE](LICENSE).

## Citation

If you use this code, please cite the associated publication (forthcoming).
