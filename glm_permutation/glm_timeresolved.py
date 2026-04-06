"""
Time-resolved windowed GLM.

Slides a window across the trial period to reveal WHEN encoding of each
variable peaks. Uses the simplified windowed design matrix (single-column
regressors per window).

Typical parameters:
  WIN_SIZE = 1.0 s
  WIN_STEP = 0.25 s
  Range: -1.0 to 6.0 s from stimulus onset

Output per mouse:
  - sig_fracs: (n_windows, n_predictors) -- % significant neurons
  - mean_dr2: (n_windows, n_predictors) -- mean delta-R2 among sig neurons
"""
import numpy as np
import time as timer
from glm_permutation.glm_core import build_windowed_dm, compute_pvalues_circular_shift


CHUNK_NAMES = ['taste', 'lick_rate', 'lick_x_taste', 'spout_on', 'trial_num']
CHUNK_LABELS = ['Taste', 'Lick Rate', 'Lick x Taste', 'Spout Onset', 'Trial #']


def run_timeresolved_mouse(readers, win_size=1.0, win_step=0.25,
                            win_range=(-1.0, 6.0), alpha=0.01):
    """
    Run time-resolved GLM for one mouse.

    Parameters
    ----------
    readers : list of session reader objects
    win_size : float, window width in seconds
    win_step : float, step size in seconds
    win_range : tuple, (start, end) of window centers
    alpha : float, significance threshold

    Returns
    -------
    dict with keys:
        win_centers, win_starts, sig_fracs, mean_dr2
    """
    win_starts = np.arange(win_range[0],
                           win_range[1] - win_size + win_step / 2,
                           win_step)
    win_centers = win_starts + win_size / 2
    n_wins = len(win_starts)
    n_preds = len(CHUNK_NAMES)

    sig_fracs = np.zeros((n_wins, n_preds))
    mean_dr2 = np.zeros((n_wins, n_preds))

    for wi, ws in enumerate(win_starts):
        we = ws + win_size
        X_list, Y, boundaries = build_windowed_dm(readers, ws, we)
        if X_list is None:
            continue

        pvals, delta_r2, _ = compute_pvalues_circular_shift(
            X_list, Y, boundaries, alpha=alpha)

        for k in range(min(n_preds, pvals.shape[1])):
            sig = pvals[:, k] < alpha
            sig_fracs[wi, k] = np.mean(sig) * 100
            mean_dr2[wi, k] = (np.mean(delta_r2[sig, k]) * 100
                               if np.any(sig) else 0.0)

    return {
        'win_centers': win_centers,
        'win_starts': win_starts,
        'sig_fracs': sig_fracs,
        'mean_dr2': mean_dr2,
    }


def run_timeresolved_all(f, session_ds, n_mice, test_indices, groups,
                          win_size=1.0, win_step=0.25, alpha=0.01,
                          ReaderClass=None):
    """
    Run time-resolved GLM for all mice.

    Parameters
    ----------
    f : h5py.File
    session_ds : h5py.Dataset
    n_mice : int
    test_indices : list of int, session indices to analyze
    groups : dict mapping group_name -> list of subject_ids
    ReaderClass : class, session reader class (default: H5SessionReader)
    alpha : float

    Returns
    -------
    dict of per-mouse results
    """
    if ReaderClass is None:
        from glm_permutation.data_loader import H5SessionReader as ReaderClass

    all_results = {}
    for mi in range(n_mice):
        subject_id = mi + 1
        group = 'unknown'
        for g, ids in groups.items():
            if subject_id in ids:
                group = g
                break

        t0 = timer.time()
        readers = [ReaderClass(f, session_ds, mi, si) for si in test_indices]
        n_neurons = min(r.n_neurons for r in readers)

        result = run_timeresolved_mouse(readers, win_size=win_size,
                                         win_step=win_step, alpha=alpha)
        result['subject_id'] = subject_id
        result['group'] = group
        result['n_neurons'] = n_neurons

        elapsed = timer.time() - t0
        print(f"Mouse {mi} ({group}, n={n_neurons}): {elapsed:.1f}s")

        all_results[f'mouse_{mi}'] = result

    return all_results
