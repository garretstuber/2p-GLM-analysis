"""
2-window magnitude analysis: early (motor) vs late (sensory) encoding.

Splits the trial into two windows and independently computes delta-R2
and significance for each predictor. This enables direct comparison of
lick vs taste encoding strength and identifies neurons encoding one,
both, or neither.

Default windows (adapt to your task):
  Early: 0-2 s (motor/lick-dominant)
  Late:  3-5 s (sensory/taste-dominant)

Output per mouse:
  - early_pvals, late_pvals: (n_neurons, n_predictors)
  - early_delta_r2, late_delta_r2: (n_neurons, n_predictors)
  - early_sig, late_sig: boolean (n_neurons, n_predictors)
  - early_sig_fracs, late_sig_fracs: (n_predictors,)
"""
import numpy as np
import time as timer
from glm_permutation.glm_core import build_windowed_dm, compute_pvalues_circular_shift


CHUNK_NAMES = ['taste', 'lick_rate', 'lick_x_taste', 'spout_on', 'trial_num']


def run_magnitude_mouse(readers, early_window=(0.0, 2.0),
                         late_window=(3.0, 5.0), alpha=0.01):
    """
    Run 2-window magnitude analysis for one mouse.

    Parameters
    ----------
    readers : list of session reader objects
    early_window : tuple (start, end) in seconds
    late_window : tuple (start, end) in seconds
    alpha : float

    Returns
    -------
    dict with per-window pvals, delta_r2, sig masks, and sig fractions
    """
    windows = {'early': early_window, 'late': late_window}
    result = {'n_neurons': min(r.n_neurons for r in readers)}

    for wname, (ws, we) in windows.items():
        X_list, Y, boundaries = build_windowed_dm(readers, ws, we)
        pvals, delta_r2, sig_fracs = compute_pvalues_circular_shift(
            X_list, Y, boundaries, alpha=alpha)

        sig = pvals < alpha
        result[f'{wname}_pvals'] = pvals
        result[f'{wname}_delta_r2'] = delta_r2
        result[f'{wname}_sig'] = sig
        result[f'{wname}_sig_fracs'] = sig_fracs

        # Per-predictor mean delta-R2 among significant neurons
        for ci, cn in enumerate(CHUNK_NAMES):
            if ci >= pvals.shape[1]:
                break
            sig_mask = sig[:, ci]
            result[f'{wname}_{cn}_mean_dr2_sig'] = (
                np.mean(delta_r2[sig_mask, ci]) if np.any(sig_mask) else 0)
            result[f'{wname}_{cn}_mean_dr2_all'] = np.mean(delta_r2[:, ci])

    return result


def run_magnitude_all(f, session_ds, n_mice, test_indices, groups,
                       early_window=(0.0, 2.0), late_window=(3.0, 5.0),
                       alpha=0.01, ReaderClass=None):
    """
    Run 2-window magnitude analysis for all mice.

    Returns dict of per-mouse results keyed as 'mouse_0', 'mouse_1', etc.
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

        result = run_magnitude_mouse(readers, early_window=early_window,
                                      late_window=late_window, alpha=alpha)
        result['subject_id'] = subject_id
        result['group'] = group

        elapsed = timer.time() - t0
        print(f"Mouse {mi} ({group}, n={result['n_neurons']}): {elapsed:.1f}s")
        print(f"  Early: taste={result['early_sig_fracs'][0]:.1f}% "
              f"lick={result['early_sig_fracs'][1]:.1f}%")
        print(f"  Late:  taste={result['late_sig_fracs'][0]:.1f}% "
              f"lick={result['late_sig_fracs'][1]:.1f}%")

        all_results[f'mouse_{mi}'] = result

    return all_results
