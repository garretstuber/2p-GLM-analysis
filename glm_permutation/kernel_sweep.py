"""
Kernel parameter sensitivity sweep.

Validates that GLM results are robust to kernel parameter choices by sweeping:
  1. Gaussian sigma: controls temporal smoothing of lick events
  2. Time-shift range: controls how far the model looks ahead/behind
  3. GCaMP kernel: alternative biophysically-motivated kernel shape

If encoding proportions are stable across parameter choices, the findings
are not an artifact of kernel selection.

Typical sweep:
  sigma_values = [0.10, 0.15, 0.30, 0.50]
  shift_values = [100, 300, 750]  # ms
  kernel_types = ['gaussian', 'gcamp']
"""
import numpy as np
import time as timer
from glm_permutation.glm_core import (build_full_dm, circular_shift_null,
                       compute_pvalues_circular_shift)


def run_single_config(readers, sigma=0.15, shift_ms=300,
                       kernel_type='gaussian', alpha=0.01):
    """
    Run GLM for one mouse with one kernel configuration.

    Returns sig_fracs array (% significant per predictor).
    """
    from glm_permutation.glm_core import build_full_dm

    X_list, Y, boundaries = build_full_dm(
        readers, sigma=sigma, shift_ms=shift_ms, kernel_type=kernel_type)

    _, _, sig_fracs = compute_pvalues_circular_shift(
        X_list, Y, boundaries, alpha=alpha)

    return sig_fracs


def run_sigma_sweep(f, session_ds, n_mice, test_indices, groups,
                     sigma_values, fixed_shift_ms=300, alpha=0.01,
                     ReaderClass=None):
    """
    Sweep Gaussian sigma with fixed shift range.

    Returns dict mapping sigma -> {'fracs': (n_mice, n_preds), 'meta': list}
    """
    if ReaderClass is None:
        from glm_permutation.data_loader import H5SessionReader as ReaderClass

    results = {}
    for sigma in sigma_values:
        print(f"\n--- sigma={sigma:.2f} ---")
        mouse_fracs, mouse_meta = [], []
        for mi in range(n_mice):
            subject_id = mi + 1
            group = 'unknown'
            for g, ids in groups.items():
                if subject_id in ids:
                    group = g
                    break
            t0 = timer.time()
            readers = [ReaderClass(f, session_ds, mi, si)
                       for si in test_indices]
            fracs = run_single_config(readers, sigma=sigma,
                                       shift_ms=fixed_shift_ms,
                                       kernel_type='gaussian', alpha=alpha)
            elapsed = timer.time() - t0
            mouse_fracs.append(fracs)
            mouse_meta.append({'subject_id': subject_id, 'group': group})
            print(f"  Mouse {mi} ({group}): {elapsed:.1f}s, "
                  f"taste={fracs[0]:.1f}%, lick={fracs[1]:.1f}%")

        results[sigma] = {
            'fracs': np.array(mouse_fracs),
            'meta': mouse_meta,
        }
    return results


def run_shift_sweep(f, session_ds, n_mice, test_indices, groups,
                     shift_values, fixed_sigma=0.15, alpha=0.01,
                     ReaderClass=None):
    """
    Sweep time-shift range with fixed sigma.

    Returns dict mapping shift_ms -> {'fracs': (n_mice, n_preds), 'meta': list}
    """
    if ReaderClass is None:
        from glm_permutation.data_loader import H5SessionReader as ReaderClass

    results = {}
    for shift_ms in shift_values:
        print(f"\n--- shift={shift_ms}ms ---")
        mouse_fracs, mouse_meta = [], []
        for mi in range(n_mice):
            subject_id = mi + 1
            group = 'unknown'
            for g, ids in groups.items():
                if subject_id in ids:
                    group = g
                    break
            t0 = timer.time()
            readers = [ReaderClass(f, session_ds, mi, si)
                       for si in test_indices]
            fracs = run_single_config(readers, sigma=fixed_sigma,
                                       shift_ms=shift_ms,
                                       kernel_type='gaussian', alpha=alpha)
            elapsed = timer.time() - t0
            mouse_fracs.append(fracs)
            mouse_meta.append({'subject_id': subject_id, 'group': group})
            print(f"  Mouse {mi} ({group}): {elapsed:.1f}s, "
                  f"taste={fracs[0]:.1f}%, lick={fracs[1]:.1f}%")

        results[shift_ms] = {
            'fracs': np.array(mouse_fracs),
            'meta': mouse_meta,
        }
    return results


def run_gcamp_comparison(f, session_ds, n_mice, test_indices, groups,
                          shift_ms=300, alpha=0.01, ReaderClass=None):
    """
    Run GLM with GCaMP impulse-response kernel for comparison.

    Returns {'fracs': (n_mice, n_preds), 'meta': list}
    """
    if ReaderClass is None:
        from glm_permutation.data_loader import H5SessionReader as ReaderClass

    print("\n--- GCaMP kernel ---")
    mouse_fracs, mouse_meta = [], []
    for mi in range(n_mice):
        subject_id = mi + 1
        group = 'unknown'
        for g, ids in groups.items():
            if subject_id in ids:
                group = g
                break
        t0 = timer.time()
        readers = [ReaderClass(f, session_ds, mi, si)
                   for si in test_indices]
        fracs = run_single_config(readers, sigma=0.15, shift_ms=shift_ms,
                                   kernel_type='gcamp', alpha=alpha)
        elapsed = timer.time() - t0
        mouse_fracs.append(fracs)
        mouse_meta.append({'subject_id': subject_id, 'group': group})
        print(f"  Mouse {mi} ({group}): {elapsed:.1f}s, "
              f"taste={fracs[0]:.1f}%, lick={fracs[1]:.1f}%")

    return {
        'fracs': np.array(mouse_fracs),
        'meta': mouse_meta,
    }
