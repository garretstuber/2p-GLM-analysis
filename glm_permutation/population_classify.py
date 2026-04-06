"""
Neuron population classification from 2-window GLM results.

Classifies neurons into functional populations based on their encoding
in early (motor/lick) and late (sensory/taste) time windows:
  - Class 0: Neither (no significant encoding in either window)
  - Class 1: Lick-only (significant lick in early, no taste in late)
  - Class 2: Taste-only (significant taste in late, no lick in early)
  - Class 3: Both (significant in both windows)

Also provides statistical comparisons:
  - Chi-squared tests on proportions across groups
  - Mann-Whitney U tests on effect sizes
  - KS tests on delta-R2 distributions
"""
import numpy as np
from scipy import stats


CLASS_NAMES = {0: 'Neither', 1: 'Lick only', 2: 'Taste only', 3: 'Both'}
CLASS_KEYS = ['neither', 'lick_only', 'taste_only', 'both']


def classify_neurons(mouse_result, early_pred_idx=1, late_pred_idx=0):
    """
    Classify neurons from 2-window magnitude results.

    Parameters
    ----------
    mouse_result : dict
        Must contain 'early_sig' and 'late_sig' boolean arrays of shape
        (n_neurons, n_predictors).
    early_pred_idx : int
        Column index for the motor/lick predictor in the early window.
        Default 1 (lick_rate).
    late_pred_idx : int
        Column index for the sensory/taste predictor in the late window.
        Default 0 (taste).

    Returns
    -------
    classes : (n_neurons,) int array
        0=neither, 1=lick_only, 2=taste_only, 3=both
    """
    early_sig = mouse_result['early_sig'][:, early_pred_idx]
    late_sig = mouse_result['late_sig'][:, late_pred_idx]

    classes = np.zeros(len(early_sig), dtype=int)
    classes[early_sig & ~late_sig] = 1   # lick only
    classes[~early_sig & late_sig] = 2   # taste only
    classes[early_sig & late_sig] = 3    # both
    return classes


def population_proportions(mice_results, group_ids):
    """
    Compute per-mouse population proportions for a group.

    Parameters
    ----------
    mice_results : dict
        Results dict with keys like 'mouse_0', 'mouse_1', etc.
    group_ids : list of int
        Subject IDs belonging to this group.

    Returns
    -------
    proportions : (n_mice, 4) array
        Columns: [neither%, lick_only%, taste_only%, both%]
    """
    props = []
    for key, m in mice_results.items():
        if not isinstance(m, dict):
            continue
        if m.get('subject_id') not in group_ids:
            continue
        classes = classify_neurons(m)
        n = len(classes)
        mouse_props = [np.sum(classes == c) / n * 100 for c in range(4)]
        props.append(mouse_props)
    return np.array(props) if props else np.zeros((0, 4))


def chi_squared_test(mice_results, group1_ids, group2_ids):
    """
    Chi-squared test on population proportions between two groups.

    Returns overall chi2/p and pairwise tests per class.
    """
    counts = {}
    for gids, gname in [(group1_ids, 'g1'), (group2_ids, 'g2')]:
        c = {0: 0, 1: 0, 2: 0, 3: 0}
        for key, m in mice_results.items():
            if not isinstance(m, dict):
                continue
            if m.get('subject_id') not in gids:
                continue
            classes = classify_neurons(m)
            for cls in range(4):
                c[cls] += np.sum(classes == cls)
        counts[gname] = c

    # 2x4 contingency table
    table = np.array([[counts['g1'][c] for c in range(4)],
                      [counts['g2'][c] for c in range(4)]])
    chi2, p, dof, expected = stats.chi2_contingency(table)

    # Pairwise per class
    pairwise = {}
    for cls in range(4):
        t = np.array([
            [counts['g1'][cls], sum(counts['g1'].values()) - counts['g1'][cls]],
            [counts['g2'][cls], sum(counts['g2'].values()) - counts['g2'][cls]],
        ])
        chi2_pw, p_pw, _, _ = stats.chi2_contingency(t)
        pairwise[cls] = {'chi2': chi2_pw, 'p': p_pw}

    return {
        'overall': {'chi2': chi2, 'p': p, 'dof': dof},
        'pairwise': pairwise,
        'counts': counts,
    }
