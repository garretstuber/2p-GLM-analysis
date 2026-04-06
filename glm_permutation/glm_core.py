"""
Core GLM engine: circular-shift permutation with SVD-based efficient computation.

This module provides the reusable GLM components:
  - Regressor builders (taste kernel, lick rate, interaction, nuisance)
  - Design matrix construction
  - SVD-based delta-R2 computation
  - Circular-shift permutation null generation
  - F-statistic and p-value computation

All functions are dataset-agnostic and operate on numpy arrays.
"""
import numpy as np
from scipy.linalg import svd


# ====================================================================
# REGRESSOR BUILDERS
# ====================================================================

def make_event_kernel(time_axis, event_time, kernel_width=1.0):
    """
    Create a boxcar event kernel (indicator columns spanning kernel_width
    seconds after event_time).

    Returns (T, n_bins) array where each column is a time-lagged indicator.
    This allows the GLM to fit a flexible temporal response shape.
    """
    dt = np.mean(np.diff(time_axis))
    n_bins = max(1, int(kernel_width / dt))
    event_frame = np.argmin(np.abs(time_axis - event_time))
    kernel = np.zeros((len(time_axis), n_bins))
    for b in range(n_bins):
        idx = event_frame + b
        if 0 <= idx < len(time_axis):
            kernel[idx, b] = 1.0
    return kernel


def make_lick_rate_regressor(lick_times, time_axis, sigma=0.15, shift_ms=300):
    """
    Gaussian-smoothed lick rate with time-shift expansion.

    Each lick event is convolved with a Gaussian (sigma seconds), then
    the resulting trace is shifted +/- shift_ms to create a bank of
    time-lagged regressors. This captures neurons that lead or lag licking.

    Parameters
    ----------
    lick_times : array-like
        Lick timestamps in seconds.
    time_axis : array-like
        Trial time vector.
    sigma : float
        Gaussian kernel width in seconds.
    shift_ms : int
        Maximum shift in milliseconds (symmetric: +/- shift_ms).

    Returns
    -------
    lick_matrix : (T, 2*n_shift+1) array
    """
    dt = np.mean(np.diff(time_axis))
    n_shift = int(shift_ms / 1000.0 / dt)
    T = len(time_axis)

    # Build continuous lick rate
    lick_trace = np.zeros(T)
    if len(lick_times) > 0:
        for lt in lick_times:
            lick_trace += np.exp(-0.5 * ((time_axis - lt) / sigma) ** 2)

    # Time-shift expansion
    n_shifts_total = 2 * n_shift + 1
    lick_matrix = np.zeros((T, n_shifts_total))
    for i, s in enumerate(range(-n_shift, n_shift + 1)):
        shifted = np.roll(lick_trace, s)
        if s > 0:
            shifted[:s] = 0
        elif s < 0:
            shifted[s:] = 0
        lick_matrix[:, i] = shifted
    return lick_matrix


def make_gcamp_lick_regressor(lick_times, time_axis, tau_rise=0.05,
                               tau_decay=0.4, shift_ms=300):
    """
    GCaMP impulse-response kernel for lick events.

    Models each lick as triggering a GCaMP transient:
        h(t) = (1 - exp(-t/tau_rise)) * exp(-t/tau_decay)  for t >= 0

    Then applies the same time-shift expansion as the Gaussian version.
    """
    dt = np.mean(np.diff(time_axis))
    n_shift = int(shift_ms / 1000.0 / dt)
    T = len(time_axis)

    # GCaMP impulse response function (2s support)
    t_kernel = np.arange(0, 2.0, dt)
    irf = (1 - np.exp(-t_kernel / tau_rise)) * np.exp(-t_kernel / tau_decay)
    irf /= np.max(irf) if np.max(irf) > 0 else 1.0

    # Convolve each lick with the IRF
    lick_trace = np.zeros(T)
    if len(lick_times) > 0:
        for lt in lick_times:
            onset_frame = np.argmin(np.abs(time_axis - lt))
            for ki, kv in enumerate(irf):
                idx = onset_frame + ki
                if 0 <= idx < T:
                    lick_trace[idx] += kv

    # Time-shift expansion
    n_shifts_total = 2 * n_shift + 1
    lick_matrix = np.zeros((T, n_shifts_total))
    for i, s in enumerate(range(-n_shift, n_shift + 1)):
        shifted = np.roll(lick_trace, s)
        if s > 0:
            shifted[:s] = 0
        elif s < 0:
            shifted[s:] = 0
        lick_matrix[:, i] = shifted
    return lick_matrix


# ====================================================================
# DESIGN MATRIX CONSTRUCTION
# ====================================================================

def build_full_dm(readers, sigma=0.15, shift_ms=300, kernel_type='gaussian',
                  taste_kernel_width=3.0, spout_kernel_width=0.5):
    """
    Build concatenated design matrix from multiple session readers.

    Regressors (chunks):
      0: taste     -- event kernel for taste trials (boxcar, taste_kernel_width s)
      1: lick_rate -- Gaussian or GCaMP smoothed, time-shifted
      2: lick_x_taste -- lick_rate gated to taste trials only
      3: spout_on  -- spout onset event kernel
      4: trial_num -- linear trial drift
      5: session   -- session indicator dummies

    Parameters
    ----------
    readers : list of H5SessionReader (or compatible)
    sigma, shift_ms, kernel_type : lick kernel parameters
    taste_kernel_width : float, seconds
    spout_kernel_width : float, seconds

    Returns
    -------
    X_list : list of (T_total, n_cols_k) arrays, one per predictor chunk
    Y : (T_total, n_neurons) array
    boundaries : list of (start, end) tuples for session blocks
    """
    n_sessions = len(readers)
    n_neurons = min(r.n_neurons for r in readers)

    session_Ys = []
    taste_blocks, lick_blocks, lxt_blocks = [], [], []
    spout_blocks, trial_blocks = [], []
    session_lengths = []

    for reader in readers:
        time_axis = reader.time_axis
        T_trial = reader.n_timepoints_per_trial
        n_trials = reader.n_trials
        T_sess = n_trials * T_trial

        Y_sess = np.zeros((T_sess, n_neurons), dtype=np.float32)
        taste_c, lick_c, lxt_c, spout_c, trial_c = [], [], [], [], []

        for t in range(n_trials):
            sl = slice(t * T_trial, (t + 1) * T_trial)
            spk = reader.get_trial_spikes(t)
            Y_sess[sl, :] = spk[:, :n_neurons]

            sol_id = reader.get_trial_sol_id(t)
            lick_times = reader.get_trial_lick_times(t)

            # Taste kernel
            tk = make_event_kernel(time_axis, 0.0, kernel_width=taste_kernel_width)
            taste_c.append(tk if sol_id == 1 else np.zeros_like(tk))

            # Lick rate
            if kernel_type == 'gaussian':
                lr = make_lick_rate_regressor(lick_times, time_axis,
                                              sigma=sigma, shift_ms=shift_ms)
            elif kernel_type == 'gcamp':
                lr = make_gcamp_lick_regressor(lick_times, time_axis,
                                               shift_ms=shift_ms)
            else:
                raise ValueError(f"Unknown kernel_type: {kernel_type}")
            lick_c.append(lr)

            # Lick x taste interaction
            lxt_c.append(lr.copy() if sol_id == 1 else np.zeros_like(lr))

            # Spout onset
            spout_c.append(make_event_kernel(time_axis, 0.0,
                                             kernel_width=spout_kernel_width))

            # Trial number (linear drift)
            trial_frac = t / max(n_trials - 1, 1)
            tn = make_event_kernel(time_axis, 0.0,
                                   kernel_width=taste_kernel_width)
            trial_c.append(tn * trial_frac)

        session_Ys.append(Y_sess)
        taste_blocks.append(np.vstack(taste_c))
        lick_blocks.append(np.vstack(lick_c))
        lxt_blocks.append(np.vstack(lxt_c))
        spout_blocks.append(np.vstack(spout_c))
        trial_blocks.append(np.vstack(trial_c))
        session_lengths.append(T_sess)

    T_total = sum(session_lengths)
    Y = np.vstack(session_Ys)

    # Session indicators (n_sessions - 1 dummy columns)
    n_sess_cols = max(1, n_sessions - 1)
    X_session = np.zeros((T_total, n_sess_cols), dtype=np.float32)
    if n_sessions > 1:
        offset = session_lengths[0]
        for si in range(1, n_sessions):
            X_session[offset:offset + session_lengths[si], si - 1] = 1.0
            offset += session_lengths[si]

    X_list = [
        np.vstack(taste_blocks),
        np.vstack(lick_blocks),
        np.vstack(lxt_blocks),
        np.vstack(spout_blocks),
        np.vstack(trial_blocks),
        X_session,
    ]

    boundaries = []
    offset = 0
    for sl in session_lengths:
        boundaries.append((offset, offset + sl))
        offset += sl

    return X_list, Y, boundaries


def build_windowed_dm(readers, win_start, win_end, sigma=0.15):
    """
    Build design matrix using only timepoints within [win_start, win_end].

    Simplified regressors for short windows (single column each):
      0: taste -- 1/0 indicator
      1: lick_rate -- Gaussian-smoothed (no time shifts)
      2: lick_x_taste -- lick_rate gated to taste trials
      3: spout_on -- 1 if t=0 falls in window
      4: trial_num -- scaled trial fraction
    """
    n_neurons = min(r.n_neurons for r in readers)
    session_Ys = []
    taste_b, lick_b, lxt_b, spout_b, trial_b = [], [], [], [], []
    session_lengths = []

    for reader in readers:
        time_axis = reader.time_axis
        win_mask = (time_axis >= win_start) & (time_axis < win_end)
        win_idx = np.where(win_mask)[0]
        T_win = len(win_idx)
        if T_win == 0:
            continue

        time_axis_win = time_axis[win_idx]
        n_trials = reader.n_trials
        T_sess = n_trials * T_win

        Y_sess = np.zeros((T_sess, n_neurons), dtype=np.float32)
        for t in range(n_trials):
            sl = slice(t * T_win, (t + 1) * T_win)
            spk = reader.get_trial_spikes(t)
            Y_sess[sl, :] = spk[win_idx, :n_neurons]

            sol_id = reader.get_trial_sol_id(t)
            lick_times = reader.get_trial_lick_times(t)

            # Taste indicator
            taste_b.append(np.ones((T_win, 1)) if sol_id == 1
                           else np.zeros((T_win, 1)))

            # Lick rate (single column, no shifts)
            lick_trace = np.zeros(T_win)
            if len(lick_times) > 0:
                for lt in lick_times:
                    lick_trace += np.exp(
                        -0.5 * ((time_axis_win - lt) / sigma) ** 2)
            lick_col = lick_trace.reshape(-1, 1)
            lick_b.append(lick_col)

            # Lick x taste
            lxt_b.append(lick_col.copy() if sol_id == 1
                         else np.zeros_like(lick_col))

            # Spout onset
            spout_val = 1.0 if (win_start <= 0 < win_end) else 0.0
            spout_b.append(np.full((T_win, 1), spout_val))

            # Trial number
            trial_frac = t / max(n_trials - 1, 1)
            trial_b.append(np.full((T_win, 1), trial_frac))

        session_Ys.append(Y_sess)
        session_lengths.append(T_sess)

    if not session_Ys:
        return None, None, None

    Y = np.vstack(session_Ys)
    X_list = [np.vstack(taste_b), np.vstack(lick_b), np.vstack(lxt_b),
              np.vstack(spout_b), np.vstack(trial_b)]

    boundaries = []
    offset = 0
    for sl in session_lengths:
        boundaries.append((offset, offset + sl))
        offset += sl

    return X_list, Y, boundaries


# ====================================================================
# CIRCULAR SHIFT NULL GENERATION
# ====================================================================

def circular_shift_null(Y, boundaries, seed=42):
    """
    Generate circularly shifted null data.

    For each neuron, independently pick a random shift within each session
    and wrap the activity vector. This preserves temporal autocorrelation
    but destroys alignment with task events.

    Parameters
    ----------
    Y : (T, N) array, neural activity
    boundaries : list of (start, end) tuples for session blocks
    seed : int, random seed for reproducibility

    Returns
    -------
    Y_null : (T, N) array, shifted neural activity
    """
    T, n_neurons = Y.shape
    Y_null = np.empty_like(Y)
    rng = np.random.default_rng(seed=seed)
    for start, end in boundaries:
        T_block = end - start
        for i in range(n_neurons):
            q = rng.integers(1, T_block)
            block = Y[start:end, i]
            Y_null[start:end, i] = np.concatenate([block[q:], block[:q]])
    return Y_null


# ====================================================================
# GLM COMPUTATION (SVD-based)
# ====================================================================

def compute_delta_r2(X_list, Y):
    """
    Compute per-neuron delta-R2 for each predictor chunk.

    Uses SVD of the design matrix for efficient RSS computation:
        RSS = ||y||^2 - ||U'y||^2

    Parameters
    ----------
    X_list : list of (T, n_cols_k) arrays
    Y : (T, N) array

    Returns
    -------
    delta_r2 : (N, K) array, unique variance explained per predictor per neuron
    """
    T, n_neurons = Y.shape
    K = len(X_list)

    # Full model
    X_full = np.ones((T, 1), dtype=np.float64)
    for Xc in X_list:
        X_full = np.hstack((X_full, Xc.astype(np.float64)))
    U_full = svd(X_full, full_matrices=False)[0]

    # Reduced models (leave one chunk out)
    U_reds = []
    for k in range(K):
        X_red = np.ones((T, 1), dtype=np.float64)
        for kt in range(K):
            if kt != k:
                X_red = np.hstack((X_red, X_list[kt].astype(np.float64)))
        U_reds.append(svd(X_red, full_matrices=False)[0])

    delta_r2 = np.zeros((n_neurons, K))
    for cell in range(n_neurons):
        y = Y[:, cell].astype(np.float64)
        ss_tot = np.sum((y - np.mean(y))**2)
        if ss_tot == 0:
            continue
        rss_full = np.sum(y**2) - np.sum((U_full.T @ y)**2)
        r2_full = 1.0 - rss_full / ss_tot

        for k in range(K):
            rss_red = np.sum(y**2) - np.sum((U_reds[k].T @ y)**2)
            r2_red = 1.0 - rss_red / ss_tot
            delta_r2[cell, k] = r2_full - r2_red

    return delta_r2


def compute_pvalues_circular_shift(X_list, Y, boundaries, seed=42, alpha=0.01):
    """
    Full circular-shift permutation test.

    Generates one circular-shift null, computes F-statistics for both
    observed and null data, and derives p-values from the empirical null.

    Parameters
    ----------
    X_list : list of (T, n_cols_k) arrays
    Y : (T, N) array
    boundaries : list of (start, end) tuples
    seed : int
    alpha : float, significance threshold

    Returns
    -------
    pvals : (N, K) array
    delta_r2 : (N, K) array
    sig_fracs : (K,) array, percentage of significant neurons per predictor
    """
    T, n_neurons = Y.shape
    K = len(X_list)

    # Null data
    Y_null = circular_shift_null(Y, boundaries, seed=seed)

    # Build full and reduced model SVDs
    X_full = np.ones((T, 1), dtype=np.float64)
    for Xc in X_list:
        X_full = np.hstack((X_full, Xc.astype(np.float64)))
    M = X_full.shape[1]

    # Handle rank deficiency (can happen in short windows)
    rank = np.linalg.matrix_rank(X_full)
    if rank < M:
        var = np.var(X_full, axis=0)
        keep = var > 1e-10
        keep[0] = True
        X_full = X_full[:, keep]
        M = X_full.shape[1]

    U = svd(X_full, full_matrices=False)[0]

    U_reds, m_ks = [], []
    for k in range(K):
        m_k = X_list[k].shape[1]
        m_ks.append(m_k)
        X_red = np.ones((T, 1), dtype=np.float64)
        for kt in range(K):
            if kt != k:
                X_red = np.hstack((X_red, X_list[kt].astype(np.float64)))
        var_red = np.var(X_red, axis=0)
        keep_red = var_red > 1e-10
        keep_red[0] = True
        X_red = X_red[:, keep_red]
        U_reds.append(svd(X_red, full_matrices=False)[0])

    # Compute F-statistics
    def get_fstats(Y_mat):
        n = Y_mat.shape[1]
        f_vals = np.empty((n, K))
        for k in range(K):
            for cell in range(n):
                y = Y_mat[:, cell].astype(np.float64)
                y_sq = np.sum(y**2)
                rss = y_sq - np.sum((U.T @ y)**2)
                rss_red = y_sq - np.sum((U_reds[k].T @ y)**2)
                denom = rss / (T - M)
                f_vals[cell, k] = (((rss_red - rss) / m_ks[k]) / denom
                                   if denom > 0 else 0)
        return f_vals

    f_null = get_fstats(Y_null)
    f_obs = get_fstats(Y)

    # P-values from empirical null
    pvals = np.empty((n_neurons, K))
    for cell in range(n_neurons):
        for k in range(K):
            pvals[cell, k] = ((np.sum(f_null[:, k] >= f_obs[cell, k]) + 1)
                              / (n_neurons + 1))

    # Delta-R2
    delta_r2 = compute_delta_r2(X_list, Y)

    # Significance fractions
    sig_fracs = np.mean(pvals < alpha, axis=0) * 100

    return pvals, delta_r2, sig_fracs
