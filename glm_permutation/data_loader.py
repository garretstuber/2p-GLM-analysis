"""
Data loader for HDF5 (MATLAB v7.3) calcium imaging data.

Handles the nested object-reference structure used by MATLAB's save() with -v7.3.
Key pitfall: MATLAB empty cells [] are stored as uint64 HDF5 object references,
NOT empty float64 arrays. Always check dtype before reading.

Usage:
    Subclass or adapt H5SessionReader for your data structure.
    The interface methods (get_trial_spikes, get_trial_sol_id, get_trial_lick_times)
    are used by all downstream GLM scripts.
"""
import h5py
import numpy as np


class H5SessionReader:
    """
    Reads one session from a MATLAB v7.3 HDF5 file with structure:
        root/subject/session -> (n_mice, 1) object refs
        each mouse -> trials -> (n_sessions, 1) object refs
        each session -> spikes, sol_id, lick_times, time_axis, neural_traces

    Parameters
    ----------
    f : h5py.File
        Open HDF5 file handle.
    session_ds : h5py.Dataset
        The root/subject/session dataset (n_mice x 1 object refs).
    mouse_idx : int
        Index of the mouse (0-based).
    session_idx : int
        Index of the session (0-based). For CTA data:
        0=Hab, 1=Cond1, 2=Inj1, 3=Cond2, 4=Inj2, 5=Test1, 6=Test2, 7=Test3
    """
    def __init__(self, f, session_ds, mouse_idx, session_idx):
        self.f = f
        ms = f[session_ds[mouse_idx, 0]]
        self.sess = f[ms['trials'][session_idx, 0]]
        self.session_idx = session_idx
        self.n_trials = self.sess['neural_traces'].shape[0]
        ref = self.sess['spikes'][0, 0]
        self.n_neurons = f[ref].shape[1]
        ref_ta = self.sess['time_axis'][0, 0]
        self.time_axis = f[ref_ta][:, 0]
        self.n_timepoints_per_trial = len(self.time_axis)

    def get_trial_spikes(self, trial_idx):
        """Return (T, N) array of deconvolved spikes for one trial."""
        ref = self.sess['spikes'][trial_idx, 0]
        return self.f[ref][:, :]

    def get_trial_sol_id(self, trial_idx):
        """Return integer trial type (e.g., 1=taste, 2=water)."""
        ref = self.sess['sol_id'][trial_idx, 0]
        return int(self.f[ref][0, 0])

    def get_trial_lick_times(self, trial_idx):
        """
        Return 1D array of lick timestamps (seconds relative to trial start).

        CRITICAL: MATLAB empty cells [] are stored as uint64 object references
        in HDF5, not as empty float64 arrays. We detect this by checking dtype.
        """
        ref = self.sess['lick_times'][trial_idx, 0]
        raw = self.f[ref]
        # Empty MATLAB cell -> dtype is uint64 (object reference), not float64
        if raw.dtype != np.float64:
            return np.array([])
        data = raw[:]
        if data.size == 0:
            return np.array([])
        return data.flatten()


def load_rois(f, roi_ds, mouse_idx):
    """
    Load ROI boundary polygons for spatial mapping.

    ROI data structure (triple dereference):
        roi_ds[mouse_idx, 0] -> (N_neurons, 1) object refs
        each neuron -> (1, 2) object refs -> [x_coords_ref, y_coords_ref]
        each coord ref -> 1D array of polygon vertices

    Returns
    -------
    list of np.ndarray
        Each element is (n_vertices, 2) array of [x, y] polygon coordinates.
    """
    roi_mouse = f[roi_ds[mouse_idx, 0]]
    n_neurons = roi_mouse.shape[0]
    polygons = []
    for ci in range(n_neurons):
        cell_ref = roi_mouse[ci, 0]
        cell = f[cell_ref]
        xd = f[cell[0, 0]][:].flatten()
        yd = f[cell[0, 1]][:].flatten()
        polygons.append(np.column_stack([xd, yd]))
    return polygons


def open_stuber_lab_data(mat_path, root_key='full_data_struct_w_spks_min_np'):
    """
    Open a Stuber lab HDF5 file and return the file handle and session dataset.

    Parameters
    ----------
    mat_path : str
        Path to the .mat file (HDF5 v7.3 format).
    root_key : str
        Top-level key in the HDF5 file.

    Returns
    -------
    f : h5py.File
    session_ds : h5py.Dataset
        The subject/session dataset for creating H5SessionReader instances.
    roi_ds : h5py.Dataset or None
        The subject/roi_edges dataset if it exists.
    """
    f = h5py.File(mat_path, 'r')
    root = f[root_key]
    session_ds = root['subject']['session']
    try:
        roi_ds = root['subject']['roi_edges']
    except KeyError:
        roi_ds = None
    return f, session_ds, roi_ds
