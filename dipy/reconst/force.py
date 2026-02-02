"""
FORCE Reconstruction Module

Dictionary-based matching for diffusion MRI microstructure estimation.
Implements both standard matching and posterior-based inference.
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from dipy.reconst.base import ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit


def normalize_signals(signals):
    """
    L2-normalize signal array for cosine similarity search.

    Parameters
    ----------
    signals : ndarray (N, M)
        Signal array with N samples and M measurements.

    Returns
    -------
    normalized : ndarray (N, M)
        L2-normalized signals.
    """
    signals = np.asarray(signals, dtype=np.float32)
    norms = np.linalg.norm(signals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(signals / norms)


def softmax_stable(x, axis=1):
    """
    Numerically stable softmax.

    Parameters
    ----------
    x : ndarray
        Input array.
    axis : int
        Axis along which to compute softmax.

    Returns
    -------
    softmax : ndarray
        Softmax probabilities.
    """
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


def compute_uncertainty_ambiguity(scores):
    """
    Compute uncertainty and ambiguity metrics from match scores.

    Parameters
    ----------
    scores : ndarray (N, K)
        Penalized scores for K neighbors.

    Returns
    -------
    uncertainty : ndarray (N,)
        IQR of scores (Eq 14 from FORCE paper).
    ambiguity : ndarray (N,)
        Fraction above half-max (Eq 15 from FORCE paper).
    """
    p75 = np.percentile(scores, 75, axis=1)
    p25 = np.percentile(scores, 25, axis=1)
    uncertainty = (p75 - p25).astype(np.float32)

    half = 0.5 * np.max(scores, axis=1)
    ambiguity = (np.sum(scores > half[:, None], axis=1) / scores.shape[1]).astype(np.float32)
    return uncertainty, ambiguity


def labels_to_peak_indices(labels_binary, max_peaks=3):
    """
    Convert binary peak labels to compact index array.

    Parameters
    ----------
    labels_binary : ndarray (N, D)
        Binary array with 1s at peak directions.
    max_peaks : int
        Maximum number of peaks to store.

    Returns
    -------
    peak_idx : ndarray (N, max_peaks)
        Indices of peak directions, -1 for missing.
    """
    nsims = labels_binary.shape[0]
    peak_idx = np.full((nsims, max_peaks), -1, dtype=np.int16)

    rows, cols = np.nonzero(labels_binary)
    counts = np.zeros(nsims, dtype=np.int8)

    for r, c in zip(rows, cols):
        j = counts[r]
        if j < max_peaks:
            peak_idx[r, j] = c
            counts[r] = j + 1

    return peak_idx


def pick_top_peaks_from_weights(
    weights,
    sphere_dirs,
    n_peaks=5,
    top_m=30,
    min_separation_angle=45.0
):
    """
    Extract discrete peaks from directional weights.

    Parameters
    ----------
    weights : ndarray (D,)
        Non-negative directional weights.
    sphere_dirs : ndarray (D, 3)
        Unit sphere directions.
    n_peaks : int
        Number of peaks to extract.
    top_m : int
        Number of top candidates to consider.
    min_separation_angle : float
        Minimum angular separation in degrees.

    Returns
    -------
    peak_dirs : ndarray (n_peaks, 3)
        Peak directions.
    peak_inds : ndarray (n_peaks,)
        Peak indices.
    peak_vals : ndarray (n_peaks,)
        Peak values.
    """
    peak_dirs = np.zeros((n_peaks, 3), dtype=np.float32)
    peak_inds = np.full((n_peaks,), -1, dtype=np.int32)
    peak_vals = np.zeros((n_peaks,), dtype=np.float32)

    mx = float(np.max(weights))
    if mx <= 0.0:
        return peak_dirs, peak_inds, peak_vals

    top_m = min(top_m, weights.shape[0])
    cand = np.argpartition(weights, -top_m)[-top_m:]
    cand = cand[np.argsort(weights[cand])[::-1]]

    cos_thr = np.cos(np.deg2rad(min_separation_angle))
    selected = []

    for idx in cand:
        if len(selected) >= n_peaks:
            break
        if float(weights[idx]) <= 0.0:
            break

        d = sphere_dirs[idx]
        ok = True
        for sidx in selected:
            ds = sphere_dirs[sidx]
            if abs(float(np.dot(d, ds))) > cos_thr:
                ok = False
                break
        if ok:
            selected.append(int(idx))

    for j, idx in enumerate(selected):
        peak_inds[j] = idx
        peak_vals[j] = float(weights[idx])
        peak_dirs[j] = sphere_dirs[idx].astype(np.float32, copy=False)

    return peak_dirs, peak_inds, peak_vals


class IndexFlatIP:
    """
    Flat index for inner product similarity search.

    Pure NumPy implementation providing FAISS-like interface.
    For production use, compile the Cython _force_search module.

    Parameters
    ----------
    d : int
        Dimension of vectors.
    """

    def __init__(self, d):
        if d <= 0:
            raise ValueError(f"Dimension must be positive, got {d}")
        self.d = int(d)
        self.ntotal = 0
        self._xb = None

    def add(self, x):
        """Add vectors to the index."""
        x = np.ascontiguousarray(x, dtype=np.float32)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        if x.shape[1] != self.d:
            raise ValueError(
                f"Vector dimension {x.shape[1]} != index dimension {self.d}"
            )

        if self._xb is None:
            self._xb = x.copy()
        else:
            self._xb = np.vstack([self._xb, x])

        self.ntotal = len(self._xb)

    def search(self, x, k):
        """
        Search for k nearest neighbors by inner product.

        Parameters
        ----------
        x : ndarray (n, d)
            Query vectors.
        k : int
            Number of neighbors.

        Returns
        -------
        distances : ndarray (n, k)
            Inner products (descending).
        indices : ndarray (n, k)
            Neighbor indices.
        """
        if self.ntotal == 0:
            raise RuntimeError("Cannot search empty index")

        x = np.ascontiguousarray(x, dtype=np.float32)
        if x.ndim == 1:
            x = x.reshape(1, -1)

        k = min(k, self.ntotal)

        scores = x @ self._xb.T
        indices = np.argsort(-scores, axis=1)[:, :k]
        distances = np.take_along_axis(scores, indices, axis=1)

        return distances.astype(np.float32), indices.astype(np.int64)

    def reset(self):
        """Remove all vectors."""
        self._xb = None
        self.ntotal = 0


def create_faiss_index(signals_norm):
    """
    Create FAISS-like index for cosine similarity search.

    Parameters
    ----------
    signals_norm : ndarray (N, M)
        L2-normalized library signals.

    Returns
    -------
    index : IndexFlatIP
        Search index.
    """
    dimension = signals_norm.shape[1]
    index = IndexFlatIP(dimension)
    index.add(signals_norm)
    return index


def force_search(
    index,
    query_signals_norm,
    penalty_array,
    n_neighbors=50
):
    """
    Perform FORCE matching search.

    Parameters
    ----------
    index : IndexFlatIP
        Search index with library signals.
    query_signals_norm : ndarray (N, M)
        L2-normalized query signals.
    penalty_array : ndarray
        Penalty values for each library entry.
    n_neighbors : int
        Number of neighbors to retrieve.

    Returns
    -------
    indices : ndarray (N, n_neighbors)
        Matched library indices.
    scores : ndarray (N, n_neighbors)
        Penalized match scores.
    """
    D, I = index.search(query_signals_norm, k=n_neighbors)
    S = D - penalty_array[I]
    return I.astype(np.int32), S.astype(np.float32)


class FORCEModel(ReconstModel):
    """
    FORCE reconstruction model.

    Dictionary-based matching for microstructure estimation.

    Parameters
    ----------
    gtab : GradientTable
        Gradient table.
    dictionary : dict
        Pre-computed FORCE dictionary with signals and parameters.
    penalty : float, optional
        Penalty weight for fiber complexity. Default is 1e-5.
    n_neighbors : int, optional
        Number of neighbors for matching. Default is 50.
    use_posterior : bool, optional
        Use posterior averaging instead of best match. Default is False.
    posterior_beta : float, optional
        Softmax temperature for posterior. Default is 2000.0.
    compute_odf : bool, optional
        Compute posterior ODF maps. Default is False.
    """

    def __init__(
        self,
        gtab,
        dictionary,
        penalty=1e-5,
        n_neighbors=50,
        use_posterior=False,
        posterior_beta=2000.0,
        compute_odf=False,
    ):
        self.gtab = gtab
        self.dictionary = dictionary
        self.penalty = penalty
        self.n_neighbors = n_neighbors
        self.use_posterior = use_posterior
        self.posterior_beta = posterior_beta
        self.compute_odf = compute_odf

        # Prepare library
        self._prepare_library()

    def _prepare_library(self):
        """Prepare library for matching."""
        signals = self.dictionary["signals"]

        # Normalize library signals
        lib_norm = np.linalg.norm(signals, axis=1, keepdims=True)
        lib_norm[lib_norm == 0] = 1.0
        self.signals_norm = np.ascontiguousarray(
            (signals / lib_norm).astype(np.float32)
        )

        # Build index
        self.index = create_faiss_index(self.signals_norm)

        # Penalty array
        num_fibers = self.dictionary.get("num_fibers", np.zeros(len(signals)))
        self.penalty_array = (self.penalty * num_fibers).astype(np.float32)

    def fit(self, data, mask=None):
        """
        Fit model to data.

        Parameters
        ----------
        data : ndarray
            Diffusion data.
        mask : ndarray, optional
            Brain mask.

        Returns
        -------
        fit : FORCEFit
            Fitted model.
        """
        return FORCEFit(self, data, mask)


class FORCEFit:
    """
    FORCE model fit results.

    Parameters
    ----------
    model : FORCEModel
        The FORCE model.
    data : ndarray
        Diffusion data.
    mask : ndarray, optional
        Brain mask.
    """

    def __init__(self, model, data, mask=None):
        self.model = model
        self.data = data
        self.mask = mask if mask is not None else np.ones(data.shape[:-1], dtype=bool)

        self._fit()

    def _fit(self):
        """Perform matching."""
        shape = self.data.shape[:-1]
        n_voxels = np.prod(shape)
        n_gradients = self.data.shape[-1]

        # Flatten data
        data_flat = self.data.reshape(-1, n_gradients).astype(np.float32)
        mask_flat = self.mask.reshape(-1)
        valid_idx = np.where(mask_flat)[0]

        # Normalize query signals
        valid_data = data_flat[valid_idx]
        norms = np.linalg.norm(valid_data, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query_norm = np.ascontiguousarray(valid_data / norms)

        # Perform search
        I, S = force_search(
            self.model.index,
            query_norm,
            self.model.penalty_array,
            n_neighbors=self.model.n_neighbors
        )

        # Initialize output maps
        self._init_output_maps(n_voxels, shape)

        if self.model.use_posterior:
            self._posterior_aggregation(valid_idx, I, S)
        else:
            self._best_match(valid_idx, I)

        # Reshape to original shape
        self._reshape_outputs(shape)

    def _init_output_maps(self, n_voxels, shape):
        """Initialize output arrays."""
        d = self.model.dictionary

        self._fa = np.zeros(n_voxels, dtype=np.float32)
        self._md = np.zeros(n_voxels, dtype=np.float32)
        self._rd = np.zeros(n_voxels, dtype=np.float32)
        self._wm_fraction = np.zeros(n_voxels, dtype=np.float32)
        self._gm_fraction = np.zeros(n_voxels, dtype=np.float32)
        self._csf_fraction = np.zeros(n_voxels, dtype=np.float32)
        self._num_fibers = np.zeros(n_voxels, dtype=np.float32)
        self._dispersion = np.zeros(n_voxels, dtype=np.float32)

        if "ufa_wm" in d:
            self._ufa_wm = np.zeros(n_voxels, dtype=np.float32)
            self._ufa_voxel = np.zeros(n_voxels, dtype=np.float32)

    def _best_match(self, valid_idx, I):
        """Use best match for each voxel."""
        d = self.model.dictionary
        best_idx = I[:, 0]

        self._fa[valid_idx] = d["fa"][best_idx]
        self._md[valid_idx] = d["md"][best_idx]
        self._rd[valid_idx] = d["rd"][best_idx]
        self._wm_fraction[valid_idx] = d["wm_fraction"][best_idx]
        self._gm_fraction[valid_idx] = d["gm_fraction"][best_idx]
        self._csf_fraction[valid_idx] = d["csf_fraction"][best_idx]
        self._num_fibers[valid_idx] = d["num_fibers"][best_idx]
        self._dispersion[valid_idx] = d["dispersion"][best_idx]

        if "ufa_wm" in d:
            self._ufa_wm[valid_idx] = d["ufa_wm"][best_idx]
            self._ufa_voxel[valid_idx] = d["ufa_voxel"][best_idx]

    def _posterior_aggregation(self, valid_idx, I, S):
        """Use posterior averaging."""
        d = self.model.dictionary
        w = softmax_stable(self.model.posterior_beta * S, axis=1)

        self._fa[valid_idx] = np.sum(w * d["fa"][I], axis=1)
        self._md[valid_idx] = np.sum(w * d["md"][I], axis=1)
        self._rd[valid_idx] = np.sum(w * d["rd"][I], axis=1)
        self._wm_fraction[valid_idx] = np.sum(w * d["wm_fraction"][I], axis=1)
        self._gm_fraction[valid_idx] = np.sum(w * d["gm_fraction"][I], axis=1)
        self._csf_fraction[valid_idx] = np.sum(w * d["csf_fraction"][I], axis=1)
        self._num_fibers[valid_idx] = np.sum(w * d["num_fibers"][I], axis=1)
        self._dispersion[valid_idx] = np.sum(w * d["dispersion"][I], axis=1)

        if "ufa_wm" in d:
            self._ufa_wm[valid_idx] = np.sum(w * d["ufa_wm"][I], axis=1)
            self._ufa_voxel[valid_idx] = np.sum(w * d["ufa_voxel"][I], axis=1)

    def _reshape_outputs(self, shape):
        """Reshape outputs to original data shape."""
        self._fa = self._fa.reshape(shape)
        self._md = self._md.reshape(shape)
        self._rd = self._rd.reshape(shape)
        self._wm_fraction = self._wm_fraction.reshape(shape)
        self._gm_fraction = self._gm_fraction.reshape(shape)
        self._csf_fraction = self._csf_fraction.reshape(shape)
        self._num_fibers = self._num_fibers.reshape(shape)
        self._dispersion = self._dispersion.reshape(shape)

        if hasattr(self, "_ufa_wm"):
            self._ufa_wm = self._ufa_wm.reshape(shape)
            self._ufa_voxel = self._ufa_voxel.reshape(shape)

    @property
    def fa(self):
        """Fractional anisotropy map."""
        return self._fa

    @property
    def md(self):
        """Mean diffusivity map."""
        return self._md

    @property
    def rd(self):
        """Radial diffusivity map."""
        return self._rd

    @property
    def wm_fraction(self):
        """White matter fraction map."""
        return self._wm_fraction

    @property
    def gm_fraction(self):
        """Gray matter fraction map."""
        return self._gm_fraction

    @property
    def csf_fraction(self):
        """CSF fraction map."""
        return self._csf_fraction

    @property
    def num_fibers(self):
        """Number of fibers map."""
        return self._num_fibers

    @property
    def dispersion(self):
        """Orientation dispersion map."""
        return self._dispersion

    @property
    def ufa_wm(self):
        """microFA in white matter."""
        if hasattr(self, "_ufa_wm"):
            return self._ufa_wm
        return None

    @property
    def ufa_voxel(self):
        """Voxel-averaged microFA."""
        if hasattr(self, "_ufa_voxel"):
            return self._ufa_voxel
        return None


def compute_entropy(weights):
    """
    Compute entropy of posterior weights.

    Parameters
    ----------
    weights : ndarray (N, K)
        Posterior weights for K neighbors.

    Returns
    -------
    entropy : ndarray (N,)
        Shannon entropy for each sample.
    """
    return (-np.sum(weights * np.log(weights + 1e-12), axis=1)).astype(np.float32)


def posterior_mean_signal(signals, weights, indices):
    """
    Compute posterior mean signal from neighbors.

    Parameters
    ----------
    signals : ndarray (N_lib, M)
        Library signals.
    weights : ndarray (N_query, K)
        Posterior weights.
    indices : ndarray (N_query, K)
        Neighbor indices.

    Returns
    -------
    mean_signal : ndarray (N_query, M)
        Posterior mean signals.
    """
    n_query = indices.shape[0]
    n_grad = signals.shape[1]
    k = indices.shape[1]

    result = np.zeros((n_query, n_grad), dtype=np.float32)
    for kk in range(k):
        result += weights[:, kk:kk+1] * signals[indices[:, kk]]

    return result
