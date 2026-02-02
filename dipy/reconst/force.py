"""
FORCE Reconstruction Module

Signal matching based reconstruction for diffusion MRI microstructure
estimation. Implements both standard matching and posterior-based inference.

The FORCEModel class provides a DIPY-compatible interface for fitting
the FORCE model to diffusion MRI data. It supports:

- Configurable number of neighbors (n_neighbors, default 50)
- Standard best-match or posterior averaging (use_posterior flag)
- Configurable posterior temperature (posterior_beta)
- Optional ODF map computation (compute_odf flag)

Examples
--------
>>> from dipy.reconst.force import FORCEModel
>>> from dipy.sims.force import load_force_simulations
>>>
>>> simulations = load_force_simulations('simulated_data.npz')
>>> model = FORCEModel(gtab, simulations, n_neighbors=50)
>>> fit = model.fit(data, mask=brain_mask)
>>> fa_map = fit.fa

References
----------
FORCE methodology paper (in preparation)
"""

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from dipy.reconst.base import ReconstModel
from dipy.data import default_sphere

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs):
        return x

# Detect available search backend
_SEARCH_BACKEND = "numpy"
try:
    from dipy.reconst._force_search import search_inner_product as _cython_search
    _SEARCH_BACKEND = "cython-blas"
except ImportError:
    _cython_search = None


class SignalIndex:
    """
    Index for inner product similarity search.

    Pure NumPy implementation. For high-performance search,
    compile the Cython _force_search module.

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
        """
        Add vectors to the index.

        Parameters
        ----------
        x : array-like (n, d)
            Vectors to add, will be converted to float32 C-contiguous.
        """
        x = np.ascontiguousarray(x, dtype=np.float32)

        if x.ndim == 1:
            if len(x) != self.d:
                raise ValueError(
                    f"Vector dimension {len(x)} != index dimension {self.d}"
                )
            x = x.reshape(1, -1)

        if x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {x.ndim}D")

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
        x : array-like (n, d) or (d,)
            Query vectors.
        k : int
            Number of neighbors.

        Returns
        -------
        distances : ndarray (n, k)
            Inner products (descending order).
        indices : ndarray (n, k)
            Neighbor indices.
        """
        if self.ntotal == 0:
            raise RuntimeError("Cannot search empty index")

        x = np.ascontiguousarray(x, dtype=np.float32)

        if x.ndim == 1:
            if len(x) != self.d:
                raise ValueError(
                    f"Query dimension {len(x)} != index dimension {self.d}"
                )
            x = x.reshape(1, -1)

        if x.ndim != 2:
            raise ValueError(f"Expected 1D or 2D array, got {x.ndim}D")

        if x.shape[1] != self.d:
            raise ValueError(
                f"Query dimension {x.shape[1]} != index dimension {self.d}"
            )

        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        k = min(k, self.ntotal)

        # Use optimized Cython search if available (SciPy BLAS + streaming heap)
        if _cython_search is not None:
            distances, indices = _cython_search(x, self._xb, k)
        else:
            # Fallback to NumPy implementation
            scores = x @ self._xb.T
            indices = np.argsort(-scores, axis=1)[:, :k]
            distances = np.take_along_axis(scores, indices, axis=1)
            distances = distances.astype(np.float32)
            indices = indices.astype(np.int64)

        return distances, indices

    def reset(self):
        """Remove all vectors from the index."""
        self._xb = None
        self.ntotal = 0

    def __repr__(self):
        return f"SignalIndex(d={self.d}, ntotal={self.ntotal})"

    @staticmethod
    def get_backend():
        """Return the search backend being used ('cython-blas' or 'numpy')."""
        return _SEARCH_BACKEND


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


def create_signal_index(signals_norm):
    """
    Create index for cosine similarity search.

    Parameters
    ----------
    signals_norm : ndarray (N, M)
        L2-normalized library signals.

    Returns
    -------
    index : SignalIndex
        Search index.
    """
    dimension = signals_norm.shape[1]
    index = SignalIndex(dimension)
    index.add(signals_norm)
    return index


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
        IQR of scores.
    ambiguity : ndarray (N,)
        Fraction above half-max.
    """
    p75 = np.percentile(scores, 75, axis=1)
    p25 = np.percentile(scores, 25, axis=1)
    uncertainty = (p75 - p25).astype(np.float32)

    half = 0.5 * np.max(scores, axis=1)
    ambiguity = (np.sum(scores > half[:, None], axis=1) / scores.shape[1]).astype(
        np.float32
    )
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
    weights, sphere_dirs, n_peaks=5, top_m=30, min_separation_angle=45.0
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


def postprocess_peaks(preds, target_sphere, fracs):
    """
    Convert binary peak masks to exactly 5 peaks per sample.

    Parameters
    ----------
    preds : ndarray (N, D)
        Binary peak masks.
    target_sphere : ndarray (D, 3)
        Sphere directions.
    fracs : ndarray (N, max_peaks)
        Fiber fractions.

    Returns
    -------
    peaks_output : ndarray (N, 5, 3)
        Peak directions.
    peak_indices : ndarray (N, 5)
        Peak indices.
    peak_values : ndarray (N, 5)
        Peak values.
    """
    n = preds.shape[0]
    peaks_output = np.zeros((n, 5, 3), dtype=np.float32)
    peak_indices = np.full((n, 5), -1, dtype=np.int32)
    peak_vals = np.zeros((n, 5), dtype=np.float32)

    for i in range(n):
        coords = target_sphere[preds[i] == 1]
        indices = np.where(preds[i] == 1)[0]

        num = min(len(coords), 5)
        if num > 0:
            peaks_output[i, :num] = coords[:num]
            peak_indices[i, :num] = indices[:num]
        num_fracs = min(5, fracs[i].shape[0])
        peak_vals[i, :num_fracs] = fracs[i][:num_fracs]

    return peaks_output, peak_indices, peak_vals


class FORCEModel(ReconstModel):
    """
    FORCE reconstruction model.

    Signal matching based microstructure estimation.

    Parameters
    ----------
    gtab : GradientTable
        Gradient table.
    simulations : dict or None
        Pre-computed FORCE simulations with signals and parameters.
        If None, call generate() to create simulations.
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
    verbose : bool, optional
        Show progress bar and status messages. Default is False.
    """

    def __init__(
        self,
        gtab,
        simulations=None,
        penalty=1e-5,
        n_neighbors=50,
        use_posterior=False,
        posterior_beta=2000.0,
        compute_odf=False,
        verbose=False,
    ):
        self.gtab = gtab
        self.simulations = simulations
        self.penalty = penalty
        self.n_neighbors = n_neighbors
        self.use_posterior = use_posterior
        self.posterior_beta = posterior_beta
        self.compute_odf = compute_odf
        self.verbose = verbose

        self._index = None
        self._penalty_array = None
        self._signals_norm = None

        if simulations is not None:
            self._prepare_library()

    def generate(
        self,
        num_simulations=100000,
        output_path=None,
        num_cpus=1,
        wm_threshold=0.5,
        tortuosity=False,
        odi_range=(0.01, 0.3),
        diffusivity_config=None,
        compute_dti=True,
        compute_dki=False,
        verbose=False,
    ):
        """
        Generate simulations for matching.

        Parameters
        ----------
        num_simulations : int
            Number of simulated voxels. Default is 100000.
        output_path : str, optional
            Path to save simulations (.npz).
        num_cpus : int
            Number of CPU cores for parallel processing.
        wm_threshold : float
            Minimum WM fraction to include fiber labels.
        tortuosity : bool
            Use tortuosity constraint for perpendicular diffusivity.
        odi_range : tuple
            (min, max) orientation dispersion index range.
        diffusivity_config : dict, optional
            Custom diffusivity ranges.
        compute_dti : bool
            Compute DTI metrics (FA, MD, RD).
        compute_dki : bool
            Compute DKI metrics (AK, RK, MK, KFA).
        verbose : bool
            Enable progress output.

        Returns
        -------
        self : FORCEModel
            Model with simulations loaded.
        """
        from dipy.sims.force import generate_force_simulations, save_force_simulations

        self.simulations = generate_force_simulations(
            self.gtab,
            num_simulations=num_simulations,
            num_cpus=num_cpus,
            wm_threshold=wm_threshold,
            tortuosity=tortuosity,
            odi_range=odi_range,
            diffusivity_config=diffusivity_config,
            compute_dti=compute_dti,
            compute_dki=compute_dki,
            verbose=verbose,
        )

        if output_path is not None:
            save_force_simulations(self.simulations, output_path)

        self._prepare_library()
        return self

    def _prepare_library(self):
        """Prepare library for matching."""
        signals = self.simulations["signals"]

        # Normalize library signals
        lib_norm = np.linalg.norm(signals, axis=1, keepdims=True)
        lib_norm[lib_norm == 0] = 1.0
        self._signals_norm = np.ascontiguousarray(
            (signals / lib_norm).astype(np.float32)
        )

        # Build index
        self._index = create_signal_index(self._signals_norm)

        # Penalty array
        num_fibers = self.simulations.get(
            "num_fibers", np.zeros(len(signals), dtype=np.float32)
        )
        self._penalty_array = (self.penalty * num_fibers).astype(np.float32)

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
        if self.simulations is None:
            raise RuntimeError(
                "No simulations loaded. Call generate() or provide simulations."
            )

        if self.verbose:
            backend = SignalIndex.get_backend()
            mode = "posterior" if self.use_posterior else "best-match"
            print(f"FORCE fitting: {mode} mode, k={self.n_neighbors}, "
                  f"search backend={backend}")

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
        self.mask = (
            mask if mask is not None else np.ones(data.shape[:-1], dtype=bool)
        )

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
        n_valid = len(valid_idx)

        verbose = self.model.verbose

        # Normalize query signals
        if verbose:
            print(f"Processing {n_valid:,} voxels...")
        valid_data = data_flat[valid_idx]
        norms = np.linalg.norm(valid_data, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        query_norm = np.ascontiguousarray(valid_data / norms)

        # Initialize output maps
        self._init_output_maps(n_voxels, shape)

        # Process in batches with progress bar
        batch_size = 50000
        n_batches = (n_valid + batch_size - 1) // batch_size

        iterator = range(n_batches)
        if verbose and HAS_TQDM:
            iterator = tqdm(iterator, desc="Matching", unit="batch")

        for batch_idx in iterator:
            start = batch_idx * batch_size
            end = min(start + batch_size, n_valid)
            batch_query = query_norm[start:end]
            batch_valid_idx = valid_idx[start:end]

            # Perform search for this batch
            D, I = self.model._index.search(batch_query, k=self.model.n_neighbors)

            # Apply penalty
            S = D - self.model._penalty_array[I]

            if self.model.use_posterior:
                self._posterior_aggregation(batch_valid_idx, I, S)
            else:
                self._best_match(batch_valid_idx, I, S)

        # Reshape to original shape
        self._reshape_outputs(shape)

    def _init_output_maps(self, n_voxels, shape):
        """Initialize output arrays."""
        d = self.model.simulations

        self._fa = np.zeros(n_voxels, dtype=np.float32)
        self._md = np.zeros(n_voxels, dtype=np.float32)
        self._rd = np.zeros(n_voxels, dtype=np.float32)
        self._wm_fraction = np.zeros(n_voxels, dtype=np.float32)
        self._gm_fraction = np.zeros(n_voxels, dtype=np.float32)
        self._csf_fraction = np.zeros(n_voxels, dtype=np.float32)
        self._num_fibers = np.zeros(n_voxels, dtype=np.float32)
        self._dispersion = np.zeros(n_voxels, dtype=np.float32)
        self._nd = np.zeros(n_voxels, dtype=np.float32)

        if "ufa_wm" in d:
            self._ufa_wm = np.zeros(n_voxels, dtype=np.float32)
            self._ufa_voxel = np.zeros(n_voxels, dtype=np.float32)

    def _best_match(self, valid_idx, I, S):
        """Use best match for each voxel."""
        d = self.model.simulations

        # Find best match after penalty
        best = np.argmax(S, axis=1)
        best_idx = I[np.arange(len(best)), best]

        self._fa[valid_idx] = d["fa"][best_idx]
        self._md[valid_idx] = d["md"][best_idx]
        self._rd[valid_idx] = d["rd"][best_idx]
        self._wm_fraction[valid_idx] = d["wm_fraction"][best_idx]
        self._gm_fraction[valid_idx] = d["gm_fraction"][best_idx]
        self._csf_fraction[valid_idx] = d["csf_fraction"][best_idx]
        self._num_fibers[valid_idx] = d["num_fibers"][best_idx]
        self._dispersion[valid_idx] = d["dispersion"][best_idx]
        self._nd[valid_idx] = d["nd"][best_idx]

        if "ufa_wm" in d:
            self._ufa_wm[valid_idx] = d["ufa_wm"][best_idx]
            self._ufa_voxel[valid_idx] = d["ufa_voxel"][best_idx]

    def _posterior_aggregation(self, valid_idx, I, S):
        """Use posterior averaging."""
        d = self.model.simulations
        w = softmax_stable(self.model.posterior_beta * S, axis=1)

        self._fa[valid_idx] = np.sum(w * d["fa"][I], axis=1)
        self._md[valid_idx] = np.sum(w * d["md"][I], axis=1)
        self._rd[valid_idx] = np.sum(w * d["rd"][I], axis=1)
        self._wm_fraction[valid_idx] = np.sum(w * d["wm_fraction"][I], axis=1)
        self._gm_fraction[valid_idx] = np.sum(w * d["gm_fraction"][I], axis=1)
        self._csf_fraction[valid_idx] = np.sum(w * d["csf_fraction"][I], axis=1)
        self._num_fibers[valid_idx] = np.sum(w * d["num_fibers"][I], axis=1)
        self._dispersion[valid_idx] = np.sum(w * d["dispersion"][I], axis=1)
        self._nd[valid_idx] = np.sum(w * d["nd"][I], axis=1)

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
        self._nd = self._nd.reshape(shape)

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
    def nd(self):
        """Neurite density map."""
        return self._nd

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
        result += weights[:, kk : kk + 1] * signals[indices[:, kk]]

    return result


def posterior_odf(odfs, weights, indices, n_dirs):
    """
    Compute posterior ODF from neighbors.

    Parameters
    ----------
    odfs : ndarray (N_lib, D)
        Library ODFs.
    weights : ndarray (N_query, K)
        Posterior weights.
    indices : ndarray (N_query, K)
        Neighbor indices.
    n_dirs : int
        Number of sphere directions.

    Returns
    -------
    odf : ndarray (N_query, D)
        Posterior mean ODFs.
    """
    n_query = indices.shape[0]
    k = indices.shape[1]

    result = np.zeros((n_query, n_dirs), dtype=np.float32)
    for kk in range(k):
        odf_k = odfs[indices[:, kk]].astype(np.float32)
        odf_k /= np.max(odf_k, axis=1, keepdims=True) + 1e-12
        result += weights[:, kk : kk + 1] * odf_k

    result /= np.max(result, axis=1, keepdims=True) + 1e-12
    return result
