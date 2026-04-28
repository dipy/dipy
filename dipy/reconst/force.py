import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np

from dipy.core.gradients import unique_bvals_tolerance
from dipy.direction.peaks import PeaksAndMetrics
from dipy.reconst._force_search import search_inner_product as _cython_search
from dipy.reconst.base import ReconstFit, ReconstModel
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.shm import sf_to_sh
from dipy.sims.force import (
    generate_force_simulations,
    get_default_diffusivity_config,
    load_force_simulations,
    save_force_simulations,
)
from dipy.utils.logging import logger

# Named constants
EPSILON = 1e-12


def _get_force_cache_dir():
    """Return the FORCE simulation cache directory inside .dipy.

    Uses ``DIPY_HOME`` environment variable if set, otherwise defaults
    to ``~/.dipy/force_simulations``.

    Returns
    -------
    cache_dir : Path
        Path to the cache directory (created if it does not exist).
    """
    if "DIPY_HOME" in os.environ:
        dipy_home = Path(os.environ["DIPY_HOME"])
    else:
        dipy_home = Path("~").expanduser() / ".dipy"
    cache_dir = dipy_home / "force_simulations"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _gtab_matches(entry_bvals, entry_bvecs, gtab, *, bval_tol=10.0, bvec_tol=1e-3):
    """Check whether stored bvals/bvecs match a GradientTable.

    Parameters
    ----------
    entry_bvals : list
        Stored b-values from the cache registry.
    entry_bvecs : list of list
        Stored b-vectors from the cache registry.
    gtab : GradientTable
        Gradient table to compare against.
    bval_tol : float, optional
        Absolute tolerance for b-value comparison.
    bvec_tol : float, optional
        Absolute tolerance for b-vector coordinate comparison.

    Returns
    -------
    match : bool
        True if the stored and passed bvals/bvecs agree within tolerance.
    """
    stored_bvals = np.asarray(entry_bvals, dtype=np.float64)
    stored_bvecs = np.asarray(entry_bvecs, dtype=np.float64)
    current_bvals = np.asarray(gtab.bvals, dtype=np.float64)
    current_bvecs = np.asarray(gtab.bvecs, dtype=np.float64)

    if stored_bvals.shape != current_bvals.shape:
        return False
    if stored_bvecs.shape != current_bvecs.shape:
        return False

    return np.allclose(stored_bvals, current_bvals, atol=bval_tol) and np.allclose(
        stored_bvecs, current_bvecs, atol=bvec_tol
    )


def _diffusivity_matches(entry_config, current_config):
    """Check whether two diffusivity configurations are equivalent.

    Parameters
    ----------
    entry_config : dict
        Stored diffusivity configuration.
    current_config : dict
        Current diffusivity configuration.

    Returns
    -------
    match : bool
        True if all keys and values are identical.
    """
    if set(entry_config.keys()) != set(current_config.keys()):
        return False
    for key in entry_config:
        stored = entry_config[key]
        current = current_config[key]
        # Both may be lists/tuples (ranges) or scalars
        if isinstance(stored, (list, tuple)):
            if not isinstance(current, (list, tuple)):
                return False
            if len(stored) != len(current):
                return False
            if not all(np.isclose(s, c) for s, c in zip(stored, current)):
                return False
        else:
            if not np.isclose(stored, current):
                return False
    return True


def _load_cache_registry(cache_dir):
    """Load the cache registry JSON from *cache_dir*.

    Returns an empty list if the file does not exist yet.
    """
    registry_path = cache_dir / "cache_registry.json"
    if registry_path.exists():
        with open(registry_path, "r") as f:
            return json.load(f)
    return []


def _save_cache_registry(cache_dir, registry):
    """Persist *registry* as JSON in *cache_dir*."""
    registry_path = cache_dir / "cache_registry.json"
    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2)


def _locked_registry_update(cache_dir, update_fn):
    """Read-modify-write the cache registry under an exclusive file lock.

    Parameters
    ----------
    cache_dir : Path
        Cache directory.
    update_fn : callable
        Function that receives the current registry list and returns the
        updated list.
    """
    lock_path = cache_dir / "cache_registry.lock"
    with open(lock_path, "w") as lock_fh:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(lock_fh.fileno(), msvcrt.LK_LOCK, 1)
            try:
                registry = _load_cache_registry(cache_dir)
                registry = update_fn(registry)
                _save_cache_registry(cache_dir, registry)
            finally:
                lock_fh.seek(0)
                msvcrt.locking(lock_fh.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_fh, fcntl.LOCK_EX)
            try:
                registry = _load_cache_registry(cache_dir)
                registry = update_fn(registry)
                _save_cache_registry(cache_dir, registry)
            finally:
                fcntl.flock(lock_fh, fcntl.LOCK_UN)


def _find_cached_simulation(cache_dir, gtab, diffusivity_config, num_simulations):
    """Search the registry for a simulation matching the given parameters.

    Parameters
    ----------
    cache_dir : Path
        Cache directory.
    gtab : GradientTable
        Gradient table.
    diffusivity_config : dict
        Diffusivity ranges used for generation.
    num_simulations : int
        Number of simulations requested.

    Returns
    -------
    path : str or None
        Path to the cached ``.npz`` file, or None if no match found.
    """
    registry = _load_cache_registry(cache_dir)
    for entry in registry:
        if entry["num_simulations"] != num_simulations:
            continue
        if not _gtab_matches(entry["bvals"], entry["bvecs"], gtab):
            continue
        if not _diffusivity_matches(entry["diffusivity_config"], diffusivity_config):
            continue
        candidate = cache_dir / entry["filename"]
        if candidate.exists():
            return str(candidate)
    return None


def _register_cached_simulation(
    cache_dir, gtab, diffusivity_config, num_simulations, filename
):
    """Add a new entry to the cache registry.

    Parameters
    ----------
    cache_dir : Path
        Cache directory.
    gtab : GradientTable
        Gradient table.
    diffusivity_config : dict
        Diffusivity ranges.
    num_simulations : int
        Number of simulations.
    filename : str
        Filename of the saved ``.npz`` inside *cache_dir*.
    """

    # Convert numpy types to plain Python for JSON serialisation
    def _to_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, tuple):
            return list(obj)
        return obj

    config_json = {}
    for k, v in diffusivity_config.items():
        config_json[k] = _to_json(v)

    entry = {
        "bvals": np.asarray(gtab.bvals, dtype=np.float64).tolist(),
        "bvecs": np.asarray(gtab.bvecs, dtype=np.float64).tolist(),
        "diffusivity_config": config_json,
        "num_simulations": int(num_simulations),
        "filename": filename,
    }

    def _append(registry):
        registry.append(entry)
        return registry

    _locked_registry_update(cache_dir, _append)


class SignalIndex:
    """Index for inner product similarity search.

    Uses optimized Cython BLAS for fast matrix multiplication
    and streaming heap for memory-efficient top-k selection.

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
        """Add vectors to the index.

        Parameters
        ----------
        x : array-like (n, d)
            Vectors to add, will be converted to float32 C-contiguous.

        Notes
        -----
        Each call reallocates the internal array via ``np.vstack``.  This
        method is designed for a single bulk load; repeated small ``add``
        calls will exhibit O(n²) memory allocation cost.
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
        """Search for k nearest neighbors by inner product.

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

        if k > self.ntotal:
            warnings.warn(
                f"k={k} exceeds index size ({self.ntotal}); "
                f"clamping to {self.ntotal}.",
                UserWarning,
                stacklevel=2,
            )
            k = self.ntotal

        # Use optimized Cython search (SciPy BLAS + streaming heap)
        distances, indices = _cython_search(x, self._xb, k)

        return distances, indices

    def reset(self):
        """Remove all vectors from the index."""
        self._xb = None
        self.ntotal = 0

    def __repr__(self):
        return f"SignalIndex(d={self.d}, ntotal={self.ntotal})"


def normalize_signals(signals):
    """L2-normalize signal array for cosine similarity search.

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
    """Create index for cosine similarity search.

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


def softmax_stable(x, *, axis=1):
    """Numerically stable softmax.

    Parameters
    ----------
    x : ndarray
        Input array.
    axis : int, optional
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
    """Compute uncertainty and ambiguity metrics from match scores.

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

    s_max = np.max(scores, axis=1)
    s_min = np.min(scores, axis=1)
    half = 0.5 * (s_max + s_min)
    ambiguity = (np.sum(scores > half[:, None], axis=1) / scores.shape[1]).astype(
        np.float32
    )
    return uncertainty, ambiguity


MICRO_PARAMS = (
    "fa",
    "md",
    "rd",
    "wm_fraction",
    "gm_fraction",
    "csf_fraction",
    "num_fibers",
    "dispersion",
    "nd",
    "ufa_voxel",
    "ak",
    "rk",
    "mk",
    "kfa",
)


def _weighted_percentile(vals, weights, q):
    """Batch weighted percentile.

    Parameters
    ----------
    vals : ndarray (N, K)
        Parameter values across K neighbors per voxel.
    weights : ndarray (N, K)
        Posterior weights.
    q : float
        Quantile in [0, 1].

    Returns
    -------
    result : ndarray (N,)
        Weighted percentile for each voxel.
    """
    idx = np.argsort(vals, axis=1)
    sv = np.take_along_axis(vals, idx, axis=1)
    sw = np.take_along_axis(weights, idx, axis=1)
    cw = np.cumsum(sw, axis=1)
    cw = cw / (cw[:, -1:] + EPSILON)
    j = np.argmax(cw >= q, axis=1)
    return sv[np.arange(sv.shape[0]), j]


def _fwhm_kde_batch(vals, weights, *, n_grid=100, batch_size=2000):
    """Batch FWHM via weighted KDE.

    Estimates the full width at half maximum of the weighted posterior
    density for each voxel using a Gaussian kernel density estimate.

    Parameters
    ----------
    vals : ndarray (N, K)
        Parameter values across K neighbors.
    weights : ndarray (N, K)
        Posterior weights.
    n_grid : int, optional
        Number of grid points for KDE evaluation.
    batch_size : int, optional
        Internal batch size to limit memory usage.

    Returns
    -------
    fwhm : ndarray (N,)
        FWHM in parameter units.
    """
    N, K = vals.shape
    fwhm = np.empty(N, dtype=np.float32)

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        vals_batch = vals[start:end].astype(np.float32)
        weights_batch = weights[start:end].astype(np.float32)
        n = end - start

        lo = vals_batch.min(axis=1)
        hi = vals_batch.max(axis=1)
        span = hi - lo
        pad = 0.15 * (span + EPSILON)
        lo_p, hi_p = lo - pad, hi + pad
        span_p = hi_p - lo_p

        t = np.linspace(0, 1, n_grid, dtype=np.float32)
        grid = lo_p[:, None] + t[None, :] * span_p[:, None]

        bw = span_p / max(K**0.5, 4.0)
        bw = np.maximum(bw, EPSILON)

        diff = (grid[:, :, None] - vals_batch[:, None, :]) / bw[:, None, None]
        density = np.sum(np.exp(-0.5 * diff**2) * weights_batch[:, None, :], axis=2)

        peak = density.max(axis=1)
        half_peak = 0.5 * peak
        above = density >= half_peak[:, None]

        any_above = np.any(above, axis=1)
        first = np.argmax(above, axis=1)
        last = n_grid - 1 - np.argmax(above[:, ::-1], axis=1)

        f = grid[np.arange(n), last] - grid[np.arange(n), first]
        f[~any_above] = 0.0
        fwhm[start:end] = f

    return fwhm


def compute_microstructure_uncertainty_ambiguity(vals, weights, prior_range):
    """Compute per-microstructure uncertainty and ambiguity.

    Uncertainty is the weighted interquartile range of the posterior density,
    normalized by the prior range. Ambiguity is the full width at half
    maximum (FWHM) of the weighted posterior density, similarly normalized.
    Both are in [0, 1] where 0 means no spread and 1 means the posterior
    spans the entire prior range.

    Parameters
    ----------
    vals : ndarray (N, K)
        Parameter values for K neighbors per voxel.
    weights : ndarray (N, K)
        Posterior weights (softmax).
    prior_range : float
        Max minus min of this parameter across the simulation library.

    Returns
    -------
    uncertainty : ndarray (N,)
        Weighted IQR / prior_range (fraction in [0, 1]).
    ambiguity : ndarray (N,)
        FWHM / prior_range (fraction in [0, 1]).
    """
    q75 = _weighted_percentile(vals, weights, 0.75)
    q25 = _weighted_percentile(vals, weights, 0.25)
    uncertainty = ((q75 - q25) / (prior_range + EPSILON)).astype(np.float32)

    fwhm = _fwhm_kde_batch(vals, weights)
    ambiguity = (fwhm / (prior_range + EPSILON)).astype(np.float32)

    return uncertainty, ambiguity


def postprocess_peaks(preds, target_sphere, fracs):
    original_shape = preds.shape[:-1]
    preds_flat = preds.reshape(-1, preds.shape[-1])
    fracs_flat = fracs.reshape(-1, fracs.shape[-1])

    n_voxels = preds_flat.shape[0]
    vertices = target_sphere.vertices

    # Initialize outputs using the total number of voxels
    peaks_output = np.zeros((n_voxels, 5, 3), dtype=np.float32)
    peak_indices = np.full((n_voxels, 5), -1, dtype=np.int32)
    peak_vals = np.zeros((n_voxels, 5), dtype=np.float32)

    for i in range(n_voxels):
        mask = preds_flat[i] == 1
        coords = vertices[mask]
        indices = np.where(mask)[0]

        num = min(len(coords), 5)
        if num > 0:
            peaks_output[i, :num] = coords[:num]
            peak_indices[i, :num] = indices[:num]

        num_fracs = min(5, fracs_flat[i].shape[0])
        peak_vals[i, :num_fracs] = fracs_flat[i][:num_fracs]

    peaks_output = peaks_output.reshape((*original_shape, 5, 3))
    peak_indices = peak_indices.reshape((*original_shape, 5))
    peak_vals = peak_vals.reshape((*original_shape, 5))

    return peaks_output, peak_indices, peak_vals


class FORCEModel(ReconstModel):
    """FORCE reconstruction model for signal matching based microstructure."""

    def __init__(
        self,
        gtab,
        *,
        simulations=None,
        penalty=1e-5,
        n_neighbors=50,
        use_posterior=False,
        posterior_beta=2000.0,
        compute_odf=False,
        verbose=False,
    ):
        r"""
        FORCE (FORward modeling for Complex microstructure Estimation) model
        :footcite:p:`Shah2025`.

        FORCE is a forward modeling paradigm that reframes how diffusion data
        is analyzed. Instead of inverting the measured signal, FORCE simulates
        a large set of biologically plausible intra-voxel fiber configurations
        and tissue compositions. It then identifies the best-matching simulation
        for each voxel by operating directly in the signal space.

        Parameters
        ----------
        gtab : GradientTable
            Gradient table.
        simulations : dict or None, optional
            Pre-computed FORCE simulations with signals and parameters.
            If None, call generate() to create simulations.
        penalty : float, optional
            Penalty weight for fiber complexity.
        n_neighbors : int, optional
            Number of neighbors for matching.
        use_posterior : bool, optional
            Use posterior averaging instead of best match.
        posterior_beta : float, optional
            Softmax temperature for posterior.
        compute_odf : bool, optional
            Compute posterior ODF maps.
        verbose : bool, optional
            Show progress bar and status messages.

        Notes
        -----
        The fit method uses the @multi_voxel_fit decorator which supports
        parallel execution. Pass `engine` and `n_jobs` kwargs to the fit method:

        Available engines: "serial", "ray", "joblib", "dask".

        References
        ----------
        .. footbibliography::
        """
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

        if simulations is not None:
            self._prepare_library()

    def generate(
        self,
        *,
        num_simulations=500000,
        output_path=None,
        num_cpus=1,
        wm_threshold=0.5,
        tortuosity=False,
        odi_range=(0.01, 0.3),
        diffusivity_config=None,
        compute_dti=True,
        compute_dki=False,
        verbose=False,
        use_cache=True,
    ):
        """Generate simulations for matching.

        When ``output_path`` is ``None`` and ``use_cache`` is ``True``,
        simulations are cached in ``~/.dipy/force_simulations/`` (or
        ``$DIPY_HOME``). A registry file (``cache_registry.json``) keeps
        track of the bvals, bvecs, diffusivity configuration and number of
        simulations for each cached file. If a cached simulation that matches
        the current gradient table (within tolerance) and diffusivity
        configuration already exists, it is loaded from disk and generation
        is skipped.

        Set ``use_cache=False`` to force regeneration even when a matching
        cached simulation exists.

        Parameters
        ----------
        num_simulations : int, optional
            Number of simulated voxels.
        output_path : str, optional
            Path to save simulations (.npz). When None, saves to
            ``~/.dipy/force_simulations/`` and uses caching.
        num_cpus : int, optional
            Number of CPU cores for parallel processing.
        wm_threshold : float, optional
            Minimum WM fraction to include fiber labels.
        tortuosity : bool, optional
            Use tortuosity constraint for perpendicular diffusivity.
        odi_range : tuple, optional
            (min, max) orientation dispersion index range.
        diffusivity_config : dict, optional
            Custom diffusivity ranges.
        compute_dti : bool, optional
            Compute DTI metrics (FA, MD, RD).
        compute_dki : bool, optional
            Compute DKI metrics (AK, RK, MK, KFA).
        verbose : bool, optional
            Enable progress output.
        use_cache : bool, optional
            Whether to use cached simulations when ``output_path`` is
            None. Set to ``False`` to always regenerate.

        Returns
        -------
        self : FORCEModel
            Model with simulations loaded.
        """

        b0_thr = getattr(self.gtab, "b0_threshold", 50)
        unique_bvals = unique_bvals_tolerance(self.gtab.bvals, tol=50)
        n_nonzero_shells = int(np.sum(unique_bvals > b0_thr))

        if compute_dki and n_nonzero_shells < 2:
            warnings.warn(
                f"compute_dki=True but only {n_nonzero_shells} non-zero "
                "b-value shell found (need at least 2). "
                "Disabling DKI computation.",
                UserWarning,
                stacklevel=2,
            )
            compute_dki = False
        elif not compute_dki and n_nonzero_shells >= 2:
            logger.info(
                f"Found {n_nonzero_shells} non-zero b-value shells. "
                "You can compute DKI metrics (AK, RK, MK, KFA) by "
                "setting compute_dki=True."
            )

        # Resolve the diffusivity config that will actually be used
        resolved_config = (
            diffusivity_config
            if diffusivity_config is not None
            else get_default_diffusivity_config()
        )

        # --- Cache logic when no explicit output_path is given ----------
        if output_path is None and use_cache:
            cache_dir = _get_force_cache_dir()
            cached = _find_cached_simulation(
                cache_dir,
                self.gtab,
                resolved_config,
                num_simulations,
            )
            if cached is not None:
                if verbose:
                    print(f"[FORCE] Loading cached simulations from {cached}")
                self.simulations = load_force_simulations(cached)
                self._prepare_library()
                return self

        # --- Generate new simulations -----------------------------------
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
        else:
            # Save into the .dipy cache and register.
            # filename is generated inside the lock to avoid races between
            # concurrent processes reading the same registry length.
            cache_dir = _get_force_cache_dir()
            filename_holder = {}

            def _append_and_name(registry):
                idx = len(registry)
                fname = f"force_sim_{idx}.npz"
                filename_holder["filename"] = fname
                return registry  # entry added by _register_cached_simulation

            _locked_registry_update(cache_dir, _append_and_name)
            filename = filename_holder["filename"]
            save_force_simulations(self.simulations, str(cache_dir / filename))
            _register_cached_simulation(
                cache_dir,
                self.gtab,
                resolved_config,
                num_simulations,
                filename,
            )
            if verbose:
                print(f"[FORCE] Cached simulations to {cache_dir / filename}")

        self._prepare_library()
        return self

    def load(self, input_path):
        """Load pre-computed simulations from file.

        Parameters
        ----------
        input_path : str
            Path to simulations file (.npz).

        Returns
        -------
        self : FORCEModel
            Model with simulations loaded.

        """
        self.simulations = load_force_simulations(input_path)
        self._prepare_library()
        return self

    def _prepare_library(self):
        """Prepare library for matching."""
        signals = self.simulations["signals"]

        # Normalize library signals
        lib_norm = np.linalg.norm(signals, axis=1, keepdims=True)
        lib_norm[lib_norm == 0] = 1.0
        signals_norm = np.ascontiguousarray((signals / lib_norm).astype(np.float32))

        # Build index
        self._index = create_signal_index(signals_norm)

        # Penalty array
        num_fibers = self.simulations.get(
            "num_fibers", np.zeros(len(signals), dtype=np.float32)
        )
        self._penalty_array = (self.penalty * num_fibers).astype(np.float32)

        # Prior ranges for microstructure uncertainty normalization
        d = self.simulations
        self._prior_ranges = {}
        for param in MICRO_PARAMS:
            if param in d:
                arr = d[param]
                self._prior_ranges[param] = float(arr.max() - arr.min())

    @staticmethod
    def _fetch_params_batched(lib_idx, d):
        """Vectorised parameter look-up for best-match indices.

        Parameters
        ----------
        lib_idx : ndarray (N,)
            Library indices of the best match per voxel.
        d : dict
            Simulation dictionary.

        Returns
        -------
        params : dict of ndarray
        """
        params = {
            "fa": d["fa"][lib_idx].astype(np.float32),
            "md": d["md"][lib_idx].astype(np.float32),
            "rd": d["rd"][lib_idx].astype(np.float32),
            "wm_fraction": d["wm_fraction"][lib_idx].astype(np.float32),
            "gm_fraction": d["gm_fraction"][lib_idx].astype(np.float32),
            "csf_fraction": d["csf_fraction"][lib_idx].astype(np.float32),
            "num_fibers": d["num_fibers"][lib_idx].astype(np.float32),
            "dispersion": d["dispersion"][lib_idx].astype(np.float32),
            "nd": d["nd"][lib_idx].astype(np.float32),
            "labels": d["labels"][lib_idx].astype(np.int8),
            "fracs": d["fraction_array"][lib_idx].astype(np.float32),
        }
        if "ufa_wm" in d:
            params["ufa_wm"] = d["ufa_wm"][lib_idx].astype(np.float32)
            params["ufa_voxel"] = d["ufa_voxel"][lib_idx].astype(np.float32)
        if "ak" in d:
            params["ak"] = d["ak"][lib_idx].astype(np.float32)
            params["rk"] = d["rk"][lib_idx].astype(np.float32)
            params["mk"] = d["mk"][lib_idx].astype(np.float32)
            params["kfa"] = d["kfa"][lib_idx].astype(np.float32)
        params["odf"] = d["odfs"][lib_idx].astype(np.float32) if "odfs" in d else None
        params["predicted_signal"] = d["signals"][lib_idx].astype(np.float32)
        return params

    @staticmethod
    def _posterior_params_batched(neighbors, W, d, lib_idx):
        """Vectorised posterior-averaging over neighbours.

        Parameters
        ----------
        neighbors : ndarray (N, K)
            Neighbour indices.
        W : ndarray (N, K)
            Posterior weights.
        d : dict
            Simulation dictionary.
        lib_idx : int
            Index of the exact match in the simulation

        Returns
        -------
        params : dict of ndarray
        """

        def _wavg(field):
            return np.sum(W * d[field][neighbors], axis=1).astype(np.float32)

        params = {
            "fa": _wavg("fa"),
            "md": _wavg("md"),
            "rd": _wavg("rd"),
            "wm_fraction": _wavg("wm_fraction"),
            "gm_fraction": _wavg("gm_fraction"),
            "csf_fraction": _wavg("csf_fraction"),
            "num_fibers": _wavg("num_fibers"),
            "dispersion": _wavg("dispersion"),
            "nd": _wavg("nd"),
            "labels": d["labels"][lib_idx].astype(np.int8),
            "fracs": d["fraction_array"][lib_idx].astype(np.float32),
        }
        if "ufa_wm" in d:
            params["ufa_wm"] = _wavg("ufa_wm")
            params["ufa_voxel"] = _wavg("ufa_voxel")
        if "ak" in d:
            params["ak"] = _wavg("ak")
            params["rk"] = _wavg("rk")
            params["mk"] = _wavg("mk")
            params["kfa"] = _wavg("kfa")

        # Posterior ODF
        if "odfs" in d:
            K = neighbors.shape[1]
            odf = np.zeros((neighbors.shape[0], d["odfs"].shape[1]), dtype=np.float32)
            for kk in range(K):
                odf_k = d["odfs"][neighbors[:, kk]].astype(np.float32)
                odf_k /= np.max(odf_k, axis=1, keepdims=True) + EPSILON
                odf += W[:, kk : kk + 1] * odf_k
            odf /= np.max(odf, axis=1, keepdims=True) + EPSILON
            params["odf"] = odf
        else:
            params["odf"] = None

        # Posterior mean signal
        params["predicted_signal"] = posterior_mean_signal(d["signals"], W, neighbors)
        return params

    @multi_voxel_fit(
        batched=True,
        shared_obj=("_penalty_array", "_index", "simulations"),
        chunk_size={"serial": 10_000, "ray": "auto"},
    )
    def fit(self, data, *, mask=None, **kwargs):
        """Fit model to data.

        Parameters
        ----------
        data : ndarray
            Diffusion data for a single voxel (1D) or multiple voxels (ND).
        mask : ndarray, optional
            Brain mask (for multi-voxel data).
        **kwargs : dict
            Additional arguments passed to multi_voxel_fit decorator:
            - engine : str, optional
                Parallel engine: "serial", "ray", "joblib", "dask".
            - n_jobs : int, optional
                Number of parallel jobs.
            - verbose : bool, optional
                Show progress bar.

        Returns
        -------
        fit : FORCEFit or ndarray of FORCEFit
            Fitted model for a single voxel (1-D input) or an object array
            of fitted models for a batch of voxels (2-D input).

        Notes
        -----
        This method is decorated with @multi_voxel_fit(batched=True)
        which handles multi-voxel dispatch, mask application, and aggregation
        into a MultiVoxelFit.  The method itself handles both 1-D
        (single voxel) and 2-D (batch) inputs directly.

        For parallel execution, use engine="ray", n_jobs=4 arguments in model
        fit() call.


        **Memory warning (joblib / dask engines):** When engine="joblib"
        or engine="dask", the full simulation library (including the
        signal matrix and search index, ~120-400 MB for 100k simulations) is
        pickled and sent to *every* worker chunk.  With 8 workers this can
        consume several gigabytes of RAM.  For num_simulations > ~10 000
        use engine="ray" instead, which places the library in a shared
        object store and avoids redundant copies across workers.
        """
        if self.simulations is None:
            raise RuntimeError(
                "No simulations loaded. Call generate() or provide simulations."
            )
        if self._index is None:
            raise RuntimeError(
                "Signal index is not prepared. Call _prepare_library() or "
                "reload simulations via generate() or load()."
            )

        single = data.ndim == 1
        data2d = data.reshape(1, -1) if single else data
        data2d = np.ascontiguousarray(data2d, dtype=np.float32)

        norms = np.linalg.norm(data2d, axis=1, keepdims=True).astype(np.float32)
        norms[norms == 0] = 1.0
        query_norm = np.ascontiguousarray(data2d / norms)

        D, neighbors = self._index.search(query_norm, k=self.n_neighbors)
        S = D - self._penalty_array[neighbors]

        U, A = compute_uncertainty_ambiguity(S)

        d = self.simulations
        n_vox = data2d.shape[0]
        best = np.argmax(S, axis=1)
        lib_idx = neighbors[np.arange(n_vox), best]

        W = softmax_stable(self.posterior_beta * S, axis=1)

        if self.use_posterior:
            entropy = -np.sum(W * np.log(W + EPSILON), axis=1)

            params_arrays = self._posterior_params_batched(neighbors, W, d, lib_idx)
            params_arrays["uncertainty"] = U
            params_arrays["ambiguity"] = A
            params_arrays["entropy"] = entropy.astype(np.float32)
        else:
            params_arrays = self._fetch_params_batched(lib_idx, d)
            params_arrays["uncertainty"] = U
            params_arrays["ambiguity"] = A
            params_arrays["entropy"] = np.full(n_vox, np.nan, dtype=np.float32)

        # Per-microstructure uncertainty and ambiguity from posterior density
        prior_ranges = getattr(self, "_prior_ranges", {})
        for param in MICRO_PARAMS:
            if param in d and param in prior_ranges:
                vals = d[param][neighbors].astype(np.float32)
                u, a = compute_microstructure_uncertainty_ambiguity(
                    vals, W, prior_ranges[param]
                )
                params_arrays[f"uncertainty_{param}"] = u
                params_arrays[f"ambiguity_{param}"] = a

        if kwargs.pop("_raw", False):
            return params_arrays

        fits = np.empty(n_vox, dtype=object)
        keys = list(params_arrays.keys())
        for i in range(n_vox):
            p = {}
            for k in keys:
                val = params_arrays[k]
                if val is None:
                    p[k] = None
                else:
                    v = val[i]
                    if isinstance(v, np.ndarray) and v.ndim == 0:
                        p[k] = float(v)
                    elif isinstance(v, (np.floating, np.integer)):
                        p[k] = float(v)
                    else:
                        p[k] = v
            entropy_val = p.get("entropy", 0.0)
            if entropy_val is not None and np.isnan(entropy_val):
                p["entropy"] = None
            fits[i] = FORCEFit(None, p)

        return fits[0] if single else fits


class FORCEFit(ReconstFit):
    """FORCE model fit results for a single voxel."""

    def __init__(self, model, params):
        """Initialize a FORCEFit class instance."""
        if (
            "entropy" in params
            and params["entropy"] is not None
            and np.isnan(params["entropy"])
        ):
            params["entropy"] = None
        self.model = model
        self._params = params

    @property
    def fa(self):
        """Fractional anisotropy."""
        return self._params["fa"]

    @property
    def md(self):
        """Mean diffusivity."""
        return self._params["md"]

    @property
    def rd(self):
        """Radial diffusivity."""
        return self._params["rd"]

    @property
    def wm_fraction(self):
        """White matter fraction."""
        return self._params["wm_fraction"]

    @property
    def gm_fraction(self):
        """Gray matter fraction."""
        return self._params["gm_fraction"]

    @property
    def csf_fraction(self):
        """CSF fraction."""
        return self._params["csf_fraction"]

    @property
    def num_fibers(self):
        """Number of fibers."""
        return self._params["num_fibers"]

    @property
    def dispersion(self):
        """Orientation dispersion."""
        return self._params["dispersion"]

    @property
    def nd(self):
        """Neurite density."""
        return self._params["nd"]

    @property
    def ufa_wm(self):
        """microFA in white matter."""
        return self._params.get("ufa_wm", None)

    @property
    def ufa_voxel(self):
        """Voxel-averaged microFA."""
        return self._params.get("ufa_voxel", None)

    @property
    def ak(self):
        """Axial kurtosis (DKI)."""
        return self._params.get("ak", None)

    @property
    def rk(self):
        """Radial kurtosis (DKI)."""
        return self._params.get("rk", None)

    @property
    def mk(self):
        """Mean kurtosis (DKI)."""
        return self._params.get("mk", None)

    @property
    def kfa(self):
        """Kurtosis fractional anisotropy (DKI)."""
        return self._params.get("kfa", None)

    @property
    def odf(self):
        """Orientation distribution function."""
        return self._params.get("odf", None)

    @property
    def predicted_signal(self):
        """Predicted signal from matched library entry (cleaned DWI)."""
        return self._params.get("predicted_signal", None)

    @property
    def uncertainty(self):
        """Uncertainty (IQR of penalized scores)."""
        return self._params["uncertainty"]

    @property
    def ambiguity(self):
        """Ambiguity (fraction above half-max)."""
        return self._params["ambiguity"]

    @property
    def entropy(self):
        """Entropy (posterior mode only)."""
        return self._params.get("entropy", None)

    @property
    def label(self):
        """Fiber configuration label."""
        return self._params.get("labels", None)

    @property
    def fracs(self):
        """Fiber fractions."""
        return self._params.get("fracs", None)

    @property
    def uncertainty_fa(self):
        """Per-microstructure uncertainty for FA (fraction of prior range)."""
        return self._params.get("uncertainty_fa", None)

    @property
    def ambiguity_fa(self):
        """Per-microstructure ambiguity for FA (fraction of prior range)."""
        return self._params.get("ambiguity_fa", None)

    @property
    def uncertainty_md(self):
        """Per-microstructure uncertainty for MD (fraction of prior range)."""
        return self._params.get("uncertainty_md", None)

    @property
    def ambiguity_md(self):
        """Per-microstructure ambiguity for MD (fraction of prior range)."""
        return self._params.get("ambiguity_md", None)

    @property
    def uncertainty_rd(self):
        """Per-microstructure uncertainty for RD (fraction of prior range)."""
        return self._params.get("uncertainty_rd", None)

    @property
    def ambiguity_rd(self):
        """Per-microstructure ambiguity for RD (fraction of prior range)."""
        return self._params.get("ambiguity_rd", None)

    @property
    def uncertainty_wm_fraction(self):
        """Per-microstructure uncertainty for WM fraction (fraction of prior range)."""
        return self._params.get("uncertainty_wm_fraction", None)

    @property
    def ambiguity_wm_fraction(self):
        """Per-microstructure ambiguity for WM fraction (fraction of prior range)."""
        return self._params.get("ambiguity_wm_fraction", None)

    @property
    def uncertainty_gm_fraction(self):
        """Per-microstructure uncertainty for GM fraction (fraction of prior range)."""
        return self._params.get("uncertainty_gm_fraction", None)

    @property
    def ambiguity_gm_fraction(self):
        """Per-microstructure ambiguity for GM fraction (fraction of prior range)."""
        return self._params.get("ambiguity_gm_fraction", None)

    @property
    def uncertainty_csf_fraction(self):
        """Per-microstructure uncertainty for CSF fraction (fraction of prior range)."""
        return self._params.get("uncertainty_csf_fraction", None)

    @property
    def ambiguity_csf_fraction(self):
        """Per-microstructure ambiguity for CSF fraction (fraction of prior range)."""
        return self._params.get("ambiguity_csf_fraction", None)

    @property
    def uncertainty_num_fibers(self):
        """Per-microstructure uncertainty for num_fibers (fraction of prior range)."""
        return self._params.get("uncertainty_num_fibers", None)

    @property
    def ambiguity_num_fibers(self):
        """Per-microstructure ambiguity for num_fibers (fraction of prior range)."""
        return self._params.get("ambiguity_num_fibers", None)

    @property
    def uncertainty_dispersion(self):
        """Per-microstructure uncertainty for dispersion (fraction of prior range)."""
        return self._params.get("uncertainty_dispersion", None)

    @property
    def ambiguity_dispersion(self):
        """Per-microstructure ambiguity for dispersion (fraction of prior range)."""
        return self._params.get("ambiguity_dispersion", None)

    @property
    def uncertainty_nd(self):
        """Per-microstructure uncertainty for ND (fraction of prior range)."""
        return self._params.get("uncertainty_nd", None)

    @property
    def ambiguity_nd(self):
        """Per-microstructure ambiguity for ND (fraction of prior range)."""
        return self._params.get("ambiguity_nd", None)

    @property
    def uncertainty_ufa_voxel(self):
        """Uncertainty for microFA voxel (fraction of prior range)."""
        return self._params.get("uncertainty_ufa_voxel", None)

    @property
    def ambiguity_ufa_voxel(self):
        """Per-microstructure ambiguity for microFA voxel (fraction of prior range)."""
        return self._params.get("ambiguity_ufa_voxel", None)

    @property
    def uncertainty_ak(self):
        """Per-microstructure uncertainty for AK (fraction of prior range)."""
        return self._params.get("uncertainty_ak", None)

    @property
    def ambiguity_ak(self):
        """Per-microstructure ambiguity for AK (fraction of prior range)."""
        return self._params.get("ambiguity_ak", None)

    @property
    def uncertainty_rk(self):
        """Per-microstructure uncertainty for RK (fraction of prior range)."""
        return self._params.get("uncertainty_rk", None)

    @property
    def ambiguity_rk(self):
        """Per-microstructure ambiguity for RK (fraction of prior range)."""
        return self._params.get("ambiguity_rk", None)

    @property
    def uncertainty_mk(self):
        """Per-microstructure uncertainty for MK (fraction of prior range)."""
        return self._params.get("uncertainty_mk", None)

    @property
    def ambiguity_mk(self):
        """Per-microstructure ambiguity for MK (fraction of prior range)."""
        return self._params.get("ambiguity_mk", None)

    @property
    def uncertainty_kfa(self):
        """Per-microstructure uncertainty for KFA (fraction of prior range)."""
        return self._params.get("uncertainty_kfa", None)

    @property
    def ambiguity_kfa(self):
        """Per-microstructure ambiguity for KFA (fraction of prior range)."""
        return self._params.get("ambiguity_kfa", None)


# Resolve forward reference: FORCEModel is defined before FORCEFit.
FORCEModel._fit_class = FORCEFit


def compute_entropy(weights):
    """Compute entropy of posterior weights.

    Parameters
    ----------
    weights : ndarray (N, K)
        Posterior weights for K neighbors.

    Returns
    -------
    entropy : ndarray (N,)
        Shannon entropy for each sample.
    """
    return (-np.sum(weights * np.log(weights + EPSILON), axis=1)).astype(np.float32)


def posterior_mean_signal(signals, weights, indices):
    """Compute posterior mean signal from neighbors.

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
    """Compute posterior ODF from neighbors.

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
        odf_k /= np.max(odf_k, axis=1, keepdims=True) + EPSILON
        result += weights[:, kk : kk + 1] * odf_k

    result /= np.max(result, axis=1, keepdims=True) + EPSILON
    return result


def force_peaks(fitted_object, *, mask=None, sh_order=8):
    """Create a PeaksAndMetrics object from a FORCEFit or MultiVoxelFit.

    Parameters
    ----------
    fitted_object : FORCEFit or MultiVoxelFit
        The result of model.fit().
    mask : ndarray, optional
        Optional brain mask.
    sh_order : int, optional
        Spherical harmonics order for the coefficients.
    """
    from dipy.sims.force import default_sphere

    labels = fitted_object.label
    fracs = fitted_object.fracs
    odf = fitted_object.odf

    is_multi_voxel = labels.ndim > 1

    if not is_multi_voxel:
        # Single Voxel Case
        p_out, p_ind, p_val = postprocess_peaks(
            labels[None, :], default_sphere, fracs[None, :]
        )
        res_dirs, res_inds, res_vals = p_out[0], p_ind[0], p_val[0]
        res_sh = (
            sf_to_sh(odf, default_sphere, sh_order=sh_order)
            if odf is not None
            else None
        )
    else:
        # Multi-Voxel / CLI Case
        original_shape = labels.shape[:-1]  # (X, Y, Z)

        if mask is not None:
            labels_to_proc = labels[mask]
            fracs_to_proc = fracs[mask]
        else:
            labels_to_proc = labels.reshape(-1, labels.shape[-1])
            fracs_to_proc = fracs.reshape(-1, fracs.shape[-1])

        p_out, p_ind, p_val = postprocess_peaks(
            labels_to_proc, default_sphere, fracs_to_proc
        )

        res_dirs = np.zeros((*original_shape, 5, 3), dtype=np.float32)
        res_inds = np.full((*original_shape, 5), -1, dtype=np.int32)
        res_vals = np.zeros((*original_shape, 5), dtype=np.float32)

        if mask is not None:
            res_dirs[mask] = p_out
            res_inds[mask] = p_ind
            res_vals[mask] = p_val
        else:
            res_dirs = p_out.reshape((*original_shape, 5, 3))
            res_inds = p_ind.reshape((*original_shape, 5))
            res_vals = p_val.reshape((*original_shape, 5))

        res_sh = None
        if odf is not None and np.issubdtype(
            getattr(odf, "dtype", type(None)), np.floating
        ):
            v_max = np.max(odf, axis=-1, keepdims=True)
            v_min = np.min(odf, axis=-1, keepdims=True)
            mask = v_max > 1.0
            denom = (v_max - v_min) + 1e-12
            normalized_odf = (odf - v_min) / denom
            odf = np.where(mask, normalized_odf, odf)
            res_sh = sf_to_sh(odf, default_sphere, sh_order=sh_order)
    peaks = PeaksAndMetrics()
    peaks.peak_dirs = res_dirs
    peaks.peak_values = res_vals
    peaks.peak_indices = res_inds
    peaks.shm_coeff = res_sh
    peaks.sphere = default_sphere

    return peaks
