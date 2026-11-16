from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import warnings

import numpy as np

from dipy.core.gradients import unique_bvals_tolerance
from dipy.direction.peaks import PeaksAndMetrics, _pam_from_attrs
from dipy.reconst._force_align import (
    alignment_system as _alignment_system,
    coherence_sweeps as _coherence_sweeps,
    fiber_edges as _fiber_edges,
    kernel_signal as _kernel_signal,
)
from dipy.reconst._force_search import search_inner_product as _cython_search
from dipy.reconst.base import ReconstFit, ReconstModel
from dipy.reconst.dirspeed import peak_directions
from dipy.reconst.multi_voxel import MultiVoxelFit, multi_voxel_fit
from dipy.reconst.shm import real_sh_descoteaux, sf_to_sh, sh_to_sf_matrix
from dipy.sims._force_core import (
    DEFAULT_THREE_FIBER_MIN_ANGLE,
    DEFAULT_TWO_FIBER_MIN_ANGLE,
)
from dipy.sims.force import (
    DEFAULT_FORCE_SEED,
    DEFAULT_NUM_ODI_VALUES,
    DEFAULT_ODI_RANGE,
    generate_force_simulations,
    get_default_diffusivity_config,
    load_force_simulations,
    resolve_num_odi_values,
    save_force_simulations,
)
from dipy.utils.logging import logger
from dipy.utils.multiproc import get_usable_cpu_affinity
from dipy.utils.optpkg import optional_package

ray, has_ray, _ = optional_package("ray")

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
        with open(registry_path) as f:
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


def _odi_grid_matches(entry, odi_range, num_odi_values):
    """Check whether a registry *entry*'s ODI grid matches the request.

    Legacy entries written before the ODI grid was part of the cache key are
    treated as having been generated with the historical defaults
    (``DEFAULT_ODI_RANGE`` / ``DEFAULT_NUM_ODI_VALUES``), so existing caches
    keep matching default-grid requests.
    """
    entry_range = entry.get("odi_range", list(DEFAULT_ODI_RANGE))
    entry_num = entry.get("num_odi_values", DEFAULT_NUM_ODI_VALUES)
    if len(entry_range) != 2:
        return False
    if not (
        np.isclose(entry_range[0], odi_range[0])
        and np.isclose(entry_range[1], odi_range[1])
    ):
        return False
    return int(entry_num) == int(num_odi_values)


def _min_angles_match(entry, min_crossing_angles):
    """Check whether a registry *entry*'s crossing angles match the request.

    Entries written before this became part of the cache key are read as
    using the defaults.
    """
    entry_angles = (
        entry.get("two_fiber_min_angle", DEFAULT_TWO_FIBER_MIN_ANGLE),
        entry.get("three_fiber_min_angle", DEFAULT_THREE_FIBER_MIN_ANGLE),
    )
    return all(
        np.isclose(a, b) for a, b in zip(entry_angles, min_crossing_angles, strict=True)
    )


def _seed_matches(entry, seed):
    """Check whether a registry *entry*'s generation seed matches the request.

    Entries written before the seed became part of the cache key carry no
    ``seed`` field; like entries generated with ``seed=None``, they only
    match requests with ``seed=None``.
    """
    entry_seed = entry.get("seed")
    if entry_seed is None or seed is None:
        return entry_seed is None and seed is None
    return int(entry_seed) == int(seed)


def _find_cached_simulation(
    cache_dir,
    gtab,
    diffusivity_config,
    num_simulations,
    odi_range,
    num_odi_values,
    min_crossing_angles,
    seed,
):
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
    odi_range : tuple
        ``(min, max)`` orientation-dispersion-index range.
    num_odi_values : int
        Resolved number of ODI grid points.
    min_crossing_angles : tuple
        ``(two_fiber_min_angle, three_fiber_min_angle)`` in degrees.
    seed : int or None
        Generation seed requested for the library.

    Returns
    -------
    path : str or None
        Path to the cached ``.npz`` file, or None if no match found.
    """
    registry = _load_cache_registry(cache_dir)
    legacy_seed_cache_reported = False
    for entry in registry:
        if entry["num_simulations"] != num_simulations:
            continue
        if not _gtab_matches(entry["bvals"], entry["bvecs"], gtab):
            continue
        if not _diffusivity_matches(entry["diffusivity_config"], diffusivity_config):
            continue
        if not _odi_grid_matches(entry, odi_range, num_odi_values):
            continue
        if not _min_angles_match(entry, min_crossing_angles):
            continue
        if "seed" not in entry and seed is not None:
            candidate = cache_dir / entry["filename"]
            if candidate.exists() and not legacy_seed_cache_reported:
                logger.info(
                    f"Cached FORCE simulations at {candidate} predate seed "
                    "tracking and cannot guarantee reproducibility. A new "
                    "seeded library will be generated."
                )
                legacy_seed_cache_reported = True
        if not _seed_matches(entry, seed):
            continue
        candidate = cache_dir / entry["filename"]
        if candidate.exists():
            return str(candidate)
    return None


def _register_cached_simulation(
    cache_dir,
    gtab,
    diffusivity_config,
    num_simulations,
    filename,
    odi_range,
    num_odi_values,
    min_crossing_angles,
    seed,
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
    odi_range : tuple
        ``(min, max)`` orientation-dispersion-index range.
    num_odi_values : int
        Resolved number of ODI grid points.
    min_crossing_angles : tuple
        ``(two_fiber_min_angle, three_fiber_min_angle)`` in degrees.
    seed : int or None
        Generation seed used for the library.
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
        "odi_range": [float(odi_range[0]), float(odi_range[1])],
        "num_odi_values": int(num_odi_values),
        "two_fiber_min_angle": float(min_crossing_angles[0]),
        "three_fiber_min_angle": float(min_crossing_angles[1]),
        "seed": None if seed is None else int(seed),
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
                f"k={k} exceeds index size ({self.ntotal}); clamping to {self.ntotal}.",
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


def postprocess_peaks(
    fits,
    support,
    *,
    model=None,
    max_uncertainty=15.0,
    wm_threshold=0.5,
    coherence_angle=5.0,
    robust_angle=25.0,
    continuity_alpha=0.25,
    continuity_power=4.0,
    max_link_angle=45.0,
):
    """Peaks of FORCE fits, aligned to the data and coherent across voxels.

    The fibers of the library entry matched to each voxel keep their
    fractions, diffusivities and dispersion; their axes are those aligned to
    the voxel's data by ``FORCEModel.fit``, off the sphere grid. Fibers the
    data cannot orient to within ``max_uncertainty`` are not reported.
    White-matter voxels left without a peak take the peaks of their posterior
    ODF. In a 3D volume the axes are then regularized by fiber continuity
    between neighboring voxels, each axis yielding to its neighbors only as
    far as its own data leave it undetermined; isotropic voxels are assumed.

    Parameters
    ----------
    fits : ndarray of FORCEFit
        Flattened fits, in the C order of ``support``.
    support : ndarray of bool
        Voxels for which peaks are returned.
    model : FORCEModel, optional
        Model of the fits; needed for the spatial stage.
    max_uncertainty : float, optional
        Peaks whose axis the data leave uncertain by more than this (RMS, in
        degrees) are not reported.
    wm_threshold : float, optional
        Voxels with at least this white-matter fraction that are left without
        a peak take the peaks of their posterior ODF. Use the ``wm_threshold``
        the library was generated with (see ``FORCEModel.generate``).
    coherence_angle : float, optional
        Expected angle, in degrees, between a fiber and its continuation in
        the next voxel; sets the strength of the continuity term.
    robust_angle : float, optional
        Scale, in degrees, of the robust (Welsch) continuity penalty.
    continuity_alpha, continuity_power : float, optional
        Fiber-continuity kernel ``alpha + (1 - alpha) |offset . axis|^power``
        weighting each neighbor by how well it lies along the fiber.
    max_link_angle : float, optional
        Fibers of neighboring voxels further apart than this, in degrees, are
        not linked.

    Returns
    -------
    peak_dirs : ndarray (..., 5, 3)
        Peak axes, by decreasing fiber fraction.
    peak_indices : ndarray (..., 5)
        Closest sphere vertex of each peak, -1 where there is none.
    peak_values : ndarray (..., 5)
        Fiber fraction of each peak.
    """
    from dipy.sims.force import default_sphere

    flat = np.flatnonzero(support)
    fits = fits[flat]
    peak_dirs = np.zeros((support.size, 5, 3), dtype=np.float32)
    peak_indices = np.full((support.size, 5), -1, dtype=np.int32)
    peak_vals = np.zeros((support.size, 5), dtype=np.float32)
    if len(fits):
        fibers = {
            key: np.stack([fit._params[f"fiber_{key}"] for fit in fits])
            for key in ("dirs", "ref", "keep", "fractions", "hess", "grad", "aligned")
        }
        keep = fibers.pop("keep")
        aligned = fibers.pop("aligned").astype(bool)
        n_fib = keep.shape[1]
        max_spread = np.deg2rad(max_uncertainty)
        spread = _fiber_uncertainty(fibers["hess"], keep)
        keep &= ~aligned[:, None] | (spread <= max_spread)

        wm = np.array([fit._params["wm_fraction"] for fit in fits])
        prior = np.eye(2 * n_fib) / max_spread**2
        for row in np.flatnonzero(~keep.any(axis=1) & (wm >= wm_threshold)):
            odf = fits[row].odf
            if odf is None:
                continue
            axes, values, _ = peak_directions(
                np.asarray(odf, dtype=np.float64),
                default_sphere,
                relative_peak_threshold=0.5,
                min_separation_angle=25,
            )
            count = min(len(axes), n_fib)
            fibers["dirs"][row] = 0.0
            fibers["dirs"][row, :count] = axes[:count]
            fibers["ref"][row] = fibers["dirs"][row]
            fibers["fractions"][row] = 0.0
            fibers["fractions"][row, :count] = values[:count] / values[:count].sum()
            keep[row] = np.arange(n_fib) < count
            fibers["hess"][row] = prior * np.repeat(keep[row], 2)[:, None]
            fibers["grad"][row] = 0.0
        fibers["valid"] = np.linalg.norm(fibers["ref"], axis=-1) > 0

        dirs = fibers["dirs"]
        if support.ndim == 3 and keep.any() and aligned.any():
            fibers["frames"] = _tangent_frames(dirs)
            dirs = _align_coherent(
                fibers,
                keep,
                _neighbor_graph(support, (1, 1, 1)),
                max_angle=model.max_angle,
                coherence_angle=coherence_angle,
                robust_angle=robust_angle,
                continuity_alpha=continuity_alpha,
                continuity_power=continuity_power,
                max_link_angle=max_link_angle,
            )

        rank = np.argsort(
            -np.where(keep, fibers["fractions"], -1.0), axis=1, kind="stable"
        )
        kept = np.take_along_axis(keep, rank, axis=1)
        axes = np.take_along_axis(dirs, rank[..., None], axis=1)
        values = np.take_along_axis(fibers["fractions"], rank, axis=1)
        nearest = _nearest_vertex(axes, default_sphere.vertices)
        peak_dirs[flat, :n_fib] = np.where(kept[..., None], axes, 0.0)
        peak_vals[flat, :n_fib] = np.where(kept, values, 0.0)
        peak_indices[flat, :n_fib] = np.where(kept, nearest, -1)

    return (
        peak_dirs.reshape((*support.shape, 5, 3)),
        peak_indices.reshape((*support.shape, 5)),
        peak_vals.reshape((*support.shape, 5)),
    )


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
        max_angle=25.0,
        max_step_angle=8.0,
        match_dof=12,
        max_basis_condition=20.0,
        verbose=False,
    ):
        r"""
        FORCE (FORward matching for Complex microstructure Estimation) model
        :footcite:p:`Shah2026`.

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
        max_angle : float, optional
            Largest rotation of a fiber axis away from its library axis, in
            degrees, during the alignment and the spatial stage of
            :func:`force_peaks`.
        max_step_angle : float, optional
            Largest rotation of a fiber axis in one Gauss-Newton iteration of
            the alignment, in degrees.
        match_dof : int, optional
            Effective number of parameters absorbed by the match when the
            residual is turned into a noise level.
        max_basis_condition : float, optional
            Largest condition number accepted for the spherical harmonic basis
            of a shell; shells with a worse basis are not used to align fibers.
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
        self.max_angle = max_angle
        self.max_step_angle = max_step_angle
        self.match_dof = match_dof
        self.max_basis_condition = max_basis_condition
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
        seed=DEFAULT_FORCE_SEED,
        wm_threshold=0.5,
        tortuosity=False,
        odi_range=DEFAULT_ODI_RANGE,
        num_odi_values=None,
        two_fiber_min_angle=DEFAULT_TWO_FIBER_MIN_ANGLE,
        three_fiber_min_angle=DEFAULT_THREE_FIBER_MIN_ANGLE,
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
        track of the bvals, bvecs, diffusivity configuration, ODI grid,
        minimum crossing angles, seed and number of simulations for each
        cached file. If a cached simulation that matches
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
        seed : int or None, optional
            Seed for reproducible library generation (see
            :func:`~dipy.sims.force.generate_force_simulations`). Identical
            seeds yield identical libraries for any ``num_cpus``. The seed
            is part of the cache key. When None, each newly generated library
            uses fresh entropy; a matching cached library may still be reused.
        wm_threshold : float, optional
            Minimum WM fraction to include fiber labels.
        tortuosity : bool, optional
            Use tortuosity constraint for perpendicular diffusivity.
        odi_range : tuple, optional
            (min, max) orientation dispersion index range.
        num_odi_values : int or None, optional
            Number of points in the ODI sampling grid. When ``None`` (default),
            the grid is autoscaled from ``odi_range`` so its sampling density
            matches the default grid.
        two_fiber_min_angle : float, optional
            Minimum crossing angle for two-fiber simulations, in degrees. The
            library never contains crossings below this angle, so the fit
            cannot report them either. Use ``0`` to allow any crossing.
        three_fiber_min_angle : float, optional
            Minimum pairwise crossing angle for three-fiber simulations, in
            degrees.
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

        # Resolve the ODI grid that will actually be used, so both the cache
        # key and the generated library agree on it.
        resolved_odi_range = (float(odi_range[0]), float(odi_range[1]))
        resolved_num_odi = resolve_num_odi_values(
            resolved_odi_range, num_odi_values=num_odi_values
        )
        min_crossing_angles = (float(two_fiber_min_angle), float(three_fiber_min_angle))
        resolved_seed = None if seed is None else int(seed)

        # --- Cache logic when no explicit output_path is given ----------
        if output_path is None and use_cache:
            cache_dir = _get_force_cache_dir()
            cached = _find_cached_simulation(
                cache_dir,
                self.gtab,
                resolved_config,
                num_simulations,
                resolved_odi_range,
                resolved_num_odi,
                min_crossing_angles,
                resolved_seed,
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
            seed=resolved_seed,
            wm_threshold=wm_threshold,
            tortuosity=tortuosity,
            odi_range=resolved_odi_range,
            num_odi_values=resolved_num_odi,
            two_fiber_min_angle=two_fiber_min_angle,
            three_fiber_min_angle=three_fiber_min_angle,
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
                resolved_odi_range,
                resolved_num_odi,
                min_crossing_angles,
                resolved_seed,
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

        # Fiber axes of the matched entry aligned to each voxel's data, with
        # their precision; force_peaks filters and makes them coherent.
        aligned = _align_voxelwise(data2d, lib_idx, self)
        n_dof = max(len(self.gtab.bvals) - self.match_dof, 1)
        noise = np.maximum(aligned["cost"], EPSILON) / n_dof
        params_arrays["fiber_dirs"] = aligned["dirs"]
        params_arrays["fiber_ref"] = aligned["ref"]
        params_arrays["fiber_keep"] = aligned["valid"]
        params_arrays["fiber_fractions"] = aligned["fractions"]
        params_arrays["fiber_hess"] = aligned["hess"] / noise[:, None, None]
        params_arrays["fiber_grad"] = aligned["grad"] / noise[:, None]
        params_arrays["fiber_aligned"] = np.full(
            n_vox, bool(_alignment_context(self)["ops"]["shells"])
        )

        # Per-microstructure uncertainty and ambiguity from posterior density.
        # The FWHM-KDE ambiguity is the dominant serial cost; numpy releases the
        # GIL on its large reductions, so fan the per-parameter loop over threads.
        prior_ranges = getattr(self, "_prior_ranges", {})
        todo = [p for p in MICRO_PARAMS if p in d and p in prior_ranges]

        def _micro(param):
            vals = d[param][neighbors].astype(np.float32)
            return param, compute_microstructure_uncertainty_ambiguity(
                vals, W, prior_ranges[param]
            )

        # One thread inside a Ray worker: process-level parallelism already
        # uses the cores, so threading here would oversubscribe.
        if has_ray and ray.is_initialized():
            n_threads = 1
        else:
            n_threads = min(len(MICRO_PARAMS), len(get_usable_cpu_affinity()))
        if n_threads > 1 and len(todo) > 1:
            with ThreadPoolExecutor(max_workers=min(n_threads, len(todo))) as ex:
                micro_results = list(ex.map(_micro, todo))
        else:
            micro_results = [_micro(p) for p in todo]

        for param, (u, a) in micro_results:
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


def force_peaks(
    fitted_object,
    *,
    mask=None,
    sh_order=8,
    max_uncertainty=15.0,
    wm_threshold=0.5,
    coherence_angle=5.0,
    robust_angle=25.0,
    continuity_alpha=0.25,
    continuity_power=4.0,
    max_link_angle=45.0,
):
    """Create a PeaksAndMetrics object from a FORCEFit or MultiVoxelFit.

    Parameters
    ----------
    fitted_object : FORCEFit or MultiVoxelFit
        The result of model.fit().
    mask : ndarray, optional
        Optional brain mask.
    sh_order : int, optional
        Spherical harmonics order for the coefficients.
    max_uncertainty : float, optional
        Peaks whose axis the data leave uncertain by more than this (RMS, in
        degrees) are not reported.
    wm_threshold : float, optional
        Voxels with at least this white-matter fraction that are left without
        a peak take the peaks of their posterior ODF. Use the ``wm_threshold``
        the library was generated with (see ``FORCEModel.generate``).
    coherence_angle : float, optional
        Expected angle, in degrees, between a fiber and its continuation in
        the next voxel; sets the strength of the continuity term.
    robust_angle : float, optional
        Scale, in degrees, of the robust (Welsch) continuity penalty.
    continuity_alpha, continuity_power : float, optional
        Fiber-continuity kernel ``alpha + (1 - alpha) |offset . axis|^power``
        weighting each neighbor by how well it lies along the fiber.
    max_link_angle : float, optional
        Fibers of neighboring voxels further apart than this, in degrees, are
        not linked.

    Notes
    -----
    Multi-voxel ODF conversion uses 10,000-voxel blocks to bound temporary
    memory. See :func:`postprocess_peaks` for how the peaks are obtained.
    """
    from dipy.sims.force import default_sphere

    peak_params = {
        "max_uncertainty": max_uncertainty,
        "wm_threshold": wm_threshold,
        "coherence_angle": coherence_angle,
        "robust_angle": robust_angle,
        "continuity_alpha": continuity_alpha,
        "continuity_power": continuity_power,
        "max_link_angle": max_link_angle,
    }
    if not isinstance(fitted_object, MultiVoxelFit):
        # Single Voxel Case
        odf = fitted_object.odf
        p_out, p_ind, p_val = postprocess_peaks(
            np.array([fitted_object], dtype=object),
            np.ones(1, dtype=bool),
            **peak_params,
        )
        res_dirs, res_inds, res_vals = p_out[0], p_ind[0], p_val[0]
        res_sh = (
            sf_to_sh(odf, default_sphere, sh_order_max=sh_order)
            if odf is not None
            else None
        )
    else:
        # Bound conversion temporaries; retain only the final peaks and SH.
        original_shape = fitted_object.shape
        fits = fitted_object.fit_array.reshape(-1)
        fit_mask = fitted_object.mask.reshape(-1)
        res_sh = invB = None

        for start in range(0, fits.size, 10_000):
            chunk = slice(start, start + 10_000)
            if not fit_mask[chunk].any():
                continue
            block = MultiVoxelFit(fitted_object.model, fits[chunk], fit_mask[chunk])
            odf = block.odf
            if np.issubdtype(odf.dtype, np.floating):
                if invB is None:
                    _, invB = sh_to_sf_matrix(default_sphere, sh_order_max=sh_order)
                    res_sh = np.zeros((fits.size, invB.shape[1]), dtype=invB.dtype)
                v_max = np.max(odf, axis=-1, keepdims=True)
                v_min = np.min(odf, axis=-1, keepdims=True)
                odf = np.where(
                    v_max > 1.0, (odf - v_min) / (v_max - v_min + 1e-12), odf
                )
                res_sh[chunk] = np.dot(odf, invB)
            del block, odf

        support = fit_mask if mask is None else fit_mask & mask.reshape(-1)
        res_dirs, res_inds, res_vals = postprocess_peaks(
            fits,
            support.reshape(original_shape),
            model=fitted_object.model,
            **peak_params,
        )
        if res_sh is not None:
            res_sh = res_sh.reshape((*original_shape, res_sh.shape[-1]))
    return _pam_from_attrs(
        PeaksAndMetrics,
        default_sphere,
        res_inds,
        res_vals,
        res_dirs,
        None,
        None,
        res_sh,
        None,
        None,
    )


_NEIGHBOR_OFFSETS = np.array(
    [
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if (i, j, k) != (0, 0, 0)
    ],
    dtype=np.int64,
)


def _legendre_monomials(degree):
    """Coefficients of the even powers of ``t`` in the Legendre polynomial."""
    coeffs = np.zeros(degree + 1)
    coeffs[degree] = 1.0
    return np.polynomial.legendre.leg2poly(coeffs)[::2]


def _alignment_operators(
    gtab, sphere, *, sh_order_max=8, shell_tol=50.0, max_condition=20.0
):
    """Operators that expose the fiber kernels hidden in a library signal.

    Parameters
    ----------
    gtab : GradientTable
        Gradient table of the library.
    sphere : Sphere
        Sphere whose vertices carry the library's fiber axes.
    sh_order_max : int, optional
        Largest spherical harmonic degree used on a shell. Shells with too
        few directions for it use the largest degree they can support.
    shell_tol : float, optional
        b-values closer than this are treated as one shell.
    max_condition : float, optional
        Largest condition number accepted for the SH basis of a shell.

    Returns
    -------
    ops : dict
        ``shell`` (M,) shell index of each measurement (``-1`` where the
        measurement carries no orientation information), ``gdirs`` (M, 3),
        ``n_term`` number of polynomial terms of a kernel, and ``shells``,
        a list with, for every usable shell: ``idx`` measurement indices,
        ``pinv`` signal-to-SH operator, ``basis`` SH basis on the sphere
        vertices, ``starts`` first SH coefficient of each even degree,
        ``degrees`` and ``mono`` monomial coefficients of ``P_l``.
    """
    from dipy.core.geometry import cart2sphere

    bvals = np.asarray(gtab.bvals, dtype=np.float64)
    bvecs = np.asarray(gtab.bvecs, dtype=np.float64)
    dwi = bvals > gtab.b0_threshold
    rounded = np.round(bvals / shell_tol) * shell_tol
    vertices = np.asarray(sphere.vertices, dtype=np.float64)

    def basis(points, order):
        _, theta, phi = cart2sphere(points[:, 0], points[:, 1], points[:, 2])
        mat, _, degree = real_sh_descoteaux(order, theta, phi, legacy=False)
        return mat, np.asarray(degree)

    shells = []
    shell = np.full(len(bvals), -1, dtype=np.int32)
    for b in np.unique(rounded[dwi]):
        idx = np.flatnonzero(dwi & (rounded == b))
        # Largest order the directions of this shell determine well; shells
        # that cannot resolve any anisotropy carry no orientation information.
        order = 0
        for candidate in range(sh_order_max - sh_order_max % 2, 1, -2):
            if (candidate + 1) * (candidate + 2) // 2 > len(idx):
                continue
            mat, degree_of = basis(bvecs[idx], candidate)
            singular = np.linalg.svd(mat, compute_uv=False)
            if singular[0] <= max_condition * singular[-1]:
                order = candidate
                break
        if order < 2:
            continue
        degrees = np.arange(2, order + 2, 2)
        mono = np.zeros((len(degrees), sh_order_max // 2 + 1))
        for i, degree in enumerate(degrees):
            mono[i, : degree // 2 + 1] = _legendre_monomials(degree)
        shell[idx] = len(shells)
        shells.append(
            {
                "idx": idx,
                "pinv": np.linalg.pinv(mat),
                "basis": basis(vertices, order)[0],
                "starts": np.array(
                    [np.flatnonzero(degree_of == d)[0] for d in degrees]
                ),
                "degrees": degrees,
                "mono": mono,
            }
        )
    return {
        "shell": shell,
        "gdirs": np.ascontiguousarray(bvecs),
        "n_term": sh_order_max // 2 + 1,
        "shells": shells,
    }


def _fiber_vertices(labels, *, max_fibers=3):
    """Sphere vertex of each fiber from binary label rows.

    Parameters
    ----------
    labels : ndarray (N, n_vertices)
        Non-zero at the vertices carrying a fiber.
    max_fibers : int, optional
        Number of fiber slots returned.

    Returns
    -------
    vertex : ndarray of int (N, max_fibers)
        Vertex index of each fiber, ascending; ``0`` in unused slots.
    valid : ndarray of bool (N, max_fibers)
        True where the slot holds a fiber.
    """
    present = np.asarray(labels) > 0
    vertex = np.argsort(~present, axis=1, kind="stable")[:, :max_fibers]
    valid = np.take_along_axis(present, vertex, axis=1)
    return np.where(valid, vertex, 0), valid


def _pair_fractions(vertex, valid, odf, fractions):
    """Give every fiber vertex the fraction it was simulated with.

    The library stores fiber fractions in simulation order but fiber vertices
    as an unordered mask. All fibers of an entry share one dispersion, so the
    ODF amplitude at a fiber's vertex ranks the fibers by fraction.

    Parameters
    ----------
    vertex : ndarray of int (N, F)
        Vertex index of each fiber.
    valid : ndarray of bool (N, F)
        True where the slot holds a fiber.
    odf : ndarray (N, n_vertices) or None
        ODF of each entry. Without it the stored order is kept.
    fractions : ndarray (N, >=F)
        Fiber fractions in simulation order.

    Returns
    -------
    paired : ndarray (N, F)
        Fraction of the fiber in each slot, zero in unused slots.
    """
    n_fib = valid.shape[1]
    fractions = np.asarray(fractions, dtype=np.float64)[:, :n_fib]
    if odf is None:
        return np.where(valid, fractions, 0.0)
    amplitude = np.take_along_axis(np.asarray(odf, dtype=np.float32), vertex, axis=1)
    amplitude = np.where(valid, amplitude, -np.inf)
    by_amplitude = np.argsort(-amplitude, axis=1, kind="stable")
    paired = np.zeros(valid.shape)
    np.put_along_axis(paired, by_amplitude, -np.sort(-fractions, axis=1), axis=1)
    return np.where(valid, paired, 0.0)


def _fiber_kernels(signals, vertex, valid, sphere, ops, *, prior=None, ridge=1e-3):
    r"""Zonal kernel of every fiber of every library entry.

    On each shell an entry's signal is a sum of axially symmetric kernels, one
    per fiber, plus an isotropic part,

    .. math::

        S(\mathbf{g}) = R + \sum_k K_k(\mathbf{g} \cdot \mathbf{d}_k), \qquad
        K_k(t) = \sum_{l = 2, 4, \dots} \kappa_{lk} P_l(t) .

    The degree-:math:`l` part of the signal evaluated at the fiber axes,
    :math:`f_l(\mathbf{d}_k)`, obeys
    :math:`\sum_j P_l(\mathbf{d}_j \cdot \mathbf{d}_k) \kappa_{lj} =
    f_l(\mathbf{d}_k)` (addition theorem), which determines the
    :math:`\kappa_{lk}` of all fibers from the entry's signal alone.

    Parameters
    ----------
    signals : ndarray (N, M)
        Library signals.
    vertex : ndarray of int (N, F)
        Sphere vertex of each fiber.
    valid : ndarray of bool (N, F)
        True where the slot holds a fiber.
    sphere : Sphere
        Sphere carrying the fiber axes.
    ops : dict
        Output of :func:`_alignment_operators`.
    prior : ndarray (N, F, S, D), optional
        Expected :math:`\kappa_{lk}`; fills the components the signal leaves
        undetermined (e.g. degree 2 of three mutually orthogonal fibers).
    ridge : float, optional
        Weight of ``prior`` relative to the signal equations.

    Returns
    -------
    poly : ndarray (N, F, S, P)
        Kernels as polynomial coefficients in :math:`t^2`.
    kappa : ndarray (N, F, S, D)
        Legendre coefficients :math:`\kappa_{lk}` (zero-padded over degrees).
    """
    n_vox, n_fib = valid.shape
    n_shell = len(ops["shells"])
    n_deg = ops["n_term"] - 1
    axes = np.asarray(sphere.vertices, dtype=np.float64)[vertex]
    cosines = np.clip(np.einsum("nja,nka->njk", axes, axes), -1.0, 1.0)
    pair = valid[:, :, None] & valid[:, None, :]
    unused = np.eye(n_fib, dtype=bool)[None] & ~pair
    eye = np.eye(n_fib)[None]
    poly = np.zeros((n_vox, n_fib, n_shell, ops["n_term"]))
    kappa = np.zeros((n_vox, n_fib, n_shell, n_deg))
    for s, shell in enumerate(ops["shells"]):
        coef = signals[:, shell["idx"]].astype(np.float64) @ shell["pinv"].T
        at_axes = shell["basis"][vertex] * coef[:, None, :]
        f_l = np.add.reduceat(at_axes, shell["starts"], axis=2)
        for i, degree in enumerate(shell["degrees"]):
            order = np.zeros(degree + 1)
            order[degree] = 1.0
            gram = np.polynomial.legendre.legval(cosines, order)
            gram = np.where(pair, gram, 0.0)
            gram[np.broadcast_to(unused, gram.shape)] = 1.0
            rhs = np.einsum("njk,nk->nj", gram, np.where(valid, f_l[:, :, i], 0.0))
            if prior is not None:
                rhs += ridge * np.where(valid, prior[:, :, s, i], 0.0)
            k_l = np.linalg.solve(gram @ gram + ridge * eye, rhs[..., None])[..., 0]
            kappa[:, :, s, i] = k_l
            poly[:, :, s, :] += k_l[..., None] * shell["mono"][i]
    return poly, kappa


def _wm_dispersion(simulations, index):
    """Orientation dispersion of the white-matter compartment of entries."""
    wm = simulations["wm_fraction"][index].astype(np.float64)
    iso = simulations["gm_fraction"][index].astype(np.float64)
    iso += simulations["csf_fraction"][index]
    dispersion = simulations["dispersion"][index].astype(np.float64)
    return (dispersion - iso) / np.maximum(wm, 1e-6)


def _kernel_prior_table(simulations, sphere, ops, *, max_entries=30000):
    """Typical single-fiber kernel for each dispersion level of a library.

    Single-fiber entries expose their kernel directly. Their median, per unit
    white-matter fraction and for every dispersion value, is what a
    multi-fiber entry is expected to hold for each of its fibers.

    Parameters
    ----------
    simulations : dict
        FORCE simulation library.
    sphere : Sphere
        Sphere carrying the fiber axes.
    ops : dict
        Output of :func:`_alignment_operators`.
    max_entries : int, optional
        Number of single-fiber entries used.

    Returns
    -------
    table : dict
        ``grid`` (G,) dispersion values and ``kappa`` (G, S, D) the median
        kernel coefficients per unit white-matter fraction.
    """
    n_shell = len(ops["shells"])
    n_deg = ops["n_term"] - 1
    labels = simulations["labels"]
    single = np.flatnonzero(
        (simulations["num_fibers"] == 1) & (np.count_nonzero(labels, axis=1) == 1)
    )[:max_entries]
    if len(single) == 0 or n_shell == 0:
        return {"grid": np.zeros(1), "kappa": np.zeros((1, n_shell, n_deg))}
    vertex, valid = _fiber_vertices(labels[single], max_fibers=1)
    _, kappa = _fiber_kernels(
        simulations["signals"][single], vertex, valid, sphere, ops, ridge=0.0
    )
    unit = kappa[:, 0] / simulations["wm_fraction"][single, None, None]
    dispersion = _wm_dispersion(simulations, single)
    grid = np.unique(np.round(dispersion, 3))
    nearest = np.abs(dispersion[:, None] - grid[None, :]).argmin(axis=1)
    table = np.stack([np.median(unit[nearest == g], axis=0) for g in range(len(grid))])
    return {"grid": grid, "kappa": table}


def _alignment_context(model):
    """Operators and kernel prior of a FORCE model (cached on the model)."""
    from dipy.sims.force import default_sphere

    simulations = model.simulations
    cached = getattr(model, "_alignment_cache", None)
    if cached is not None and cached["signals"] is simulations["signals"]:
        return cached
    ops = _alignment_operators(
        model.gtab, default_sphere, max_condition=model.max_basis_condition
    )
    context = {
        "signals": simulations["signals"],
        "sphere": default_sphere,
        "ops": ops,
        "prior": _kernel_prior_table(simulations, default_sphere, ops),
    }
    model._alignment_cache = context
    return context


def _retract(dirs, frames, step):
    """Move unit vectors along their tangent frames and renormalize."""
    moved = dirs + np.einsum("nfca,nfc->nfa", frames, step)
    norm = np.linalg.norm(moved, axis=-1, keepdims=True)
    return moved / np.maximum(norm, EPSILON)


def _clamp_to_cone(dirs, center, max_angle):
    """Keep each axis within ``max_angle`` (radians) of its ``center`` axis."""
    cos_max = np.cos(max_angle)
    cosine = np.einsum("nfa,nfa->nf", dirs, center)
    dirs = dirs * np.where(cosine < 0, -1.0, 1.0)[..., None]
    cosine = np.abs(cosine)
    outside = cosine < cos_max
    if outside.any():
        perp = dirs - cosine[..., None] * center
        perp /= np.maximum(np.linalg.norm(perp, axis=-1, keepdims=True), EPSILON)
        edge = cos_max * center + np.sin(max_angle) * perp
        dirs = np.where(outside[..., None], edge, dirs)
    return dirs


def _align_voxelwise(signals, entries, model, *, n_iter=4):
    """Maximum-likelihood fiber axes of each voxel, matched entry frozen.

    Every fiber of the matched library entry keeps its kernel (fraction,
    diffusivities, dispersion); its axis moves to maximize the cosine
    similarity between the measured signal and the entry's signal, by damped
    Gauss-Newton steps in the tangent planes of the axes.

    Parameters
    ----------
    signals : ndarray (N, M)
        Measured signals.
    entries : ndarray of int (N,)
        Matched library entry of each voxel.
    model : FORCEModel
        Model holding the library.
    n_iter : int, optional
        Number of Gauss-Newton iterations.

    Returns
    -------
    result : dict
        ``dirs`` (N, F, 3) aligned axes, ``ref`` (N, F, 3) library axes,
        ``valid`` (N, F), ``fractions`` (N, F), ``hess`` (N, 2F, 2F) and
        ``grad`` (N, 2F) Gauss-Newton system at ``dirs`` in the tangent
        ``frames`` (N, F, 2, 3), and ``cost`` (N,) the final ``1 - rho^2``.
    """
    context = _alignment_context(model)
    ops, sphere = context["ops"], context["sphere"]
    simulations = model.simulations
    vertex, valid = _fiber_vertices(simulations["labels"][entries])
    n_vox, n_fib = valid.shape
    odf = simulations["odfs"][entries] if "odfs" in simulations else None
    fractions = _pair_fractions(
        vertex, valid, odf, simulations["fraction_array"][entries]
    )
    ref = np.where(valid[..., None], np.asarray(sphere.vertices)[vertex], 0.0)
    ref = np.ascontiguousarray(ref, dtype=np.float64)
    out = {
        "dirs": ref.copy(),
        "ref": ref,
        "valid": valid,
        "fractions": fractions,
        "hess": np.zeros((n_vox, 2 * n_fib, 2 * n_fib)),
        "grad": np.zeros((n_vox, 2 * n_fib)),
        "frames": np.zeros((n_vox, n_fib, 2, 3)),
        "cost": np.ones(n_vox),
    }
    if n_vox == 0 or not ops["shells"]:
        return out

    table = context["prior"]
    nearest = np.abs(
        _wm_dispersion(simulations, entries)[:, None] - table["grid"][None, :]
    ).argmin(axis=1)
    weight = simulations["wm_fraction"][entries].astype(np.float64)[:, None] * fractions
    prior = weight[:, :, None, None] * table["kappa"][nearest][:, None]
    matched = simulations["signals"][entries]
    poly, _ = _fiber_kernels(matched, vertex, valid, sphere, ops, prior=prior)
    poly = np.ascontiguousarray(poly)
    valid8 = np.ascontiguousarray(valid, dtype=np.uint8)
    shell, gdirs = ops["shell"], ops["gdirs"]
    rest = matched.astype(np.float32) - _kernel_signal(ref, valid8, poly, shell, gdirs)
    signals = np.ascontiguousarray(signals, dtype=np.float32)

    def system(dirs):
        return _alignment_system(signals, rest, dirs, valid8, poly, shell, gdirs)

    dirs = ref.copy()
    hess, grad, cost, frames = system(dirs)
    free = np.repeat(valid, 2, axis=1)
    eye = np.eye(2 * n_fib)[None]
    pinned = (~free)[:, :, None] * eye
    damping = np.full(n_vox, 1e-2)
    step_limit = np.tan(np.deg2rad(model.max_step_angle))
    for _ in range(n_iter):
        diag = np.einsum("npp->np", hess)
        lhs = hess + (damping[:, None] * np.maximum(diag, EPSILON))[:, :, None] * eye
        step = np.linalg.solve(lhs + pinned, grad[..., None])[..., 0]
        step = step.reshape(n_vox, n_fib, 2)
        length = np.linalg.norm(step, axis=-1, keepdims=True)
        step *= np.minimum(1.0, step_limit / np.maximum(length, EPSILON))
        trial = _clamp_to_cone(
            _retract(dirs, frames, step), ref, np.deg2rad(model.max_angle)
        )
        trial = np.ascontiguousarray(np.where(valid[..., None], trial, 0.0))
        hess_t, grad_t, cost_t, frames_t = system(trial)
        better = cost_t < cost
        damping = np.where(better, damping / 3.0, damping * 10.0)
        dirs[better] = trial[better]
        hess[better] = hess_t[better]
        grad[better] = grad_t[better]
        frames[better] = frames_t[better]
        cost[better] = cost_t[better]
    out.update(dirs=dirs, hess=hess, grad=grad, frames=frames, cost=cost)
    return out


def _fiber_uncertainty(precision, valid):
    """Angular uncertainty of each fiber axis from the data alone.

    Parameters
    ----------
    precision : ndarray (N, 2F, 2F)
        Precision of the aligned axes in tangent coordinates: Gauss-Newton
        Hessian of ``(1 - rho^2) / 2`` divided by the noise variance.
    valid : ndarray of bool (N, F)
        True where the slot holds a fiber.

    Returns
    -------
    sd : ndarray (N, F)
        Root-mean-square angular error of each axis (Laplace approximation),
        in radians; ``inf`` in unused slots.
    """
    n_fib = valid.shape[1]
    eye = np.eye(2 * n_fib)[None]
    free = np.repeat(valid, 2, axis=1)
    precision = precision + (~free)[:, :, None] * eye
    # The floor keeps an axis the data cannot orient at a large, finite value.
    cov = np.linalg.inv(precision + 1e-6 * eye)
    var = np.einsum("npp->np", cov).reshape(-1, n_fib, 2).sum(axis=-1)
    return np.where(valid, np.sqrt(np.maximum(var, 0.0)), np.inf)


def _neighbor_graph(mask, voxel_size):
    """26-neighborhood of the voxels selected by a 3D mask.

    Parameters
    ----------
    mask : ndarray of bool (X, Y, Z)
        Voxels of the graph, in C order.
    voxel_size : sequence of 3 floats
        Voxel dimensions.

    Returns
    -------
    graph : dict
        ``nbr`` (V, 26) neighbor index or ``-1``, ``unit`` (26, 3) unit
        offsets, ``weight`` (26,) inverse squared distance (1 for the closest
        neighbors), ``order`` (V,) voxels sorted by color and
        ``color_start`` (9,); voxels of one color are never neighbors.
    """
    coords = np.argwhere(mask)
    index = np.full(mask.shape, -1, dtype=np.int64)
    index[tuple(coords.T)] = np.arange(len(coords))
    shape = np.array(mask.shape)
    nbr = np.full((len(coords), len(_NEIGHBOR_OFFSETS)), -1, dtype=np.int64)
    for k, offset in enumerate(_NEIGHBOR_OFFSETS):
        shifted = coords + offset
        inside = np.all((shifted >= 0) & (shifted < shape), axis=1)
        nbr[inside, k] = index[tuple(shifted[inside].T)]
    physical = _NEIGHBOR_OFFSETS * np.asarray(voxel_size, dtype=np.float64)
    dist2 = np.sum(physical**2, axis=1)
    color = (coords % 2) @ np.array([1, 2, 4])
    order = np.argsort(color, kind="stable").astype(np.int64)
    return {
        "nbr": nbr,
        "unit": np.ascontiguousarray(physical / np.sqrt(dist2)[:, None]),
        "weight": np.ascontiguousarray(dist2.min() / dist2),
        "order": order,
        "color_start": np.searchsorted(color[order], np.arange(9)).astype(np.int64),
    }


def _align_coherent(
    fibers,
    keep,
    graph,
    *,
    max_angle,
    coherence_angle,
    robust_angle,
    continuity_alpha,
    continuity_power,
    max_link_angle,
    n_sweeps=10,
):
    """Spatially coherent fiber axes under the frozen-microstructure data term.

    Minimizes the second-order model of the data term around the voxel-wise
    solution plus a robust continuity term between linked fibers of
    neighboring voxels.

    Parameters
    ----------
    fibers : dict
        Per-voxel ``dirs``, ``ref``, ``valid`` and ``frames`` of the fibers,
        and the precision ``hess`` and linear term ``grad`` of their data term.
    keep : ndarray of bool (V, F)
        Fibers that take part in the continuity term.
    graph : dict
        Output of :func:`_neighbor_graph`.
    max_angle, coherence_angle, robust_angle : float
        Trust-region radius, expected angle between linked fibers and robust
        scale, in degrees (see :func:`force_peaks`).
    continuity_alpha, continuity_power : float
        Fiber-continuity kernel (see :func:`force_peaks`).
    max_link_angle : float
        Largest angle, in degrees, between linked fibers.
    n_sweeps : int, optional
        Number of Gauss-Seidel sweeps.

    Returns
    -------
    dirs : ndarray (V, F, 3)
        Coherent fiber axes.
    """
    valid = fibers["valid"]
    n_fib = valid.shape[1]
    base = np.ascontiguousarray(fibers["dirs"], dtype=np.float64)
    partner, weight = _fiber_edges(
        base,
        np.ascontiguousarray(keep, dtype=np.uint8),
        graph["nbr"],
        graph["unit"],
        graph["weight"],
        continuity_alpha,
        continuity_power,
        float(np.cos(np.deg2rad(max_link_angle))),
    )
    eye = np.eye(2 * n_fib)[None]
    free = np.repeat(valid, 2, axis=1)
    hess = fibers["hess"] + (~free)[:, :, None] * eye
    lin = fibers["grad"].copy()
    # A voxel whose system is not finite stays where the data put it.
    usable = np.isfinite(hess).all(axis=(1, 2)) & np.isfinite(lin).all(axis=1)
    hess[~usable] = eye[0] / EPSILON
    lin[~usable] = 0.0
    dirs = base.copy()
    _coherence_sweeps(
        dirs,
        base,
        np.ascontiguousarray(fibers["frames"], dtype=np.float64),
        np.ascontiguousarray(hess),
        np.ascontiguousarray(lin),
        np.ascontiguousarray(valid, dtype=np.uint8),
        np.ascontiguousarray(fibers["ref"], dtype=np.float64),
        graph["nbr"],
        partner,
        weight,
        graph["order"],
        graph["color_start"],
        1.0 / (2.0 * np.deg2rad(coherence_angle) ** 2),
        np.deg2rad(robust_angle),
        np.deg2rad(max_angle),
        n_sweeps,
        np.deg2rad(0.05),
    )
    return dirs


def _nearest_vertex(axes, vertices, *, block_size=50_000):
    """Index of the sphere vertex closest to each axis, up to sign.

    Parameters
    ----------
    axes : ndarray (..., 3)
        Unit vectors.
    vertices : ndarray (n_vertices, 3)
        Sphere vertices.
    block_size : int, optional
        Number of axes compared with the sphere at a time.

    Returns
    -------
    index : ndarray of int (...)
        Closest vertex of every axis.
    """
    vertices = np.ascontiguousarray(vertices, dtype=np.float32)
    flat = np.ascontiguousarray(axes, dtype=np.float32).reshape(-1, 3)
    index = np.empty(len(flat), dtype=np.int64)
    for start in range(0, len(flat), block_size):
        cosine = flat[start : start + block_size] @ vertices.T
        index[start : start + block_size] = np.abs(cosine, out=cosine).argmax(axis=1)
    return index.reshape(np.shape(axes)[:-1])


def _tangent_frames(dirs):
    """Two orthonormal vectors spanning the tangent plane at each unit vector.

    Same construction as ``_tangent_frame`` in ``_force_align.pyx``.

    Parameters
    ----------
    dirs : ndarray (..., 3)
        Unit vectors.

    Returns
    -------
    frames : ndarray (..., 2, 3)
        Tangent frame of every vector.
    """
    d = np.asarray(dirs, dtype=np.float64)
    zero = np.zeros(d.shape[:-1])
    e1 = np.where(
        (np.abs(d[..., 2]) < 0.9)[..., None],
        np.stack([d[..., 1], -d[..., 0], zero], axis=-1),
        np.stack([zero, d[..., 2], -d[..., 1]], axis=-1),
    )
    e1 /= np.maximum(np.linalg.norm(e1, axis=-1, keepdims=True), EPSILON)
    return np.stack([e1, np.cross(d, e1)], axis=-2)
