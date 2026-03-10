import json
import os
from pathlib import Path

import numpy as np

from dipy.reconst.base import ReconstModel, ReconstFit
from dipy.data import default_sphere
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst._force_search import search_inner_product as _cython_search


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


def _gtab_matches(entry_bvals, entry_bvecs, gtab, bval_tol=10.0,
                  bvec_tol=1e-3):
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

    return (np.allclose(stored_bvals, current_bvals, atol=bval_tol) and
            np.allclose(stored_bvecs, current_bvecs, atol=bvec_tol))


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


def _find_cached_simulation(cache_dir, gtab, diffusivity_config,
                            num_simulations):
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
        if not _diffusivity_matches(entry["diffusivity_config"],
                                    diffusivity_config):
            continue
        candidate = cache_dir / entry["filename"]
        if candidate.exists():
            return str(candidate)
    return None


def _register_cached_simulation(cache_dir, gtab, diffusivity_config,
                                num_simulations, filename):
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
    registry = _load_cache_registry(cache_dir)

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
    registry.append(entry)
    _save_cache_registry(cache_dir, registry)


class SignalIndex:
    """
    Index for inner product similarity search.

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

    s_max = np.max(scores, axis=1)
    s_min = np.min(scores, axis=1)
    half = 0.5 * (s_max + s_min)
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
    """FORCE reconstruction model for signal matching based microstructure."""

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

        >>> fit = model.fit(data, mask=mask, engine="ray", n_jobs=4)

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
        self._signals_norm = None

        if simulations is not None:
            self._prepare_library()

    def generate(
        self,
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
        """
        Generate simulations for matching.

        When ``output_path`` is ``None`` (the default) and
        ``use_cache`` is ``True``, simulations are cached in
        ``~/.dipy/force_simulations/`` (or ``$DIPY_HOME``).  A
        registry file (``cache_registry.json``) keeps track of the
        bvals, bvecs, diffusivity configuration and number of
        simulations for each cached file.  If a cached simulation that
        matches the current gradient table (within tolerance) and
        diffusivity configuration already exists, it is loaded from
        disk and generation is skipped.

        Set ``use_cache=False`` to force regeneration even when a
        matching cached simulation exists.

        Parameters
        ----------
        num_simulations : int
            Number of simulated voxels.
        output_path : str, optional
            Path to save simulations (.npz).  When None, defaults to
            ``~/.dipy/force_simulations/`` and uses caching.
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
        use_cache : bool, optional
            Whether to use cached simulations when ``output_path`` is
            None.  Set to ``False`` to always regenerate.

        Returns
        -------
        self : FORCEModel
            Model with simulations loaded.
        """
        from dipy.sims.force import (generate_force_simulations,
                                     get_default_diffusivity_config,
                                     load_force_simulations,
                                     save_force_simulations)

        # Resolve the diffusivity config that will actually be used
        resolved_config = (diffusivity_config if diffusivity_config is not None
                           else get_default_diffusivity_config())

        # --- Cache logic when no explicit output_path is given ----------
        if output_path is None and use_cache:
            cache_dir = _get_force_cache_dir()
            cached = _find_cached_simulation(
                cache_dir, self.gtab, resolved_config, num_simulations,
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
            # Save into the .dipy cache and register
            cache_dir = _get_force_cache_dir()
            idx = len(_load_cache_registry(cache_dir))
            filename = f"force_sim_{idx}.npz"
            save_force_simulations(self.simulations,
                                   str(cache_dir / filename))
            _register_cached_simulation(
                cache_dir, self.gtab, resolved_config,
                num_simulations, filename,
            )
            if verbose:
                print(f"[FORCE] Cached simulations to "
                      f"{cache_dir / filename}")

        self._prepare_library()
        return self

    def load(self, input_path):
        """
        Load pre-computed simulations from file.

        Parameters
        ----------
        input_path : str
            Path to simulations file (.npz).

        Returns
        -------
        self : FORCEModel
            Model with simulations loaded.

        Examples
        --------
        >>> model = FORCEModel(gtab, n_neighbors=50)
        >>> model.load('simulated_data.npz')
        >>> fit = model.fit(data, mask=mask)
        """
        from dipy.sims.force import load_force_simulations

        self.simulations = load_force_simulations(input_path)
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

    @multi_voxel_fit
    def fit(self, data, *, mask=None, **kwargs):
        """
        Fit model to data.

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
        fit : FORCEFit
            Fitted model for a single voxel.

        Notes
        -----
        This method is decorated with @multi_voxel_fit which handles:
        - Multi-voxel iteration (serial or parallel)
        - Mask application
        - Results aggregation into MultiVoxelFit

        For parallel execution, use:
        >>> fit = model.fit(data, mask=mask, engine="ray", n_jobs=4)
        """
        if self.simulations is None:
            raise RuntimeError(
                "No simulations loaded. Call generate() or provide simulations."
            )

        # Normalize the single voxel signal
        data = data.astype(np.float32)
        norm = np.linalg.norm(data)
        if norm == 0:
            norm = 1.0
        query_norm = (data / norm).reshape(1, -1)

        # Perform k-NN search for this single voxel
        D, I = self._index.search(query_norm, k=self.n_neighbors)
        S = D - self._penalty_array[I]

        # Compute diagnostics
        U, A = compute_uncertainty_ambiguity(S)

        if self.use_posterior:
            # Posterior averaging
            w = softmax_stable(self.posterior_beta * S, axis=1)
            entropy = float(-np.sum(w * np.log(w + 1e-12)))
            
            # Weighted average of library parameters
            params = self._compute_posterior_params(I[0], w[0])
            params["uncertainty"] = float(U[0])
            params["ambiguity"] = float(A[0])
            params["entropy"] = entropy
        else:
            # Best match
            best = np.argmax(S[0])
            lib_idx = I[0, best]
            
            params = self._fetch_library_params(lib_idx)
            params["uncertainty"] = float(U[0])
            params["ambiguity"] = float(A[0])
            params["entropy"] = None

        return FORCEFit(self, params)

    def _fetch_library_params(self, lib_idx):
        """Fetch parameters for a single library entry."""
        d = self.simulations
        params = {
            "fa": float(d["fa"][lib_idx]),
            "md": float(d["md"][lib_idx]),
            "rd": float(d["rd"][lib_idx]),
            "wm_fraction": float(d["wm_fraction"][lib_idx]),
            "gm_fraction": float(d["gm_fraction"][lib_idx]),
            "csf_fraction": float(d["csf_fraction"][lib_idx]),
            "num_fibers": float(d["num_fibers"][lib_idx]),
            "dispersion": float(d["dispersion"][lib_idx]),
            "nd": float(d["nd"][lib_idx]),
        }

        if "ufa_wm" in d:
            params["ufa_wm"] = float(d["ufa_wm"][lib_idx])
            params["ufa_voxel"] = float(d["ufa_voxel"][lib_idx])

        if "ak" in d:
            params["ak"] = float(d["ak"][lib_idx])
            params["rk"] = float(d["rk"][lib_idx])
            params["mk"] = float(d["mk"][lib_idx])
            params["kfa"] = float(d["kfa"][lib_idx])

        # ODF (if requested and available)
        if self.compute_odf and "odfs" in d:
            params["odf"] = d["odfs"][lib_idx].astype(np.float32)
        else:
            params["odf"] = None

        # Predicted signal (cleaned_dwi) - the matched library signal
        params["predicted_signal"] = d["signals"][lib_idx].astype(np.float32)

        return params

    def _compute_posterior_params(self, indices, weights):
        """Compute weighted average of library parameters."""
        d = self.simulations
        params = {
            "fa": float(np.sum(weights * d["fa"][indices])),
            "md": float(np.sum(weights * d["md"][indices])),
            "rd": float(np.sum(weights * d["rd"][indices])),
            "wm_fraction": float(np.sum(weights * d["wm_fraction"][indices])),
            "gm_fraction": float(np.sum(weights * d["gm_fraction"][indices])),
            "csf_fraction": float(np.sum(weights * d["csf_fraction"][indices])),
            "num_fibers": float(np.sum(weights * d["num_fibers"][indices])),
            "dispersion": float(np.sum(weights * d["dispersion"][indices])),
            "nd": float(np.sum(weights * d["nd"][indices])),
        }

        if "ufa_wm" in d:
            params["ufa_wm"] = float(np.sum(weights * d["ufa_wm"][indices]))
            params["ufa_voxel"] = float(np.sum(weights * d["ufa_voxel"][indices]))

        if "ak" in d:
            params["ak"] = float(np.sum(weights * d["ak"][indices]))
            params["rk"] = float(np.sum(weights * d["rk"][indices]))
            params["mk"] = float(np.sum(weights * d["mk"][indices]))
            params["kfa"] = float(np.sum(weights * d["kfa"][indices]))

        # Posterior ODF (if requested and available)
        if self.compute_odf and "odfs" in d:
            # Weighted average of ODFs
            odf = np.zeros(d["odfs"].shape[1], dtype=np.float32)
            for i, idx in enumerate(indices):
                odf_i = d["odfs"][idx].astype(np.float32)
                odf_i /= np.max(odf_i) + 1e-12  # Normalize
                odf += weights[i] * odf_i
            odf /= np.max(odf) + 1e-12  # Normalize final
            params["odf"] = odf
        else:
            params["odf"] = None

        # Posterior mean signal (cleaned_dwi)
        n_grad = d["signals"].shape[1]
        predicted_signal = np.zeros(n_grad, dtype=np.float32)
        for i, idx in enumerate(indices):
            predicted_signal += weights[i] * d["signals"][idx]
        params["predicted_signal"] = predicted_signal

        return params


class FORCEFit(ReconstFit):
    """FORCE model fit results for a single voxel."""

    def __init__(self, model, params):
        """Initialize a FORCEFit class instance."""
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
