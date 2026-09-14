from dataclasses import dataclass
import importlib
import logging

import numpy as np

from dipy.core.sphere import HemiSphere
from dipy.data import default_sphere
from dipy.tracking.simplet import gen_streamlines_prob, get_num_streamlines_prob

logger = logging.getLogger("dipy")

_PARAM_FIELDS = (
    "dimx",
    "dimy",
    "dimz",
    "dimt",
    "nedges",
    "sphere_symm",
    "relative_peak_thresh",
    "min_separation_angle",
    "step_size",
    "max_angle",
    "pmf_threshold",
    "max_sline_len",
)


@dataclass(frozen=True)
class _SimpleTrackerData:
    dataf: np.ndarray
    metric_map: np.ndarray
    sphere_vertices: np.ndarray
    sphere_edges: np.ndarray
    sphere_symm: bool
    dimx: int
    dimy: int
    dimz: int
    dimt: int
    max_angle: float
    stop_threshold: float
    step_size: float
    relative_peak_thresh: float
    min_separation_angle: float
    pmf_threshold: float
    nedges: int
    min_steps: int
    max_steps: int
    max_sline_len: int
    random_seed: int
    chunk_size: int
    n_procs: int
    real_dtype: type


@dataclass(frozen=True)
class _SimpleBackendSpec:
    module: str
    factory: str
    probe: str | None
    float32_only: bool
    multi_device: bool


SIMPLE_BACKENDS = {
    "cython": _SimpleBackendSpec(
        module="dipy.tracking.simpletracker",
        factory="_cython_gen_streamlines_prob",
        probe=None,
        float32_only=False,
        multi_device=False,
    ),
    "cuda": _SimpleBackendSpec(
        module="dipy.tracking.cudasimplet",
        factory="cuda_gen_streamlines_prob",
        probe="cuda_available",
        float32_only=True,
        multi_device=True,
    ),
    "metal": _SimpleBackendSpec(
        module="dipy.tracking.metalsimplet",
        factory="metal_gen_streamlines_prob",
        probe="metal_available",
        float32_only=True,
        multi_device=False,
    ),
    "webgpu": _SimpleBackendSpec(
        module="dipy.tracking.webgpusimplet",
        factory="webgpu_gen_streamlines_prob",
        probe="webgpu_available",
        float32_only=True,
        multi_device=False,
    ),
}


def prepare_simple_tracker_data(
    pmf,
    stop_map,
    stop_threshold,
    sphere,
    *,
    max_angle=1.0472,  # 60 degrees in radians
    step_size=0.5,
    min_steps=0,
    max_steps=500,
    relative_peak_thresh=0.25,
    min_separation_angle=0.785398,  # 45 degrees in radians
    pmf_threshold=0.1,
    random_seed=0,
    chunk_size=25000,
    precision="float64",
    n_procs=1,
):
    """
    Prepare a simple tracker. Simple trackers assume:
        the entire SF can be loaded into memory (not SH)
        isotropic voxels
        simplified stopping criteria (ie, threshold on a scalar map)
        fixed (large) max SL length (500 steps by default)
        generic probabilistic direction getting
    Simplified trackers are implemented in CUDA, WebGPU, Metal and
    Cython and tend to be faster.

    Parameters
    ----------
    pmf : np.ndarray, shape (dimx, dimy, dimz, dimt)
        PMF volume.
    stop_map : np.ndarray, shape (dimx, dimy, dimz)
        Stopping metric (e.g. GFA or FA).
    stop_threshold : float
        Voxels with stop_map <= stop_threshold are endpoints.
    sphere: Sphere
        Sphere defining the directions in the PMF.
        If None, uses default_sphere.
    max_angle : float
        Maximum turning angle in radians
    step_size : float
        Step size in voxels.
    min_steps : int
        Minimum streamline length (steps) to keep.
    max_steps : float
        Maximum streamline length (steps) to keep.
    relative_peak_thresh : float
        Relative peak threshold for direction selection.
    min_separation_angle : float
        Minimum separation angle (radians) between peaks.
    pmf_threshold : float
        Minimum PMF value (relative to max) to consider a valid direction.
    random_seed : int, optional
        Seed for random number generator
    chunk_size : int
        Seeds per chunk in simple_sl_generator()
    precision : str
        "float32" or "float64". GPU backends only support "float32".
    n_procs : int
        Number of devices to use. Must be 1 for the Cython, Metal and WebGPU
        backends; for CUDA this is the number of GPUs each chunk is split
        across.
    """
    if precision not in ("float32", "float64"):
        raise ValueError(f"Unsupported precision: {precision}")
    real_dtype = np.float32 if precision == "float32" else np.float64

    if sphere is None:
        sphere = default_sphere

    dataf = np.ascontiguousarray(pmf, dtype=real_dtype)
    sphere_vertices = np.ascontiguousarray(sphere.vertices, dtype=real_dtype)

    if sphere_vertices.shape[0] != dataf.shape[3]:
        raise ValueError(
            f"Number of vertices in sphere ({sphere_vertices.shape[0]}) "
            f"must match 4th dimension of PMF ({dataf.shape[3]})"
        )

    dimx, dimy, dimz, dimt = pmf.shape

    return _SimpleTrackerData(
        dataf=dataf,
        metric_map=np.ascontiguousarray(stop_map, dtype=real_dtype),
        sphere_vertices=sphere_vertices,
        sphere_edges=np.ascontiguousarray(sphere.edges, dtype=np.int32),
        # This assumes that if you pass a sphere which is not
        # a HemiSphere, then it should be treated as asymmetric.
        sphere_symm=isinstance(sphere, HemiSphere),
        dimx=dimx,
        dimy=dimy,
        dimz=dimz,
        dimt=dimt,
        max_angle=float(max_angle),
        stop_threshold=float(stop_threshold),
        step_size=float(step_size),
        relative_peak_thresh=float(relative_peak_thresh),
        min_separation_angle=float(min_separation_angle),
        pmf_threshold=float(pmf_threshold),
        nedges=int(sphere.edges.shape[0]),
        min_steps=min_steps,
        max_steps=max_steps,
        max_sline_len=int(max_steps),
        random_seed=random_seed,
        chunk_size=int(chunk_size),
        n_procs=int(n_procs),
        real_dtype=real_dtype,
    )


def _prob_seed_directions(std, seeds, params, *, seed_directions=None, nbr_threads=0):
    """
    Find the initial tracking directions for a chunk of seeds.

    Returns
    -------
    peak_dirs : ndarray, shape (nseed * dimt, 3)
        Initial directions, ``dimt`` slots per seed.
    sline_offsets : ndarray, shape (nseed + 1,), int32
        Exclusive prefix sum of the number of directions per seed.
    """
    nseed = len(seeds)
    peak_dirs = np.zeros((nseed * std.dimt, 3), dtype=std.real_dtype)
    sline_offsets = np.zeros(nseed + 1, dtype=np.int32)

    if seed_directions is not None:
        peak_dirs.reshape(nseed, std.dimt, 3)[:, 0, :] = seed_directions
        sline_offsets[:nseed] = 1
    else:
        get_num_streamlines_prob(
            seeds,
            std.dataf,
            std.sphere_vertices,
            std.sphere_edges,
            peak_dirs,
            sline_offsets,
            params,
            nbr_threads,
        )

    counts = sline_offsets[:nseed].copy()
    sline_offsets[0] = 0
    np.cumsum(counts, out=sline_offsets[1:])
    return peak_dirs, sline_offsets


def _cython_gen_streamlines_prob(std, *, nbr_threads=0):
    """
    Set up the Cython probabilistic streamline generation kernel.
    """
    params = _simple_tracker_params(std)

    def gen_streamlines(seeds, sline_offsets, peak_dirs):
        n_slines = int(sline_offsets[-1])
        sline_seed = np.full(n_slines, -1, dtype=np.int32)
        sline_len = np.zeros(n_slines, dtype=np.int32)
        sline = np.zeros((n_slines * std.max_sline_len * 2, 3), dtype=std.real_dtype)
        gen_streamlines_prob(
            seeds,
            std.dataf,
            std.metric_map,
            std.sphere_vertices,
            sline_offsets,
            peak_dirs,
            sline_seed,
            sline_len,
            sline,
            params,
            std.random_seed,
            nbr_threads,
        )
        return sline, sline_len, n_slines

    return gen_streamlines, None


def _simple_tracker_params(std):
    """Convert _SimpleTrackerData to dict"""
    params = {name: getattr(std, name) for name in _PARAM_FIELDS}
    params["tc_threshold"] = std.stop_threshold
    return params


def _get_simple_backend_spec(simple_backend):
    try:
        return SIMPLE_BACKENDS[simple_backend]
    except KeyError:
        raise ValueError(
            f"Unknown simple_backend {simple_backend!r}, "
            f"expected one of {list(SIMPLE_BACKENDS)}"
        ) from None


def simple_backend_available(simple_backend):
    """
    Check whether a simple tracker backend can be used on this machine.

    Parameters
    ----------
    simple_backend : str
        One of "cython", "cuda", "metal", "webgpu".
    """
    spec = _get_simple_backend_spec(simple_backend)
    if spec.probe is None:
        return True
    return getattr(importlib.import_module(spec.module), spec.probe)()


def simple_sl_generator(
    simple_tracker_data,
    seeds,
    *,
    simple_backend="auto",
    seed_directions=None,
    nbr_threads=0,
):
    """
    Generate streamlines in chunks from simple tracker data and seeds.

    Parameters
    ----------
    simple_tracker_data : _SimpleTrackerData
        Output of :func:`prepare_simple_tracker_data`.
    seeds : ndarray, shape (N, 3)
        Seed points in voxel space.
    simple_backend : str
        One of "metal" (Apple Silicon, requires
        ``dipy[metal]``), "cuda" (requires
        ``dipy[cu12]`` or ``dipy[cu13]``),
        "webgpu" (requires ``dipy[webgpu]``), or "cython" (CPU, OpenMP).
        "auto" will select the first available backend in that order.
        Default: "auto".
    seed_directions : ndarray, shape (N, 3), optional
        Initial direction for each seed. If None, the peaks of the PMF
        at each seed are used.
    nbr_threads : int, optional
        Number of OpenMP threads (0 means all available).
    """
    std = simple_tracker_data
    if simple_backend == "auto":
        for name in ("metal", "cuda", "webgpu"):
            if simple_backend_available(name):
                simple_backend = name
                break
        else:
            simple_backend = "cython"

    spec = _get_simple_backend_spec(simple_backend)

    if spec.float32_only and std.real_dtype != np.float32:
        raise ValueError(
            f"The {simple_backend} simple tracker only supports float32. "
            'Use precision="float32" when preparing the tracker data.'
        )
    if not spec.multi_device and std.n_procs != 1:
        raise ValueError(
            f"The {simple_backend} simple tracker only supports a single "
            "device. Set n_procs=1 when preparing the tracker data."
        )

    factory = getattr(importlib.import_module(spec.module), spec.factory)
    if simple_backend == "cython":
        gen_streamlines, close = factory(std, nbr_threads=nbr_threads)
    else:
        gen_streamlines, close = factory(std)

    return _sl_generator(
        std,
        seeds,
        gen_streamlines,
        close=close,
        seed_directions=seed_directions,
        nbr_threads=nbr_threads,
    )


def _sl_generator(
    std, seeds, gen_streamlines, *, seed_directions=None, nbr_threads=0, close=None
):
    """
    Yield streamlines chunk by chunk from a backend function.

    ``gen_streamlines(seeds, sline_offsets, peak_dirs)`` must return
    ``(sline, sline_len, n_slines)``; ``close``, if given, is called when
    the generator is exhausted or closed.
    """
    params = _simple_tracker_params(std)
    step = std.max_sline_len * 2
    nchunks = (seeds.shape[0] + std.chunk_size - 1) // std.chunk_size

    try:
        for idx in range(nchunks):
            lo = idx * std.chunk_size
            chunk = np.ascontiguousarray(
                seeds[lo : lo + std.chunk_size], dtype=std.real_dtype
            )
            chunk_dirs = None
            if seed_directions is not None:
                chunk_dirs = np.ascontiguousarray(
                    seed_directions[lo : lo + len(chunk)], dtype=std.real_dtype
                )

            peak_dirs, sline_offsets = _prob_seed_directions(
                std, chunk, params, seed_directions=chunk_dirs, nbr_threads=nbr_threads
            )
            sline, sline_len, n_slines = gen_streamlines(
                chunk, sline_offsets, peak_dirs
            )

            for ii in range(n_slines):
                npts = int(sline_len[ii])
                if std.min_steps <= npts <= std.max_steps:
                    yield np.asarray(
                        sline[ii * step : ii * step + npts], dtype=std.real_dtype
                    )
    finally:
        if close is not None:
            close()
