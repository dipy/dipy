from dataclasses import dataclass, replace
import importlib
import logging

import numpy as np

from dipy.core.sphere import HemiSphere
from dipy.data import default_sphere
from dipy.direction.pmf import SimplePmfGen
from dipy.tracking.stopping_criterion import ThresholdStoppingCriterion
from dipy.tracking.tracker_parameters import generate_tracking_parameters
from dipy.tracking.tractogen import generate_tractogram, seed_peaks

logger = logging.getLogger("dipy")


@dataclass(frozen=True)
class _SimpleTrackerData:
    dataf: np.ndarray
    metric_map: np.ndarray
    sphere: object
    sphere_vertices: np.ndarray
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
    min_steps: int
    max_steps: int
    max_sline_len: int
    random_seed: int
    chunk_size: int
    n_procs: int


@dataclass(frozen=True)
class _SimpleBackendSpec:
    module: str
    factory: str
    probe: str | None
    multi_device: bool


SIMPLE_BACKENDS = {
    "cython": _SimpleBackendSpec(
        module="dipy.tracking.simpletracker",
        factory=None,
        probe=None,
        multi_device=False,
    ),
    "cuda": _SimpleBackendSpec(
        module="dipy.tracking.cudasimplet",
        factory="cuda_gen_streamlines_prob",
        probe="cuda_available",
        multi_device=True,
    ),
    "metal": _SimpleBackendSpec(
        module="dipy.tracking.metalsimplet",
        factory="metal_gen_streamlines_prob",
        probe="metal_available",
        multi_device=False,
    ),
    "webgpu": _SimpleBackendSpec(
        module="dipy.tracking.webgpusimplet",
        factory="webgpu_gen_streamlines_prob",
        probe="webgpu_available",
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
    n_procs=1,
):
    """
    Prepare a simple tracker. Simple trackers assume:
        the entire SF can be loaded into memory (not SH)
        isotropic voxels
        simplified stopping criteria (ie, threshold on a scalar map)
        fixed (large) max SL length (500 steps by default)
        generic probabilistic direction getting
    Simplified trackers are implemented in CUDA, WebGPU and Metal; the
    "cython" backend is :func:`dipy.tracking.tractogen.generate_tractogram`.

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
    n_procs : int
        Number of devices to use. Must be 1 for the Cython, Metal and WebGPU
        backends; for CUDA this is the number of GPUs each chunk is split
        across.
    """
    if sphere is None:
        sphere = default_sphere

    dataf = np.ascontiguousarray(pmf, dtype=float)
    sphere_vertices = np.ascontiguousarray(sphere.vertices, dtype=float)

    if sphere_vertices.shape[0] != dataf.shape[3]:
        raise ValueError(
            f"Number of vertices in sphere ({sphere_vertices.shape[0]}) "
            f"must match 4th dimension of PMF ({dataf.shape[3]})"
        )

    dimx, dimy, dimz, dimt = pmf.shape

    return _SimpleTrackerData(
        dataf=dataf,
        metric_map=np.ascontiguousarray(stop_map, dtype=float),
        sphere=sphere,
        sphere_vertices=sphere_vertices,
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
        min_steps=min_steps,
        max_steps=max_steps,
        max_sline_len=int(max_steps),
        random_seed=random_seed,
        chunk_size=int(chunk_size),
        n_procs=int(n_procs),
    )


def _tracking_objects(std):
    """PmfGen, StoppingCriterion and TrackerParameters matching ``std``."""
    pmf_gen = SimplePmfGen(std.dataf, std.sphere)
    sc = ThresholdStoppingCriterion(std.metric_map, std.stop_threshold)
    params = generate_tracking_parameters(
        "prob",
        max_len=std.max_steps * std.step_size,
        min_len=std.min_steps * std.step_size,
        step_size=std.step_size,
        voxel_size=np.ones(3),
        max_angle=np.rad2deg(std.max_angle),
        pmf_threshold=std.pmf_threshold,
        random_seed=std.random_seed,
        is_symmetric=std.sphere_symm,
    )
    return pmf_gen, sc, params


def _prob_seed_directions(std, seeds, pmf_gen, params, *, nbr_threads=0):
    """
    Initial tracking directions of a chunk of seeds, in the layout expected
    by the GPU kernels: ``dimt`` direction slots per seed.
    """
    offsets, dirs = seed_peaks(
        seeds,
        pmf_gen,
        params,
        nbr_threads,
        -1,
        std.relative_peak_thresh,
        np.rad2deg(std.min_separation_angle),
    )
    nseed = len(seeds)
    peak_dirs = np.zeros((nseed * std.dimt, 3), dtype=np.float32)
    seed_idx = np.repeat(np.arange(nseed), np.diff(offsets))
    peak_dirs[seed_idx * std.dimt + np.arange(len(dirs)) - offsets[seed_idx]] = dirs
    return peak_dirs, offsets.astype(np.int32)


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
    save_seeds=False,
    affine=None,
):
    """
    Generate streamlines in chunks from simple tracker data and seeds.

    Parameters
    ----------
    simple_tracker_data : _SimpleTrackerData
        Output of :func:`prepare_simple_tracker_data`.
    seeds : ndarray, shape (N, 3)
        Seed points, in the space of ``affine`` (voxel space by default).
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
    save_seeds : bool, optional
        True to yield ``(streamline, seed)`` pairs.
    affine : ndarray, shape (4, 4), optional
        Voxel-to-world affine of the seeds and of the output streamlines.

    Yields
    ------
    streamline : ndarray, shape (npts, 3)
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
    if not spec.multi_device and std.n_procs != 1:
        raise ValueError(
            f"The {simple_backend} simple tracker only supports a single "
            "device. Set n_procs=1 when preparing the tracker data."
        )

    affine = np.eye(4) if affine is None else affine
    pmf_gen, sc, params = _tracking_objects(std)
    if simple_backend == "cython":
        return generate_tractogram(
            np.asarray(seeds, dtype=float),
            seed_directions,
            sc,
            params,
            pmf_gen,
            affine,
            nbr_threads=nbr_threads,
            chunk_size=std.chunk_size,
            save_seeds=save_seeds,
            relative_peak_threshold=std.relative_peak_thresh,
            min_separation_angle=np.rad2deg(std.min_separation_angle),
        )

    # GPU kernels are float32
    std = replace(
        std,
        dataf=std.dataf.astype(np.float32),
        metric_map=std.metric_map.astype(np.float32),
        sphere_vertices=std.sphere_vertices.astype(np.float32),
    )
    factory = getattr(importlib.import_module(spec.module), spec.factory)
    gen_streamlines, close = factory(std)
    inv_affine = np.linalg.inv(affine)
    return _sl_generator(
        std,
        np.dot(seeds, inv_affine[:3, :3].T) + inv_affine[:3, 3],
        gen_streamlines,
        pmf_gen,
        params,
        affine,
        close=close,
        seed_directions=seed_directions,
        nbr_threads=nbr_threads,
        save_seeds=save_seeds,
    )


def _sl_generator(
    std,
    seeds,
    gen_streamlines,
    pmf_gen,
    params,
    affine,
    *,
    seed_directions=None,
    nbr_threads=0,
    close=None,
    save_seeds=False,
):
    """
    Yield streamlines chunk by chunk from a GPU backend function. ``seeds``
    are in voxel space; output is mapped through ``affine``.

    ``gen_streamlines(seeds, sline_offsets, peak_dirs)`` must return
    ``(sline, sline_len, n_slines)``; ``close``, if given, is called when
    the generator is exhausted or closed.
    """
    step = std.max_sline_len * 2
    nchunks = (seeds.shape[0] + std.chunk_size - 1) // std.chunk_size
    lin_T = affine[:3, :3].T.copy()
    offset = affine[:3, 3].copy()

    try:
        for idx in range(nchunks):
            lo = idx * std.chunk_size
            chunk = np.ascontiguousarray(
                seeds[lo : lo + std.chunk_size], dtype=np.float32
            )
            if seed_directions is not None:
                nseed = len(chunk)
                peak_dirs = np.zeros((nseed * std.dimt, 3), dtype=np.float32)
                peak_dirs[:: std.dimt] = seed_directions[lo : lo + nseed]
                sline_offsets = np.arange(nseed + 1, dtype=np.int32)
            else:
                peak_dirs, sline_offsets = _prob_seed_directions(
                    std,
                    np.asarray(chunk, dtype=float),
                    pmf_gen,
                    params,
                    nbr_threads=nbr_threads,
                )
            sline, sline_len, n_slines = gen_streamlines(
                chunk, sline_offsets, peak_dirs
            )
            sl_seed = np.repeat(np.arange(len(chunk)), np.diff(sline_offsets))

            for ii in range(n_slines):
                npts = int(sline_len[ii])
                if std.min_steps <= npts <= std.max_steps:
                    sl = np.dot(sline[ii * step : ii * step + npts], lin_T) + offset
                    if save_seeds:
                        yield sl, np.dot(chunk[sl_seed[ii]], lin_T) + offset
                    else:
                        yield sl
    finally:
        if close is not None:
            close()
