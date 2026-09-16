from dataclasses import dataclass, replace
import importlib
import logging

import numpy as np

from dipy.core.sphere import HemiSphere
from dipy.data import default_sphere
from dipy.direction.pmf import SimplePmfGen
from dipy.tracking._utils import _iter_chunk
from dipy.tracking.tractogen import compact_chunk, seed_peaks
from dipy.utils.omp import determine_num_threads

logger = logging.getLogger("dipy")


@dataclass(frozen=True)
class _SimpleTrackerData:
    pmf: np.ndarray
    stop_map: np.ndarray
    sphere: object
    sphere_vertices: np.ndarray
    is_symmetric: bool
    shape: tuple
    max_angle: float
    stop_threshold: float
    step_size: float
    relative_peak_thresh: float
    min_separation_angle: float
    max_cross: int
    pmf_threshold: float
    min_steps: int
    max_steps: int
    random_seed: int
    chunk_size: int
    n_procs: int


@dataclass(frozen=True)
class _SimpleBackendSpec:
    module: str
    factory: str
    probe: str
    multi_device: bool


SIMPLE_BACKENDS = {
    "metal": _SimpleBackendSpec(
        module="dipy.tracking.simplet.metal",
        factory="metal_streamline_generator",
        probe="metal_available",
        multi_device=False,
    ),
    "cuda": _SimpleBackendSpec(
        module="dipy.tracking.simplet.cuda",
        factory="cuda_streamline_generator",
        probe="cuda_available",
        multi_device=True,
    ),
    "webgpu": _SimpleBackendSpec(
        module="dipy.tracking.simplet.webgpu",
        factory="webgpu_streamline_generator",
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
    max_cross=1,
    pmf_threshold=0.1,
    random_seed=0,
    chunk_size=25000,
    n_procs=1,
    is_symmetric=None,
):
    """
    Prepare a simple tracker. Simple trackers assume:
        the entire SF can be loaded into memory (not SH)
        isotropic voxels
        simplified stopping criteria (ie, threshold on a scalar map)
        fixed (large) max SL length
        generic probabilistic direction getting
    Simplified trackers are implemented in CUDA, WebGPU and Metal. On the
    CPU, use :func:`dipy.tracking.tracker.probabilistic_tracking`.

    Parameters
    ----------
    pmf : ndarray, shape (dimx, dimy, dimz, dimt)
        PMF volume.
    stop_map : ndarray, shape (dimx, dimy, dimz)
        Stopping metric (e.g. GFA or FA).
    stop_threshold : float
        Voxels with stop_map <= stop_threshold are endpoints.
    sphere : Sphere or None
        Sphere defining the directions in the PMF.
        If None, uses default_sphere.
    max_angle : float, optional
        Maximum turning angle in radians
    step_size : float, optional
        Step size in voxels.
    min_steps : int, optional
        Minimum streamline length (steps) to keep.
    max_steps : int, optional
        Maximum streamline length (steps) to keep.
    relative_peak_thresh : float, optional
        Relative peak threshold for direction selection.
    min_separation_angle : float, optional
        Minimum separation angle (radians) between peaks.
    max_cross : int, optional
        Maximum number of peaks tracked per seed (all peaks if <= 0).
    pmf_threshold : float, optional
        Minimum PMF value (relative to max) to consider a valid direction.
    random_seed : int, optional
        Seed for random number generator
    chunk_size : int, optional
        Seeds per chunk in simple_sl_generator()
    n_procs : int, optional
        Number of devices to use. Must be 1 for the Metal and WebGPU
        backends; for CUDA this is the number of GPUs each chunk is split
        across.
    is_symmetric : bool, optional
        Whether the PMF is a symmetric spherical function (antipodal
        directions folded together when selecting directions). If None,
        the PMF is treated as symmetric when ``sphere`` is a HemiSphere.
    """
    if sphere is None:
        sphere = default_sphere
    if is_symmetric is None:
        is_symmetric = isinstance(sphere, HemiSphere)

    pmf = np.asarray(pmf, dtype=float, order="C")
    sphere_vertices = np.asarray(sphere.vertices, dtype=float, order="C")

    if sphere_vertices.shape[0] != pmf.shape[3]:
        raise ValueError(
            f"Number of vertices in sphere ({sphere_vertices.shape[0]}) "
            f"must match 4th dimension of PMF ({pmf.shape[3]})"
        )

    return _SimpleTrackerData(
        pmf=pmf,
        stop_map=np.asarray(stop_map, dtype=float, order="C"),
        sphere=sphere,
        sphere_vertices=sphere_vertices,
        is_symmetric=bool(is_symmetric),
        shape=tuple(int(dim) for dim in pmf.shape),
        max_angle=float(max_angle),
        stop_threshold=float(stop_threshold),
        step_size=float(step_size),
        relative_peak_thresh=float(relative_peak_thresh),
        min_separation_angle=float(min_separation_angle),
        max_cross=int(max_cross),
        pmf_threshold=float(pmf_threshold),
        min_steps=int(min_steps),
        max_steps=int(max_steps),
        random_seed=int(random_seed),
        chunk_size=int(chunk_size),
        n_procs=int(n_procs),
    )


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
        One of "cuda", "metal", "webgpu".
    """
    spec = _get_simple_backend_spec(simple_backend)
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
    chunked=False,
):
    """
    Generate streamlines in chunks from simple tracker data and seeds.

    Parameters
    ----------
    simple_tracker_data : _SimpleTrackerData
        Output of :func:`prepare_simple_tracker_data`.
    seeds : ndarray, shape (N, 3)
        Seed points, in the space of ``affine`` (voxel space by default).
    simple_backend : str, optional
        One of "metal" (Apple Silicon, requires
        ``dipy[metal]``), "cuda" (requires
        ``dipy[cu12]`` or ``dipy[cu13]``),
        or "webgpu" (requires ``dipy[webgpu]``).
        "auto" will select the first available backend in that order.
    seed_directions : ndarray, shape (N, 3), optional
        Initial direction for each seed. If None, the peaks of the PMF
        at each seed are used.
    nbr_threads : int, optional
        Number of OpenMP threads (0 means all available).
    save_seeds : bool, optional
        True to yield ``(streamline, seed)`` pairs.
    affine : ndarray, shape (4, 4), optional
        Voxel-to-world affine of the seeds and of the output streamlines.
    chunked : bool, optional
        Yield ``(points, lengths[, seeds])`` per chunk instead of single
        streamlines (see :func:`dipy.tracking.tractogen.generate_tractogram`).

    Yields
    ------
    streamline : ndarray, shape (npts, 3)
    """
    tracker_data = simple_tracker_data
    if simple_backend == "auto":
        for name in SIMPLE_BACKENDS:
            if simple_backend_available(name):
                simple_backend = name
                break
        else:
            raise RuntimeError(
                "No simple tracker backend available "
                f"({', '.join(SIMPLE_BACKENDS)}); use probabilistic_tracking()."
            )

    spec = _get_simple_backend_spec(simple_backend)
    if not spec.multi_device and tracker_data.n_procs != 1:
        raise ValueError(
            f"The {simple_backend} simple tracker only supports a single "
            "device. Set n_procs=1 when preparing the tracker data."
        )

    affine = np.eye(4) if affine is None else affine
    pmf_gen = SimplePmfGen(tracker_data.pmf, tracker_data.sphere)
    # GPU kernels are float32
    tracker_data = replace(
        tracker_data,
        pmf=tracker_data.pmf.astype(np.float32),
        stop_map=tracker_data.stop_map.astype(np.float32),
        sphere_vertices=tracker_data.sphere_vertices.astype(np.float32),
    )
    factory = getattr(importlib.import_module(spec.module), spec.factory)
    if spec.multi_device:
        generate_streamlines, close = factory(tracker_data, n_gpus=tracker_data.n_procs)
    else:
        generate_streamlines, close = factory(tracker_data)
    inv_affine = np.linalg.inv(affine)
    return _streamline_generator(
        tracker_data,
        np.dot(seeds, inv_affine[:3, :3].T) + inv_affine[:3, 3],
        generate_streamlines,
        pmf_gen,
        affine,
        close=close,
        seed_directions=seed_directions,
        nbr_threads=nbr_threads,
        save_seeds=save_seeds,
        chunked=chunked,
    )


def _streamline_generator(
    tracker_data,
    seeds,
    generate_streamlines,
    pmf_gen,
    affine,
    *,
    seed_directions=None,
    nbr_threads=0,
    close=None,
    save_seeds=False,
    chunked=False,
):
    """
    Yield streamlines chunk by chunk from a GPU backend function. ``seeds``
    are in voxel space; output is mapped through ``affine``.

    ``generate_streamlines(seeds, streamline_offsets, dirs)`` must return
    ``(streamline_buffer, streamline_lengths, n_streamlines)``, where
    ``dirs[streamline_offsets[i]:streamline_offsets[i + 1]]`` are the initial
    directions of seed ``i``; ``close``, if given, is called when the
    generator is exhausted or closed.
    """
    step = tracker_data.max_steps * 2
    chunk_size = tracker_data.chunk_size
    n_chunks = (seeds.shape[0] + chunk_size - 1) // chunk_size
    lin_T = np.asarray(affine[:3, :3].T, dtype=float, order="C")
    offset = np.asarray(affine[:3, 3], dtype=float, order="C")
    if nbr_threads <= 0:
        nbr_threads = determine_num_threads(None)

    try:
        for idx in range(n_chunks):
            start = idx * chunk_size
            chunk = np.asarray(
                seeds[start : start + chunk_size], dtype=np.float32, order="C"
            )
            if seed_directions is not None:
                streamline_offsets = np.arange(len(chunk) + 1, dtype=np.int32)
                dirs = seed_directions[start : start + len(chunk)]
            else:
                streamline_offsets, dirs = seed_peaks(
                    np.asarray(chunk, dtype=float),
                    pmf_gen,
                    tracker_data.is_symmetric,
                    nbr_threads,
                    tracker_data.max_cross,
                    tracker_data.relative_peak_thresh,
                    np.rad2deg(tracker_data.min_separation_angle),
                )
                streamline_offsets = streamline_offsets.astype(np.int32)
            streamline_buffer, streamline_lengths, n_streamlines = generate_streamlines(
                chunk,
                streamline_offsets,
                np.asarray(dirs, dtype=np.float32, order="C"),
            )
            seed_of_streamline = np.repeat(
                np.arange(len(chunk)), np.diff(streamline_offsets)
            )
            lengths = np.asarray(streamline_lengths[:n_streamlines])
            keep = np.flatnonzero(
                (lengths >= tracker_data.min_steps)
                & (lengths <= tracker_data.max_steps)
            )
            lengths = lengths[keep]
            out_offsets = np.zeros(len(keep) + 1, dtype=np.intp)
            np.cumsum(lengths, out=out_offsets[1:])
            points = np.empty((out_offsets[-1], 3))
            compact_chunk(
                streamline_buffer,
                (keep * step).astype(np.intp),
                out_offsets,
                points,
                lin_T,
                offset,
                nbr_threads,
            )
            seeds_out = None
            if save_seeds:
                seeds_out = np.dot(chunk[seed_of_streamline[keep]], lin_T) + offset
            if chunked:
                yield (
                    (points, lengths)
                    if seeds_out is None
                    else (points, lengths, seeds_out)
                )
            else:
                yield from _iter_chunk(points, lengths, seeds_out)
    finally:
        if close is not None:
            close()
