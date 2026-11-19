from dataclasses import dataclass
import functools
import importlib

from nibabel.affines import apply_affine
import numpy as np

from dipy.core.sphere import HemiSphere
from dipy.data import default_sphere
from dipy.direction.pmf import SimplePmfGen, pmf_peaks
from dipy.tracking.tractogen import SLINE_BUFFER_BYTES, finalize_chunk, seed_batches
from dipy.utils.omp import determine_num_threads


@dataclass(frozen=True)
class _GpuTrackerData:
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
class _GpuBackendSpec:
    module: str
    factory: str
    probe: str
    multi_device: bool


# in priority order for backend="auto": Metal (Apple Silicon), CUDA, then
# WebGPU, which runs on most GPUs but is a generic layer.
ACCEL_BACKENDS = {
    "metal": _GpuBackendSpec(
        module="dipy.tracking.accel.metal",
        factory="metal_streamline_generator",
        probe="metal_available",
        multi_device=False,
    ),
    "cuda": _GpuBackendSpec(
        module="dipy.tracking.accel.cuda",
        factory="cuda_streamline_generator",
        probe="cuda_available",
        multi_device=True,
    ),
    "webgpu": _GpuBackendSpec(
        module="dipy.tracking.accel.webgpu",
        factory="webgpu_streamline_generator",
        probe="webgpu_available",
        multi_device=False,
    ),
}


def ceil_div(a, b):
    """Ceiling of ``a / b`` for positive integers."""
    return (a + b - 1) // b


def block_rows(starts, step):
    """First row of each streamline in a buffer of ``step`` rows per streamline.

    Parameters
    ----------
    starts : ndarray, shape (n,)
        First point of each streamline in its block.
    step : int
        Rows reserved per streamline.

    Returns
    -------
    rows : ndarray of intp, shape (n,)
    """
    starts = np.asarray(starts, dtype=np.intp)
    return np.arange(starts.shape[0], dtype=np.intp) * step + starts


def kernel_params(tracker_data, *, n_seeds, sl_base):
    """Pack the ``ProbTrackingParams`` struct of kernel.wgsl / kernel.metal.

    Parameters
    ----------
    tracker_data : _GpuTrackerData
        Output of :func:`prepare_gpu_tracker_data`.
    n_seeds : int
        Number of seeds of the launch.
    sl_base : int
        Index in the tractogram of the launch's first streamline.

    Returns
    -------
    params : ndarray of uint32, shape (14,)
        4 f32, 4 u32 and 6 i32 fields, bit-cast to uint32.
    """
    seed = tracker_data.random_seed
    params = np.zeros(14, dtype=np.uint32)
    params[:4] = np.array(
        [
            tracker_data.max_angle,
            tracker_data.stop_threshold,
            tracker_data.step_size,
            tracker_data.pmf_threshold,
        ],
        dtype=np.float32,
    ).view(np.uint32)
    params[4:] = [
        seed & 0xFFFFFFFF,
        (seed >> 32) & 0xFFFFFFFF,
        sl_base & 0xFFFFFFFF,
        (sl_base >> 32) & 0xFFFFFFFF,
        n_seeds,
        *tracker_data.shape,
        tracker_data.max_steps,
    ]
    return params


def prepare_gpu_tracker_data(
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
    Prepare a GPU tracker. GPU trackers assume:

    - the entire SF can be loaded into memory (not SH)
    - isotropic voxels
    - simplified stopping criteria (ie, threshold on a scalar map)
    - fixed (large) max SL length
    - generic probabilistic direction getting

    GPU trackers are implemented in CUDA, WebGPU and Metal. On the
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
        Seed of the random number generator. All backends use the same
        generator, Philox4x32-10 keyed by ``random_seed`` with one stream
        per streamline (its index in the tractogram), which is also the
        generator of :func:`dipy.tracking.tracker.probabilistic_tracking`
        with ``rng="philox"``. Every backend therefore draws the same random
        numbers, whatever the chunk size or number of devices.
    chunk_size : int, optional
        Seeds per chunk in gpu_streamline_generator()
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

    # Kept in float64 so that it is shared with the CPU seed peak finder; the
    # backends convert to float32 while uploading.
    pmf = np.asarray(pmf, dtype=float, order="C")
    sphere_vertices = np.asarray(sphere.vertices, dtype=float, order="C")

    if sphere_vertices.shape[0] != pmf.shape[3]:
        raise ValueError(
            f"Number of vertices in sphere ({sphere_vertices.shape[0]}) "
            f"must match 4th dimension of PMF ({pmf.shape[3]})"
        )

    return _GpuTrackerData(
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
        random_seed=int(random_seed) % 2**64,
        chunk_size=int(chunk_size),
        n_procs=int(n_procs),
    )


def _get_gpu_backend_spec(backend):
    try:
        return ACCEL_BACKENDS[backend]
    except KeyError:
        raise ValueError(
            f"Unknown GPU backend {backend!r}, expected one of {list(ACCEL_BACKENDS)}"
        ) from None


@functools.cache
def accel_backend_available(backend):
    """
    Check whether a GPU tracker backend can be used on this machine.

    The answer is cached for the life of the process.

    Parameters
    ----------
    backend : str
        One of "cuda", "metal", "webgpu".

    Returns
    -------
    available : bool
    """
    spec = _get_gpu_backend_spec(backend)
    return bool(getattr(importlib.import_module(spec.module), spec.probe)())


def select_accel_backend():
    """
    First available GPU tracker backend, in the order of ``ACCEL_BACKENDS``.

    Returns
    -------
    backend : str or None
        "metal", "cuda" or "webgpu", or None if no backend is available.
    """
    for name in ACCEL_BACKENDS:
        if accel_backend_available(name):
            return name
    return None


def gpu_streamline_generator(
    tracker_data,
    seeds,
    *,
    backend="auto",
    seed_directions=None,
    nbr_threads=0,
    save_seeds=False,
    affine=None,
    chunked=False,
    return_all=True,
):
    """
    Generate streamlines in chunks from GPU tracker data and seeds.

    Parameters
    ----------
    tracker_data : _GpuTrackerData
        Output of :func:`prepare_gpu_tracker_data`.
    seeds : ndarray, shape (N, 3)
        Seed points, in the space of ``affine`` (voxel space by default).
    backend : str, optional
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
    return_all : bool, optional
        True to return all the streamlines, False to return only those whose
        both ends reached the stopping map.

    Yields
    ------
    streamline : ndarray, shape (npts, 3)
    """
    if backend == "auto":
        backend = select_accel_backend()
        if backend is None:
            raise RuntimeError(
                "No GPU tracker backend available "
                f"({', '.join(ACCEL_BACKENDS)}); use probabilistic_tracking()."
            )

    spec = _get_gpu_backend_spec(backend)
    if not spec.multi_device and tracker_data.n_procs != 1:
        raise ValueError(
            f"The {backend} GPU tracker only supports a single "
            "device. Set n_procs=1 when preparing the tracker data."
        )

    affine = np.eye(4) if affine is None else affine
    # shares tracker_data.pmf (float64, C order): no copy
    pmf_gen = SimplePmfGen(tracker_data.pmf, tracker_data.sphere)
    factory = getattr(importlib.import_module(spec.module), spec.factory)
    if spec.multi_device:
        generate_streamlines, close = factory(tracker_data, n_gpus=tracker_data.n_procs)
    else:
        generate_streamlines, close = factory(tracker_data)
    return _streamline_generator(
        tracker_data,
        apply_affine(np.linalg.inv(affine), np.asarray(seeds, dtype=float)),
        generate_streamlines,
        pmf_gen,
        affine,
        close=close,
        seed_directions=seed_directions,
        nbr_threads=nbr_threads,
        save_seeds=save_seeds,
        chunked=chunked,
        return_all=return_all,
        max_seeds=getattr(importlib.import_module(spec.module), "MAX_SEEDS", None),
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
    max_seeds=None,
    return_all=True,
):
    """
    Yield streamlines batch by batch from a GPU backend function. ``seeds``
    are in voxel space; output is mapped through ``affine``.

    ``generate_streamlines(seeds, streamline_offsets, dirs, sl_base)`` must
    return ``(streamline_buffer, streamline_rows, streamline_lengths)``,
    where ``dirs[streamline_offsets[i]:streamline_offsets[i + 1]]`` are the
    initial directions of seed ``i`` and ``sl_base`` is the index, in the
    tractogram, of the batch's first streamline (it selects the random
    streams). It is only called with at least one streamline. Streamline
    ``j`` has ``abs(streamline_lengths[j])`` points, the sign telling whether
    both ends reached the stopping map (positive) or not, starting at row
    ``streamline_rows[j]`` of ``streamline_buffer``, which only needs to
    stay valid until the next call. ``close``, if given, is called when the
    generator is exhausted or closed. ``max_seeds`` caps
    the number of seeds per call (e.g. a dispatch size limit). With
    ``return_all=False``, only streamlines with both ends on the stopping map
    are kept.
    """
    step = tracker_data.max_steps * 2
    chunk_size = tracker_data.chunk_size
    max_sl = max(1, SLINE_BUFFER_BYTES // (step * 3 * np.dtype(np.float32).itemsize))
    lin_T = np.asarray(affine[:3, :3].T, dtype=float, order="C")
    offset = np.asarray(affine[:3, 3], dtype=float, order="C")
    if nbr_threads <= 0:
        nbr_threads = determine_num_threads(None)

    sl_base = 0
    try:
        for start in range(0, seeds.shape[0], chunk_size):
            chunk = np.ascontiguousarray(seeds[start : start + chunk_size], dtype=float)
            if seed_directions is not None:
                offsets = np.arange(len(chunk) + 1)
                dirs = seed_directions[start : start + len(chunk)]
            else:
                offsets, dirs = pmf_peaks(
                    chunk,
                    pmf_gen,
                    is_symmetric=tracker_data.is_symmetric,
                    nbr_threads=nbr_threads,
                    max_peaks=tracker_data.max_cross,
                    relative_peak_threshold=tracker_data.relative_peak_thresh,
                    min_separation_angle=np.rad2deg(tracker_data.min_separation_angle),
                )
            for s0, s1 in seed_batches(offsets, max_sl, max_seeds=max_seeds):
                batch_offsets = (offsets[s0 : s1 + 1] - offsets[s0]).astype(np.int32)
                n_streamlines = int(batch_offsets[-1])
                buffer, rows, lengths = generate_streamlines(
                    np.asarray(chunk[s0:s1], dtype=np.float32),
                    batch_offsets,
                    np.asarray(
                        dirs[offsets[s0] : offsets[s1]], dtype=np.float32, order="C"
                    ),
                    sl_base,
                )
                sl_base += n_streamlines
                sl_seed = np.repeat(np.arange(s0, s1), np.diff(offsets[s0 : s1 + 1]))
                lengths = np.asarray(lengths[:n_streamlines], dtype=np.intp)
                valid = (lengths > 0) | return_all
                yield from finalize_chunk(
                    buffer,
                    rows=rows[:n_streamlines],
                    lengths=np.abs(lengths),
                    valid=valid,
                    min_pts=tracker_data.min_steps,
                    max_pts=tracker_data.max_steps,
                    seeds=chunk,
                    sl_seed=sl_seed,
                    lin_T=lin_T,
                    offset=offset,
                    nbr_threads=nbr_threads,
                    save_seeds=save_seeds,
                    chunked=chunked,
                )
    finally:
        if close is not None:
            close()
