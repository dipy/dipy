from collections.abc import Callable
from dataclasses import dataclass
import logging
import math

from nibabel.streamlines.array_sequence import MEGABYTE, ArraySequence
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
from tqdm import tqdm
from trx.trx_file_memmap import TrxFile

from dipy.core.sphere import HemiSphere
from dipy.data import default_sphere
from dipy.io.stateful_tractogram import Space, StatefulTractogram

logger = logging.getLogger("GPUStreamlines")


@dataclass(frozen=True)
class JITTrackerData:
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
    tc_threshold: float
    step_size: float
    relative_peak_thresh: float
    min_separation_angle: float
    pmf_threshold: float
    nedges: int
    min_steps: int
    max_steps: int
    max_sline_len: int
    chunk_size: int
    n_procs: int
    real_dtype: type


@dataclass(frozen=True)
class StreamlineChunk:
    n_slines: int
    slines: np.ndarray
    sline_lens: np.ndarray
    step: int
    min_steps: int
    max_steps: int
    real_dtype: type


@dataclass(frozen=True)
class ChunkGenerator:
    propagate: Callable[[np.ndarray], StreamlineChunk]
    chunk_size: int
    n_procs: int = 1


def _iter_streamlines_result(result):
    slines = result.slines
    sline_lens = result.sline_lens
    step = result.step
    for i in range(result.n_slines):
        npts = int(sline_lens[i])
        if npts < result.min_steps or npts > result.max_steps:
            continue
        yield np.asarray(slines[i * step : i * step + npts], dtype=result.real_dtype)


def _buffer_size_mb(result):
    """Return estimated buffer size in MB"""
    if result.n_slines == 0:
        return 0
    total_pts = sum(
        len_
        for len_ in result.sline_lens[: result.n_slines]
        if result.min_steps <= len_ <= result.max_steps
    )
    itemsize = np.dtype(result.real_dtype).itemsize
    return math.ceil(total_pts * 3 * itemsize / MEGABYTE)


def _divide_chunks(chunk_generator, seeds):
    global_chunk_sz = chunk_generator.chunk_size * chunk_generator.n_procs
    nchunks = (seeds.shape[0] + global_chunk_sz - 1) // global_chunk_sz
    return global_chunk_sz, nchunks


def _to_array_sequence(result):
    return ArraySequence(_iter_streamlines_result(result), _buffer_size_mb(result))


def prepare_jit_tracker_data(
    pmf,
    stop_map,
    stop_threshold,
    sphere,
    max_angle=1.0472,  # 60 degrees in radians
    step_size=0.5,
    min_steps=0,
    max_steps=500,
    relative_peak_thresh=0.25,
    min_separation_angle=0.785398,  # 45 degrees in radians
    pmf_threshold=0.1,
    rng_seed=0,
    chunk_size=25000,
    precision="float64",
    n_procs=1,
):
    """
    Prepare a generic JIT tracker.

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
    rng_seed : int, optional
        Seed for random number generator
    chunk_size : int
        Seeds per propagate() call in generate_sft()
    precision : str
        "float32" or "float64".
    n_procs : int
        Number of processes to use for parallelization
    """
    if precision not in ("float32", "float64"):
        raise ValueError(f"Unsupported precision: {precision}")
    if precision == "float32":
        real_dtype = np.float32
    else:
        real_dtype = np.float64

    if sphere is None:
        sphere = default_sphere

    dataf = np.ascontiguousarray(pmf, dtype=real_dtype)
    metric_map = np.ascontiguousarray(stop_map, dtype=real_dtype)
    sphere_vertices = np.ascontiguousarray(sphere.vertices, dtype=real_dtype)
    sphere_edges = np.ascontiguousarray(sphere.edges, dtype=np.int32)

    if sphere_vertices.shape[0] != dataf.shape[3]:
        raise ValueError(
            f"Number of vertices in sphere ({sphere_vertices.shape[0]}) "
            f"must match 4th dimension of PMF ({dataf.shape[3]})"
        )

    # This assumes that if you pass a sphere which is not
    # a HemiSphere, then it should be treated as asymmetric.
    sphere_symm = isinstance(sphere, HemiSphere)

    dimx, dimy, dimz, dimt = pmf.shape
    nedges = int(sphere_edges.shape[0])
    max_sline_len = int(max_steps)

    if rng_seed != 0:
        np.random.seed(rng_seed)

    return JITTrackerData(
        dataf=dataf,
        metric_map=metric_map,
        sphere_vertices=sphere_vertices,
        sphere_edges=sphere_edges,
        sphere_symm=sphere_symm,
        dimx=dimx,
        dimy=dimy,
        dimz=dimz,
        dimt=dimt,
        max_angle=float(max_angle),
        tc_threshold=float(stop_threshold),
        step_size=float(step_size),
        relative_peak_thresh=float(relative_peak_thresh),
        min_separation_angle=float(min_separation_angle),
        pmf_threshold=float(pmf_threshold),
        nedges=nedges,
        min_steps=min_steps,
        max_steps=max_steps,
        max_sline_len=max_sline_len,
        chunk_size=int(chunk_size),
        n_procs=int(n_procs),
        real_dtype=real_dtype,
    )


def streamline_generator(chunk_generator, seeds):
    global_chunk_sz, nchunks = _divide_chunks(chunk_generator, seeds)
    for idx in range(nchunks):
        chunk = seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
        result = chunk_generator.propagate(chunk)
        yield from _iter_streamlines_result(result)


def generate_array_sequence(chunk_generator, seeds):
    global_chunk_sz, nchunks = _divide_chunks(chunk_generator, seeds)
    buffer_size = 0
    results = []

    with tqdm(total=seeds.shape[0]) as pbar:
        for idx in range(nchunks):
            chunk = seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
            result = chunk_generator.propagate(chunk)
            buffer_size += _buffer_size_mb(result)
            results.append(result)
            pbar.update(chunk.shape[0])

    return ArraySequence(
        (item for r in results for item in _iter_streamlines_result(r)), buffer_size
    )


def generate_tractogram(chunk_generator, seeds, affine):
    return Tractogram(
        generate_array_sequence(chunk_generator, seeds), affine_to_rasmm=affine
    )


def generate_sft(chunk_generator, seeds, ref_img):
    return StatefulTractogram(
        generate_array_sequence(chunk_generator, seeds), ref_img, Space.VOX
    )


def generate_trx(chunk_generator, seeds, ref_img):
    global_chunk_sz, nchunks = _divide_chunks(chunk_generator, seeds)

    # Will resize by a factor of 2 if these are exceeded
    sl_len_guess = 100
    sl_per_seed_guess = 2
    n_sls_guess = sl_per_seed_guess * seeds.shape[0]

    # trx files use memory mapping
    trx_reference = TrxFile(reference=ref_img)
    trx_reference.streamlines._data = trx_reference.streamlines._data.astype(np.float32)
    trx_reference.streamlines._offsets = trx_reference.streamlines._offsets.astype(
        np.uint64
    )

    trx_file = TrxFile(
        nb_streamlines=n_sls_guess,
        nb_vertices=n_sls_guess * sl_len_guess,
        init_as=trx_reference,
    )
    offsets_idx = 0
    sls_data_idx = 0

    with tqdm(total=seeds.shape[0]) as pbar:
        for idx in range(int(nchunks)):
            chunk = seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
            result = chunk_generator.propagate(chunk)
            tractogram = Tractogram(
                _to_array_sequence(result),
                affine_to_rasmm=ref_img.affine,
            )
            tractogram.to_world()
            sls = tractogram.streamlines

            new_offsets_idx = offsets_idx + len(sls._offsets)
            new_sls_data_idx = sls_data_idx + len(sls._data)

            if (
                new_offsets_idx > trx_file.header["NB_STREAMLINES"]
                or new_sls_data_idx > trx_file.header["NB_VERTICES"]
            ):
                logger.info("TRX resizing...")
                trx_file.resize(
                    nb_streamlines=new_offsets_idx * 2,
                    nb_vertices=new_sls_data_idx * 2,
                )

            # TRX uses memmaps here
            trx_file.streamlines._data[sls_data_idx:new_sls_data_idx] = sls._data
            trx_file.streamlines._offsets[offsets_idx:new_offsets_idx] = (
                sls_data_idx + sls._offsets
            )
            trx_file.streamlines._lengths[offsets_idx:new_offsets_idx] = sls._lengths

            offsets_idx = new_offsets_idx
            sls_data_idx = new_sls_data_idx
            pbar.update(chunk.shape[0])
    trx_file.resize()

    return trx_file
