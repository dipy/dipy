from dataclasses import dataclass
import logging

import numpy as np
from tqdm import tqdm
from trx.trx_file_memmap import TrxFile

from dipy.core.sphere import HemiSphere
from dipy.data import default_sphere

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
    stop_threshold: float
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
        stop_threshold=float(stop_threshold),
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


def streamline_generator(propagate, chunk_size, n_procs, seeds, close):
    global_chunk_sz = chunk_size * n_procs
    nchunks = (seeds.shape[0] + global_chunk_sz - 1) // global_chunk_sz
    try:
        for idx in range(nchunks):
            chunk = seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
            result = propagate(chunk)
            slines = result.slines
            sline_lens = result.sline_lens
            step = result.step
            for i in range(result.n_slines):
                npts = int(sline_lens[i])
                if npts < result.min_steps or npts > result.max_steps:
                    continue
                yield np.asarray(
                    slines[i * step : i * step + npts], dtype=result.real_dtype
                )

    finally:
        close()


def generate_trx(
    sl_generator,
    ref_img,
    nb_streamlines_estimate=None,
    nb_vertices_estimate=None,
    offset_dtype=np.uint64,
    data_dtype=np.float16,
):
    if nb_streamlines_estimate is None:
        nb_streamlines_estimate = int(1e6)
    if nb_vertices_estimate is None:
        nb_vertices_estimate = nb_streamlines_estimate * 100

    trx_reference = TrxFile(reference=ref_img)
    trx_reference.streamlines._data = trx_reference.streamlines._data.astype(data_dtype)
    trx_reference.streamlines._offsets = trx_reference.streamlines._offsets.astype(
        offset_dtype
    )

    trx_file = TrxFile(
        nb_streamlines=nb_streamlines_estimate,
        nb_vertices=nb_vertices_estimate,
        init_as=trx_reference,
    )

    affine = ref_img.affine
    aff_A = affine[:3, :3].T
    aff_b = affine[:3, 3]

    sl_idx = 0
    data_idx = 0

    with tqdm(total=nb_streamlines_estimate) as pbar:
        for sl in sl_generator:
            n = sl.shape[0]
            new_data_idx = data_idx + n

            if (
                sl_idx + 1 > trx_file.header["NB_STREAMLINES"]
                or new_data_idx > trx_file.header["NB_VERTICES"]
            ):
                logger.info("TRX resizing...")
                trx_file.resize(
                    nb_streamlines=(sl_idx + 1) * 2,
                    nb_vertices=new_data_idx * 2,
                )

            trx_file.streamlines._data[data_idx:new_data_idx] = sl.dot(aff_A) + aff_b
            trx_file.streamlines._offsets[sl_idx] = data_idx
            trx_file.streamlines._lengths[sl_idx] = n

            sl_idx += 1
            data_idx = new_data_idx
            pbar.update(1)

    if (
        sl_idx < trx_file.header["NB_STREAMLINES"]
        or data_idx < trx_file.header["NB_VERTICES"]
    ):
        trx_file.resize()
    return trx_file
