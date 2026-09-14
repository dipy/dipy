from dataclasses import dataclass
import logging

import numpy as np
from tqdm import tqdm
from trx.trx_file_memmap import TrxFile

from dipy.core.sphere import HemiSphere
from dipy.data import default_sphere
from dipy.tracking.simplet import gen_streamlines_prob, get_num_streamlines_prob

logger = logging.getLogger("dipy")


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
class StreamlineChunk:
    """
    A chunk of streamlines and useful information
    about their size and ordering within the wider
    tractogram.
    """

    n_slines: int
    slines: np.ndarray
    sline_lens: np.ndarray
    step: int
    min_steps: int
    max_steps: int
    real_dtype: type


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

    return _SimpleTrackerData(
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
        random_seed=random_seed,
        chunk_size=int(chunk_size),
        n_procs=int(n_procs),
        real_dtype=real_dtype,
    )


def streamline_generator(propagate, chunk_size, seeds, *, close=None):
    """
    Generate streamlines in chunks from a propagate function and
    seeds.

    Parameters
    ----------
    propagate : function
        A function that takes a chunk of seeds and returns a StreamlineChunk.
    chunk_size : int
        Number of seeds to process in each chunk.
    seeds : np.ndarray
        Array of seed points to generate streamlines from.
    close : function, optional
        A function to call when the generator is closed, for cleanup.
        Default: None
    """

    nchunks = (seeds.shape[0] + chunk_size - 1) // chunk_size
    try:
        for idx in range(nchunks):
            chunk = seeds[idx * chunk_size : (idx + 1) * chunk_size]
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
        if close is not None:
            close()


def generate_trx(
    sl_generator,
    ref_img,
    *,
    nb_streamlines_estimate=None,
    nb_vertices_estimate=None,
    offset_dtype=np.uint64,
    data_dtype=np.float16,
):
    """
    Generate a TRX file from a streamline generator.

    Parameters
    ----------
    sl_generator : generator
        A generator that yields streamlines (numpy arrays of shape (N, 3)).
    ref_img : nibabel.Nifti1Image
        Reference image for the TRX file.
    nb_streamlines_estimate : int, optional
        Estimated total number of streamlines, useful
        for preallocating the TRX file on disk.
        If None, defaults to 1e6.
        Default: None
    nb_vertices_estimate : int, optional
        Estimated total number of vertices, useful
        for preallocating the TRX file on disk.
        If None, defaults to nb_streamlines_estimate * 100.
        Default: None
    offset_dtype : data-type, optional
        Data type for the offsets array in the TRX file.
        Default: np.uint64
    data_dtype : data-type, optional
        Data type for the data array in the TRX file.
        Default: np.float16
    """
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


def cython_simple_sl_generator(
    simple_tracker_data, seeds, *, seed_directions=None, nbr_threads=0
):
    """
    Cython-based streamline generator for simple tracker data.
    """

    if simple_tracker_data.n_procs != 1:
        raise ValueError(
            "Cython simple tracker does not support multiprocessing, "
            "only multithreading. Set n_procs=1 when preparing the tracker data."
        )

    params = {
        "dimx": simple_tracker_data.dimx,
        "dimy": simple_tracker_data.dimy,
        "dimz": simple_tracker_data.dimz,
        "dimt": simple_tracker_data.dimt,
        "nedges": simple_tracker_data.nedges,
        "sphere_symm": simple_tracker_data.sphere_symm,
        "relative_peak_thresh": float(simple_tracker_data.relative_peak_thresh),
        "min_separation_angle": float(simple_tracker_data.min_separation_angle),
        "step_size": float(simple_tracker_data.step_size),
        "max_angle": float(simple_tracker_data.max_angle),
        "tc_threshold": float(simple_tracker_data.stop_threshold),
        "pmf_threshold": float(simple_tracker_data.pmf_threshold),
        "max_sline_len": simple_tracker_data.max_sline_len,
    }

    chunk_offset = 0

    def propagate(seeds):
        nonlocal chunk_offset
        seeds = np.ascontiguousarray(seeds, dtype=simple_tracker_data.real_dtype)
        nseed = len(seeds)

        peak_dirs = np.zeros(
            (nseed, simple_tracker_data.dimt, 3), dtype=simple_tracker_data.real_dtype
        )
        sline_offsets = np.zeros(nseed + 1, dtype=np.int32)

        if seed_directions is not None:
            start = chunk_offset
            chunk_dirs = np.ascontiguousarray(
                seed_directions[start : start + nseed],
                dtype=simple_tracker_data.real_dtype,
            )
            peak_dirs[:, 0, :] = chunk_dirs
            sline_offsets[:nseed] = 1
            chunk_offset += nseed
        else:
            get_num_streamlines_prob(
                seeds,
                simple_tracker_data.dataf,
                simple_tracker_data.sphere_vertices,
                simple_tracker_data.sphere_edges,
                peak_dirs.reshape(-1, 3),
                sline_offsets,
                params,
                nbr_threads,
            )

        counts = sline_offsets[:nseed].copy()
        sline_offsets[0] = 0
        np.cumsum(counts, out=sline_offsets[1:])

        nSlines = int(sline_offsets[-1])
        slineSeed = np.full(nSlines, -1, dtype=np.int32)
        sline_len = np.zeros(nSlines, dtype=np.int32)
        sline = np.zeros(
            (nSlines * simple_tracker_data.max_sline_len * 2, 3),
            dtype=simple_tracker_data.real_dtype,
        )

        gen_streamlines_prob(
            seeds,
            simple_tracker_data.dataf,
            simple_tracker_data.metric_map,
            simple_tracker_data.sphere_vertices,
            sline_offsets,
            peak_dirs.reshape(-1, 3),
            slineSeed,
            sline_len,
            sline,
            params,
            simple_tracker_data.random_seed,
            nbr_threads,
        )

        return StreamlineChunk(
            n_slines=nSlines,
            slines=sline,
            sline_lens=sline_len,
            step=simple_tracker_data.max_sline_len * 2,
            min_steps=simple_tracker_data.min_steps,
            max_steps=simple_tracker_data.max_steps,
            real_dtype=simple_tracker_data.real_dtype,
        )

    return streamline_generator(
        propagate=propagate,
        chunk_size=simple_tracker_data.chunk_size,
        seeds=seeds,
    )
