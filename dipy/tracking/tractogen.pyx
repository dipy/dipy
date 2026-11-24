# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

from cython.parallel import prange, threadid
from nibabel.affines import apply_affine
import numpy as np
cimport numpy as cnp

from dipy.align.fused_types cimport floating
from dipy.direction.pmf cimport PmfGen
from dipy.direction.pmf import pmf_peaks
from dipy.tracking._utils import _iter_chunk
from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.utils cimport fast_numpy
from dipy.utils.deprecator import deprecated_params
from dipy.utils.omp import determine_num_threads

from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT,
                                               VALIDSTREAMLINE,
                                               INVALIDSTREAMLINE)
from dipy.tracking.tracker_parameters cimport (TrackerParameters,
                                               TrackerStatus)

from libc.string cimport memset


# Size of the buffer receiving the raw streamlines of one batch. Each
# streamline reserves 2 * max_nbr_pts points whatever its final length, so
# the number of streamlines tracked at once is capped by this budget rather
# than by ``chunk_size`` alone.
SLINE_BUFFER_BYTES = 256 * 1024 * 1024


@deprecated_params("buffer_frac", since="1.13.0", until="2.0.0",
                   alternative="chunk_size")
def generate_tractogram(double[:, ::1] seed_positions,
                        seed_directions,
                        StoppingCriterion sc,
                        TrackerParameters params,
                        PmfGen pmf_gen,
                        affine,
                        int nbr_threads=0,
                        int chunk_size=25000,
                        buffer_frac=None,
                        bint save_seeds=0,
                        int max_cross=-1,
                        double relative_peak_threshold=0.5,
                        double min_separation_angle=25,
                        bint chunked=0):
    """Generate a tractogram from a set of seed points and directions.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions for the streamlines.
    seed_directions : ndarray or None
        Seed directions for the streamlines. If None, the peaks of the pmf at
        each seed are used (see ``max_cross``).
    sc : StoppingCriterion
        Stopping criterion for the streamlines.
    params : TrackerParameters
        Parameters for the streamline generation.
    pmf_gen : PmfGen
        Probability mass function generator.
    affine : ndarray
        Affine transformation for the streamlines.
    nbr_threads : int, optional
        Number of threads to use for streamline generation.
    chunk_size : int, optional
        Number of seeds whose initial directions are computed at once. The
        streamlines of a chunk are then tracked in batches whose raw buffer
        stays under ``SLINE_BUFFER_BYTES``.
    buffer_frac : float, optional
        Deprecated and ignored. Use ``chunk_size``.
    save_seeds : bool, optional
        If True, return seeds alongside streamlines
    max_cross : int, optional
        Maximum number of peaks tracked per seed when ``seed_directions`` is
        None. Use all peaks when <= 0.
    relative_peak_threshold, min_separation_angle : float, optional
        Peak selection parameters when ``seed_directions`` is None (see
        :func:`dipy.direction.peak_directions`).
    chunked : bool, optional
        If True, yield one ``(points, lengths)`` tuple per batch of
        streamlines (``(points, lengths, seeds)`` with ``save_seeds``), where
        ``points`` is the concatenation of the batch's streamlines. Much
        cheaper to consume than one streamline at a time, e.g. with
        :func:`dipy.io.streamline.save_trx_from_generator`.

    Yields
    ------
    streamline : ndarray, shape (N, 3)
        Streamline in world space, or ``(streamline, seed)`` with
        ``save_seeds``, or chunk tuples with ``chunked`` (see above).

    """
    cdef:
        cnp.npy_intp nseed = seed_positions.shape[0]
        cnp.npy_intp step = 2 * params.max_nbr_pts
        cnp.npy_intp start, n, s0, s1, nsl
        cnp.npy_uint64 rng_seed
        double[:, ::1] seeds, dirs, scratch
        cnp.npy_intp[::1] offsets

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    if nbr_threads <= 0:
        nbr_threads = determine_num_threads(None)
    max_sl = max(1, SLINE_BUFFER_BYTES // (step * 3 * 8))

    if params.random_seed > 0:
        rng_seed = params.random_seed
    else:
        rng_seed = np.random.default_rng().integers(0, 2**63, dtype=np.uint64)

    # per-thread scratch: pmf and propagator state (>= 100 doubles for PTT)
    scratch = np.empty((nbr_threads, max(100, pmf_gen.pmf.shape[0])))
    sline_buf = np.empty((0, 3))
    stream_idx_buf = np.empty((0, 2), dtype=np.intc)
    status_buf = np.empty(0, dtype=np.intc)

    lin_T = np.asarray(affine[:3, :3].T, dtype=float, order="C")
    offset = np.asarray(affine[:3, 3], dtype=float, order="C")
    seed_positions = np.ascontiguousarray(
        apply_affine(np.linalg.inv(affine), np.asarray(seed_positions)), dtype=float)
    if seed_directions is not None:
        seed_directions = np.asarray(seed_directions, dtype=float, order="C")

    for start in range(0, nseed, chunk_size):
        n = min(chunk_size, nseed - start)
        seeds = seed_positions[start:start + n]
        if seed_directions is None:
            offsets, dirs = pmf_peaks(
                seeds, pmf_gen, is_symmetric=params.is_symmetric,
                nbr_threads=nbr_threads, max_peaks=max_cross,
                relative_peak_threshold=relative_peak_threshold,
                min_separation_angle=min_separation_angle)
        else:
            offsets = np.arange(n + 1)
            dirs = seed_directions[start:start + n]

        for s0, s1 in seed_batches(offsets, max_sl):
            nsl = offsets[s1] - offsets[s0]
            if nsl * step > sline_buf.shape[0]:
                sline_buf = np.empty((nsl * step, 3))
                stream_idx_buf = np.empty((nsl, 2), dtype=np.intc)
                status_buf = np.empty(nsl, dtype=np.intc)
            sl_seed = np.repeat(np.arange(s0, s1),
                                np.diff(np.asarray(offsets)[s0:s1 + 1]))
            generate_tractogram_c(seeds, dirs[offsets[s0]:offsets[s1]],
                                  sl_seed, rng_seed, nbr_threads,
                                  sc, params, pmf_gen, scratch,
                                  sline_buf[:nsl * step],
                                  stream_idx_buf[:nsl], status_buf[:nsl])

            idx = stream_idx_buf[:nsl]
            lengths = idx[:, 1] - idx[:, 0] + 1
            valid = (status_buf[:nsl] == <int>VALIDSTREAMLINE) | params.return_all
            yield from finalize_chunk(
                sline_buf, rows=np.arange(nsl) * step + idx[:, 0],
                lengths=lengths, valid=valid, min_pts=params.min_nbr_pts,
                max_pts=params.max_nbr_pts, seeds=seeds, sl_seed=sl_seed,
                lin_T=lin_T, offset=offset, nbr_threads=nbr_threads,
                save_seeds=save_seeds,
                chunked=chunked)


def seed_batches(offsets, max_streamlines, *, max_seeds=None):
    """Split seeds into consecutive ranges of at most ``max_streamlines``.

    Parameters
    ----------
    offsets : ndarray, shape (nseed + 1,)
        Exclusive prefix sum of the number of streamlines per seed.
    max_streamlines : int
        Maximum number of streamlines per range. A single seed with more
        streamlines still gets its own range.
    max_seeds : int, optional
        Maximum number of seeds per range.

    Yields
    ------
    start, stop : int
        Seed range ``[start, stop)`` with at least one streamline.
    """
    offsets = np.asarray(offsets)
    n = offsets.shape[0] - 1
    start = 0
    while start < n:
        stop = int(np.searchsorted(offsets, offsets[start] + max_streamlines,
                                   side="right")) - 1
        stop = min(max(stop, start + 1), n)
        if max_seeds is not None:
            stop = min(stop, start + max_seeds)
        if offsets[stop] > offsets[start]:
            yield start, stop
        start = stop


def finalize_chunk(sline, *, rows, lengths, valid, min_pts, max_pts,
                   seeds, sl_seed, lin_T, offset, nbr_threads,
                   save_seeds=False, chunked=False):
    """Select, transform and emit the streamlines of a tracked batch.

    Shared by the CPU tracker and the GPU backends of
    :mod:`dipy.tracking.accel`.

    Parameters
    ----------
    sline : ndarray, shape (nrows, 3)
        Raw streamline buffer, float32 or float64.
    rows : ndarray, shape (nsl,)
        First row of each streamline in ``sline``.
    lengths : ndarray, shape (nsl,)
        Number of points of each streamline.
    valid : ndarray of bool, shape (nsl,) or bool
        Status filter (e.g. ``return_all`` or a valid ending).
    min_pts, max_pts : int
        Range of the number of points of the kept streamlines.
    seeds : ndarray, shape (nseed, 3)
        Seeds in voxel space.
    sl_seed : ndarray, shape (nsl,)
        Index into ``seeds`` of each streamline.
    lin_T, offset : ndarray
        Transposed linear part and translation of the voxel-to-world affine.
    nbr_threads : int
        Number of threads for the copy.
    save_seeds : bool, optional
        Also emit the world-space seed of each streamline.
    chunked : bool, optional
        Emit one ``(points, lengths[, seeds])`` tuple instead of single
        streamlines.

    Yields
    ------
    streamline or chunk tuple
        See :func:`generate_tractogram`.
    """
    lengths = np.asarray(lengths, dtype=np.intp)
    keep = np.flatnonzero(valid & (lengths >= min_pts) & (lengths <= max_pts))
    if keep.size == 0:
        return
    lengths = lengths[keep]
    out_offsets = np.zeros(keep.size + 1, dtype=np.intp)
    np.cumsum(lengths, out=out_offsets[1:])
    points = np.empty((out_offsets[keep.size], 3))
    compact_chunk(sline, np.ascontiguousarray(np.asarray(rows, dtype=np.intp)[keep]),
                  out_offsets, points, lin_T, offset, nbr_threads)
    seeds_out = None
    if save_seeds:
        seeds_out = np.dot(np.asarray(seeds)[np.asarray(sl_seed)[keep]],
                           lin_T) + offset
    if chunked:
        if seeds_out is None:
            yield (points, lengths)
        else:
            yield (points, lengths, seeds_out)
    else:
        yield from _iter_chunk(points, lengths, seeds_out)


def compact_chunk(const floating[:, ::1] sline,
                  cnp.npy_intp[::1] starts,
                  cnp.npy_intp[::1] out_offsets,
                  double[:, ::1] out,
                  double[:, ::1] lin_T,
                  double[::1] offset,
                  int nbr_threads):
    """Copy the kept streamlines out of the chunk buffer, applying the affine.

    ``starts[i]`` is the first row of streamline ``i`` in ``sline``; its points
    land at ``out[out_offsets[i]:out_offsets[i + 1]]``. ``sline`` may be
    float32 (GPU backends) or float64.
    """
    cdef cnp.npy_intp n = starts.shape[0], i, j, k, src, dst
    cdef double x, y, z

    for i in prange(n, nogil=True, num_threads=nbr_threads, schedule="static"):
        src = starts[i]
        dst = out_offsets[i]
        for j in range(out_offsets[i + 1] - dst):
            x = sline[src + j, 0]
            y = sline[src + j, 1]
            z = sline[src + j, 2]
            for k in range(3):
                out[dst + j, k] = (x * lin_T[0, k] + y * lin_T[1, k]
                                   + z * lin_T[2, k] + offset[k])


cdef void generate_tractogram_c(
    double[:, ::1] seeds,
    double[:, ::1] directions,
    cnp.npy_intp[::1] sl_seed,
    cnp.npy_uint64 rng_seed,
    int nbr_threads,
    StoppingCriterion sc,
    TrackerParameters params,
    PmfGen pmf_gen,
    double[:, ::1] scratch,
    double[:, ::1] sline,
    int[:, ::1] stream_idx,
    int[::1] status,
):
    """Generate a tractogram from a set of seed points and directions.

    This is the C implementation of the generate_tractogram function.

    Parameters
    ----------
    seeds : ndarray
        Seed positions (voxel coordinates).
    directions : ndarray, shape (nsl, 3)
        Seed direction of each streamline.
    sl_seed : ndarray, shape (nsl,)
        Index into ``seeds`` of each streamline.
    rng_seed : int
        Base random seed.
    nbr_threads : int
        Number of threads to use for streamline generation.
    sc : StoppingCriterion
        Stopping criterion for the streamlines.
    params : TrackerParameters
        Parameters for the streamline generation.
    pmf_gen : PmfGen
        Probability mass function generator.
    scratch : ndarray, shape (nbr_threads, >= max(100, len(pmf)))
        Per-thread scratch buffer.
    sline : ndarray, shape (nsl * 2 * max_nbr_pts, 3)
        Buffer receiving the streamline points.
    stream_idx : ndarray, shape (nsl, 2)
        First and last point of each streamline in its ``sline`` block.
    status : ndarray, shape (nsl,)
        Status of each streamline.

    """
    cdef:
        cnp.npy_intp nsl = directions.shape[0]
        cnp.npy_intp step = 2 * params.max_nbr_pts
        cnp.npy_intp i

    for i in prange(
        nsl, nogil=True, num_threads=nbr_threads, schedule="dynamic", chunksize=64
    ):
        status[i] = generate_local_streamline(&seeds[sl_seed[i], 0],
                                              &directions[i, 0],
                                              &sline[i * step, 0],
                                              &stream_idx[i, 0],
                                              &scratch[threadid(), 0],
                                              rng_seed,
                                              sc,
                                              params,
                                              pmf_gen)


cdef StreamlineStatus generate_local_streamline(double* seed,
                                                double* direction,
                                                double* stream,
                                                int* stream_idx,
                                                double* stream_data,
                                                cnp.npy_uint64 rng_seed,
                                                StoppingCriterion sc,
                                                TrackerParameters params,
                                                PmfGen pmf_gen) noexcept nogil:
    """Generate a unique streamline from a seed point and direction.

    This is the C implementation.

    Parameters
    ----------
    seed : ndarray
        Seed point for the streamline.
    direction : ndarray
        Seed direction for the streamline.
    stream : ndarray
        Buffer to store the generated streamline.
    stream_idx : ndarray
        Buffer to store the indices of the generated streamline.
    stream_data : double*
        Scratch buffer (>= max(100, len(pmf)) doubles): pmf and propagator
        state.
    rng_seed : int
        Base random seed of the tractogram. With the seed position and
        direction, it keys the Philox stream of the streamline, as on the GPU
        backends (see ``fast_numpy.seed_philox_streamline``).
    sc : StoppingCriterion
        Stopping criterion for the streamline.
    params : TrackerParameters
        Parameters for the streamline generation.
    pmf_gen : PmfGen
        Probability mass function generator.

    """
    cdef:
        cnp.npy_intp i, j
        double[3] point
        double[3] voxdir
        double voxdir_norm
        StreamlineStatus status_forward, status_backward
        fast_numpy.RNGState rng

    fast_numpy.seed_philox_streamline(&rng, rng_seed, seed, direction)

    # set the initial position
    fast_numpy.copy_point(seed, point)
    fast_numpy.copy_point(direction, voxdir)
    fast_numpy.copy_point(seed, &stream[params.max_nbr_pts * 3])
    stream_idx[0] = stream_idx[1] = params.max_nbr_pts

    # the input direction is invalid
    voxdir_norm = fast_numpy.norm(voxdir)
    if voxdir_norm < 0.99 or voxdir_norm > 1.01:
        return INVALIDSTREAMLINE

    # forward tracking
    memset(stream_data, 0, 100 * sizeof(double))
    status_forward = TRACKPOINT
    for i in range(1, params.max_nbr_pts):
        if (
            params.tracker(&point[0], &voxdir[0], params, stream_data, pmf_gen, &rng)
            == TrackerStatus.FAIL
        ):
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] * params.inv_voxel_size[j] * params.step_size
        fast_numpy.copy_point(point, &stream[(params.max_nbr_pts + i)* 3])

        status_forward = sc.check_point_c(point, &rng)
        if (
            status_forward == ENDPOINT
            or status_forward == INVALIDPOINT
            or status_forward == OUTSIDEIMAGE
        ):
            break
    stream_idx[1] = params.max_nbr_pts + i - 1

    # backward tracking
    memset(stream_data, 0, 100 * sizeof(double))

    fast_numpy.copy_point(seed, point)
    fast_numpy.copy_point(direction, voxdir)
    if i > 1:
        # Use the first selected orientation for the backward tracking segment
        for j in range(3):
            voxdir[j] = (stream[(params.max_nbr_pts + 1) * 3 + j]
                         - stream[params.max_nbr_pts * 3 + j])
        fast_numpy.normalize(voxdir)

    # flip the initial direction for backward streamline segment
    for j in range(3):
        voxdir[j] = voxdir[j] * -1

    status_backward = TRACKPOINT
    for i in range(1, params.max_nbr_pts):
        if (
            params.tracker(&point[0], &voxdir[0], params, stream_data, pmf_gen, &rng)
            == TrackerStatus.FAIL
        ):
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] * params.inv_voxel_size[j] * params.step_size
        fast_numpy.copy_point(point, &stream[(params.max_nbr_pts - i)* 3])

        status_backward = sc.check_point_c(point, &rng)
        if (
            status_backward == ENDPOINT
            or status_backward == INVALIDPOINT
            or status_backward == OUTSIDEIMAGE
        ):
            break
    stream_idx[0] = params.max_nbr_pts - i + 1

    # check for valid streamline ending status
    if (
        (status_backward == ENDPOINT or status_backward == OUTSIDEIMAGE)
        and (status_forward == ENDPOINT or status_forward == OUTSIDEIMAGE)
    ):
        return VALIDSTREAMLINE
    return INVALIDSTREAMLINE
