# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

cimport ctime
from cython.parallel import prange, threadid
import numpy as np
cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.reconst.dirspeed cimport peak_directions_c
from dipy.tracking._utils import _gather_chunk, _iter_chunk
from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.utils cimport fast_numpy
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

from libc.stdlib cimport malloc, free
from libc.string cimport memset


def generate_tractogram(double[:, ::1] seed_positions,
                        seed_directions,
                        StoppingCriterion sc,
                        TrackerParameters params,
                        PmfGen pmf_gen,
                        affine,
                        int nbr_threads=0,
                        int chunk_size=25000,
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
        Number of seeds tracked at once. Lower it to reduce memory usage.
    save_seeds : bool, optional
        If True, return seeds alongside streamlines
    max_cross : int, optional
        Maximum number of peaks tracked per seed when ``seed_directions`` is
        None. Use all peaks when <= 0.
    relative_peak_threshold, min_separation_angle : float, optional
        Peak selection parameters when ``seed_directions`` is None (see
        :func:`dipy.direction.peak_directions`).
    chunked : bool, optional
        If True, yield one ``(points, lengths)`` tuple per chunk of seeds
        (``(points, lengths, seeds)`` with ``save_seeds``), where ``points``
        is the concatenation of the chunk's streamlines. Much cheaper to
        consume than one streamline at a time, e.g. with
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
        cnp.npy_intp start, n, nsl
        double[:, ::1] seeds, dirs, sline
        cnp.npy_intp[::1] offsets, sl_seed
        int[:, ::1] stream_idx
        int[::1] status

    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0.")
    if nbr_threads <= 0:
        nbr_threads = determine_num_threads(None)

    lin_T = affine[:3, :3].T.copy()
    offset = affine[:3, 3].copy()

    inv_affine = np.linalg.inv(affine)
    seed_positions = np.dot(seed_positions, inv_affine[:3, :3].T.copy())
    seed_positions += inv_affine[:3, 3]
    if seed_directions is not None:
        seed_directions = np.ascontiguousarray(seed_directions, dtype=float)

    for start in range(0, nseed, chunk_size):
        n = min(chunk_size, nseed - start)
        seeds = seed_positions[start:start + n]
        if seed_directions is None:
            offsets, dirs = seed_peaks(seeds, pmf_gen, params, nbr_threads,
                                       max_cross, relative_peak_threshold,
                                       min_separation_angle)
        else:
            offsets = np.arange(n + 1)
            dirs = seed_directions[start:start + n]
        nsl = offsets[n]
        sl_seed = np.repeat(np.arange(n), np.diff(offsets))
        sline = np.empty((nsl * step, 3))
        stream_idx = np.empty((nsl, 2), dtype=np.intc)
        status = np.empty(nsl, dtype=np.intc)

        generate_tractogram_c(seeds, dirs, sl_seed, nbr_threads, sc, params,
                              pmf_gen, sline, stream_idx, status)

        idx = np.asarray(stream_idx)
        lengths = idx[:, 1] - idx[:, 0] + 1
        keep = np.flatnonzero(
            ((np.asarray(status) == <int>VALIDSTREAMLINE) | params.return_all)
            & (lengths >= params.min_nbr_pts) & (lengths <= params.max_nbr_pts))
        lengths = lengths[keep]
        points = _gather_chunk(sline, keep * step + idx[keep, 0], lengths,
                               lin_T, offset)
        seeds_out = None
        if save_seeds:
            seeds_out = np.dot(
                np.asarray(seeds)[
                    np.asarray(sl_seed)[keep]], lin_T) + offset
        if chunked:
            if seeds_out is None:
                yield (points, lengths)
            else:
                yield (points, lengths, seeds_out)
        else:
            yield from _iter_chunk(points, lengths, seeds_out)


def seed_peaks(double[:, ::1] seeds,
               PmfGen pmf_gen,
               TrackerParameters params,
               int nbr_threads=0,
               int max_cross=-1,
               double relative_peak_threshold=0.5,
               double min_separation_angle=25):
    """Peaks of the pmf at each seed (voxel coordinates), in parallel.

    Returns
    -------
    offsets : ndarray, shape (nseed + 1,)
        Exclusive prefix sum of the number of peaks per seed.
    directions : ndarray, shape (offsets[-1], 3)
        Peak directions, sorted by decreasing peak value within each seed.
    """
    cdef:
        cnp.npy_intp n = seeds.shape[0], dimt = pmf_gen.pmf.shape[0]
        cnp.npy_intp cap = dimt if max_cross <= 0 else max_cross
        cnp.npy_intp i, j, k, t
        double[:, ::1] verts = np.ascontiguousarray(pmf_gen.vertices)
        cnp.uint16_t[:, ::1] edges = np.ascontiguousarray(
            pmf_gen.sphere.edges, dtype=np.uint16)
        cnp.npy_intp[::1] counts, offsets
        double[:, ::1] dirs

    if nbr_threads <= 0:
        nbr_threads = determine_num_threads(None)

    # per-thread scratch for peak_directions_c
    cdef:
        double[:, ::1] pmf = np.empty((nbr_threads, dimt))
        double[:, ::1] values = np.empty((nbr_threads, dimt))
        cnp.npy_intp[:, ::1] indices = np.empty((nbr_threads, dimt), dtype=np.intp)
        double[:, :, ::1] out_dirs = np.empty((nbr_threads, dimt, 3))
        double[:, :, ::1] uniq = np.empty((nbr_threads, dimt, 3))
        cnp.uint16_t[:, ::1] mapping = np.empty((nbr_threads, dimt), dtype=np.uint16)
        cnp.uint16_t[:, ::1] index = np.empty((nbr_threads, dimt), dtype=np.uint16)

    counts = np.empty(n, dtype=np.intp)
    for i in prange(n, nogil=True, num_threads=nbr_threads, schedule="dynamic",
                    chunksize=64):
        t = threadid()
        counts[i] = _peaks_at(&seeds[i, 0], pmf_gen, verts, edges, params,
                              relative_peak_threshold, min_separation_angle,
                              cap, pmf[t], out_dirs[t], values[t], indices[t],
                              uniq[t], mapping[t], index[t])

    offsets = np.zeros(n + 1, dtype=np.intp)
    np.cumsum(counts, out=np.asarray(offsets)[1:])
    dirs = np.empty((offsets[n], 3))
    for i in prange(n, nogil=True, num_threads=nbr_threads, schedule="dynamic",
                    chunksize=64):
        t = threadid()
        k = _peaks_at(&seeds[i, 0], pmf_gen, verts, edges, params,
                      relative_peak_threshold, min_separation_angle, cap,
                      pmf[t], out_dirs[t], values[t], indices[t], uniq[t],
                      mapping[t], index[t])
        for j in range(k):
            dirs[offsets[i] + j, 0] = verts[indices[t, j], 0]
            dirs[offsets[i] + j, 1] = verts[indices[t, j], 1]
            dirs[offsets[i] + j, 2] = verts[indices[t, j], 2]
    return np.asarray(offsets), np.asarray(dirs)


cdef cnp.npy_intp _peaks_at(double* seed,
                            PmfGen pmf_gen,
                            double[:, ::1] verts,
                            cnp.uint16_t[:, ::1] edges,
                            TrackerParameters params,
                            double relative_peak_threshold,
                            double min_separation_angle,
                            cnp.npy_intp cap,
                            double[::1] pmf,
                            double[:, ::1] out_dirs,
                            double[::1] values,
                            cnp.npy_intp[::1] indices,
                            double[:, ::1] uniq,
                            cnp.uint16_t[::1] mapping,
                            cnp.uint16_t[::1] index) noexcept nogil:
    cdef cnp.npy_intp k
    pmf_gen.get_pmf_c(seed, &pmf[0])
    k = peak_directions_c(pmf, verts, edges, relative_peak_threshold,
                          min_separation_angle, params.is_symmetric, out_dirs,
                          values, indices, uniq, mapping, index)
    return k if k < cap else cap


cdef void generate_tractogram_c(
    double[:, ::1] seeds,
    double[:, ::1] directions,
    cnp.npy_intp[::1] sl_seed,
    int nbr_threads,
    StoppingCriterion sc,
    TrackerParameters params,
    PmfGen pmf_gen,
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
    nbr_threads : int
        Number of threads to use for streamline generation.
    sc : StoppingCriterion
        Stopping criterion for the streamlines.
    params : TrackerParameters
        Parameters for the streamline generation.
    pmf_gen : PmfGen
        Probability mass function generator.
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
                                              sc,
                                              params,
                                              pmf_gen)


cdef StreamlineStatus generate_local_streamline(double* seed,
                                                double* direction,
                                                double* stream,
                                                int* stream_idx,
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
    sc : StoppingCriterion
        Stopping criterion for the streamline.
    params : TrackerParameters
        Parameters for the streamline generation.
    pmf_gen : PmfGen
        Probability mass function generator.

    """
    cdef:
        cnp.npy_intp i, j
        cnp.npy_intp n_data = max(100, pmf_gen.pmf.shape[0])
        cnp.npy_uint32 s_random_seed
        double[3] point
        double[3] voxdir
        double voxdir_norm
        double* stream_data
        StreamlineStatus status_forward, status_backward
        fast_numpy.RNGState rng

    # set the random generator
    if params.random_seed > 0:
        s_random_seed = int(
            (seed[0] * 2 + seed[1] * 3 + seed[2] * 5) * params.random_seed
            )
    else:
        s_random_seed = <cnp.npy_uint32>ctime.time_ns()

    fast_numpy.seed_rng(&rng, s_random_seed)

    # set the initial position
    fast_numpy.copy_point(seed, point)
    fast_numpy.copy_point(direction, voxdir)
    fast_numpy.copy_point(seed, &stream[params.max_nbr_pts * 3])
    stream_idx[0] = stream_idx[1] = params.max_nbr_pts

    # the input direction is invalid
    voxdir_norm = fast_numpy.norm(voxdir)
    if voxdir_norm < 0.99 or voxdir_norm > 1.01:
        return INVALIDSTREAMLINE

    # forward tracking. stream_data also serves as pmf scratch (>= len_pmf)
    stream_data = <double*> malloc(n_data * sizeof(double))
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
    free(stream_data)

    # check for valid streamline ending status
    if (
        (status_backward == ENDPOINT or status_backward == OUTSIDEIMAGE)
        and (status_forward == ENDPOINT or status_forward == OUTSIDEIMAGE)
    ):
        return VALIDSTREAMLINE
    return INVALIDSTREAMLINE


cdef void prepare_pmf(double* pmf,
                      double* point,
                      PmfGen pmf_gen,
                      double pmf_threshold,
                      int pmf_len) noexcept nogil:
    """Prepare the probability mass function for streamline generation.

    Parameters
    ----------
    pmf : ndarray
        Probability mass function.
    point : ndarray
        Current tracking position.
    pmf_gen : PmfGen
        Probability mass function generator.
    pmf_threshold : float
        Threshold for the probability mass function.
    pmf_len : int
        Length of the probability mass function.

    """
    pmf_gen.get_pmf_c(point, pmf)
    threshold_pmf(pmf, pmf_len, pmf_threshold)


cdef void threshold_pmf(double* pmf, int pmf_len, double pmf_threshold) noexcept nogil:
    """Zero pmf entries below ``pmf_threshold * max(pmf)``."""
    cdef:
        cnp.npy_intp i
        double max_pmf=0

    for i in range(pmf_len):
        if pmf[i] > max_pmf:
            max_pmf = pmf[i]
    max_pmf *= pmf_threshold

    for i in range(pmf_len):
        if pmf[i] < max_pmf:
            pmf[i] = 0.0
