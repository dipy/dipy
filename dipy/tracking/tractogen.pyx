# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

from libc.stdio cimport printf

cimport ctime
cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.utils cimport fast_numpy

from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT,
                                               VALIDSTREAMLIME,
                                               INVALIDSTREAMLIME)
from dipy.tracking.tracker_parameters cimport (TrackerParameters,
                                               TrackerStatus,
                                               func_ptr)

from nibabel.streamlines import ArraySequence as Streamlines

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy, memset
from libc.math cimport ceil
from libc.stdio cimport printf


def generate_tractogram(double[:,::1] seed_positions,
                        double[:,::1] seed_directions,
                        StoppingCriterion sc,
                        TrackerParameters params,
                        PmfGen pmf_gen,
                        affine,
                        int nbr_threads=0,
                        float buffer_frac=1.0,
                        bint save_seeds=0):
    """Generate a tractogram from a set of seed points and directions.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions for the streamlines.
    seed_directions : ndarray
        Seed directions for the streamlines.
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
    buffer_frac : float, optional
        Fraction of the seed points to process in each iteration.
    save_seeds : bool, optional
        If True, return seeds alongside streamlines

    Yields
    ------
    streamlines : Streamlines
        Streamlines generated from the seed points.
    seeds : ndarray, optional
        seed points associated with the generated streamlines.

    """
    cdef:
        cnp.npy_intp _len = seed_positions.shape[0]
        cnp.npy_intp _plen = int(ceil(_len * buffer_frac))
        cnp.npy_intp i, seed_start, seed_end
        double** streamlines_arr
        int* length_arr
        StreamlineStatus* status_arr

    if buffer_frac <=0 or buffer_frac > 1:
        raise ValueError("buffer_frac must > 0 and <= 1.")

    lin_T = affine[:3, :3].T.copy()
    offset = affine[:3, 3].copy()

    inv_affine = np.linalg.inv(affine)
    seed_positions = np.dot(seed_positions, inv_affine[:3, :3].T.copy())
    seed_positions += inv_affine[:3, 3]

    seed_start = 0
    seed_end = _plen
    while seed_start < _len:
        streamlines_arr = <double**> malloc(_plen * sizeof(double*))
        length_arr = <int*> malloc(_plen * sizeof(int))
        status_arr = <StreamlineStatus*> malloc(_plen * sizeof(int))

        if streamlines_arr == NULL or length_arr == NULL or status_arr == NULL:
            raise MemoryError("Memory allocation failed")

        generate_tractogram_c(seed_positions[seed_start:seed_end],
                              seed_directions[seed_start:seed_end],
                              nbr_threads, sc, params, pmf_gen,
                              streamlines_arr, length_arr, status_arr)

        for i in range(seed_end - seed_start):
            if ((status_arr[i] == VALIDSTREAMLIME or params.return_all)
                and (length_arr[i] >= params.min_nbr_pts
                     and length_arr[i] <= params.max_nbr_pts)):
                s = np.asarray(<cnp.float_t[:length_arr[i]*3]> streamlines_arr[i])
                track = s.copy().reshape((-1,3))
                if save_seeds:
                    yield np.dot(track, lin_T) + offset, np.dot(seed_positions[seed_start + i], lin_T) + offset
                else:
                    yield np.dot(track, lin_T) + offset
            free(streamlines_arr[i])

        free(streamlines_arr)
        free(length_arr)
        free(status_arr)

        seed_start += _plen
        seed_end += _plen
        if seed_end > _len:
            seed_end = _len


cdef void generate_tractogram_c(double[:,::1] seed_positions,
                                double[:,::1] seed_directions,
                               int nbr_threads,
                               StoppingCriterion sc,
                               TrackerParameters params,
                               PmfGen pmf_gen,
                               double** streamlines,
                               int* lengths,
                               StreamlineStatus* status):
    """Generate a tractogram from a set of seed points and directions.

    This is the C implementation of the generate_tractogram function.

    Parameters
    ----------
    seed_positions : ndarray
        Seed positions for the streamlines.
    seed_directions : ndarray
        Seed directions for the streamlines.
    nbr_threads : int
        Number of threads to use for streamline generation.
    sc : StoppingCriterion
        Stopping criterion for the streamlines.
    params : TrackerParameters
        Parameters for the streamline generation.
    pmf_gen : PmfGen
        Probability mass function generator.
    streamlines : list
        List to store the generated streamlines.
    lengths : list
        List to store the lengths of the generated streamlines.
    status : list
        List to store the status of the generated streamlines.

    """
    cdef:
        cnp.npy_intp _len=seed_positions.shape[0]
        cnp.npy_intp i

    if nbr_threads<= 0:
        nbr_threads = 0
    for i in prange(_len, nogil=True, num_threads=nbr_threads):
        stream = <double*> malloc((params.max_nbr_pts * 3 * 2 + 1) * sizeof(double))
        stream_idx = <int*> malloc(2 * sizeof(int))
        status[i] = generate_local_streamline(&seed_positions[i][0],
                                              &seed_directions[i][0],
                                              stream,
                                              stream_idx,
                                              sc,
                                              params,
                                              pmf_gen)

        # copy the streamlines points from the buffer to a 1d vector of the streamline length
        lengths[i] = stream_idx[1] - stream_idx[0] + 1
        streamlines[i] = <double*> malloc(lengths[i] * 3 * sizeof(double))
        memcpy(&streamlines[i][0], &stream[stream_idx[0] * 3], lengths[i] * 3 * sizeof(double))
        free(stream)
        free(stream_idx)


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
        return INVALIDSTREAMLIME

    # forward tracking
    stream_data = <double*> malloc(100 * sizeof(double))
    memset(stream_data, 0, 100 * sizeof(double))
    status_forward = TRACKPOINT
    for i in range(1, params.max_nbr_pts):
        if params.tracker(&point[0], &voxdir[0], params, stream_data, pmf_gen, &rng) == TrackerStatus.FAIL:
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] * params.inv_voxel_size[j] * params.step_size
        fast_numpy.copy_point(point, &stream[(params.max_nbr_pts + i )* 3])

        status_forward = sc.check_point_c(point, &rng)
        if (status_forward == ENDPOINT or
            status_forward == INVALIDPOINT or
            status_forward == OUTSIDEIMAGE):
            break
    stream_idx[1] = params.max_nbr_pts + i - 1
    free(stream_data)

    # backward tracking
    stream_data = <double*> malloc(100 * sizeof(double))
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
        if params.tracker(&point[0], &voxdir[0], params, stream_data, pmf_gen, &rng) == TrackerStatus.FAIL:
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] * params.inv_voxel_size[j] * params.step_size
        fast_numpy.copy_point(point, &stream[(params.max_nbr_pts - i )* 3])

        status_backward = sc.check_point_c(point, &rng)
        if (status_backward == ENDPOINT or
            status_backward == INVALIDPOINT or
            status_backward == OUTSIDEIMAGE):
            break
    stream_idx[0] = params.max_nbr_pts - i + 1
    free(stream_data)

    # check for valid streamline ending status
    if ((status_backward == ENDPOINT or status_backward == OUTSIDEIMAGE)
        and (status_forward == ENDPOINT or status_forward == OUTSIDEIMAGE)):
        return VALIDSTREAMLIME
    return INVALIDSTREAMLIME


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
    cdef:
        cnp.npy_intp i
        double absolute_pmf_threshold
        double max_pmf=0

    pmf = pmf_gen.get_pmf_c(point, pmf)

    for i in range(pmf_len):
        if pmf[i] > max_pmf:
            max_pmf = pmf[i]
    absolute_pmf_threshold = pmf_threshold * max_pmf

    for i in range(pmf_len):
        if pmf[i] < absolute_pmf_threshold:
            pmf[i] = 0.0
