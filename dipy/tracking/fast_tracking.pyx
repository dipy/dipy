# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

from libc.stdio cimport printf

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as cnp

from dipy.core.interpolation cimport trilinear_interpolate4d_c
from dipy.direction.pmf cimport PmfGen
from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.utils.fast_numpy cimport (copy_point, cumsum, norm, normalize,
                                    where_to_insert, random)

from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion,
                                               TRACKPOINT,
                                               ENDPOINT,
                                               OUTSIDEIMAGE,
                                               INVALIDPOINT)
from dipy.tracking.tracking_parameters cimport TrackingParameters, func_ptr
from dipy.tracking.tracker_probabilistic cimport probabilistic_tracker
from nibabel.streamlines import ArraySequence as Streamlines

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport floor, ceil


def generate_tractogram(double[:,::1] seed_positions,
                        double[:,::1] seed_directions,
                        StoppingCriterion sc,
                        TrackingParameters params,
                        PmfGen pmf_gen,
                        int nbr_threads=0,
                        float buffer_frac=1.0):

    cdef:
        cnp.npy_intp _len = seed_positions.shape[0]
        cnp.npy_intp _plen = int(ceil(_len * buffer_frac))
        cnp.npy_intp i,
        double** streamlines_arr = <double**> malloc(_len * sizeof(double*))
        int* length_arr = <int*> malloc(_len * sizeof(int))
        int* status_arr = <int*> malloc(_len * sizeof(double))

    if streamlines_arr == NULL or length_arr == NULL or status_arr == NULL:
        raise MemoryError("Memory allocation failed")

    # Todo: Check if probalistic parameters are set if using probabilistic
    # tracking. Same for PTT
    seed_start = 0
    seed_end = seed_start + _plen
    while seed_start < _len:
        generate_tractogram_c(seed_positions[seed_start:seed_end],
                              seed_directions[seed_start:seed_end],
                              nbr_threads, sc, params,
                              pmf_gen,
                              streamlines_arr, length_arr, status_arr)
        seed_start += _plen
        seed_end += _plen
        streamlines = []
        try:
            for i in range(_len):
                if length_arr[i] > 1:
                    s = np.asarray(<cnp.float_t[:length_arr[i]*3]> streamlines_arr[i])
                    streamlines.append(s.copy().reshape((-1,3)))
                    if streamlines_arr[i] == NULL:
                        continue
                    free(streamlines_arr[i])
        finally:
            free(streamlines_arr)
            free(length_arr)
            free(status_arr)

        for s in streamlines:
            yield s


cdef int generate_tractogram_c(double[:,::1] seed_positions,
                               double[:,::1] seed_directions,
                               int nbr_threads,
                               StoppingCriterion sc,
                               TrackingParameters params,
                               PmfGen pmf_gen,
                               double** streamlines,
                               int* lengths,
                               int* status):
    cdef:
        cnp.npy_intp _len=seed_positions.shape[0]
        cnp.npy_intp i, j, k

    if nbr_threads<= 0:
        nbr_threads = 0
    for i in prange(_len, nogil=True, num_threads=nbr_threads):
        stream = <double*> malloc((params.max_len * 3 * 2 + 1) * sizeof(double))
        stream_idx = <int*> malloc(2 * sizeof(int))
        status[i] = generate_local_streamline(&seed_positions[i][0],
                                              &seed_directions[i][0],
                                              stream,
                                              stream_idx,
                                              sc,
                                              params,
                                              pmf_gen)

        # copy the streamlines points from the buffer to a 1d vector of the streamline length
        lengths[i] = stream_idx[1] - stream_idx[0]
        if lengths[i] > 1:
            streamlines[i] = <double*> malloc(lengths[i] * 3 * sizeof(double))
            memcpy(&streamlines[i][0], &stream[stream_idx[0] * 3], lengths[i] * 3 * sizeof(double))
        free(stream)
        free(stream_idx)

    return 0


cdef int generate_local_streamline(double* seed,
                                   double* direction,
                                   double* stream,
                                   int* stream_idx,
                                   StoppingCriterion sc,
                                   TrackingParameters params,
                                   PmfGen pmf_gen) noexcept nogil:
    cdef:
        cnp.npy_intp i, j
        double[3] point
        double[3] voxdir
        StreamlineStatus stream_status_forward, stream_status_backward

    # set the initial position
    copy_point(seed, point)
    copy_point(direction, voxdir)
    copy_point(seed, &stream[params.max_len * 3])

    # forward tracking
    stream_status_forward = TRACKPOINT
    for i in range(1, params.max_len):
        if <func_ptr>params.tracker(&point[0], &voxdir[0], params, pmf_gen):  # probabilistic_tracker
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] * params.inv_voxel_size[j] * params.step_size
        copy_point(point, &stream[(params.max_len + i )* 3])

        stream_status_forward = sc.check_point_c(point)
        if (stream_status_forward == ENDPOINT or
            stream_status_forward == INVALIDPOINT or
            stream_status_forward == OUTSIDEIMAGE):
            break
    stream_idx[1] = params.max_len + i -1

    # # backward tracking
    copy_point(seed, point)
    copy_point(direction, voxdir)
    for j in range(3):
        voxdir[j] = voxdir[j] * -1
    stream_status_backward = TRACKPOINT
    for i in range(1, params.max_len):
        ##### VOXDIR should be the real first direction #####
        if <func_ptr>params.tracker(&point[0], &voxdir[0], params, pmf_gen):  # probabilistic_tracker
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] * params.inv_voxel_size[j] * params.step_size
        copy_point(point, &stream[(params.max_len - i )* 3])

        stream_status_backward = sc.check_point_c(point)
        if (stream_status_backward == ENDPOINT or
            stream_status_backward == INVALIDPOINT or
            stream_status_backward == OUTSIDEIMAGE):
            break
    stream_idx[0] = params.max_len - i + 1
    # # need to handle stream status
    return 0 #stream_status


cdef int get_pmf(double* pmf,
                 double* point,
                 PmfGen pmf_gen,
                 double pmf_threshold,
                 int pmf_len) noexcept nogil:
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

    return 0








