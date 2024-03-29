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

from nibabel.streamlines import ArraySequence as Streamlines

from libc.stdlib cimport malloc, free
from libc.string cimport memcpy
from libc.math cimport floor


def generate_tracking_parameters(algo_name, *,
    int max_len=500, double step_size=0.5, double[:] voxel_size,
    double max_angle=30, double pmf_threshold=0.1, double probe_length=0.5,
    double probe_radius=0, int probe_quality=3, int probe_count=1,
    double data_support_exponent=1):

    algo_name = algo_name.lower()

    if algo_name in ['deterministic', 'det']:
        return TrackingParameters(max_len=max_len, step_size=step_size,
                                  voxel_size=voxel_size)
    elif algo_name in ['probabilistic', 'prob']:
        return TrackingParameters(max_len=max_len, step_size=step_size,
                                  voxel_size=voxel_size,
                                  pmf_threshold=pmf_threshold,
                                  max_angle=max_angle)
    elif algo_name == 'pft':
        return TrackingParameters(max_len=max_len, step_size=step_size,
                                  voxel_size=voxel_size)
    elif algo_name == 'ptt':
            return TrackingParameters(max_len=max_len, step_size=step_size,
                                      voxel_size=voxel_size, probe_length=probe_length,
                                      probe_radius=probe_radius, probe_quality=probe_quality,
                                      probe_count=probe_count, data_support_exponent=data_support_exponent)
    elif algo_name == 'eudx':
        return TrackingParameters(max_len=max_len, step_size=step_size,
                                  voxel_size=voxel_size)
    else:
        raise ValueError("Invalid algorithm name")


def generate_tractogram(double[:,::1] seed_positions,
                        double[:,::1] seed_directions,
                        StoppingCriterion sc,
                        TrackingParameters params,
                        PmfGen pmf_gen,
                        int nbr_threads=0):

    cdef:
        cnp.npy_intp _len=seed_positions.shape[0]
        cnp.npy_intp i
        double** streamlines_arr = <double**> malloc(_len * sizeof(double*))
        int* length_arr = <int*> malloc(_len * sizeof(int))
        int* status_arr = <int*> malloc(_len * sizeof(double))

    if streamlines_arr == NULL or length_arr == NULL or status_arr == NULL:
        raise MemoryError("Memory allocation failed")

    # Todo: Check if probalistic parameters are set if using probabilistic
    # tracking. Same for PTT
    generate_tractogram_c(seed_positions, seed_directions, nbr_threads, sc, params,
                          pmf_gen, &probabilistic_tracker, #&deterministic_maximum_tracker,
                          streamlines_arr, length_arr, status_arr)
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

    return streamlines


cdef int generate_tractogram_c(double[:,::1] seed_positions,
                               double[:,::1] seed_directions,
                               int nbr_threads,
                               StoppingCriterion sc,
                               TrackingParameters params,
                               PmfGen pmf_gen,
                               func_ptr tracker,
                               double** streamlines,
                               int* lengths,
                               int* status):
    cdef:
        cnp.npy_intp _len=seed_positions.shape[0]
        cnp.npy_intp i, j, k

    if nbr_threads<= 0:
        nbr_threads = 0
    for i in prange(_len, nogil=True, num_threads=nbr_threads):
    #for i in range(_len):
        stream = <double*> malloc((params.max_len * 3 * 2 + 1) * sizeof(double))
        stream_idx = <int*> malloc(2 * sizeof(int))
        status[i] = generate_local_streamline(&seed_positions[i][0],
                                              &seed_directions[i][0],
                                              stream,
                                              stream_idx,
                                              tracker,
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
                                   func_ptr tracker,
                                   StoppingCriterion sc,
                                   TrackingParameters params,
                                   PmfGen pmf_gen) noexcept nogil:
    cdef:
        cnp.npy_intp i, j
        double point[3]
        double voxdir[3]
        StreamlineStatus stream_status_forward, stream_status_backward

    # set the initial position
    copy_point(seed, point)
    copy_point(direction, voxdir)
    copy_point(seed, &stream[params.max_len * 3])

    # forward tracking
    stream_status_forward = TRACKPOINT
    for i in range(1, params.max_len):
        if tracker(&point[0], &voxdir[0], params, pmf_gen):
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
        if tracker(&point[0], &voxdir[0], params, pmf_gen):
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

    if trilinear_interpolate4d_c(pmf_gen.data, point, pmf):
        return 1

    for i in range(pmf_len):
        if pmf[i] > max_pmf:
            max_pmf = pmf[i]
    absolute_pmf_threshold = pmf_threshold * max_pmf

    for i in range(pmf_len):
        if pmf[i] < absolute_pmf_threshold:
            pmf[i] = 0.0

    return 0


cdef int probabilistic_tracker(double* point,
                               double* direction,
                               TrackingParameters params,
                               PmfGen pmf_gen) noexcept nogil:
    cdef:
        cnp.npy_intp i, idx
        double* newdir
        double* pmf
        double last_cdf, cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]

    pmf = <double*> malloc(len_pmf * sizeof(double))
    # printf("len_pmf %f\n", params.probabilistic.pmf_threshold)
    if get_pmf(pmf, point, pmf_gen, params.probabilistic.pmf_threshold, len_pmf):
        free(pmf)
        return 1
    if norm(direction) == 0:
        free(pmf)
        return 1
    normalize(direction)

    for i in range(len_pmf):
        cos_sim = pmf_gen.vertices[i][0] * direction[0] \
                + pmf_gen.vertices[i][1] * direction[1] \
                + pmf_gen.vertices[i][2] * direction[2]
        if cos_sim < 0:
            cos_sim = cos_sim * -1
        if cos_sim < params.probabilistic.cos_similarity:
            pmf[i] = 0

    cumsum(pmf, pmf, len_pmf)
    last_cdf = pmf[len_pmf - 1]
    if last_cdf == 0:
        free(pmf)
        return 1

    idx = where_to_insert(pmf, random() * last_cdf, len_pmf)
    newdir = &pmf_gen.vertices[idx][0]
    # Update direction
    if (direction[0] * newdir[0]
        + direction[1] * newdir[1]
        + direction[2] * newdir[2] > 0):
        copy_point(newdir, direction)
    else:
        copy_point(newdir, direction)
        direction[0] = direction[0] * -1
        direction[1] = direction[1] * -1
        direction[2] = direction[2] * -1
    free(pmf)
    return 0


cdef int deterministic_maximum_tracker(double* point,
                                       double* direction,
                                       TrackingParameters params,
                                       PmfGen pmf_gen) noexcept nogil:
    cdef:
        cnp.npy_intp i, max_idx
        double max_value=0
        double* newdir
        double* pmf
        double cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]

    pmf = <double*> malloc(len_pmf * sizeof(double))
    if get_pmf(pmf, point, pmf_gen, params.probabilistic.pmf_threshold, len_pmf):
        free(pmf)
        return 1
    if norm(direction) == 0:
        free(pmf)
        return 1
    normalize(direction)

    for i in range(len_pmf):
        cos_sim = pmf_gen.vertices[i][0] * direction[0] \
                + pmf_gen.vertices[i][1] * direction[1] \
                + pmf_gen.vertices[i][2] * direction[2]
        if cos_sim < 0:
            cos_sim = cos_sim * -1
        if cos_sim > params.probabilistic.cos_similarity and pmf[i] > max_value:
            max_idx = i
            max_value = pmf[i]

    if max_value <= 0:
        free(pmf)
        return 1

    newdir = &pmf_gen.vertices[max_idx][0]
    # Update direction
    if (direction[0] * newdir[0]
        + direction[1] * newdir[1]
        + direction[2] * newdir[2] > 0):
        copy_point(newdir, direction)
    else:
        copy_point(newdir, direction)
        direction[0] = direction[0] * -1
        direction[1] = direction[1] * -1
        direction[2] = direction[2] * -1
    free(pmf)
    return 0

#get_direction_c of the DG
cdef int parallel_transport_tracker(double* point,
                                    double* direction,
                                    TrackingParameters params,
                                    PmfGen pmf_gen) noexcept nogil:
    # update point and dir with new position and direction

    # return 1 if the propagation failed.

    return 1


cdef class TrackingParameters:

    def __init__(self, max_len, step_size, voxel_size,
                 max_angle=None, pmf_threshold=None, probe_length=None,
                 probe_radius=None, probe_quality=None, probe_count=None,
                 data_support_exponent=None):
        cdef cnp.npy_intp i

        self.max_len = max_len
        self.step_size = step_size
        for i in range(3):
            self.voxel_size[i] = voxel_size[i]
            self.inv_voxel_size[i] = 1. / voxel_size[i]

        self.probabilistic = None
        self.ptt = None

        if max_angle is not None and pmf_threshold is not None:
            self.probabilistic = ProbabilisticTrackingParameters(max_angle, pmf_threshold)

        if probe_length is not None and probe_radius is not None and probe_quality is not None and probe_count is not None and data_support_exponent is not None:
            self.ptt = PttTrackingParameters(probe_length, probe_radius, probe_quality, probe_count, data_support_exponent)

cdef class ProbabilisticTrackingParameters:

    def __init__(self, max_angle, pmf_threshold):
        self.cos_similarity = np.cos(np.deg2rad(max_angle))
        self.pmf_threshold = pmf_threshold

cdef class PttTrackingParameters:

    def __init__(self, double probe_length, double probe_radius,
                int probe_quality, int probe_count, double data_support_exponent):
        self.probe_length = probe_length
        self.probe_radius = probe_radius
        self.probe_quality = probe_quality
        self.probe_count = probe_count
        self.data_support_exponent = data_support_exponent


