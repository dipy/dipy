# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as cnp

#from dipy.core.interpolation cimport trilinear_interpolate4d_c
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

from libc.stdlib cimport malloc, free
from libc.math cimport floor
from libc.time cimport time, time_t


#need cpdef to access cdef functions?
cpdef list generate_tractogram(double[:,::1] seed_positions,
                               double[:,::1] seed_directions,
                               StoppingCriterion sc,
                               DeterministicTrackingParameters params,
                               PmfGen pmf_gen):
    cdef:
        cnp.npy_intp _len=seed_positions.shape[0]
        double[:,:,:] streamlines_arr
        double[:] status_arr
        cnp.npy_intp i, j
        time_t t1, t2

    # temporary array to store generated streamlines
    streamlines_arr =  np.array(np.zeros((_len, params.max_len * 2 + 1, 3)), order='C')
    status_arr =  np.array(np.zeros((_len)), order='C')
    print("1")
    t1 = time(NULL)
    generate_tractogram_c(seed_positions, seed_directions, _len, sc, params,
                          pmf_gen, &deterministic_maximum_tracker, streamlines_arr, status_arr)
    print("2")
    t2 = time(NULL)
    print(t2-t1)
    streamlines = []
    for i in range(_len):
        # array to list
        s = []
        for j in range(params.max_len):
            if norm(&streamlines_arr[i,j,0]) > 0:
                s.append(list(np.copy(streamlines_arr[i,j])))
            else:
                break
        if len(s) > 1:
            streamlines.append(np.array(s))

    return streamlines


cdef int generate_tractogram_c(double[:,::1] seed_positions,
                               double[:,::1] seed_directions,
                               int nbr_seeds,
                               StoppingCriterion sc,
                               DeterministicTrackingParameters params,
                               PmfGen pmf_gen,
                               func_ptr tracker,
                               double[:,:,:] streamlines,
                               double[:] status):
    cdef:
        cnp.npy_intp i, j, k

    for i in prange(nbr_seeds, nogil=True):
    #for i in range(nbr_seeds):
        stream = <double*> malloc((params.max_len * 3 * 2 + 1) * sizeof(double))

        # initialize to 0. It will be replaced when better handling various
        # streamline lengtyh
        for j in range(params.max_len * 3 * 2 + 1):
            stream[j] = 0

        status[i] = generate_local_streamline(&seed_positions[i][0],
                                              &seed_directions[i][0],
                                              stream,
                                              tracker,
                                              sc,
                                              params,
                                              pmf_gen)
        # copy the v
        k = 0
        for j in range(params.max_len * 2 + 1):
            if (stream[j * 3] != 0
                and stream[j * 3 + 1] !=0
                and stream[j * 3 + 2] != 0):
                streamlines[i,k,0] = stream[j * 3]
                streamlines[i,k,1] = stream[j * 3 + 1]
                streamlines[i,k,2] = stream[j * 3 + 2]
                k = k + 1

        free(stream)

    return 0


cdef int generate_local_streamline(double* seed,
                                   double* direction,
                                   double* stream,
                                   func_ptr tracker,
                                   StoppingCriterion sc,
                                   DeterministicTrackingParameters params,
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
        if tracker(&point[0], &voxdir[0], params, pmf_gen):  # , pmf_gen_func):
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

    # # backward tracking
    copy_point(seed, point)
    copy_point(direction, voxdir)
    for j in range(3):
        voxdir[j] = voxdir[j] * -1
    stream_status_backward = TRACKPOINT
    for i in range(1, params.max_len):
        if tracker(&point[0], &voxdir[0], params, pmf_gen):
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] * params.inv_voxel_size[j] * params.step_size
        copy_point(point, &stream[(params.max_len + i )* 3])

        stream_status_backward = sc.check_point_c(point)
        if (stream_status_backward == ENDPOINT or
            stream_status_backward == INVALIDPOINT or
            stream_status_backward == OUTSIDEIMAGE):
            break
    # # need to handle stream status
    return 0 #stream_status


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int trilinear_interpolate4d_c(
        double[:, :, :, :] data,
        double* point,
        double* result) noexcept nogil:
    """Tri-linear interpolation along the last dimension of a 4d array

    Parameters
    ----------
    point : 1d array (3,)
        3 doubles representing a 3d point in space. If point has integer values
        ``[i, j, k]``, the result will be the same as ``data[i, j, k]``.
    data : 4d array
        Data to be interpolated.
    result : 1d array
        The result of interpolation. Should have length equal to the
        ``data.shape[3]``.
    Returns
    -------
    err : int
         0 : successful interpolation.
        -1 : point is outside the data area, meaning round(point) is not a
             valid index to data.
        -2 : point has the wrong shape
        -3 : shape of data and result do not match

    """
    cdef:
        cnp.npy_intp flr, N
        double w, rem
        cnp.npy_intp index[3][2]
        double weight[3][2]

    #if data.shape[3] != result.shape[0]:
    #    return -3

    for i in range(3):
        if point[i] < -.5 or point[i] >= (data.shape[i] - .5):
            return -1

        flr = <cnp.npy_intp> floor(point[i])
        rem = point[i] - flr

        index[i][0] = flr + (flr == -1)
        index[i][1] = flr + (flr != (data.shape[i] - 1))
        weight[i][0] = 1 - rem
        weight[i][1] = rem

    N = data.shape[3]
    for i in range(N):
        result[i] = 0

    for i in range(2):
        for j in range(2):
            for k in range(2):
                w = weight[0][i] * weight[1][j] * weight[2][k]
                for L in range(N):
                    result[L] += w * data[index[0][i], index[1][j],
                                          index[2][k], L]
    return 0


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
                               ProbabilisticTrackingParameters params,
                               PmfGen pmf_gen) noexcept nogil:
    cdef:
        cnp.npy_intp i, idx
        double* newdir
        double* pmf
        double last_cdf, cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]
        # This requires the GIL
        # The problem is instanciating double[:] without GIL
        #double[:] pmf = cython.view.array(size = len_pmf,
        #                                  itemsize = sizeof(double),
        #                                  format = "double",
        #                                  allocate_buffer = True)

    pmf = <double*> malloc(len_pmf * sizeof(double))
    if get_pmf(pmf, point, pmf_gen, params.pmf_threshold, len_pmf):
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
        if cos_sim < params.cos_similarity:
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
                                       DeterministicTrackingParameters params,
                                       PmfGen pmf_gen) noexcept nogil:
    cdef:
        cnp.npy_intp i, max_idx
        double max_value=0
        double* newdir
        double* pmf
        double cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]

    pmf = <double*> malloc(len_pmf * sizeof(double))
    if get_pmf(pmf, point, pmf_gen, params.pmf_threshold, len_pmf):
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
        if cos_sim > params.cos_similarity and pmf[i] > max_value:
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
                                    ParallelTransportTrackingParameters params,
                                    PmfGen pmf_gen) noexcept nogil:
    # update point and dir with new position and direction

    # return 1 if the propagation failed.

    return 1



cdef class ProbabilisticTrackingParameters(TrackingParameters):

    def __cinit__(self, int max_len, double step_size, double[:] voxel_size,
                  double cos_similarity, double pmf_threshold):
        cdef:
            cnp.npy_intp i
        self.max_len = max_len
        self.step_size = step_size
        self.voxel_size = voxel_size
        self.cos_similarity = cos_similarity
        self.pmf_threshold = pmf_threshold

        for i in range(3):
            self.inv_voxel_size[i] = 1. / voxel_size[i]
