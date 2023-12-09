import numpy as np
cimport numpy as cnp

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


#need cpdef to access cdef functions?
cpdef list generate_tractogram(double[:,::1] seed_positons,
                               double[:,::1] seed_directions,
                               TrackingParameters params):
    cdef:
        cnp.npy_intp _len=seed_positons.shape[0]
        double[:,:,:] streamlines_arr
        double[:] status_arr
        cnp.npy_intp i, j

    #print(_len, flush=True)

    # temporary array to store generated streamlines
    streamlines_arr =  np.array(np.zeros((_len, params.max_len * 2 + 1, 3)), order='C')
    status_arr =  np.array(np.zeros((_len)), order='C')

    generate_tractogram_c(seed_positons,
                          seed_directions,
                          _len, params, streamlines_arr, status_arr)
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


cdef int generate_tractogram_c(double[:,::1] seed_positons,
                               double[:,::1] seed_directions,
                               int nbr_seeds,
                               TrackingParameters params,
                               double[:,:,:] streamlines,
                               double[:] status):
    cdef:
        cnp.npy_intp i, j, k


    # <<cython.parallel.prange>>
    for i in range(nbr_seeds):
        stream = <double*> malloc((params.max_len * 3 * 2 + 1) * sizeof(double))

        # initialize to 0. It will be replaced when better handling various
        # streamline lengtyh
        for j in range(params.max_len * 3 * 2 + 1):
            stream[j] = 0

        status[i] = generate_local_streamline(&seed_positons[i][0],
                                              &seed_directions[i][0],
                                              stream,
                                              params)
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
                                   TrackingParameters params):
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
        if probabilistic_tracker(&point[0], &voxdir[0], params):
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] / params.voxel_size[j] * params.step_size

        copy_point(point, &stream[(params.max_len + i )* 3])

        stream_status_forward = params.sc.check_point_c(point)
        if (stream_status_forward == ENDPOINT or
            stream_status_forward == INVALIDPOINT or
            stream_status_forward == OUTSIDEIMAGE):
            break

    # backward tracking
    copy_point(seed, point)
    copy_point(direction, voxdir)
    for j in range(3):
        voxdir[j] = voxdir[j] * -1
    stream_status_backward = TRACKPOINT
    for i in range(1, params.max_len):
        if probabilistic_tracker(&point[0], &voxdir[0], params):
            break
        # update position
        for j in range(3):
            point[j] += voxdir[j] / params.voxel_size[j] * params.step_size

        copy_point(point, &stream[(params.max_len + i )* 3])


        stream_status_backward = params.sc.check_point_c(point)
        if (stream_status_backward == ENDPOINT or
            stream_status_backward == INVALIDPOINT or
            stream_status_backward == OUTSIDEIMAGE):
            break
    # need to handle stream status
    return 0 #stream_status


cdef double* get_pmf(double* point,
                     PmfGen pmf_gen,
                     double pmf_threshold,
                     int pmf_len):
    cdef:
        cnp.npy_intp i
        double* pmf
        double absolute_pmf_threshold
        double max_pmf=0

    pmf = pmf_gen.get_pmf_c(point)
    for i in range(pmf_len):
        if pmf[i] > max_pmf:
            max_pmf = pmf[i]
    absolute_pmf_threshold = pmf_threshold * max_pmf

    for i in range(pmf_len):
        if pmf[i] < absolute_pmf_threshold:
            pmf[i] = 0.0
    return pmf


cdef int probabilistic_tracker(double* point,
                               double* direction,
                               ProbabilisticTrackingParameters params):
    cdef:
        cnp.npy_intp i, idx
        double[:] newdir
        double* pmf
        double last_cdf, cos_sim

    pmf = get_pmf(point, params.pmf_gen, params.pmf_threshold, params.pmf_len)

    if norm(direction) == 0:
        return 1
    normalize(direction)

    for i in range(params.pmf_len):
        cos_sim = params.vertices[i][0] * direction[0] \
                + params.vertices[i][1] * direction[1] \
                + params.vertices[i][2] * direction[2]
        if cos_sim < 0:
            cos_sim = cos_sim * -1
        if cos_sim < params.cos_similarity:
            pmf[i] = 0

    cumsum(pmf, pmf, params.pmf_len)
    last_cdf = pmf[params.pmf_len - 1]
    if last_cdf == 0:
        return 1

    idx = where_to_insert(pmf, random() * last_cdf, params.pmf_len)

    newdir = params.vertices[idx]
    # Update direction
    if (direction[0] * newdir[0]
        + direction[1] * newdir[1]
        + direction[2] * newdir[2] > 0):
        copy_point(&newdir[0], direction)
    else:
        newdir[0] = newdir[0] * -1
        newdir[1] = newdir[1] * -1
        newdir[2] = newdir[2] * -1
        copy_point(&newdir[0], direction)
    return 0

#get_direction_c of the DG
cdef int deterministic_maximum_tracker(double* point,
                                       double* direction,
                                       DeterministicTrackingParameters params):
    # update point and dir with new position and direction

    # return 1 if the propagation failed.

    return 1

#get_direction_c of the DG
cdef int paralle_transport_tracker(double* point,
                                   double* direction,
                                   ParalleTransportTrackingParameters params):
    # update point and dir with new position and direction

    # return 1 if the propagation failed.

    return 1



cdef class ProbabilisticTrackingParameters(TrackingParameters):

    def __cinit__(self, sc, max_len, step_size, voxel_size, cos_similarity,
                  pmf_threshold, pmf_gen, pmf_len, vertices):
        self.sc = sc
        self.max_len = max_len
        self.step_size = step_size
        self.voxel_size = voxel_size
        self.cos_similarity = cos_similarity
        self.pmf_threshold = pmf_threshold
        self.pmf_gen = pmf_gen
        self.pmf_len = pmf_len
        self.vertices = vertices
