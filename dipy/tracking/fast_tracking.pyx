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


cpdef list generate_tractogram(double[:,::1] seed_positons,
                               double[:,::1] seed_directions,
                               TrackingParameters params):
    cdef:
        cnp.npy_intp _len=seed_positons.shape[0]
        double[:,:,:] streamlines_arr
        double[:] status_arr

    # temporary array to store generated streamlines
    streamlines_arr =  np.array(np.zeros((_len, params.max_len, 3)), order='C')
    status_arr =  np.array(np.zeros((_len)) - 2, order='C')

    generate_tractogram_c(seed_positons,
                          seed_directions,
                          _len, params, streamlines_arr, status_arr)

    streamlines = []
    for i in range(_len):
        # array to list
        pass

    return streamlines


cdef int generate_tractogram_c(double[:,::1] seed_positons,
                               double[:,::1] seed_directions,
                               int nbr_seeds,
                               TrackingParameters params,
                               double[:,:,:] streamlines,
                               double[:] status) nogil:
    cdef:
        cnp.npy_intp i, j, k
        double *stream

    # <<cython.parallel.prange>>
    for i in range(nbr_seeds):
        stream_x = <double*> malloc(params.max_len *  sizeof(double))
        stream_y = <double*> malloc(params.max_len *  sizeof(double))
        stream_z = <double*> malloc(params.max_len *  sizeof(double))

        status[i] = generate_local_streamline(seed_positons[i],
                                              seed_directions[i],
                                              stream_x,
                                              stream_y,
                                              stream_z,
                                              params)
        for j in range(params.max_len):
                streamlines[i,j,0] = stream_x[j]
                streamlines[i,j,1] = stream_y[j]
                streamlines[i,j,2] = stream_z[j]
        free(stream_x)
        free(stream_y)
        free(stream_z)

    return 0


cdef int generate_local_streamline(double[::1] seed,
                                   double[::1] direction,
                                   double[::1] stream_x,
                                   double[::1] stream_y,
                                   double[::1] stream_z,
                                   TrackingParameters params) nogil:
    cdef:
        cnp.npy_intp i, j
        double point[3]
        double voxdir[3]
        StreamlineStatus stream_status

    # set the initial position
    copy_point(&seed[0], point)
    #copy_point(&seed[0], &streamline[0,0])
    stream_x[i] = seed[0]
    stream_y[i] = seed[1]
    stream_z[i] = seed[2]

    stream_status = TRACKPOINT
    for i in range(1, params.max_len):
        if probabilistic_tracker(point, direction, params):
            break

        # update position
        for j in range(3):
            point[j] += direction[j] / params.voxel_size[j] * params.step_size

        #copy_point(point, &streamline[i, 0])
        stream_x[i] = point[0]
        stream_y[i] = point[1]
        stream_z[i] = point[2]

        stream_status = params.sc.check_point_c(point)
        if stream_status == TRACKPOINT:
            continue
        elif (stream_status == ENDPOINT or
              stream_status == INVALIDPOINT or
              stream_status == OUTSIDEIMAGE):
            break
    else:
        # maximum length of streamline has been reached, return everything
        i = params.max_len

        # i should be return to know the length of the streamline
    return stream_status


cdef double* get_pmf(double* point,
                     PmfGen pmf_gen,
                     double pmf_threshold,
                     int pmf_len) nogil:
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
                               ProbabilisticTrackingParameters params) nogil:
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
    # Update direction and return 0 for error
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
                                       DeterministicTrackingParameters params) nogil:
    # update point and dir with new position and direction

    # return 1 if the propagation failed.

    return 1

#get_direction_c of the DG
cdef int paralle_transport_tracker(double* point,
                                   double* direction,
                                   ParalleTransportTrackingParameters params) nogil:
    # update point and dir with new position and direction

    # return 1 if the propagation failed.

    return 1

