import numpy as np
cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.utils.fast_numpy cimport (copy_point, cumsum, norm, normalize,
                                    where_to_insert, random)


cpdef list generate_tractogram(double[:,::1] seed_positons,
                               double[:,::1] seed_directions,
                               StoppingCriterion sc,
                               TrackingParameters params):
    cdef:
        cnp.npy_intp _len=seed_positons.shape[0]
        double[:,:,:] streamlines

    streamlines =  np.array(np.zeros((_len, 500, 3)), order='F')

    generate_tractogram_c(seed_positons,
                          seed_directions,
                          _len, sc, params, streamlines)

    return []


cdef int generate_tractogram_c(double[:,::1] seed_positons,
                               double[:,::1] seed_directions,
                               int nbr_seeds,
                               StoppingCriterion sc,
                               TrackingParameters params,
                               double[:,:,:] streamlines) nogil:

    cdef:
        cnp.npy_intp i


    #for i in range(nbr_seeds):

    # for loop over all seed position and directions <<cython.parallel.prange>>

        #initialize an empty streamline array
        #do while stream_status is valid forward and backward

            #call the tracker to get the next direction
            #probabilistic_tracker(point, dir, params)

            #stream_status = sc.check_point_c(new_pos)

        # copy the streamline to the results array

    return 1


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
        cnp.npy_intp i, idx,
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

