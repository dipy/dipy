# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False


cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as cnp

from dipy.direction.pmf cimport PmfGen
from dipy.utils.fast_numpy cimport (copy_point, cross, cumsum, norm, normalize, random,
                                    random_perpendicular_vector,
                                    random_point_within_circle, where_to_insert)
from dipy.tracking.propspeed cimport (
    calculate_ptt_data_support,
    initialize_ptt,
    prepare_ptt_propagator,
)
from dipy.tracking.fast_tracking cimport prepare_pmf
from dipy.tracking.tracker_parameters cimport (TrackerParameters, TrackerStatus,
                                               SUCCESS, FAIL)
from libc.stdlib cimport malloc, free
from libc.math cimport M_PI, pow, sin, cos, fabs
from libc.stdio cimport printf


cdef TrackerStatus deterministic_tracker(double* point,
                                         double* direction,
                                         TrackerParameters params,
                                         double* stream_data,
                                         PmfGen pmf_gen) noexcept nogil:
    """
    Propagate the position by step_size amount.

    The propagation use the direction of a sphere with the highest probability
    mass function (pmf).

    Parameters
    ----------
    point : double[3]
        Current tracking position.
    direction : double[3]
        Previous tracking direction.
    params : TrackerParameters
        Deterministic Tractography parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    pmf_gen : PmfGen
        Orientation data.

    Returns
    -------
    status : TrackerStatus
        Returns SUCCESS if the propagation was successful, or
        FAIL otherwise.
    """
    cdef:
        cnp.npy_intp i, max_idx
        double max_value=0
        double* newdir
        double* pmf
        double cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]

    if norm(direction) == 0:
        return FAIL
    normalize(direction)

    pmf = <double*> malloc(len_pmf * sizeof(double))
    prepare_pmf(pmf, point, pmf_gen, params.sh.pmf_threshold, len_pmf)

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
        return FAIL

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
    return SUCCESS


cdef TrackerStatus probabilistic_tracker(double* point,
                                         double* direction,
                                         TrackerParameters params,
                                         double* stream_data,
                                         PmfGen pmf_gen) noexcept nogil:
    """
    Propagates the position by step_size amount. The propagation use randomly samples
    direction of a sphere based on probability mass function (pmf).

    Parameters
    ----------
    point : double[3]
        Current tracking position.
    direction : double[3]
        Previous tracking direction.
    params : TrackerParameters
        Parallel Transport Tractography (PTT) parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    pmf_gen : PmfGen
        Orientation data.

    Returns
    -------
    status : TrackerStatus
        Returns SUCCESS if the propagation was successful, or
        FAIL otherwise.
    """

    cdef:
        cnp.npy_intp i, idx
        double* newdir
        double* pmf
        double last_cdf, cos_sim
        cnp.npy_intp len_pmf=pmf_gen.pmf.shape[0]

    if norm(direction) == 0:
        return FAIL
    normalize(direction)

    pmf = <double*> malloc(len_pmf * sizeof(double))
    prepare_pmf(pmf, point, pmf_gen, params.sh.pmf_threshold, len_pmf)

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
        return FAIL

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
    return SUCCESS


cdef TrackerStatus parallel_transport_tracker(double* point,
                                              double* direction,
                                              TrackerParameters params,
                                              double* stream_data,
                                              PmfGen pmf_gen) noexcept nogil:
    """
    Propagates the position by step_size amount. The propagation is using
    the parameters of the last candidate curve. Then, randomly generate
    curve parametrization from the current position. The walking frame
    is the same, only the k1 and k2 parameters are randomly picked.
    Rejection sampling is used to pick the next curve using the data
    support (likelihood).

    stream_data:
        0    : initialized
        1-10 : frame1,2,3
        10-19: propagator
        19-22: position
        22   : last_val
        23   : last_val_cand
        24   : k1
        25   : k2

    Parameters
    ----------
    point : double[3]
        Current tracking position.
    direction : double[3]
        Previous tracking direction.
    params : TrackerParameters
        Parallel Transport Tractography (PTT) parameters.
    stream_data : double*
        Streamline data persitant across tracking steps.
    pmf_gen : PmfGen
        Orientation data.

    Returns
    -------
    status : TrackerStatus
        Returns SUCCESS if the propagation was successful, or
        FAIL otherwise.
    """

    cdef double max_posterior = 0
    cdef double data_support = 0
    cdef double[3] tangent
    cdef int tries
    cdef int i

    if stream_data[0] == 0:
        initialize_ptt(params, stream_data, pmf_gen, point, direction)
        stream_data[0] = 1  # initialized

    prepare_ptt_propagator(params, stream_data, params.step_size)

    for i in range(3):
        #  position
        stream_data[19 + i] = (stream_data[10] * stream_data[1 + i]
                               * params.inv_voxel_size[i]
                               + stream_data[11] * stream_data[4 + i]
                               * params.inv_voxel_size[i]
                               + stream_data[12] * stream_data[7 + i]
                               * params.inv_voxel_size[i]
                               + stream_data[19 + i])
        tangent[i] = (stream_data[13] * stream_data[1 + i]
                      + stream_data[14] * stream_data[4 + i]
                      + stream_data[15] * stream_data[7 + i])
        stream_data[7 + i] = \
            (stream_data[16] * stream_data[1 + i]
            + stream_data[17] * stream_data[4 + i]
            + stream_data[18] * stream_data[7 + i])
    normalize(&tangent[0])
    cross(&stream_data[4], &stream_data[7], &tangent[0])  # frame1, frame2
    normalize(&stream_data[4])  # frame1
    cross(&stream_data[7], &tangent[0], &stream_data[4])  # frame2, tangent, frame1
    stream_data[1] = tangent[0]
    stream_data[2] = tangent[1]
    stream_data[3] = tangent[2]

    for tries in range(params.ptt.rejection_sampling_nbr_sample):
        # k1, k2
        stream_data[24], stream_data[25] = \
            random_point_within_circle(params.max_curvature)
        data_support = calculate_ptt_data_support(params, stream_data, pmf_gen)
        if data_support > max_posterior:
            max_posterior = data_support

    # Compensation for underestimation of max posterior estimate
    max_posterior = pow(2.0 * max_posterior, params.ptt.data_support_exponent)


    for tries in range(params.ptt.rejection_sampling_max_try):
        # k1, k2
        stream_data[24], stream_data[25] = \
            random_point_within_circle(params.max_curvature)
        if random() * max_posterior < calculate_ptt_data_support(params, stream_data, pmf_gen):
            stream_data[22] = stream_data[23] # last_val = last_val_cand
            # Propagation is successful if a suitable candidate can be sampled
            # within the trial limit
            # update the point and return
            copy_point(&stream_data[19], point)
            return SUCCESS

    return FAIL
