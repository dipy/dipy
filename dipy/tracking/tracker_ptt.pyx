# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

cimport cython
from cython.parallel import prange
cimport numpy as cnp
from libc.math cimport M_PI, pow, sin, cos, fabs
from dipy.direction.pmf cimport PmfGen
from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.utils.fast_numpy cimport (copy_point, cross, normalize, random,
                                    random_perpendicular_vector,
                                    random_point_within_circle)
from dipy.tracking.fast_tracking cimport TrackerParameters
from libc.stdlib cimport malloc, free



cdef int parallel_transport_tracker(double* point,
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
        PTT tracking parameters.
    stream_data : double*
        Streamline data persitant accros tracking steps.
    pmf_gen : PmfGen
        Orientation data.

    Returns
    -------
    status : int
        Returns 0 if the propagation was successful, or
        1 otherwise.
    """
    cdef double max_posterior = 0
    cdef double data_support = 0
    cdef double[3] tangent
    cdef int tries
    cdef int i

    if stream_data[0] == 0:
        initialize(params, stream_data, pmf_gen, point, direction)
        stream_data[0] = 1  # initialized

    prepare_propagator(params, stream_data, params.step_size)

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
        data_support = calculate_data_support(params, stream_data, pmf_gen)
        if data_support > max_posterior:
            max_posterior = data_support

    # Compensation for underestimation of max posterior estimate
    max_posterior = pow(2.0 * max_posterior, params.ptt.data_support_exponent)


    for tries in range(params.ptt.rejection_sampling_max_try):
        # k1, k2
        stream_data[24], stream_data[25] = \
            random_point_within_circle(params.max_curvature)
        if random() * max_posterior <= calculate_data_support(params, stream_data, pmf_gen):
            stream_data[22] = stream_data[23] # last_val = last_val_cand
            # Propagation is successful if a suitable candidate can be sampled
            # within the trial limit
            # update the point and return
            copy_point(&stream_data[19], point)
            return 0

    return 1


cdef int initialize(TrackerParameters params,
                    double* stream_data,
                    PmfGen pmf_gen,
                    double* seed_point,
                    double* seed_direction) noexcept nogil:
        """Sample an initial curve by rejection sampling.

        Parameters
        ----------
        params : TrackerParameters
            PTT tracking parameters.
        stream_data : double*
            Streamline data persitant accros tracking steps.
        pmf_gen : PmfGen
            Orientation data.
        seed_point : double[3]
            Initial point
        seed_direction : double[3]
            Initial direction

        Returns
        -------
        status : int
            Returns 0 if the initialization was successful, or
            1 otherwise.
        """
        cdef double data_support = 0
        cdef double max_posterior = 0
        cdef int tries

        # position
        stream_data[19] = seed_point[0]
        stream_data[20] = seed_point[1]
        stream_data[21] = seed_point[2]

        for tries in range(params.ptt.rejection_sampling_nbr_sample):
            initialize_candidate(params, stream_data, pmf_gen, seed_direction)
            data_support = calculate_data_support(params, stream_data, pmf_gen)
            if data_support > max_posterior:
                max_posterior = data_support

        # Compensation for underestimation of max posterior estimate
        max_posterior = pow(2.0 * max_posterior, params.ptt.data_support_exponent)

        # Initialization is successful if a suitable candidate can be sampled
        # within the trial limit
        for tries in range(params.ptt.rejection_sampling_max_try):
            initialize_candidate(params, stream_data, pmf_gen, seed_direction)
            if (random() * max_posterior <= calculate_data_support(params, stream_data, pmf_gen)):
                stream_data[22] = stream_data[23] # last_val = last_val_cand
                return 0
        return 1

cdef void initialize_candidate(TrackerParameters params,
                               double* stream_data,
                               PmfGen pmf_gen,
                               double* init_dir) noexcept nogil:
    """"Initialize the parallel transport frame.

    After initial position is set, a parallel transport frame is set using
    the initial direction (a walking frame, i.e., 3 orthonormal vectors,
    plus 2 scalars, i.e. k1 and k2).

    A point and parallel transport frame parametrizes a curve that is named
    the "probe". Using probe parameters (probe_length, probe_radius,
    probe_quality, probe_count), a short fiber bundle segment is modelled.

    Parameters
    ----------
    params : TrackerParameters
        PTT tracking parameters.
    stream_data : double*
        Streamline data persitant accros tracking steps.
    pmf_gen : PmfGen
        Orientation data.
    init_dir : double[3]
        Initial tracking direction (tangent)

    """
    cdef double[3] position
    cdef int count
    cdef int i
    cdef double* pmf

    # Initialize Frame
    stream_data[1] = init_dir[0]
    stream_data[2] = init_dir[1]
    stream_data[3] = init_dir[2]
    random_perpendicular_vector(&stream_data[7],
                                &stream_data[1])  # frame2, frame0
    cross(&stream_data[4],
          &stream_data[7],
          &stream_data[1])  # frame1, frame2, frame0
    stream_data[24], stream_data[25] = \
        random_point_within_circle(params.max_curvature)

    stream_data[22] = 0  # last_val

    if params.ptt.probe_count == 1:
        stream_data[22] = pmf_gen.get_pmf_value_c(&stream_data[19],
                                                  &stream_data[1])  # position, frame[0]
    else:
        for count in range(params.ptt.probe_count):
            for i in range(3):
                position[i] = (stream_data[19 + i]
                              + stream_data[4 + i]
                              * params.ptt.probe_radius
                              * cos(count * params.ptt.angular_separation)
                              * params.inv_voxel_size[i]
                              + stream_data[7 + i]
                              * params.ptt.probe_radius
                              * sin(count * params.ptt.angular_separation)
                              * params.inv_voxel_size[i])

            stream_data[22] += pmf_gen.get_pmf_value_c(&stream_data[19],
                                                       &stream_data[1])


cdef void prepare_propagator(TrackerParameters params,
                             double* stream_data,
                             double arclength) noexcept nogil:
    """Prepare the propagator.

    The propagator used for transporting the moving frame forward.

    Parameters
    ----------
    params : TrackerParameters
        PTT tracking parameters.
    stream_data : double*
        Streamline data persitant accros tracking steps.
    arclength : double
        Arclenth, which is equivalent to step size along the arc

    """
    cdef double tmp_arclength
    stream_data[10] = arclength  # propagator[0]

    if (fabs(stream_data[24]) < params.ptt.k_small
        and fabs(stream_data[25]) < params.ptt.k_small):
        stream_data[11] = 0
        stream_data[12] = 0
        stream_data[13] = 1
        stream_data[14] = 0
        stream_data[15] = 0
        stream_data[16] = 0
        stream_data[17] = 0
        stream_data[18] = 1
    else:
        if fabs(stream_data[24]) < params.ptt.k_small:  # k1
            stream_data[24] = params.ptt.k_small
        if fabs(stream_data[25]) < params.ptt.k_small:  # k2
            stream_data[25] = params.ptt.k_small

        tmp_arclength  = arclength * arclength / 2.0

        # stream_data[10:18] -> propagator
        stream_data[11] = stream_data[24] * tmp_arclength
        stream_data[12] = stream_data[25] * tmp_arclength
        stream_data[13] = (1 - stream_data[25]
                          * stream_data[25] * tmp_arclength
                          - stream_data[24] * stream_data[24]
                          * tmp_arclength)
        stream_data[14] = stream_data[24] * arclength
        stream_data[15] = stream_data[25] * arclength
        stream_data[16] = -stream_data[25] * arclength
        stream_data[17] = (-stream_data[24] * stream_data[25]
                          * tmp_arclength)
        stream_data[18] = (1 - stream_data[25] * stream_data[25]
                          * tmp_arclength)


cdef double calculate_data_support(TrackerParameters params,
                                   double* stream_data,
                                   PmfGen pmf_gen) noexcept nogil:
    """Calculates data support for the candidate probe.
        Parameters
    ----------
    params : TrackerParameters
        PTT tracking parameters.
    stream_data : double*
        Streamline data persitant accros tracking steps.
    pmf_gen : PmfGen
        Orientation data.
    """

    cdef double fod_amp
    cdef double[3] position
    cdef double[3][3] frame
    cdef double[3] tangent
    cdef double[3] normal
    cdef double[3] binormal
    cdef double[3] new_position
    cdef double likelihood
    cdef int c, i, j, q
    cdef double* pmf

    prepare_propagator(params, stream_data, params.ptt.probe_step_size)

    for i in range(3):
        position[i] = stream_data[19+i]
        for j in range(3):
            frame[i][j] = stream_data[1 + i * 3 + j]

    likelihood = stream_data[22]
    for q in range(1, params.ptt.probe_quality):
        for i in range(3):
            # stream_data[10:18] : propagator
            position[i] = \
                (stream_data[10] * frame[0][i] * params.voxel_size[i]
                + stream_data[11] * frame[1][i] * params.voxel_size[i]
                + stream_data[12] * frame[2][i] * params.voxel_size[i]
                + position[i])
            tangent[i] = (stream_data[13] * frame[0][i]
                         + stream_data[14] * frame[1][i]
                         + stream_data[15] * frame[2][i])
        normalize(&tangent[0])

        if q < (params.ptt.probe_quality - 1):
            for i in range(3):
                binormal[i] = (stream_data[16] * frame[0][i]
                              + stream_data[17] * frame[1][i]
                              + stream_data[18] * frame[2][i])
            cross(&normal[0], &binormal[0], &tangent[0])

            copy_point(&tangent[0], &frame[0][0])
            copy_point(&normal[0], &frame[1][0])
            copy_point(&binormal[0], &frame[2][0])
        if params.ptt.probe_count == 1:
            fod_amp = pmf_gen.get_pmf_value_c(position, tangent)
            fod_amp = fod_amp if fod_amp > params.sh.pmf_threshold else 0
            stream_data[23] = fod_amp  # last_val_cand
            likelihood += stream_data[23]  # last_val_cand
        else:
            stream_data[23] = 0  # last_val_cand
            if q == params.ptt.probe_quality - 1:
                for i in range(3):
                    binormal[i] = (stream_data[16] * frame[0][i]
                                  + stream_data[17] * frame[1][i]
                                  + stream_data[18] * frame[2][i])
                cross(&normal[0], &binormal[0], &tangent[0])

            for c in range(params.ptt.probe_count):
                for i in range(3):
                    new_position[i] = (position[i]
                                      + normal[i] * params.ptt.probe_radius
                                      * cos(c * params.ptt.angular_separation)
                                      * params.inv_voxel_size[i]
                                      + binormal[i] * params.ptt.probe_radius
                                      * sin(c * params.ptt.angular_separation)
                                      * params.inv_voxel_size[i])
                fod_amp = pmf_gen.get_pmf_value_c(new_position, tangent)
                fod_amp = fod_amp if fod_amp > params.sh.pmf_threshold else 0
                stream_data[23] += fod_amp  # last_val_cand

            likelihood += stream_data[23]  # last_val_cand

    likelihood *= params.ptt.probe_normalizer
    if params.ptt.data_support_exponent != 1:
        likelihood = pow(likelihood, params.ptt.data_support_exponent)

    return likelihood

