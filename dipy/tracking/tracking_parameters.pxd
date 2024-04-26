from dipy.direction.pmf cimport PmfGen

ctypedef int (*func_ptr)(double* point,
                         double* direction,
                         TrackingParameters params,
                         PmfGen pmf_gen) noexcept nogil

cdef class ParallelTransportTrackingParameters:
    cdef public double angular_separation
    cdef public double data_support_exponent
    cdef public double[3][3] frame
    cdef public double k1
    cdef public double k2
    cdef public double k_small
    cdef public double last_val
    cdef public double last_val_cand
    cdef public double[3] position
    cdef public int probe_count
    cdef public double probe_length
    cdef public double probe_normalizer
    cdef public int probe_quality
    cdef public double probe_radius
    cdef public double probe_step_size
    cdef public double[9] propagator
    cdef public int rejection_sampling_max_try
    cdef public int rejection_sampling_nbr_sample

cdef class ShTrackingParameters:
    cdef public double pmf_threshold

cdef class TrackingParameters:
    cdef func_ptr tracker

    cdef public double cos_similarity
    cdef public double max_angle
    cdef public double max_curvature
    cdef public int max_len
    cdef public double step_size
    cdef public double average_voxel_size
    cdef public double[3] voxel_size
    cdef public double[3] inv_voxel_size

    cdef public ShTrackingParameters sh
    cdef public ParallelTransportTrackingParameters ptt

    cdef void set_tracker_c(self, func_ptr tracker) noexcept nogil