from dipy.direction.pmf cimport PmfGen
from dipy.utils.fast_numpy cimport RNGState


cpdef enum TrackerStatus:
    SUCCESS = 1
    FAIL = -1

ctypedef TrackerStatus (*func_ptr)(double* point,
                                   double* direction,
                                   TrackerParameters params,
                                   double* stream_data,
                                   PmfGen pmf_gen,
                                   RNGState* rng) noexcept nogil

cdef class ParallelTransportTrackerParameters:
    cdef public double angular_separation
    cdef public double data_support_exponent
    cdef public double k_small
    cdef public int probe_count
    cdef public double probe_length
    cdef public double probe_normalizer
    cdef public int probe_quality
    cdef public double probe_radius
    cdef public double probe_step_size
    cdef public int rejection_sampling_max_try
    cdef public int rejection_sampling_nbr_sample

cdef class ShTrackerParameters:
    cdef public double pmf_threshold

cdef class TrackerParameters:
    cdef func_ptr tracker

    cdef public double cos_similarity
    cdef public double max_angle
    cdef public double max_curvature
    cdef public int max_nbr_pts
    cdef public int min_nbr_pts
    cdef public int random_seed
    cdef public double step_size
    cdef public double average_voxel_size
    cdef public double[3] voxel_size
    cdef public double[3] inv_voxel_size
    cdef public bint return_all

    cdef public ShTrackerParameters sh
    cdef public ParallelTransportTrackerParameters ptt

    cdef void set_tracker_c(self, func_ptr tracker) noexcept nogil
