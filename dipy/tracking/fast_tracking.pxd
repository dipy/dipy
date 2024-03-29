cimport numpy as cnp

from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.direction.pmf cimport PmfGen


cdef class ParallelTransportTrackingParameters:
    cdef public double      angular_separation
    cdef public double      data_support_exponent
    cdef public double[3][3] frame
    cdef public double      k1
    cdef public double      k2
    cdef public double      k_small
    cdef public double      last_val
    cdef public double      last_val_cand
    cdef public double      max_angle
    cdef public double      max_curvature
    cdef public double[3]   position
    cdef public int         probe_count
    cdef public double      probe_length
    cdef public double      probe_normalizer
    cdef public int         probe_quality
    cdef public double      probe_radius
    cdef public double      probe_step_size
    cdef public double[9]   propagator
    cdef public int         rejection_sampling_max_try
    cdef public int         rejection_sampling_nbr_sample

cdef class ProbabilisticTrackingParameters:
    cdef public double       cos_similarity
    cdef public double       pmf_threshold


cdef class TrackingParameters:
    cdef public int max_len
    cdef public double step_size
    cdef public double[3] voxel_size
    cdef public double[3] inv_voxel_size

    cdef public ProbabilisticTrackingParameters probabilistic
    cdef public ParallelTransportTrackingParameters ptt


ctypedef int (*func_ptr)(double* point,
                         double* direction,
                         TrackingParameters params,
                         PmfGen pmf_gen) noexcept nogil


cdef int generate_tractogram_c(double[:,::1] seed_positions,
                               double[:,::1] seed_directions,
                               int nbr_threads,
                               StoppingCriterion sc,
                               TrackingParameters params,
                               PmfGen pmf_gen,
                               func_ptr tracker,
                               double** streamlines,
                               int* length,
                               int* status)


cdef int generate_local_streamline(double* seed,
                                   double* position,
                                   double* stream,
                                   int* stream_idx,
                                   func_ptr tracker,
                                   StoppingCriterion sc,
                                   TrackingParameters params,
                                   PmfGen pmf_gen) noexcept nogil


cdef int get_pmf(double* pmf,
                 double* point,
                 PmfGen pmf_gen,
                 double pmf_threshold,
                 int pmf_len) noexcept nogil

cdef int probabilistic_tracker(double* point,
                               double* direction,
                               TrackingParameters params,
                               PmfGen pmf_gen) noexcept nogil

cdef int deterministic_maximum_tracker(double* point,
                                       double* direction,
                                       TrackingParameters params,
                                       PmfGen pmf_gen) noexcept nogil

cdef int parallel_transport_tracker(double* point,
                                    double* direction,
                                    TrackingParameters params,
                                    PmfGen pmf_gen) noexcept nogil