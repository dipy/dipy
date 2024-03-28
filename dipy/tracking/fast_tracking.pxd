cimport numpy as cnp

from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.direction.pmf cimport PmfGen


cdef class TrackingParameters():
    cdef:
        int max_len
        double step_size
        double[:] voxel_size
        double[3] inv_voxel_size


ctypedef int (*func_ptr)(double* point,
                         double* direction,
                         DeterministicTrackingParameters,
                         PmfGen) noexcept nogil


cdef int generate_tractogram_c(double[:,::1] seed_positions,
                               double[:,::1] seed_directions,
                               int nbr_threads,
                               StoppingCriterion sc,
                               DeterministicTrackingParameters params,
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
                                   DeterministicTrackingParameters params,
                                   PmfGen pmf_gen) noexcept nogil


cdef int get_pmf(double* pmf,
                 double* point,
                 PmfGen pmf_gen,
                 double pmf_threshold,
                 int pmf_len) noexcept nogil


cdef class ProbabilisticTrackingParameters(TrackingParameters):
    cdef:
        double       cos_similarity
        double       pmf_threshold


cdef int probabilistic_tracker(double* point,
                               double* direction,
                               ProbabilisticTrackingParameters params,
                               PmfGen pmf_gen) noexcept nogil

cdef class DeterministicTrackingParameters(ProbabilisticTrackingParameters):
    pass


cdef int deterministic_maximum_tracker(double* point,
                                       double* direction,
                                       DeterministicTrackingParameters params,
                                       PmfGen pmf_gen) noexcept nogil

cdef class ParallelTransportTrackingParameters(ProbabilisticTrackingParameters):
    cdef:
        double      angular_separation
        double      data_support_exponent
        double[3][3] frame
        double      k1
        double      k2
        double      k_small
        double      last_val
        double      last_val_cand
        double      max_angle
        double      max_curvature
        double[3]   position
        int         probe_count
        double      probe_length
        double      probe_normalizer
        int         probe_quality
        double      probe_radius
        double      probe_step_size
        double[9]   propagator
        int         rejection_sampling_max_try
        int         rejection_sampling_nbr_sample


cdef int parallel_transport_tracker(double* point,
                                    double* direction,
                                    ParallelTransportTrackingParameters params,
                                    PmfGen pmf_gen) noexcept nogil