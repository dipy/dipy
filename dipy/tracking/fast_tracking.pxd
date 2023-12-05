cimport numpy as cnp

from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.direction.pmf cimport PmfGen


cdef class TrackingParameters():
    cdef:
        StoppingCriterion sc
        int max_len
        double step_size
        double[3] voxel_size
    pass


ctypedef int (*func_ptr)(double* point, double* direction, ProbabilisticTrackingParameters)

cpdef list generate_tractogram(double[:,::1] seed_positons,
                               double[:,::1] seed_directions,
                               TrackingParameters params)


cdef int generate_tractogram_c(double[:,::1] seed_positons,
                               double[:,::1] seed_directions,
                               int nbr_seeds,
                               TrackingParameters params,
                               double[:,:,:] streamlines,
                               double[:] status)


cdef int generate_local_streamline(double* seed,
                                   double* position,
                                   double* stream,
                                   func_ptr tracker,
                                #    sc_ptr stopping_criterion,
                                #    pmf_ptr pmf_gen,
                                   TrackingParameters params)


cdef double* get_pmf(double* point,
                     PmfGen pmf_gen,
                     double pmf_threshold,
                     int pmf_len)


cdef class ProbabilisticTrackingParameters(TrackingParameters):
    cdef:
        double       cos_similarity
        double       pmf_threshold
        PmfGen       pmf_gen
        int          pmf_len
        double[:, :] vertices


    pass

cdef int probabilistic_tracker(double* point,
                               double* direction,
                               ProbabilisticTrackingParameters params)

cdef class DeterministicTrackingParameters(ProbabilisticTrackingParameters):
    pass


cdef int deterministic_maximum_tracker(double* point,
                                       double* direction,
                                       DeterministicTrackingParameters params)

cdef class ParalleTransportTrackingParameters(ProbabilisticTrackingParameters):
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
        double[3]   inv_voxel_size
    pass


cdef int paralle_transport_tracker(double* point,
                                   double* direction,
                                   ParalleTransportTrackingParameters params)