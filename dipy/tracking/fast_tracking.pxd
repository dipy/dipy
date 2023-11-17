cimport numpy as cnp

from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.direction.pmf cimport PmfGen


cpdef list generate_tractogram(double[:,::1] seed_positons,
                               double[:,::1] seed_directions,
                               StoppingCriterion sc,
                               TrackingParameters params)


cdef int generate_tractogram_c(double[:, ::1] seed_positons,
                               double[:, ::1] seed_directions,
                               int nbr_seeds,
                               StoppingCriterion sc,
                               TrackingParameters params,
                               double[:,:,:] streamlines) nogil

cdef double* get_pmf(double* point,
                     PmfGen pmf_gen,
                     double pmf_threshold,
                     int pmf_len) nogil

cdef class TrackingParameters():
    pass

cdef class ProbabilisticTrackingParameters(TrackingParameters):
    cdef:
        double       cos_similarity
        double       pmf_threshold
        PmfGen       pmf_gen
        int          pmf_len
        double       step_size
        double[:, :] vertices


cdef int probabilistic_tracker(double* point,
                               double* direction,
                               ProbabilisticTrackingParameters params) nogil

cdef class DeterministicTrackingParameters(ProbabilisticTrackingParameters):
    pass


cdef int deterministic_maximum_tracker(double* point,
                                       double* direction,
                                       DeterministicTrackingParameters params) nogil

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
        double[3]   voxel_size
        double[3]   inv_voxel_size


cdef int paralle_transport_tracker(double* point,
                                   double* direction,
                                   ParalleTransportTrackingParameters params) nogil