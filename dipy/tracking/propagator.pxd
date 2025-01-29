
from dipy.direction.pmf cimport PmfGen
from dipy.tracking.tracker_parameters cimport TrackerParameters, TrackerStatus
from dipy.utils.fast_numpy cimport RNGState

cdef TrackerStatus deterministic_tracker(double* point,
                                         double* direction,
                                         TrackerParameters params,
                                         double* stream_data,
                                         PmfGen pmf_gen,
                                         RNGState* rng) noexcept nogil

cdef TrackerStatus probabilistic_tracker(double* point,
                                         double* direction,
                                         TrackerParameters params,
                                         double* stream_data,
                                         PmfGen pmf_gen,
                                         RNGState* rng) noexcept nogil

cdef TrackerStatus parallel_transport_tracker(double* point,
                                              double* direction,
                                              TrackerParameters params,
                                              double* stream_data,
                                              PmfGen pmf_gen,
                                              RNGState* rng) noexcept nogil
