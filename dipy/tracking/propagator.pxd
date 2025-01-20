from dipy.direction.pmf cimport PmfGen
from dipy.tracking.tracker_parameters cimport TrackerParameters, TrackerStatus

cdef TrackerStatus deterministic_tracker(double* point,
                                         double* direction,
                                         TrackerParameters params,
                                         double* stream_data,
                                         PmfGen pmf_gen) noexcept nogil

cdef TrackerStatus probabilistic_tracker(double* point,
                                         double* direction,
                                         TrackerParameters params,
                                         double* stream_data,
                                         PmfGen pmf_gen) noexcept nogil

cdef TrackerStatus parallel_transport_tracker(double* point,
                                              double* direction,
                                              TrackerParameters params,
                                              double* stream_data,
                                              PmfGen pmf_gen) noexcept nogil
