from dipy.direction.pmf cimport PmfGen
from dipy.tracking.fast_tracking cimport TrackingParameters

cdef int deterministic_tracker(double* point,
                               double* direction,
                               TrackingParameters params,
                               double* stream_data,
                               PmfGen pmf_gen) noexcept nogil