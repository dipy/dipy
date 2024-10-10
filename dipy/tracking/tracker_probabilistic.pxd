from dipy.direction.pmf cimport PmfGen
from dipy.tracking.tracker_parameters cimport TrackerParameters

cdef int probabilistic_tracker(double* point,
                               double* direction,
                               TrackerParameters params,
                               double* stream_data,
                               PmfGen pmf_gen) noexcept nogil
                               