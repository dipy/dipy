from dipy.direction.closest_peak_direction_getter cimport PmfGenDirectionGetter


cdef class ProbabilisticDirectionGetter(PmfGenDirectionGetter):
    cdef:
        double[:, :] vertices


cdef class DeterministicMaximumDirectionGetter(ProbabilisticDirectionGetter):
    pass
