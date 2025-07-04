from dipy.direction.closest_peak_direction_getter cimport PmfGenDirectionGetter


cdef class ProbabilisticDirectionGetter(PmfGenDirectionGetter):
    cdef:
        double[:, :] vertices


cdef class DeterministicMaximumDirectionGetter(ProbabilisticDirectionGetter):
    pass
    
cdef class FlockingDirectionGetter(PmfGenDirectionGetter):
    cdef:
        double[:, :] vertices
        int particle_count
        double r_min
        double r_max
        double delta
        double alpha
