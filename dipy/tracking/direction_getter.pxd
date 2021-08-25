
from dipy.tracking.stopping_criterion cimport (StreamlineStatus,
                                               StoppingCriterion)

cimport numpy as cnp

cdef class DirectionGetter:

    cpdef cnp.ndarray[cnp.float_t, ndim=2] initial_direction(
        self, double[::1] point)

    cpdef tuple generate_streamline(self,
                                    double[::1] seed,
                                    double[::1] dir,
                                    double[::1] voxel_size,
                                    double step_size,
                                    StoppingCriterion stopping_criterion,
                                    cnp.float_t[:, :] streamline,
                                    StreamlineStatus stream_status,
                                    int fixedstep)

    cpdef int get_direction(
        self,
        double[::1] point,
        double[::1] direction) except -1

    cdef int get_direction_c(
        self, double* point, double* direction)
