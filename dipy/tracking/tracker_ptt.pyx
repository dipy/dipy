# cython: boundscheck=False
# cython: initializedcheck=False
# cython: wraparound=False
# cython: Nonecheck=False

cimport cython
from cython.parallel import prange
import numpy as np
cimport numpy as cnp

from dipy.core.interpolation cimport trilinear_interpolate4d_c
from dipy.direction.pmf cimport PmfGen
from dipy.tracking.stopping_criterion cimport StoppingCriterion
from dipy.utils.fast_numpy cimport (copy_point, cumsum, norm, normalize,
                                    where_to_insert, random)
from dipy.tracking.fast_tracking cimport get_pmf, TrackingParameters
from libc.stdlib cimport malloc, free


cdef int parallel_transport_tracker(double* point,
                                    double* direction,
                                    TrackingParameters params,
                                    PmfGen pmf_gen) noexcept nogil:
    # update point and dir with new position and direction



    # return 1 if the propagation failed.

    return 0