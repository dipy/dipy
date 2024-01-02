from dipy.segment.cythonutils cimport Data2D, Shape
cimport numpy as cnp

cdef class Feature:
    cdef int is_order_invariant

    cdef Shape c_infer_shape(Feature self, Data2D datum) noexcept nogil
    cdef void c_extract(Feature self, Data2D datum, Data2D out) noexcept nogil

    cpdef infer_shape(Feature self, datum)
    cpdef extract(Feature self, datum)


cdef class CythonFeature(Feature):
    pass

# The IdentityFeature class returns the datum as-is. This is useful for metric
# that does not require any pre-processing.
cdef class IdentityFeature(CythonFeature):
    pass

# The ResampleFeature class returns the datum resampled. This is useful for
# metric like SumPointwiseEuclideanMetric that does require a consistent
# number of points between datum.
cdef class ResampleFeature(CythonFeature):
    cdef cnp.npy_intp nb_points
