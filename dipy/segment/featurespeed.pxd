from cythonutils cimport Data2D, Shape


cdef class Feature(object):
    cdef int is_order_invariant

    cdef Shape c_infer_shape(Feature self, Data2D datum) nogil except *
    cdef void c_extract(Feature self, Data2D datum, Data2D out) nogil except *

    cpdef infer_shape(Feature self, datum)
    cpdef extract(Feature self, datum)


cdef class CythonFeature(Feature):
    pass

# The IdentityFeature class returns the datum as-is. This is useful for metric
# like SumPointwiseEuclideanMetric that does not require any pre-processing.
cdef class IdentityFeature(CythonFeature):
    pass
