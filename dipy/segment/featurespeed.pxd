# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

from cythonutils cimport Data2D, Shape


cdef class Feature(object):
    cdef int is_order_invariant

    cdef Shape c_infer_shape(Feature self, Data2D streamline) nogil except *
    cdef void c_extract(Feature self, Data2D streamline, Data2D out) nogil except *

    cpdef infer_shape(Feature self, streamline)
    cpdef extract(Feature self, streamline)


cdef class CythonFeature(Feature):
    pass


cdef class IdentityFeature(CythonFeature):
    pass


cdef class ArcLengthFeature(CythonFeature):
    pass
