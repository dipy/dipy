cdef extern from "metricspeed.h":
    enum: MAX_NDIM

ctypedef float[:,:] float2d
ctypedef double[:,:] double2d

ctypedef float[:,:] Features
ctypedef float[:,:] Streamline

ctypedef float[:] Data1D
ctypedef float[:,:] Data2D
ctypedef float[:,:,:] Data3D
ctypedef float[:,:,:,:] Data4D
ctypedef float[:,:,:,:,:] Data5D
ctypedef float[:,:,:,:,:,:] Data6D
ctypedef float[:,:,:,:,:,:,:] Data7D

ctypedef fused Data:
    Data1D
    Data2D
    Data3D
    Data4D
    Data5D
    Data6D
    Data7D

cdef struct Shape:
   Py_ssize_t ndim
   Py_ssize_t dims[MAX_NDIM]
   Py_ssize_t size

cdef Shape shape_from_memview(Data data) nogil

cdef Shape tuple2shape(dims)

cdef shape2tuple(Shape shape)

cdef int same_shape(Shape shape1, Shape shape2) nogil


cdef class Metric(object):
    cdef int is_order_invariant

    cdef Shape c_infer_features_shape(Metric self, Streamline streamline) nogil except *
    cdef void c_extract_features(Metric self, Streamline streamline, Features out) nogil except *
    cdef float c_dist(Metric self, Features features1, Features features2) nogil except -1.0
    cdef int c_compatible(Metric self, Shape shape1, Shape shape2) nogil except -1

    cpdef infer_features_shape(Metric self, streamline)
    cpdef extract_features(Metric self, streamline)
    cpdef float dist(Metric self, features1, features2) except -1.0
    cpdef compatible(Metric self, shape1, shape2)


cdef class FeatureType(object):
    cdef int is_order_invariant

    cdef Shape c_infer_shape(FeatureType self, Streamline streamline) nogil except *
    cdef void c_extract(FeatureType self, Streamline streamline, Features out) nogil except *

    cpdef infer_shape(FeatureType self, streamline)
    cpdef extract(FeatureType self, streamline)
