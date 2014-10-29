# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np


cdef Shape shape_from_memview(Data data) nogil:
    """ Retrieves shape from a memoryview """
    cdef Shape shape
    cdef int i
    shape.ndim = 0
    shape.size = 1
    for i in range(MAX_NDIM):
        shape.dims[i] = data.shape[i]
        if shape.dims[i] > 0:
            shape.size *= shape.dims[i]
            shape.ndim += 1

    return shape


cdef Shape tuple2shape(dims):
    """ Converts a tuple to a shape """
    cdef Shape shape
    cdef int i
    shape.ndim = len(dims)
    shape.size = np.prod(dims)
    for i in range(shape.ndim):
        shape.dims[i] = dims[i]

    return shape


cdef shape2tuple(Shape shape):
    """ Converts a shape to a tuple """
    cdef int i
    dims = []
    for i in range(shape.ndim):
        dims.append(shape.dims[i])

    return tuple(dims)


cdef int same_shape(Shape shape1, Shape shape2) nogil:
    """ Checks if shape1 and shape2 has the same ndim and same dims """
    cdef int i
    cdef int same_shape = True

    same_shape &= shape1.ndim == shape2.ndim

    for i in range(shape1.ndim):
        same_shape &= shape1.dims[i] == shape2.dims[i]

    return same_shape
