# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np


cdef Shape shape_from_memview(Data data) nogil:
    """ Retrieves shape from a memoryview object.

    Parameters
    ----------
    data : memoryview object (float)
        array for which the shape informations are retrieved

    Returns
    -------
    shape : `Shape` struct
        structure containing informations about the shape of `data`
    """
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


cdef Shape tuple2shape(dims) except *:
    """ Converts a Python's tuple into a `Shape` Cython's struct.

    Parameters
    ----------
    dims : tuple of int
        size of each dimension

    Returns
    -------
    shape : `Shape` struct
        structure containing shape informations obtained from `dims`
    """
    assert len(dims) < MAX_NDIM
    cdef Shape shape
    cdef int i
    shape.ndim = len(dims)
    shape.size = np.prod(dims)
    for i in range(shape.ndim):
        shape.dims[i] = dims[i]

    return shape


cdef shape2tuple(Shape shape):
    """ Converts a `Shape` Cython's struct into a Python's tuple.

    Parameters
    ----------
    shape : `Shape` struct
        structure containing shape informations

    Returns
    -------
    dims : tuple of int
        size of each dimension
    """
    cdef int i
    dims = []
    for i in range(shape.ndim):
        dims.append(shape.dims[i])

    return tuple(dims)


cdef int same_shape(Shape shape1, Shape shape2) nogil:
    """ Checks if two shapes are the same.

    Two shapes are equals if they have the same number of dimensions
    and that each dimension's size matches.

    Parameters
    ----------
    shape1 : `Shape` struct
        structure containing shape informations
    shape2 : `Shape` struct
        structure containing shape informations

    Returns
    -------
    same_shape : int (0 or 1)
        tells if the shape are equals
    """

    """  """
    cdef int i
    cdef int same_shape = True

    same_shape &= shape1.ndim == shape2.ndim

    for i in range(shape1.ndim):
        same_shape &= shape1.dims[i] == shape2.dims[i]

    return same_shape
