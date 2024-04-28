# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as cnp

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *calloc(size_t nelem, size_t elsize)


cdef Py_ssize_t sizeof_memviewslice = 2 * sizeof(cnp.npy_intp) + 3 * sizeof(cnp.npy_intp) * 8

cdef Shape shape_from_memview(Data data) noexcept nogil:
    """ Retrieves shape from a memoryview object.

    Parameters
    ----------
    data : memoryview object (float)
        array for which the shape information is retrieved

    Returns
    -------
    shape : `Shape` struct
        structure containing information about the shape of `data`
    """
    cdef Shape shape
    cdef cnp.npy_intp i
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
        structure containing shape information obtained from `dims`
    """
    assert len(dims) < MAX_NDIM
    cdef Shape shape
    cdef cnp.npy_intp i
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
        structure containing shape information

    Returns
    -------
    dims : tuple of int
        size of each dimension
    """
    cdef cnp.npy_intp i
    dims = []
    for i in range(shape.ndim):
        dims.append(shape.dims[i])

    return tuple(dims)


cdef int same_shape(Shape shape1, Shape shape2) noexcept nogil:
    """ Checks if two shapes are the same.

    Two shapes are equals if they have the same number of dimensions
    and that each dimension's size matches.

    Parameters
    ----------
    shape1 : `Shape` struct
        structure containing shape information
    shape2 : `Shape` struct
        structure containing shape information

    Returns
    -------
    same_shape : int (0 or 1)
        tells if the shape are equals
    """

    cdef cnp.npy_intp i
    cdef int same_shape = True

    same_shape &= shape1.ndim == shape2.ndim

    for i in range(shape1.ndim):
        same_shape &= shape1.dims[i] == shape2.dims[i]

    return same_shape


cdef Data2D* create_memview_2d(Py_ssize_t buffer_size, Py_ssize_t dims[MAX_NDIM]) noexcept nogil:
    """ Create a light version of cython memory view.

    Parameters
    ----------
    buffer_size : int
        data size
    dims : array
        desired memory view shape

    Returns
    -------
    Data2D* : memview pointer
        floating pointer to memview
    """
    cdef Data2D* memview

    memview = <Data2D*> calloc(1, sizeof_memviewslice)
    memview.shape[0] = dims[0]
    memview.shape[1] = dims[1]
    memview.strides[0] = dims[1] * sizeof(float)
    memview.strides[1] = sizeof(float)
    memview.suboffsets[0] = -1
    memview.suboffsets[1] = -1
    memview._data = <char*> calloc(buffer_size, sizeof(float))

    return memview

cdef void free_memview_2d(Data2D* memview) noexcept nogil:
    """ free a light version of cython memory view

    Parameters
    ----------
    memview : Data2D*
        floating pointer to memory view pointer

    """
    free(&(memview[0][0, 0]))
    memview[0] = None  # Necessary to decrease refcount
    free(memview)
