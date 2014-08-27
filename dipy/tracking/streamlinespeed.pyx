# distutils: language = c++

import numpy as np
cimport numpy as np
import cython

from libcpp.vector cimport vector
from libc.math cimport sqrt

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *malloc(size_t size)

ctypedef float[:,:] float2d
ctypedef double[:,:] double2d

ctypedef fused Streamline:
    float2d
    double2d

ctypedef vector[Streamline] Streamlines


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _length(Streamlines streamlines, double[:] out) nogil:
    cdef:
        unsigned int i
        unsigned int idx
        Streamline streamline
        double d0, d1, d2

    for idx in range(streamlines.size()):
        streamline = streamlines[idx]
        out[idx] = 0.0

        for i in range(1, streamline.shape[0]):
            d0 = streamline[i, 0] - streamline[i-1, 0]
            d1 = streamline[i, 1] - streamline[i-1, 1]
            d2 = streamline[i, 2] - streamline[i-1, 2]
            out[idx] += sqrt(d0 * d0 + d1 * d1 + d2 * d2)


def length(streamlines):
    ''' Euclidean length of streamlines

    This will give length in mm if streamlines are expressed in world coordinates.

    Parameters
    ------------
    streamlines : one or a list of array-like shape (N,3)
       array representing x,y,z of N points in a streamline

    Returns
    ---------
    lengths : scalar or array shape (N,)
       scalar representing the length of one streamline, or
       array representing the lengths of multiple streamlines.

    Examples
    ----------
    >>> from dipy.tracking.streamline import length
    >>> streamline = np.array([[1, 1, 1], [2, 3, 4], [0, 0, 0]])
    >>> expected_length = np.sqrt([1+2**2+3**2, 2**2+3**2+4**2]).sum()
    >>> length(streamline) == expected_length
    True
    >>> streamlines = [streamline, np.vstack([streamline, streamline[:-1]])]
    >>> expected_lengths = [expected_length, 2*expected_length]
    >>> np.allclose(expected_lengths, [length(streamlines[0]), length(streamlines[1])])
    True
    >>> length([])
    0
    >>> length(np.array([[1, 2, 3]]))
    0
    '''
    only_one_streamlines = False
    if type(streamlines) is np.ndarray:
        only_one_streamlines = True
        streamlines = [streamlines]

    if len(streamlines) == 0:
        return 0.0

    dtype = streamlines[0].dtype
    for streamline in streamlines:
        if streamline.dtype != dtype:
            raise ValueError("All streamlines must have the same dtype.")

    # Cast any integer array into float.
    if dtype != np.float32 and dtype != np.float64:
        if dtype == np.int64 or dtype == np.uint64:
            dtype = np.float64
            streamlines = [streamline.astype(np.float64) for streamline in streamlines]
        else:  # Integer using less than 64bits
            dtype = np.float32
            streamlines = [streamline.astype(np.float32) for streamline in streamlines]

    # TODO: Support any number of coordinates?
    nb_coords = streamlines[0].shape[1]
    if nb_coords != 3:
        raise ValueError("Streamlines must have 3 coordinates (i.e. X,Y,Z).")

    # Allocate memory for each streamline length.
    streamlines_length = np.empty(len(streamlines), dtype=np.float64)

    if dtype == np.float32:
        _length[float2d](streamlines, streamlines_length)
    else:
        _length[double2d](streamlines, streamlines_length)

    if only_one_streamlines:
        return streamlines_length[0]
    else:
        return streamlines_length


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void _arclengths(Streamline streamlines, double* out) nogil:
    cdef int i = 0
    cdef double d0, d1, d2
    out[0] = 0.0
    for i in range(1, streamlines.shape[0]):
        d0 = streamlines[i, 0] - streamlines[i-1, 0]
        d1 = streamlines[i, 1] - streamlines[i-1, 1]
        d2 = streamlines[i, 2] - streamlines[i-1, 2]
        out[i] = out[i-1] + sqrt(d0 * d0 + d1 * d1 + d2 * d2)


@cython.wraparound(False)
@cython.cdivision(True)
@cython.boundscheck(False)
cdef void _set_number_of_points(Streamlines streamlines, Streamlines out) nogil:
    cdef:
        unsigned int N
        unsigned int newN = out[0].shape[0]
        double ratio
        double step
        double nextPoint
        unsigned int i
        unsigned int j
        unsigned int k
        unsigned int idx
        Streamline streamline

    for idx in range(streamlines.size()):
        streamline = streamlines[idx]

        # Get arclength at each point.
        arclengths = <double*> malloc(streamline.shape[0] * sizeof(double))
        _arclengths(streamline, arclengths)

        N = streamline.shape[0]
        step = arclengths[N-1] / (newN-1)

        nextPoint = 0.0
        i = 0
        j = 0
        k = 0

        while nextPoint < arclengths[N-1]:
            if nextPoint == arclengths[k]:
                out[idx][i,0] = streamline[j,0];
                out[idx][i,1] = streamline[j,1];
                out[idx][i,2] = streamline[j,2];

                nextPoint += step
                i += 1
                j += 1
                k += 1
            elif nextPoint < arclengths[k]:
                ratio = 1 - ((arclengths[k]-nextPoint) / (arclengths[k]-arclengths[k-1]))

                out[idx][i,0] = streamline[j-1,0] + ratio * (streamline[j,0] - streamline[j-1,0])
                out[idx][i,1] = streamline[j-1,1] + ratio * (streamline[j,1] - streamline[j-1,1])
                out[idx][i,2] = streamline[j-1,2] + ratio * (streamline[j,2] - streamline[j-1,2])

                nextPoint += step
                i += 1
            else:
                j += 1
                k += 1

        # Last resampled point always the one from orignal streamline.
        out[idx][newN-1,0] = streamline[N-1,0]
        out[idx][newN-1,1] = streamline[N-1,1]
        out[idx][newN-1,2] = streamline[N-1,2]

        free(arclengths)


def set_number_of_points(streamlines, nb_points=3):
    ''' change the number of points of streamlines
        (either by downsampling or upsampling)

    Change the number of points of streamlines in order to obtain
    `nb_points`-1 segments of equal length. Points of streamlines will be
    modified along the curve.

    Parameters
    ----------
    streamlines : one or a list of array-like shape (N,3)
       array representing x,y,z of N points in a streamline
    nb_points : int
       integer representing number of points wanted along the curve.

    Returns
    -------
    modified_streamlines : one or a list of array-like shape (`nb_points`,3)
       array representing x,y,z of `nb_points` points that where interpolated.

    Examples
    --------
    >>> from dipy.tracking.streamline import set_number_of_points
    >>> import numpy as np
    >>> # One streamline: a semi-circle
    >>> theta = np.pi*np.linspace(0, 1, 100)
    >>> x = np.cos(theta)
    >>> y = np.sin(theta)
    >>> z = 0 * x
    >>> streamline = np.vstack((x, y, z)).T
    >>> modified_streamline = set_number_of_points(streamline, 3)
    >>> len(modified_streamline)
    3
    >>> # Multiple streamlines
    >>> streamlines = [streamline, streamline[::2]]
    >>> modified_streamlines = set_number_of_points(streamlines, 10)
    >>> map(len, streamlines)
    [100, 50]
    >>> map(len, modified_streamlines)
    [10, 10]
    '''
    only_one_streamlines = False
    if type(streamlines) is np.ndarray:
        only_one_streamlines = True
        streamlines = [streamlines]

    if len(streamlines) == 0:
        return []

    dtype = streamlines[0].dtype
    for streamline in streamlines:
        if streamline.dtype != dtype:
            raise ValueError("All streamlines must have the same dtype.")
        if len(streamline) < 2:
            raise ValueError("All streamlines must have at least 2 points.")

    # Cast any integer array into float.
    if dtype != np.float32 and dtype != np.float64:
        if dtype == np.int64 or dtype == np.uint64:
            dtype = np.float64
            streamlines = [streamline.astype(np.float64) for streamline in streamlines]
        else:  # Integer using less than 64bits
            dtype = np.float32
            streamlines = [streamline.astype(np.float32) for streamline in streamlines]

    # TODO: Support any number of coordinates?
    nb_coords = streamlines[0].shape[1]
    if nb_coords != 3:
        raise ValueError("Streamlines must have 3 coordinates (i.e. X,Y,Z).")

    # Allocate memory for each modified streamline
    modified_streamlines = [np.empty((nb_points, streamline.shape[1]), dtype=dtype) for streamline in streamlines]

    if dtype == np.float32:
        _set_number_of_points[float2d](streamlines, modified_streamlines)
    else:
        _set_number_of_points[double2d](streamlines, modified_streamlines)

    if only_one_streamlines:
        return modified_streamlines[0]
    else:
        return modified_streamlines
