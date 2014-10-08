# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np
import cython

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


cdef double _length(Streamline streamline) nogil:
    cdef:
        np.npy_intp i
        double out = 0.0
        double dn, sum_dn_sqr

    for i in range(1, streamline.shape[0]):
        sum_dn_sqr = 0.0
        for j in range(streamline.shape[1]):
            dn = streamline[i, j] - streamline[i-1, j]
            sum_dn_sqr += dn*dn

        out += sqrt(sum_dn_sqr)

    return out


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
    >>> import numpy as np
    >>> streamline = np.array([[1, 1, 1], [2, 3, 4], [0, 0, 0]])
    >>> expected_length = np.sqrt([1+2**2+3**2, 2**2+3**2+4**2]).sum()
    >>> length(streamline) == expected_length
    True
    >>> streamlines = [streamline, np.vstack([streamline, streamline[::-1]])]
    >>> expected_lengths = [expected_length, 2*expected_length]
    >>> np.allclose(expected_lengths, [length(streamlines[0]), length(streamlines[1])])
    True
    >>> length([])
    0.0
    >>> length(np.array([[1, 2, 3]]))
    0.0

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
            dtype = None
            break

    # Allocate memory for each streamline length.
    streamlines_length = np.empty(len(streamlines), dtype=np.float64)
    cdef np.npy_intp i

    if dtype is None:
        # List of streamlines having different dtypes
        for i in range(len(streamlines)):
            streamline = streamlines[i]
            dtype = streamline.dtype
            if dtype != np.float32 and dtype != np.float64:
                dtype = np.float64 if dtype == np.int64 or dtype == np.uint64 else np.float32
                streamline = streamlines[i].astype(dtype)

            if not streamline.flags.writeable:
                streamline = streamline.astype(dtype)

            if dtype == np.float32:
                streamlines_length[i] = _length[float2d](streamline)
            else:
                streamlines_length[i] = _length[double2d](streamline)

    elif dtype == np.float32:
        # All streamlines have composed of float32 points
        for i in range(len(streamlines)):
            streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(dtype)
            streamlines_length[i] = _length[float2d](streamline)
    elif dtype == np.float64:
        # All streamlines are composed of float64 points
        for i in range(len(streamlines)):
            streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(dtype)
            streamlines_length[i] = _length[double2d](streamline)
    elif dtype == np.int64 or dtype == np.uint64:
        # All streamlines are composed of int64 or uint64 points so convert them in float64 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float64)
            streamlines_length[i] = _length[double2d](streamline)
    else:
        # All streamlines are composed of points with a dtype fitting in 32bits so convert them in float32 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float32)
            streamlines_length[i] = _length[float2d](streamline)

    if only_one_streamlines:
        return streamlines_length[0]
    else:
        return streamlines_length


cdef void _arclengths(Streamline streamline, double* out) nogil:
    cdef np.npy_intp i = 0
    cdef double dn

    out[0] = 0.0
    for i in range(1, streamline.shape[0]):
        out[i] = 0.0
        for j in range(streamline.shape[1]):
            dn = streamline[i, j] - streamline[i-1, j]
            out[i] += dn*dn

        out[i] = out[i-1] + sqrt(out[i])


cdef void _set_number_of_points(Streamline streamline, Streamline out) nogil:
    cdef:
        np.npy_intp N = streamline.shape[0]
        np.npy_intp D = streamline.shape[1]
        np.npy_intp new_N = out.shape[0]
        double ratio, step, next_point
        np.npy_intp i, j, k, dim

    # Get arclength at each point.
    arclengths = <double*> malloc(streamline.shape[0] * sizeof(double))
    _arclengths(streamline, arclengths)

    step = arclengths[N-1] / (new_N-1)

    next_point = 0.0
    i = 0
    j = 0
    k = 0

    while next_point < arclengths[N-1]:
        if next_point == arclengths[k]:
            for dim in range(D):
                out[i,dim] = streamline[j,dim];

            next_point += step
            i += 1
            j += 1
            k += 1
        elif next_point < arclengths[k]:
            ratio = 1 - ((arclengths[k]-next_point) / (arclengths[k]-arclengths[k-1]))

            for dim in range(D):
                out[i,dim] = streamline[j-1,dim] + ratio * (streamline[j,dim] - streamline[j-1,dim])

            next_point += step
            i += 1
        else:
            j += 1
            k += 1

    # Last resampled point always the one from orignal streamline.
    for dim in range(D):
        out[new_N-1,dim] = streamline[N-1,dim]

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
    >>> [len(s) for s in streamlines]
    [100, 50]
    >>> [len(s) for s in modified_streamlines]
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
            dtype = None

        if len(streamline) < 2:
            raise ValueError("All streamlines must have at least 2 points.")

    # Allocate memory for each modified streamline
    modified_streamlines = []
    cdef np.npy_intp i

    if dtype is None:
        # List of streamlines having different dtypes
        for i in range(len(streamlines)):
            streamline = streamlines[i]
            dtype = streamline.dtype
            if dtype != np.float32 and dtype != np.float64:
                dtype = np.float64 if dtype == np.int64 or dtype == np.uint64 else np.float32
                streamline = streamline.astype(dtype)

            if not streamline.flags.writeable:
                streamline = streamline.astype(dtype)

            modified_streamline = np.empty((nb_points, streamline.shape[1]), dtype=streamline.dtype)
            if dtype == np.float32:
                _set_number_of_points[float2d](streamline, modified_streamline)
            else:
                _set_number_of_points[double2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)

    elif dtype == np.float32:
        # All streamlines have composed of float32 points
        for i in range(len(streamlines)):
            streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(dtype)
            modified_streamline = np.empty((nb_points, streamline.shape[1]), dtype=streamline.dtype)
            _set_number_of_points[float2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)
    elif dtype == np.float64:
        # All streamlines are composed of float64 points
        for i in range(len(streamlines)):
            streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(dtype)
            modified_streamline = np.empty((nb_points, streamline.shape[1]), dtype=streamline.dtype)
            _set_number_of_points[double2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)
    elif dtype == np.int64 or dtype == np.uint64:
        # All streamlines are composed of int64 or uint64 points so convert them in float64 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float64)
            modified_streamline = np.empty((nb_points, streamline.shape[1]), dtype=streamline.dtype)
            _set_number_of_points[double2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)
    else:
        # All streamlines are composed of points with a dtype fitting in 32bits so convert them in float32 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float32)
            modified_streamline = np.empty((nb_points, streamline.shape[1]), dtype=streamline.dtype)
            _set_number_of_points[float2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)

    if only_one_streamlines:
        return modified_streamlines[0]
    else:
        return modified_streamlines
