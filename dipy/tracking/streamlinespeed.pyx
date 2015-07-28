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


cdef double c_length(Streamline streamline) nogil:
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
                streamlines_length[i] = c_length[float2d](streamline)
            else:
                streamlines_length[i] = c_length[double2d](streamline)

    elif dtype == np.float32:
        # All streamlines have composed of float32 points
        for i in range(len(streamlines)):
            streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(dtype)
            streamlines_length[i] = c_length[float2d](streamline)
    elif dtype == np.float64:
        # All streamlines are composed of float64 points
        for i in range(len(streamlines)):
            streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(dtype)
            streamlines_length[i] = c_length[double2d](streamline)
    elif dtype == np.int64 or dtype == np.uint64:
        # All streamlines are composed of int64 or uint64 points so convert them in float64 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float64)
            streamlines_length[i] = c_length[double2d](streamline)
    else:
        # All streamlines are composed of points with a dtype fitting in 32bits so convert them in float32 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float32)
            streamlines_length[i] = c_length[float2d](streamline)

    if only_one_streamlines:
        return streamlines_length[0]
    else:
        return streamlines_length


cdef void c_arclengths(Streamline streamline, double* out) nogil:
    cdef np.npy_intp i = 0
    cdef double dn

    out[0] = 0.0
    for i in range(1, streamline.shape[0]):
        out[i] = 0.0
        for j in range(streamline.shape[1]):
            dn = streamline[i, j] - streamline[i-1, j]
            out[i] += dn*dn

        out[i] = out[i-1] + sqrt(out[i])


cdef void c_set_number_of_points(Streamline streamline, Streamline out) nogil:
    cdef:
        np.npy_intp N = streamline.shape[0]
        np.npy_intp D = streamline.shape[1]
        np.npy_intp new_N = out.shape[0]
        double ratio, step, next_point
        np.npy_intp i, j, k, dim

    # Get arclength at each point.
    arclengths = <double*> malloc(streamline.shape[0] * sizeof(double))
    c_arclengths(streamline, arclengths)

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
    ''' Change the number of points of streamlines
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
                c_set_number_of_points[float2d](streamline, modified_streamline)
            else:
                c_set_number_of_points[double2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)

    elif dtype == np.float32:
        # All streamlines have composed of float32 points
        for i in range(len(streamlines)):
            streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(dtype)
            modified_streamline = np.empty((nb_points, streamline.shape[1]), dtype=streamline.dtype)
            c_set_number_of_points[float2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)
    elif dtype == np.float64:
        # All streamlines are composed of float64 points
        for i in range(len(streamlines)):
            streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(dtype)
            modified_streamline = np.empty((nb_points, streamline.shape[1]), dtype=streamline.dtype)
            c_set_number_of_points[double2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)
    elif dtype == np.int64 or dtype == np.uint64:
        # All streamlines are composed of int64 or uint64 points so convert them in float64 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float64)
            modified_streamline = np.empty((nb_points, streamline.shape[1]), dtype=streamline.dtype)
            c_set_number_of_points[double2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)
    else:
        # All streamlines are composed of points with a dtype fitting in 32bits so convert them in float32 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float32)
            modified_streamline = np.empty((nb_points, streamline.shape[1]), dtype=streamline.dtype)
            c_set_number_of_points[float2d](streamline, modified_streamline)
            modified_streamlines.append(modified_streamline)

    if only_one_streamlines:
        return modified_streamlines[0]
    else:
        return modified_streamlines


cdef double c_cross_product_normed(double bx, double by, double bz,
                                   double cx, double cy, double cz) nogil:
    cdef double ax, ay, az
    ax = by*cz - bz*cy
    ay = bz*cx - bx*cz
    az = bx*cy - by*cx
    return sqrt(ax*ax + ay*ay + az*az)


cdef double c_dist_to_line(Streamline streamline, np.npy_intp prev,
                           np.npy_intp next, np.npy_intp curr) nogil:

    cdef:
        double norm1, norm2
        double dn
        np.npy_intp D = streamline.shape[1]

    # Compute cross product of next-prev and curr-next
    norm1 = c_cross_product_normed(streamline[next, 0]-streamline[prev, 0],
                                   streamline[next, 1]-streamline[prev, 1],
                                   streamline[next, 2]-streamline[prev, 2],
                                   streamline[curr, 0]-streamline[next, 0],
                                   streamline[curr, 1]-streamline[next, 1],
                                   streamline[curr, 2]-streamline[next, 2])

    # Norm of next-prev
    norm2 = 0.0
    for d in range(D):
        dn = streamline[next, d]-streamline[prev, d]
        norm2 += dn*dn
    norm2 = sqrt(norm2)

    return norm1 / norm2


cdef np.npy_intp c_compress_streamline(Streamline streamline, double tol_error, Streamline out) nogil:
    cdef:
        np.npy_intp N = streamline.shape[0]
        np.npy_intp D = streamline.shape[1]
        np.npy_intp nb_points = 0
        np.npy_intp last_added_point_idx = 0
        np.npy_intp last_success = 0
        np.npy_intp last_added_point = 0
        np.npy_intp i, k, prev, next, curr
        double dn, in_between_dist

    for i in range(N):
        # Euclidean distance between last added point and current point.
        in_between_dist = 0.0
        for d in range(D):
            dn = streamline[i, d] - streamline[last_added_point, d]
            in_between_dist += dn*dn

        in_between_dist = sqrt(in_between_dist)

        if i == 0:  # First point is always added.
            for d in range(D):
                out[nb_points, d] = streamline[i, d]

            last_success = i
            nb_points += 1
        elif i < N-1:  # Perform linearization if needed.
            prev = last_added_point
            next = i+1

            # Check that each point is not offset by more than `tol_error` mm.
            for k in range(last_added_point, i):
                curr = k
                dist = c_dist_to_line(streamline, prev, next, curr)

                # If the current point's offset is greater than `tol_error`, use the latest success.
                if dist > tol_error or in_between_dist > 10:
                    for d in range(D):
                        out[nb_points, d] = streamline[last_success, d]

                    nb_points += 1
                    last_added_point = i-1
                    last_success = i-1
                    break
                # The current point's offset is ok, check the next point.
                else:
                    last_success = i

        else:  # Last point is always added.
            for d in range(D):
                out[nb_points, d] = streamline[i, d]

            nb_points += 1

    return nb_points


def compress_streamlines(streamlines, tol_error=0.5):
    """ Compress streamlines by linearizing some part of them.

    Basically, it consists in merging consecutive segments that are
    near collinear. The merging is achieved by removing the point the two
    segments have in common. The linearization process [Presseau15]_. will
    not remove a point if it causes either an offset of more than
    `tol_error`mm or a distance between two consecutive points to be more
    than 10mm [Rheault15]_.

    Parameters
    ----------
    streamlines : one or a list of array-like of shape (N,3)
        Array representing x,y,z of N points in a streamline
    tol_error : float (optional)
        Tolerance error in mm. Default is 0.5mm.

    Returns
    -------
    compressed_streamlines : one or a list of array-like
        Results of the linearization process.

    Examples
    --------
    >>> from dipy.tracking.streamline import compress_streamlines
    >>> import numpy as np
    >>> # One streamline: a wiggling line
    >>> rng = np.random.RandomState(42)
    >>> streamline = np.linspace(0, 10, 100*3).reshape((100, 3))
    >>> streamline += 0.2 * rng.rand(100, 3)
    >>> c_streamline = compress_streamlines(streamline, tol_error=0.2)
    >>> len(streamline)
    100
    >>> len(c_streamline)
    12
    >>> # Multiple streamlines
    >>> streamlines = [streamline, streamline[::2]]
    >>> c_streamlines = compress_streamlines(streamlines, tol_error=0.2)
    >>> [len(s) for s in streamlines]
    [100, 50]
    >>> [len(s) for s in c_streamlines]
    [12, 5]


    References
    ----------
    .. [Presseau15] Presseau C. et al., A new compression format for fiber
                    tracking datasets, NeuroImage, no 109, 73-83, 2015.
    .. [Rheault15] Rheault F. et al., In writing, 2015.
    """
    only_one_streamlines = False
    if type(streamlines) is np.ndarray:
        only_one_streamlines = True
        streamlines = [streamlines]

    if len(streamlines) == 0:
        return []


    compressed_streamlines = []
    cdef np.npy_intp i
    for i in range(len(streamlines)):
        dtype = streamlines[i].dtype
        streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(dtype)
        compressed_streamline = np.empty(streamline.shape, dtype)

        if dtype == np.float32:
            nb_points = c_compress_streamline[float2d](streamline, tol_error, compressed_streamline)
        else:
            nb_points = c_compress_streamline[double2d](streamline, tol_error, compressed_streamline)

        compressed_streamlines.append(compressed_streamline[:nb_points])

    if only_one_streamlines:
        return compressed_streamlines[0]
    else:
        return compressed_streamlines
