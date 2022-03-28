# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free

cimport numpy as cnp

from dipy.tracking import Streamlines


cdef extern from "dpy_math.h" nogil:
    bint dpy_isnan(double x)


cdef double c_length(Streamline streamline) nogil:
    cdef:
        cnp.npy_intp i
        double out = 0.0
        double dn, sum_dn_sqr

    for i in range(1, streamline.shape[0]):
        sum_dn_sqr = 0.0
        for j in range(streamline.shape[1]):
            dn = streamline[i, j] - streamline[i-1, j]
            sum_dn_sqr += dn*dn

        out += sqrt(sum_dn_sqr)

    return out


cdef void c_arclengths_from_arraysequence(Streamline points,
                                          cnp.npy_intp[:] offsets,
                                          cnp.npy_intp[:] lengths,
                                          double[:] arclengths) nogil:
    cdef:
        cnp.npy_intp i, j, k
        cnp.npy_intp offset
        double dn, sum_dn_sqr

    for i in range(offsets.shape[0]):
        offset = offsets[i]

        arclengths[i] = 0
        for j in range(1, lengths[i]):
            sum_dn_sqr = 0.0
            for k in range(points.shape[1]):
                dn = points[offset+j, k] - points[offset+j-1, k]
                sum_dn_sqr += dn*dn

            arclengths[i] += sqrt(sum_dn_sqr)


def length(streamlines):
    """ Euclidean length of streamlines

    Length is in mm only if streamlines are expressed in world coordinates.

    Parameters
    ----------
    streamlines : ndarray or a list or :class:`dipy.tracking.Streamlines`
        If ndarray, must have shape (N,3) where N is the number of points
        of the streamline.
        If list, each item must be ndarray shape (Ni,3) where Ni is the number
        of points of streamline i.
        If :class:`dipy.tracking.Streamlines`, its `common_shape` must be 3.

    Returns
    -------
    lengths : scalar or ndarray shape (N,)
       If there is only one streamline, a scalar representing the length of the
       streamline.
       If there are several streamlines, ndarray containing the length of every
       streamline.

    Examples
    --------
    >>> from dipy.tracking.streamline import length
    >>> import numpy as np
    >>> streamline = np.array([[1, 1, 1], [2, 3, 4], [0, 0, 0]])
    >>> expected_length = np.sqrt([1+2**2+3**2, 2**2+3**2+4**2]).sum()
    >>> length(streamline) == expected_length
    True
    >>> streamlines = [streamline, np.vstack([streamline, streamline[::-1]])]
    >>> expected_lengths = [expected_length, 2*expected_length]
    >>> lengths = [length(streamlines[0]), length(streamlines[1])]
    >>> np.allclose(lengths, expected_lengths)
    True
    >>> length([])
    0.0
    >>> length(np.array([[1, 2, 3]]))
    0.0

    """
    if isinstance(streamlines, Streamlines):
        if len(streamlines) == 0:
            return 0.0

        arclengths = np.zeros(len(streamlines), dtype=np.float64)

        if streamlines._data.dtype == np.float32:
            c_arclengths_from_arraysequence[float2d](
                                    streamlines._data,
                                    streamlines._offsets.astype(np.intp),
                                    streamlines._lengths.astype(np.intp),
                                    arclengths)
        else:
            c_arclengths_from_arraysequence[double2d](
                                      streamlines._data,
                                      streamlines._offsets.astype(np.intp),
                                      streamlines._lengths.astype(np.intp),
                                      arclengths)

        return arclengths

    only_one_streamlines = False
    if type(streamlines) is cnp.ndarray:
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
    cdef cnp.npy_intp i

    if dtype is None:
        # List of streamlines having different dtypes
        for i in range(len(streamlines)):
            dtype = streamlines[i].dtype
            # HACK: To avoid memleaks we have to recast with astype(dtype).
            streamline = streamlines[i].astype(dtype)
            if dtype != np.float32 and dtype != np.float64:
                is_integer = dtype == np.int64 or dtype == np.uint64
                dtype = np.float64 if is_integer else np.float32
                streamline = streamlines[i].astype(dtype)

            if dtype == np.float32:
                streamlines_length[i] = c_length[float2d](streamline)
            else:
                streamlines_length[i] = c_length[double2d](streamline)

    elif dtype == np.float32:
        # All streamlines have composed of float32 points
        for i in range(len(streamlines)):
            # HACK: To avoid memleaks we have to recast with astype(dtype).
            streamline = streamlines[i].astype(dtype)
            streamlines_length[i] = c_length[float2d](streamline)
    elif dtype == np.float64:
        # All streamlines are composed of float64 points
        for i in range(len(streamlines)):
            # HACK: To avoid memleaks we have to recast with astype(dtype).
            streamline = streamlines[i].astype(dtype)
            streamlines_length[i] = c_length[double2d](streamline)
    elif dtype == np.int64 or dtype == np.uint64:
        # All streamlines are composed of int64 or uint64 points so convert
        # them in float64 one at the time.
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float64)
            streamlines_length[i] = c_length[double2d](streamline)
    else:
        # All streamlines are composed of points with a dtype fitting in
        # 32 bits so convert them in float32 one at the time.
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float32)
            streamlines_length[i] = c_length[float2d](streamline)

    if only_one_streamlines:
        return streamlines_length[0]
    else:
        return streamlines_length


cdef void c_arclengths(Streamline streamline, double* out) nogil:
    cdef cnp.npy_intp i = 0
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
        cnp.npy_intp N = streamline.shape[0]
        cnp.npy_intp D = streamline.shape[1]
        cnp.npy_intp new_N = out.shape[0]
        double ratio, step, next_point, delta
        cnp.npy_intp i, j, k, dim

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
                out[i, dim] = streamline[j, dim]

            next_point += step
            i += 1
            j += 1
            k += 1
        elif next_point < arclengths[k]:
            ratio = 1 - ((arclengths[k]-next_point) /
                         (arclengths[k]-arclengths[k-1]))

            for dim in range(D):
                delta = streamline[j, dim] - streamline[j-1, dim]
                out[i, dim] = streamline[j-1, dim] + ratio * delta

            next_point += step
            i += 1
        else:
            j += 1
            k += 1

    # Last resampled point always the one from original streamline.
    for dim in range(D):
        out[new_N-1, dim] = streamline[N-1, dim]

    free(arclengths)


cdef void c_set_number_of_points_from_arraysequence(Streamline points,
                                                    cnp.npy_intp[:] offsets,
                                                    cnp.npy_intp[:] lengths,
                                                    long nb_points,
                                                    Streamline out) nogil:
    cdef:
        cnp.npy_intp i, j, k
        cnp.npy_intp offset, length
        cnp.npy_intp offset_out = 0
        double dn, sum_dn_sqr

    for i in range(offsets.shape[0]):
        offset = offsets[i]
        length = lengths[i]

        c_set_number_of_points(points[offset:offset+length, :],
                               out[offset_out:offset_out+nb_points, :])

        offset_out += nb_points


def set_number_of_points(streamlines, nb_points=3):
    """ Change the number of points of streamlines
        (either by downsampling or upsampling)

    Change the number of points of streamlines in order to obtain
    `nb_points`-1 segments of equal length. Points of streamlines will be
    modified along the curve.

    Parameters
    ----------
    streamlines : ndarray or a list or :class:`dipy.tracking.Streamlines`
        If ndarray, must have shape (N,3) where N is the number of points
        of the streamline.
        If list, each item must be ndarray shape (Ni,3) where Ni is the number
        of points of streamline i.
        If :class:`dipy.tracking.Streamlines`, its `common_shape` must be 3.

    nb_points : int
        integer representing number of points wanted along the curve.

    Returns
    -------
    new_streamlines : ndarray or a list or :class:`dipy.tracking.Streamlines`
        Results of the downsampling or upsampling process.

    Examples
    --------
    >>> from dipy.tracking.streamline import set_number_of_points
    >>> import numpy as np

    One streamline, a semi-circle:

    >>> theta = np.pi*np.linspace(0, 1, 100)
    >>> x = np.cos(theta)
    >>> y = np.sin(theta)
    >>> z = 0 * x
    >>> streamline = np.vstack((x, y, z)).T
    >>> modified_streamline = set_number_of_points(streamline, 3)
    >>> len(modified_streamline)
    3

    Multiple streamlines:

    >>> streamlines = [streamline, streamline[::2]]
    >>> new_streamlines = set_number_of_points(streamlines, 10)
    >>> [len(s) for s in streamlines]
    [100, 50]
    >>> [len(s) for s in new_streamlines]
    [10, 10]

    """
    if isinstance(streamlines, Streamlines):
        if len(streamlines) == 0:
            return Streamlines()

        nb_streamlines = len(streamlines)
        dtype = streamlines._data.dtype
        new_streamlines = Streamlines()
        new_streamlines._data = np.zeros((nb_streamlines * nb_points, 3),
                                         dtype=dtype)
        new_streamlines._offsets = nb_points * np.arange(nb_streamlines,
                                                         dtype=np.intp)
        new_streamlines._lengths = nb_points * np.ones(nb_streamlines,
                                                       dtype=np.intp)

        if dtype == np.float32:
            c_set_number_of_points_from_arraysequence[float2d](
                streamlines._data, streamlines._offsets.astype(np.intp),
                streamlines._lengths.astype(np.intp), nb_points,
                new_streamlines._data)
        else:
            c_set_number_of_points_from_arraysequence[double2d](
                streamlines._data, streamlines._offsets.astype(np.intp),
                streamlines._lengths.astype(np.intp), nb_points,
                new_streamlines._data)

        return new_streamlines

    only_one_streamlines = False
    if type(streamlines) is cnp.ndarray:
        only_one_streamlines = True
        streamlines = [streamlines]

    if len(streamlines) == 0:
        return []

    if nb_points < 2:
        raise ValueError("nb_points must be at least 2")

    dtype = streamlines[0].dtype
    for streamline in streamlines:
        if streamline.dtype != dtype:
            dtype = None

        if len(streamline) < 2:
            raise ValueError("All streamlines must have at least 2 points.")

    # Allocate memory for each modified streamline
    new_streamlines = []
    cdef cnp.npy_intp i

    if dtype is None:
        # List of streamlines having different dtypes
        for i in range(len(streamlines)):
            dtype = streamlines[i].dtype
            # HACK: To avoid memleaks we have to recast with astype(dtype).
            streamline = streamlines[i].astype(dtype)
            if dtype != np.float32 and dtype != np.float64:
                dtype = np.float32
                if dtype == np.int64 or dtype == np.uint64:
                    dtype = np.float64

                streamline = streamline.astype(dtype)

            new_streamline = np.empty((nb_points, streamline.shape[1]),
                                      dtype=dtype)
            if dtype == np.float32:
                c_set_number_of_points[float2d](streamline, new_streamline)
            else:
                c_set_number_of_points[double2d](streamline, new_streamline)

            # HACK: To avoid memleaks we have to recast with astype(dtype).
            new_streamlines.append(new_streamline.astype(dtype))

    elif dtype == np.float32:
        # All streamlines have composed of float32 points
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(dtype)
            modified_streamline = np.empty((nb_points, streamline.shape[1]),
                                           dtype=streamline.dtype)
            c_set_number_of_points[float2d](streamline, modified_streamline)
            # HACK: To avoid memleaks we have to recast with astype(dtype).
            new_streamlines.append(modified_streamline.astype(dtype))
    elif dtype == np.float64:
        # All streamlines are composed of float64 points
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(dtype)
            modified_streamline = np.empty((nb_points, streamline.shape[1]),
                                           dtype=streamline.dtype)
            c_set_number_of_points[double2d](streamline, modified_streamline)
            # HACK: To avoid memleaks we have to recast with astype(dtype).
            new_streamlines.append(modified_streamline.astype(dtype))
    elif dtype == np.int64 or dtype == np.uint64:
        # All streamlines are composed of int64 or uint64 points so convert
        # them in float64 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float64)
            modified_streamline = np.empty((nb_points, streamline.shape[1]),
                                           dtype=streamline.dtype)
            c_set_number_of_points[double2d](streamline, modified_streamline)
            # HACK: To avoid memleaks we've to recast with astype(np.float64).
            new_streamlines.append(modified_streamline.astype(np.float64))
    else:
        # All streamlines are composed of points with a dtype fitting in
        # 32bits so convert them in float32 one at the time
        for i in range(len(streamlines)):
            streamline = streamlines[i].astype(np.float32)
            modified_streamline = np.empty((nb_points, streamline.shape[1]),
                                           dtype=streamline.dtype)
            c_set_number_of_points[float2d](streamline, modified_streamline)
            # HACK: To avoid memleaks we've to recast with astype(np.float32).
            new_streamlines.append(modified_streamline.astype(np.float32))

    if only_one_streamlines:
        return new_streamlines[0]
    else:
        return new_streamlines


cdef double c_norm_of_cross_product(double bx, double by, double bz,
                                    double cx, double cy, double cz) nogil:
    """ Computes the norm of the cross-product in 3D. """
    cdef double ax, ay, az
    ax = by*cz - bz*cy
    ay = bz*cx - bx*cz
    az = bx*cy - by*cx
    return sqrt(ax*ax + ay*ay + az*az)


cdef double c_dist_to_line(Streamline streamline, cnp.npy_intp prev,
                           cnp.npy_intp next, cnp.npy_intp curr) nogil:
    """ Computes the shortest Euclidean distance between a point `curr` and
        the line passing through `prev` and `next`. """

    cdef:
        double dn, norm1, norm2
        cnp.npy_intp D = streamline.shape[1]

    # Compute cross product of next-prev and curr-next
    norm1 = c_norm_of_cross_product(streamline[next, 0]-streamline[prev, 0],
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


cdef double c_segment_length(Streamline streamline,
                             cnp.npy_intp start, cnp.npy_intp end) nogil:
    """ Computes the length of the segment going from `start` to `end`. """
    cdef:
        cnp.npy_intp D = streamline.shape[1]
        cnp.npy_intp d
        double segment_length = 0.0
        double dn

    for d in range(D):
        dn = streamline[end, d] - streamline[start, d]
        segment_length += dn*dn

    return sqrt(segment_length)


cdef cnp.npy_intp c_compress_streamline(Streamline streamline, Streamline out,
                                       double tol_error, double max_segment_length) nogil:
    """ Compresses a streamline (see function `compress_streamlines`)."""
    cdef:
        cnp.npy_intp N = streamline.shape[0]
        cnp.npy_intp D = streamline.shape[1]
        cnp.npy_intp nb_points = 0
        cnp.npy_intp d, prev, next, curr
        double segment_length

    # Copy first point since it is always kept.
    for d in range(D):
        out[0, d] = streamline[0, d]

    nb_points = 1
    prev = 0

    # Loop through the points of the streamline checking if we can use the
    # linearized segment: next-prev. We start with next=2 (third points) since
    # we already added point 0 and segment between the two firsts is linear.
    for next in range(2, N):
        # Euclidean distance between last added point and current point.
        if c_segment_length(streamline, prev, next) > max_segment_length:
            for d in range(D):
                out[nb_points, d] = streamline[next-1, d]

            nb_points += 1
            prev = next-1
            continue

        # Check that each point is not offset by more than `tol_error` mm.
        for curr in range(prev+1, next):
            if c_segment_length(streamline, curr, prev) == 0:
                continue
            dist = c_dist_to_line(streamline, prev, next, curr)

            if dpy_isnan(dist) or dist > tol_error:
                for d in range(D):
                    out[nb_points, d] = streamline[next-1, d]

                nb_points += 1
                prev = next-1
                break

    # Copy last point since it is always kept.
    for d in range(D):
        out[nb_points, d] = streamline[N-1, d]

    nb_points += 1
    return nb_points


def compress_streamlines(streamlines, tol_error=0.01, max_segment_length=10):
    """ Compress streamlines by linearization as in [Presseau15]_.

    The compression consists in merging consecutive segments that are
    nearly collinear. The merging is achieved by removing the point the two
    segments have in common.

    The linearization process [Presseau15]_ ensures that every point being
    removed are within a certain margin (in mm) of the resulting streamline.
    Recommendations for setting this margin can be found in [Presseau15]_
    (in which they called it tolerance error).

    The compression also ensures that two consecutive points won't be too far
    from each other (precisely less or equal than `max_segment_length`mm).
    This is a tradeoff to speed up the linearization process [Rheault15]_. A low
    value will result in a faster linearization but low compression, whereas
    a high value will result in a slower linearization but high compression.

    Parameters
    ----------
    streamlines : one or a list of array-like of shape (N,3)
        Array representing x,y,z of N points in a streamline.
    tol_error : float (optional)
        Tolerance error in mm (default: 0.01). A rule of thumb is to set it
        to 0.01mm for deterministic streamlines and 0.1mm for probabilitic
        streamlines.
    max_segment_length : float (optional)
        Maximum length in mm of any given segment produced by the compression.
        The default is 10mm. (In [Presseau15]_, they used a value of `np.inf`).

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
    10
    >>> # Multiple streamlines
    >>> streamlines = [streamline, streamline[::2]]
    >>> c_streamlines = compress_streamlines(streamlines, tol_error=0.2)
    >>> [len(s) for s in streamlines]
    [100, 50]
    >>> [len(s) for s in c_streamlines]
    [10, 7]


    Notes
    -----
    Be aware that compressed streamlines have variable step sizes. One needs to
    be careful when computing streamlines-based metrics [Houde15]_.

    References
    ----------
    .. [Presseau15] Presseau C. et al., A new compression format for fiber
                    tracking datasets, NeuroImage, no 109, 73-83, 2015.
    .. [Rheault15] Rheault F. et al., Real Time Interaction with Millions of
                   Streamlines, ISMRM, 2015.
    .. [Houde15] Houde J.-C. et al. How to Avoid Biased Streamlines-Based
                 Metrics for Streamlines with Variable Step Sizes, ISMRM, 2015.
    """
    only_one_streamlines = False
    if type(streamlines) is cnp.ndarray:
        only_one_streamlines = True
        streamlines = [streamlines]

    if len(streamlines) == 0:
        return []

    compressed_streamlines = []
    cdef cnp.npy_intp i
    for i in range(len(streamlines)):
        dtype = streamlines[i].dtype
        # HACK: To avoid memleaks we have to recast with astype(dtype).
        streamline = streamlines[i].astype(dtype)
        shape = streamline.shape

        if dtype != np.float32 and dtype != np.float64:
            dtype = np.float64 if dtype == np.int64 or dtype == np.uint64 else np.float32
            streamline = streamline.astype(dtype)

        if shape[0] <= 2:
            compressed_streamlines.append(streamline.copy())
            continue

        compressed_streamline = np.empty(shape, dtype)

        if dtype == np.float32:
            nb_points = c_compress_streamline[float2d](streamline, compressed_streamline,
                                                       tol_error, max_segment_length)
        else:
            nb_points = c_compress_streamline[double2d](streamline, compressed_streamline,
                                                        tol_error, max_segment_length)

        compressed_streamline.resize((nb_points, streamline.shape[1]))
        # HACK: To avoid memleaks we have to recast with astype(dtype).
        compressed_streamlines.append(compressed_streamline.astype(dtype))

    if only_one_streamlines:
        return compressed_streamlines[0]
    else:
        return compressed_streamlines
