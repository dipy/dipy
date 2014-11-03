# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from libc.math cimport sqrt

from cythonutils cimport tuple2shape, shape2tuple, shape_from_memview


cdef class Feature(object):
    """ Provides functionalities to extract features from a sequence of
    N-dimensional points represented as 2D array of shape (points, coordinates).

    Parameters
    ----------
    is_order_invariant : bool
        tells if the sequence's ordering matters for extracting features

    Notes
    -----
    By default, when inheriting from `Feature`, Python methods will call their
    Python version (e.g. `Feature.c_extract` -> `self.extract`).
    """
    def __init__(Feature self, is_order_invariant=True):
        # By default every features are order invariant.
        self.is_order_invariant = is_order_invariant

    property is_order_invariant:
        """ Tells if the sequence's ordering matters for extracting features """
        def __get__(Feature self):
            return bool(self.is_order_invariant)
        def __set__(self, int value):
            self.is_order_invariant = bool(value)

    cdef Shape c_infer_shape(Feature self, Data2D datum) nogil:
        with gil:
            return tuple2shape(self.infer_shape(np.asarray(datum)))

    cdef void c_extract(Feature self, Data2D datum, Data2D out) nogil:
        cdef Data2D features
        with gil:
            features = self.extract(np.asarray(datum)).astype(np.float32)

        out[:] = features

    cpdef infer_shape(Feature self, datum):
        """ Infers features' shape from a sequence of N-dimensional points
        represented as 2D array of shape (points, coordinates).

        Parameters
        ----------
        datum : 2D array
            sequence of N-dimensional points

        Returns
        -------
        shape : tuple
            features' shape
        """
        raise NotImplementedError("Subclass must implement this method!")

    cpdef extract(Feature self, datum):
        """ Extracts features from a sequence of N-dimensional points
        represented as 2D array of shape (points, coordinates)

        Parameters
        ----------
        datum : 2D array
            sequence of N-dimensional points

        Returns
        -------
        features : 2D array
            features extracted from `datum`
        """
        raise NotImplementedError("Subclass must implement this method!")


cdef class CythonFeature(Feature):
    """ Provides functionalities to extract features from a sequence of
    N-dimensional points represented as 2D array of shape (points, coordinates).

    Parameters
    ----------
    is_order_invariant : bool
        tells if the sequence's ordering matters for extracting features

    Notes
    -----
    By default, when inheriting from `CythonFeature`, Python methods will call their
    C version (e.g. `CythonFeature.extract` -> `self.c_extract`).
    """
    cpdef infer_shape(CythonFeature self, datum):
        """ Infers features' shape from a sequence of N-dimensional points
        represented as 2D array of shape (points, coordinates).

        *This method is calling the C version: `self.c_infer_shape`.*

        Parameters
        ----------
        datum : 2D array
            sequence of N-dimensional points

        Returns
        -------
        shape : tuple
            features' shape
        """
        if not datum.flags.writeable:
            datum = datum.astype(np.float32)

        return shape2tuple(self.c_infer_shape(datum))

    cpdef extract(CythonFeature self, datum):
        """ Extracts features from a sequence of N-dimensional points
        represented as 2D array of shape (points, coordinates)

        *This method is calling the C version: `self.c_extract`.*

        Parameters
        ----------
        datum : 2D array
            sequence of N-dimensional points

        Returns
        -------
        features : 2D array
            features extracted from `datum`
        """
        if not datum.flags.writeable:
            datum = datum.astype(np.float32)

        shape = shape2tuple(self.c_infer_shape(datum))
        cdef Data2D out = np.empty(shape, dtype=datum.dtype)
        self.c_extract(datum, out)
        return np.asarray(out)


cdef class IdentityFeature(CythonFeature):
    """ Provides functionalities to extract features from a sequence of
    N-dimensional points represented as 2D array of shape (points, coordinates).

    The features being extracted are the actual sequence's points.
    """
    def __init__(IdentityFeature self):
        super(IdentityFeature, self).__init__(is_order_invariant=False)

    cdef Shape c_infer_shape(IdentityFeature self, Data2D datum) nogil:
        return shape_from_memview(datum)

    cdef void c_extract(IdentityFeature self, Data2D datum, Data2D out) nogil:
        cdef:
            int N = datum.shape[0], D = datum.shape[1]
            int n, d

        for n in range(N):
            for d in range(D):
                out[n, d] = datum[n, d]


cdef class CenterOfMassFeature(CythonFeature):
    """ Provides functionalities to extract features from a sequence of
    N-dimensional points represented as 2D array of shape (points, coordinates).

    The feature being extracted consists in one N-dimensional point representing
    the mean of the points.
    """
    def __init__(CenterOfMassFeature self):
        super(CenterOfMassFeature, self).__init__(is_order_invariant=True)

    cdef Shape c_infer_shape(CenterOfMassFeature self, Data2D datum) nogil:
        cdef Shape shape
        shape.ndim = 2
        shape.dims[0] = 1
        shape.dims[1] = datum.shape[1]
        shape.size = datum.shape[1]
        return shape

    cdef void c_extract(CenterOfMassFeature self, Data2D datum, Data2D out) nogil:
        cdef int N = datum.shape[0], D = datum.shape[1]
        cdef int i, d

        for d in range(D):
            out[0, d] = 0

        for i in range(N):
            for d in range(D):
                out[0, d] += datum[i, d]

        for d in range(D):
            out[0, d] /= N


cdef class MidpointFeature(CythonFeature):
    """ Provides functionalities to extract features from a sequence of
    N-dimensional points represented as 2D array of shape (points, coordinates).

    The feature being extracted consist in one N-dimensional point representing
    the sequence's middle point.
    """
    def __init__(MidpointFeature self):
        super(MidpointFeature, self).__init__(is_order_invariant=False)

    cdef Shape c_infer_shape(MidpointFeature self, Data2D datum) nogil:
        cdef Shape shape = shape_from_memview(datum)
        shape.size /= shape.dims[0]
        shape.dims[0] = 1  # Features boil down to only one point.
        return shape

    cdef void c_extract(MidpointFeature self, Data2D datum, Data2D out) nogil:
        cdef:
            int N = datum.shape[0], D = datum.shape[1]
            int mid = N/2
            int d

        for d in range(D):
            out[0, d] = datum[mid, d]


cdef class ArcLengthFeature(CythonFeature):
    """ Provides functionalities to extract features from a sequence of
    N-dimensional points represented as 2D array of shape (points, coordinates).

    The feature being extracted consists in one scalars (i.e. array of shape (1, 1))
    representing the sequence's arc length.
    """
    def __init__(ArcLengthFeature self):
        super(ArcLengthFeature, self).__init__(is_order_invariant=True)

    cdef Shape c_infer_shape(ArcLengthFeature self, Data2D datum) nogil:
        cdef Shape shape
        shape.ndim = 2
        shape.dims[0] = 1
        shape.dims[1] = 1
        shape.size = 1
        return shape

    cdef void c_extract(ArcLengthFeature self, Data2D datum, Data2D out) nogil:
        cdef:
            int N = datum.shape[0], D = datum.shape[1]
            int n, d
            double dn, sum_dn_sqr

        out[0, 0] = 0.
        for n in range(1, N):
            sum_dn_sqr = 0.0
            for d in range(D):
                dn = datum[n, d] - datum[n-1, d]
                sum_dn_sqr += dn * dn

            out[0, 0] += sqrt(sum_dn_sqr)


cpdef extract(Feature feature, streamlines):
    only_one_streamlines = False
    if type(streamlines) is np.ndarray:
        only_one_streamlines = True
        streamlines = [streamlines]

    if len(streamlines) == 0:
        return []

    cdef int i
    all_same_shapes = True
    shapes = [feature.infer_shape(streamlines[0])]
    for i in range(1, len(streamlines)):
        shapes.append(feature.infer_shape(streamlines[i]))
        if shapes[0] != shapes[i]:
            all_same_shapes = False

    if all_same_shapes:
        features = np.empty((len(shapes),) + shapes[0], dtype=np.float32)
    else:
        features = [np.empty(shape, dtype=np.float32) for shape in shapes]

    for i in range(len(streamlines)):
        streamline = streamlines[i] if streamlines[i].flags.writeable else streamlines[i].astype(np.float32)
        feature.c_extract(streamline, features[i])

    if only_one_streamlines:
        return features[0]
    else:
        return features
