# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as cnp

from dipy.segment.cythonutils cimport tuple2shape, shape2tuple, shape_from_memview
from dipy.tracking.streamlinespeed cimport c_set_number_of_points, c_length


cdef class Feature(object):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    Parameters
    ----------
    is_order_invariant : bool (optional)
        tells if this feature is invariant to the sequence's ordering. This
        means starting from either extremities produces the same features.
        (Default: True)

    Notes
    -----
    When subclassing `Feature`, one only needs to override the `extract` and
    `infer_shape` methods.
    """
    def __init__(Feature self, is_order_invariant=True):
        # By default every feature is order invariant.
        self.is_order_invariant = is_order_invariant

    property is_order_invariant:
        """ Is this feature invariant to the sequence's ordering """
        def __get__(Feature self):
            return bool(self.is_order_invariant)
        def __set__(self, int value):
            self.is_order_invariant = bool(value)

    cdef Shape c_infer_shape(Feature self, Data2D datum) nogil except *:
        """ Cython version of `Feature.infer_shape`. """
        with gil:
            shape = self.infer_shape(np.asarray(datum))
            if np.asarray(shape).ndim == 0:
                return tuple2shape((1, shape))
            elif len(shape) == 1:
                return tuple2shape((1,) + shape)
            elif len(shape) == 2:
                return tuple2shape(shape)
            else:
                raise TypeError("Only scalar, 1D or 2D array features are supported!")

    cdef void c_extract(Feature self, Data2D datum, Data2D out) nogil except *:
        """ Cython version of `Feature.extract`. """
        cdef Data2D c_features
        with gil:
            features = np.asarray(self.extract(np.asarray(datum))).astype(np.float32)
            if features.ndim == 0:
                features = features[np.newaxis, np.newaxis]
            elif features.ndim == 1:
                features = features[np.newaxis]
            elif features.ndim == 2:
                pass
            else:
                raise TypeError("Only scalar, 1D or 2D array features are supported!")

            c_features = features

        out[:] = c_features

    cpdef infer_shape(Feature self, datum):
        """ Infers the shape of features extracted from a sequential datum.

        Parameters
        ----------
        datum : 2D array
            Sequence of N-dimensional points.

        Returns
        -------
        int, 1-tuple or 2-tuple
            Shape of the features.
        """
        raise NotImplementedError("Feature's subclasses must implement method `infer_shape(self, datum)`!")

    cpdef extract(Feature self, datum):
        """ Extracts features from a sequential datum.

        Parameters
        ----------
        datum : 2D array
            Sequence of N-dimensional points.

        Returns
        -------
        2D array
            Features extracted from `datum`.
        """
        raise NotImplementedError("Feature's subclasses must implement method `extract(self, datum)`!")


cdef class CythonFeature(Feature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    Parameters
    ----------
    is_order_invariant : bool, optional
        Tells if this feature is invariant to the sequence's ordering (Default: True).

    Notes
    -----
    By default, when inheriting from `CythonFeature`, Python methods will call their
    C version (e.g. `CythonFeature.extract` -> `self.c_extract`).
    """
    cpdef infer_shape(CythonFeature self, datum):
        """ Infers the shape of features extracted from a sequential datum.

        Parameters
        ----------
        datum : 2D array
            Sequence of N-dimensional points.

        Returns
        -------
        tuple
            Shape of the features.

        Notes
        -----
        This method calls its Cython version `self.c_infer_shape` accordingly.
        """
        if not datum.flags.writeable or datum.dtype is not np.float32:
            datum = datum.astype(np.float32)

        return shape2tuple(self.c_infer_shape(datum))

    cpdef extract(CythonFeature self, datum):
        """ Extracts features from a sequential datum.

        Parameters
        ----------
        datum : 2D array
            Sequence of N-dimensional points.

        Returns
        -------
        2D array
            Features extracted from `datum`.

        Notes
        -----
        This method calls its Cython version `self.c_extract` accordingly.
        """
        if not datum.flags.writeable or datum.dtype is not np.float32:
            datum = datum.astype(np.float32)

        shape = shape2tuple(self.c_infer_shape(datum))
        cdef Data2D out = np.empty(shape, dtype=datum.dtype)
        self.c_extract(datum, out)
        return np.asarray(out)


cdef class IdentityFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The features being extracted are the actual sequence's points. This is
    useful for metric that does not require any pre-processing.
    """
    def __init__(IdentityFeature self):
        super(IdentityFeature, self).__init__(is_order_invariant=False)

    cdef Shape c_infer_shape(IdentityFeature self, Data2D datum) nogil except *:
        return shape_from_memview(datum)

    cdef void c_extract(IdentityFeature self, Data2D datum, Data2D out) nogil except *:
        cdef:
            int N = datum.shape[0], D = datum.shape[1]
            int n, d

        for n in range(N):
            for d in range(D):
                out[n, d] = datum[n, d]


cdef class ResampleFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The features being extracted are the points of the sequence once resampled.
    This is useful for metrics requiring a constant number of points for all
     streamlines.
    """
    def __init__(ResampleFeature self, cnp.npy_intp nb_points):
        super(ResampleFeature, self).__init__(is_order_invariant=False)
        self.nb_points = nb_points

        if nb_points <= 0:
            raise ValueError("ResampleFeature: `nb_points` must be strictly positive: {0}".format(nb_points))

    cdef Shape c_infer_shape(ResampleFeature self, Data2D datum) nogil except *:
        cdef Shape shape = shape_from_memview(datum)
        shape.dims[0] = self.nb_points
        return shape

    cdef void c_extract(ResampleFeature self, Data2D datum, Data2D out) nogil except *:
        c_set_number_of_points(datum, out)


cdef class CenterOfMassFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The feature being extracted consists of one N-dimensional point representing
    the mean of the points, i.e. the center of mass.
    """
    def __init__(CenterOfMassFeature self):
        super(CenterOfMassFeature, self).__init__(is_order_invariant=True)

    cdef Shape c_infer_shape(CenterOfMassFeature self, Data2D datum) nogil except *:
        cdef Shape shape
        shape.ndim = 2
        shape.dims[0] = 1
        shape.dims[1] = datum.shape[1]
        shape.size = datum.shape[1]
        return shape

    cdef void c_extract(CenterOfMassFeature self, Data2D datum, Data2D out) nogil except *:
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
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The feature being extracted consists of one N-dimensional point representing
    the middle point of the sequence (i.e. `nb_points//2`th point).
    """
    def __init__(MidpointFeature self):
        super(MidpointFeature, self).__init__(is_order_invariant=False)

    cdef Shape c_infer_shape(MidpointFeature self, Data2D datum) nogil except *:
        cdef Shape shape
        shape.ndim = 2
        shape.dims[0] = 1
        shape.dims[1] = datum.shape[1]
        shape.size = datum.shape[1]
        return shape

    cdef void c_extract(MidpointFeature self, Data2D datum, Data2D out) nogil except *:
        cdef:
            int N = datum.shape[0], D = datum.shape[1]
            int mid = N/2
            int d

        for d in range(D):
            out[0, d] = datum[mid, d]


cdef class ArcLengthFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The feature being extracted consists of one scalar representing
    the arc length of the sequence (i.e. the sum of the length of all segments).
    """
    def __init__(ArcLengthFeature self):
        super(ArcLengthFeature, self).__init__(is_order_invariant=True)

    cdef Shape c_infer_shape(ArcLengthFeature self, Data2D datum) nogil except *:
        cdef Shape shape
        shape.ndim = 2
        shape.dims[0] = 1
        shape.dims[1] = 1
        shape.size = 1
        return shape

    cdef void c_extract(ArcLengthFeature self, Data2D datum, Data2D out) nogil except *:
        out[0, 0] = c_length(datum)


cdef class VectorOfEndpointsFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The feature being extracted consists of one vector in the N-dimensional
    space pointing from one end-point of the sequence to the other
    (i.e. `S[-1]-S[0]`).
    """
    def __init__(VectorOfEndpointsFeature self):
        super(VectorOfEndpointsFeature, self).__init__(is_order_invariant=False)

    cdef Shape c_infer_shape(VectorOfEndpointsFeature self, Data2D datum) nogil except *:
        cdef Shape shape
        shape.ndim = 2
        shape.dims[0] = 1
        shape.dims[1] = datum.shape[1]
        shape.size = datum.shape[1]
        return shape

    cdef void c_extract(VectorOfEndpointsFeature self, Data2D datum, Data2D out) nogil except *:
        cdef:
            int N = datum.shape[0], D = datum.shape[1]
            int d

        for d in range(D):
            out[0, d] = datum[N-1, d] - datum[0, d]


cpdef infer_shape(Feature feature, data):
    """ Infers shape of the features extracted from data.

    Parameters
    ----------
    feature : `Feature` object
        Tells how to infer shape of the features.
    data : list of 2D arrays
        List of sequences of N-dimensional points.

    Returns
    -------
    list of tuples
        Shapes of the features inferred from `data`.
    """
    single_datum = False
    if type(data) is np.ndarray:
        single_datum = True
        data = [data]

    if len(data) == 0:
        return []

    shapes = []
    cdef int i
    for i in range(0, len(data)):
        datum = data[i] if data[i].flags.writeable else data[i].astype(np.float32)
        shapes.append(shape2tuple(feature.c_infer_shape(datum)))

    if single_datum:
        return shapes[0]
    else:
        return shapes


cpdef extract(Feature feature, data):
    """ Extracts features from data.

    Parameters
    ----------
    feature : `Feature` object
        Tells how to extract features from the data.
    datum : list of 2D arrays
        List of sequence of N-dimensional points.

    Returns
    -------
    list of 2D arrays
        List of features extracted from `data`.
    """
    single_datum = False
    if type(data) is np.ndarray:
        single_datum = True
        data = [data]

    if len(data) == 0:
        return []

    shapes = infer_shape(feature, data)
    features = [np.empty(shape, dtype=np.float32) for shape in shapes]

    cdef int i
    for i in range(len(data)):
        datum = data[i] if data[i].flags.writeable else data[i].astype(np.float32)
        feature.c_extract(datum, features[i])

    if single_datum:
        return features[0]
    else:
        return features
