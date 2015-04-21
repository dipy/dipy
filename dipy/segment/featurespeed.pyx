# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from cythonutils cimport tuple2shape, shape2tuple, shape_from_memview


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
        # By default every features are order invariant.
        self.is_order_invariant = is_order_invariant

    property is_order_invariant:
        """ Is this feature invariant to the sequence's ordering """
        def __get__(Feature self):
            return bool(self.is_order_invariant)
        def __set__(self, int value):
            self.is_order_invariant = bool(value)

    cdef Shape c_infer_shape(Feature self, Data2D datum) nogil  except *:
        """ Cython version of `Feature.infer_shape`. """
        with gil:
            shape = self.infer_shape(np.asarray(datum))
            if type(shape) is int:
                return tuple2shape((1, shape))
            elif len(shape) == 1:
                return tuple2shape((1,) + shape)
            elif len(shape) == 2:
                return tuple2shape(shape)
            else:
                raise TypeError("Only scalar, 1D or 2D array features are supported!")

    cdef void c_extract(Feature self, Data2D datum, Data2D out) nogil  except *:
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
        tuple
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


cdef class CenterOfMassFeature(CythonFeature):
    """ Extracts features from a sequential datum.

    A sequence of N-dimensional points is represented as a 2D array with
    shape (nb_points, nb_dimensions).

    The feature being extracted consists in one N-dimensional point representing
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

    cdef int i
    all_same_shapes = True
    shapes = [shape2tuple(feature.c_infer_shape(data[0]))]
    for i in range(1, len(data)):
        shapes.append(shape2tuple(feature.c_infer_shape(data[i])))
        if shapes[0] != shapes[i]:
            all_same_shapes = False

    if all_same_shapes:
        features = np.empty((len(shapes),) + shapes[0], dtype=np.float32)
    else:
        features = [np.empty(shape, dtype=np.float32) for shape in shapes]

    for i in range(len(data)):
        datum = data[i] if data[i].flags.writeable else data[i].astype(np.float32)
        feature.c_extract(datum, features[i])

    if single_datum:
        return features[0]
    else:
        return features


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

    Notes
    -----
    This method calls its Cython version `self.c_extract` accordingly.
    """
    single_datum = False
    if type(data) is np.ndarray:
        single_datum = True
        data = [data]

    if len(data) == 0:
        return []

    cdef int i
    all_same_shapes = True
    shapes = [shape2tuple(feature.c_infer_shape(data[0]))]
    for i in range(1, len(data)):
        shapes.append(shape2tuple(feature.c_infer_shape(data[i])))
        if shapes[0] != shapes[i]:
            all_same_shapes = False

    if all_same_shapes:
        features = np.empty((len(shapes),) + shapes[0], dtype=np.float32)
    else:
        features = [np.empty(shape, dtype=np.float32) for shape in shapes]

    for i in range(len(data)):
        datum = data[i] if data[i].flags.writeable else data[i].astype(np.float32)
        feature.c_extract(datum, features[i])

    if single_datum:
        return features[0]
    else:
        return features
