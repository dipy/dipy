# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

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
            shape = self.infer_shape(np.asarray(datum))
            if type(shape) is int:
                return tuple2shape((1, shape))
            elif len(shape) == 1:
                return tuple2shape((1,) + shape)
            elif len(shape) == 2:
                return tuple2shape(shape)
            else:
                raise TypeError("Only scalar, 1D or 2D array features are supported!")

    cdef void c_extract(Feature self, Data2D datum, Data2D out) nogil:
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
        raise NotImplementedError("Feature subclasses must implement method `infer_shape(self, datum)`!")

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
        raise NotImplementedError("Feature subclasses must implement method `extract(self, datum)`!")


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


cpdef infer_shape(Feature feature, streamlines):
    only_one_streamlines = False
    if type(streamlines) is np.ndarray:
        only_one_streamlines = True
        streamlines = [streamlines]

    if len(streamlines) == 0:
        return []

    cdef int i
    all_same_shapes = True
    shapes = [shape2tuple(feature.c_infer_shape(streamlines[0]))]
    for i in range(1, len(streamlines)):
        shapes.append(shape2tuple(feature.c_infer_shape(streamlines[i])))
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


cpdef extract(Feature feature, streamlines):
    only_one_streamlines = False
    if type(streamlines) is np.ndarray:
        only_one_streamlines = True
        streamlines = [streamlines]

    if len(streamlines) == 0:
        return []

    cdef int i
    all_same_shapes = True
    shapes = [shape2tuple(feature.c_infer_shape(streamlines[0]))]
    for i in range(1, len(streamlines)):
        shapes.append(shape2tuple(feature.c_infer_shape(streamlines[i])))
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
