import numpy as np
from numpy.lib.stride_tricks import as_strided

class ndindex(object):
    """
    An N-dimensional iterator object to index arrays.

    Given the shape of an array, an `ndindex` instance iterates over
    the N-dimensional index of the array. At each iteration a tuple
    of indices is returned; the last dimension is iterated over first.

    Parameters
    ----------
    shape : tuple of ints
      The dimensions of the array.

    Examples
    --------
    >>> from dipy.core.ndindex import ndindex
    >>> shape = (3, 2, 1)
    >>> for index in ndindex(shape):
    ...     print index
    (0, 0, 0)
    (0, 1, 0)
    (1, 0, 0)
    (1, 1, 0)
    (2, 0, 0)
    (2, 1, 0)

    """
    def __init__(self, shape):
        x = as_strided(np.zeros(1), shape=shape, strides=np.zeros_like(shape))
        self._it = np.nditer(x, flags=['multi_index'], order='C')

    def __iter__(self):
        return self

    def next(self):
        """
        Standard iterator method, updates the index and returns the index tuple.

        Returns
        -------
        val : tuple of ints
            Returns a tuple containing the indices of the current iteration.

        """
        self._it.next()
        return self._it.multi_index
