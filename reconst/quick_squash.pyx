""" Detect common dtype across object array """

from functools import reduce

cimport numpy as cnp
cimport cython

import numpy as np

cdef enum:
    SCALAR, ARRAY

SCALAR_TYPES = np.ScalarType


@cython.boundscheck(False)
@cython.wraparound(False)
def quick_squash(obj_arr, mask=None, fill=0):
    """Try and make a standard array from an object array

    This function takes an object array and attempts to convert it to a more
    useful dtype. If array can be converted to a better dtype, Nones are
    replaced by `fill`. To make the behaviour of this function more clear, here
    are the most common cases:

    1.  `obj_arr` is an array of scalars of type `T`. Returns an array like
        `obj_arr.astype(T)`
    2.  `obj_arr` is an array of arrays. All items in `obj_arr` have the same
        shape ``S``. Returns an array with shape ``obj_arr.shape + S``
    3.  `obj_arr` is an array of arrays of different shapes. Returns `obj_arr`.
    4.  Items in `obj_arr` are not ndarrays or scalars. Returns `obj_arr`.

    Parameters
    ----------
    obj_arr : array, dtype=object
        The array to be converted.
    mask : array, dtype=bool, optional
       mask is nonzero where `obj_arr` has Nones.
    fill : number, optional
        Nones are replaced by `fill`.

    Returns
    -------
    result : array

    Examples
    --------
    >>> arr = np.empty(3, dtype=object)
    >>> arr.fill(2)
    >>> quick_squash(arr)
    array([2, 2, 2])
    >>> arr[0] = None
    >>> quick_squash(arr)
    array([0, 2, 2])
    >>> arr.fill(np.ones(2))
    >>> r = quick_squash(arr)
    >>> r.shape
    (3, 2)
    >>> r.dtype
    dtype('float64')
    """
    cdef:
        cnp.npy_intp i, j, N, dtypes_i
        object [:] flat_obj
        char [:] flat_mask
        cnp.dtype [:] dtypes
        int have_mask = not mask is None
        int search_for
        cnp.ndarray result
        cnp.dtype dtype, last_dtype
        object common_shape
    if have_mask:
        flat_mask = np.array(mask.reshape(-1), dtype=np.int8)
    N = obj_arr.size
    dtypes = np.empty((N,), dtype=object)
    flat_obj = obj_arr.reshape((-1))
    # Find first valid value
    for i in range(N):
        e = flat_obj[i]
        if ((have_mask and flat_mask[i] == 0) or
            (not have_mask and e is None)):
            continue
        t = type(e)
        if issubclass(t, np.generic) or t in SCALAR_TYPES:
            search_for = SCALAR
            common_shape = ()
            dtype = np.dtype(t)
            break
        elif t == cnp.ndarray:
            search_for = ARRAY
            common_shape = e.shape
            dtype = e.dtype
            break
        else: # something other than scalar or array
            return obj_arr
    else: # Nothing outside mask / all None
        return obj_arr
    # Check rest of values to confirm common type / shape, and collect dtypes
    last_dtype = dtype
    dtypes[0] = dtype
    dtypes_i = 1
    for j in range(i+1, N):
        e = flat_obj[j]
        if ((have_mask and flat_mask[j] == 0) or
            (not have_mask and e is None)):
            continue
        t = type(e)
        if search_for == SCALAR:
            if not issubclass(t, np.generic) and not t in SCALAR_TYPES:
                return obj_arr
            dtype = np.dtype(t)
        else: # search_for == ARRAY:
            if not t == cnp.ndarray:
                return obj_arr
            if not e.shape == common_shape:
                return obj_arr
            dtype = e.dtype
        if dtype != last_dtype:
            last_dtype = dtype
            dtypes[dtypes_i] = dtype
            dtypes_i += 1
    # Find common dtype
    unique_dtypes = set(dtypes[:dtypes_i])
    tiny_arrs = [np.zeros((1,), dtype=dt) for dt in unique_dtypes]
    dtype = reduce(np.add, tiny_arrs).dtype
    # Create and fill output array
    result = np.empty((N,) + common_shape, dtype=dtype)
    for i in range(N):
        e = flat_obj[i]
        if ((have_mask and flat_mask[i] == 0) or
            (not have_mask and e is None)):
            result[i] = fill
        else:
            result[i] = e
    return result.reshape(obj_arr.shape + common_shape)
