
import ctypes
import numpy as np
from multiprocessing import Array


_ctypes_to_numpy = {ctypes.c_char: np.dtype(np.uint8),
                    ctypes.c_wchar: np.dtype(np.int16),
                    ctypes.c_byte: np.dtype(np.int8),
                    ctypes.c_ubyte: np.dtype(np.uint8),
                    ctypes.c_short: np.dtype(np.int16),
                    ctypes.c_ushort: np.dtype(np.uint16),
                    ctypes.c_int: np.dtype(np.int32),
                    ctypes.c_uint: np.dtype(np.uint32),
                    ctypes.c_long: np.dtype(np.int64),
                    ctypes.c_ulong: np.dtype(np.uint64),
                    ctypes.c_float: np.dtype(np.float32),
                    ctypes.c_double: np.dtype(np.float64),
                    ctypes.c_void_p: np.dtype(object)}


_numpy_to_ctypes = dict(zip(_ctypes_to_numpy.values(),
                            _ctypes_to_numpy.keys()))


def mparray_as_ndarray(mp_array, shape=None):
    """
    Given a multiprocessing.Array, returns an ndarray pointing to
    the same data.

    Parameters
    ----------
    mp_array : array
        multiprocessing.Array that you want to convert.
    shape : tuple, optional
        tuple of array dimensions (default: None)

    Returns
    --------
    numpy_array : ndarray
        ndarray pointing to the same data
    """

    # support SynchronizedArray:
    if not hasattr(mp_array, '_type_'):
        mp_array = mp_array.get_obj()

    dtype = _ctypes_to_numpy[mp_array._type_]
    result = np.frombuffer(mp_array, dtype)

    if shape is not None:
        result = result.reshape(shape)

    return np.asarray(result)


def ndarray_to_mparray(arr, lock=False):
    """
    Generate an 1D multiprocessing.Array containing the data from
    the passed ndarray.  The data will be *copied* into shared
    memory.

    Parameters
    ----------
    arr : ndarray
        numpy ndarray that you want to convert.
    lock : boolean, optional
        controls the access to the shared array. When your shared
        array has a read only access, you do not need lock. Otherwise,
        any writing access need to activate the lock to be
        process-safe(default: False)

    Returns
    --------
    shm : multiprocessing.Array
        copied shared array
    """

    array1d = arr.ravel(order='A')

    try:
        c_type = _numpy_to_ctypes[array1d.dtype]
    except KeyError:
        c_type = _numpy_to_ctypes[np.dtype(array1d.dtype)]

    result = Array(c_type, array1d.size, lock=lock)
    mparray_as_ndarray(result)[:] = array1d

    return result
