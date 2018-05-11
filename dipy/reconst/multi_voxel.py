"""Tools to easily make multi voxel models"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

from dipy.core.ndindex import ndindex
from dipy.reconst.quick_squash import quick_squash as _squash
from dipy.reconst.base import ReconstFit
from multiprocessing import cpu_count, Pool, Array
import ctypes
import warnings
import itertools


_ctypes_to_numpy = {
    ctypes.c_char: np.dtype(np.uint8),
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

global shared_arr


def shm_as_ndarray(mp_array, shape=None):
    """
    Given a multiprocessing.Array, returns an ndarray pointing to
    the same data.

    Parameters
    ----------
    mp_array : array
        multiprocessing.Array that you want to convert.
    shape : tuple
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


def ndarray_to_shm(arr, lock=False):
    """
    Generate an 1D multiprocessing.Array containing the data from
    the passed ndarray.  The data will be *copied* into shared
    memory.

    Parameters
    ----------
    arr : ndarray
        numpy ndarray that you want to convert.
    lock : boolean
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
    shm_as_ndarray(result)[:] = array1d

    return result


def parallel_fit_worker(arguments):
    """
    Each pool process calls this worker.
    Fit model on chunks

    Parameters
    -----------
    arguments: tuple
        tuple should contains a model and
        list of indexes

    Returns
    --------
    result: list of tuple
        return a list of tuple(voxel index, model fitted instance)
    """
    model, input_queue, args, kwargs = arguments
    return [(idx, model.fit(shared_arr[idx], *args, **kwargs))
            for idx in input_queue]


def _init_parallel_fit_worker(arr_to_populate, shape):
    """
    Each pool process calls this initializer.
    Load the array to be populated into that process's global namespace

    Parameters
    -----------
    arr_to_populate: multiprocessing.Array
        shared array
    shape: tuple
        tuple of array dimensions
    """
    global shared_arr
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        # shared_arr = np.ctypeslib.as_array(arr_to_populate)
        # shared_arr = shared_arr.reshape(shape)
        shared_arr = shm_as_ndarray(arr_to_populate, shape)


def parallel_voxel_fit(single_voxel_fit):
    """
    Wraps single_voxel_fit method to turn a model into a parallel
    multi voxel model. Use this decorator on the fit method of
    your model to take advantage of the MultiVoxelFit.

    Parameters
    -----------
    single_voxel_fit : callable
        Should have a signature like: model [self when a model method is being
        wrapped], data [single voxel data].

    Returns
    --------
    multi_voxel_fit_function : callable

    Examples
    ---------
    >>> import numpy as np
    >>> from dipy.reconst.base import ReconstModel, ReconstFit
    >>> class BasicModel(ReconstModel):
    ...     @parallel_voxel_fit
    ...     def fit(self, single_voxel_data):
    ...         return ReconstFit(self, single_voxel_data.sum())
    ...
    >>> if __name__ == '__main__':
    ...     data = np.random.random((2, 3, 4, 5))
    ...     fit = model.fit(data)
    ...     assert np.allclose(fit.data, data.sum(-1))
    """

    def new_fit(model, data, *args, **kwargs):
        """Fit method in parallel for every voxel in data """
        # Pop the mask, if there is one
        mask = kwargs.pop('mask', None)
        # print(mask, mask.shape)
        if data.ndim == 1:
            return single_voxel_fit(model, data, *args, **kwargs)
        if mask is None:
            mask = np.ones(data.shape[:-1], bool)
        elif mask.shape != data.shape[:-1]:
            raise ValueError("mask and data shape do not match")

        # Get number of processes
        nb_processes = int(kwargs.pop('nb_processes', '0'))
        nb_processes = nb_processes if nb_processes >= 1 else cpu_count()

        if nb_processes == 1:
            return single_voxel_fit(model, data, *args, **kwargs)

        # Get non null indexes from mask
        indexes = np.argwhere(mask)
        # Convert indexes to tuple
        indexes = [tuple(v) for v in indexes]
        # Create chunks
        chunks_spacing = np.linspace(0, len(indexes), num=nb_processes + 1)
        chunks = [(indexes[int(chunks_spacing[i - 1]): int(chunks_spacing[i])])
                  for i in range(1, len(chunks_spacing))]

        # Create shared array
        shared_arr_in = ndarray_to_shm(data)

        # Start worker processes
        pool = Pool(processes=nb_processes,
                    initializer=_init_parallel_fit_worker,
                    initargs=(shared_arr_in,  data.shape))
        result = pool.map_async(parallel_fit_worker,
                                [(model, c, args, kwargs)
                                 for c in chunks])
        result.wait()

        # Create output array
        fit_array = np.empty(data.shape[:-1], dtype=object)
        # Fill output array with results
        res_flatten = itertools.chain.from_iterable(result.get())
        for idx, val in res_flatten:
            fit_array[idx] = val

        global shared_arr
        shared_arr = None
        return MultiVoxelFit(model, fit_array, mask)
    return new_fit


def multi_voxel_fit(single_voxel_fit):
    """Method decorator to turn a single voxel model fit
    definition into a multi voxel model fit definition
    """
    def new_fit(self, data, mask=None):
        """Fit method for every voxel in data"""
        # If only one voxel just return a normal fit
        if data.ndim == 1:
            return single_voxel_fit(self, data)

        # Make a mask if mask is None
        if mask is None:
            shape = data.shape[:-1]
            strides = (0,) * len(shape)
            mask = as_strided(np.array(True), shape=shape, strides=strides)
        # Check the shape of the mask if mask is not None
        elif mask.shape != data.shape[:-1]:
            raise ValueError("mask and data shape do not match")

        # Fit data where mask is True
        fit_array = np.empty(data.shape[:-1], dtype=object)
        for ijk in ndindex(data.shape[:-1]):
            if mask[ijk]:
                fit_array[ijk] = single_voxel_fit(self, data[ijk])
        return MultiVoxelFit(self, fit_array, mask)
    return new_fit


class MultiVoxelFit(ReconstFit):
    """Holds an array of fits and allows access to their attributes and
    methods"""
    def __init__(self, model, fit_array, mask):
        self.model = model
        self.fit_array = fit_array
        self.mask = mask

    @property
    def shape(self):
        return self.fit_array.shape

    def __getattr__(self, attr):
        result = CallableArray(self.fit_array.shape, dtype=object)
        for ijk in ndindex(result.shape):
            if self.mask[ijk]:
                result[ijk] = getattr(self.fit_array[ijk], attr)
        return _squash(result, self.mask)

    def __getitem__(self, index):
        item = self.fit_array[index]
        if isinstance(item, np.ndarray):
            return MultiVoxelFit(self.model, item, self.mask[index])
        else:
            return item

    def predict(self, *args, **kwargs):
        """
        Predict for the multi-voxel object using each single-object's
        prediction API, with S0 provided from an array.
        """
        S0 = kwargs.get('S0', np.ones(self.fit_array.shape))
        idx = ndindex(self.fit_array.shape)
        ijk = next(idx)

        def gimme_S0(S0, ijk):
            if isinstance(S0, np.ndarray):
                return S0[ijk]
            else:
                return S0

        kwargs['S0'] = gimme_S0(S0, ijk)
        # If we have a mask, we might have some Nones up front, skip those:
        while self.fit_array[ijk] is None:
            ijk = next(idx)

        if not hasattr(self.fit_array[ijk], 'predict'):
            msg = "This model does not have prediction implemented yet"
            raise NotImplementedError(msg)

        first_pred = self.fit_array[ijk].predict(*args, **kwargs)
        result = np.zeros(self.fit_array.shape + (first_pred.shape[-1],))
        result[ijk] = first_pred
        for ijk in idx:
            kwargs['S0'] = gimme_S0(S0, ijk)
            # If it's masked, we predict a 0:
            if self.fit_array[ijk] is None:
                result[ijk] *= 0
            else:
                result[ijk] = self.fit_array[ijk].predict(*args, **kwargs)

        return result


class CallableArray(np.ndarray):
    """An array which can be called like a function"""
    def __call__(self, *args, **kwargs):
        result = np.empty(self.shape, dtype=object)
        for ijk in ndindex(self.shape):
            item = self[ijk]
            if item is not None:
                result[ijk] = item(*args, **kwargs)
        return _squash(result)
