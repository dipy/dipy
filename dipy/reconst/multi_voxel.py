"""Tools to easily make multi voxel models"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

from dipy.core.ndindex import ndindex
from dipy.core.parallel import mparray_as_ndarray, ndarray_to_mparray
from dipy.reconst.quick_squash import quick_squash as _squash
from dipy.reconst.base import ReconstFit
from multiprocessing import cpu_count, Pool

import warnings
import itertools


global shared_arr


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
    model, chunks = arguments
    return [(idx, model.fit(shared_arr[idx]))
            for idx in chunks]


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
        shared_arr = mparray_as_ndarray(arr_to_populate, shape)


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
        mask = kwargs.get('mask', None)
        # print(mask, mask.shape)
        if data.ndim == 1:
            return single_voxel_fit(model, data)
        if mask is None:
            mask = np.ones(data.shape[:-1], bool)
        elif mask.shape != data.shape[:-1]:
            raise ValueError("mask and data shape do not match")

        # Get number of processes
        nb_cpu  = cpu_count()
        nb_processes = int(kwargs.pop('nb_processes', '0'))
        nb_processes = nb_processes if nb_cpu > nb_processes >= 1 else nb_cpu

        if nb_processes == 1:
            return multi_voxel_fit(single_voxel_fit)(model, data, *args, **kwargs)

        # Get non null indexes from mask
        indexes = np.argwhere(mask)
        # Convert indexes to tuple
        indexes = [tuple(v) for v in indexes]
        # Create chunks
        chunks_spacing = np.linspace(0, len(indexes), num=nb_processes + 1)
        chunks = [(indexes[int(chunks_spacing[i - 1]): int(chunks_spacing[i])])
                  for i in range(1, len(chunks_spacing))]

        # Create shared array
        shared_arr_in = ndarray_to_mparray(data)

        # Start worker processes
        pool = Pool(processes=nb_processes,
                    initializer=_init_parallel_fit_worker,
                    initargs=(shared_arr_in,  data.shape))
        result = pool.map_async(parallel_fit_worker,
                                [(model, c) for c in chunks])
        result.wait()
        pool.close()
        pool.join()
        
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
