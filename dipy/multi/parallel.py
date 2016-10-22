"""Classes and functions to implement multiprocessing."""

from __future__ import division

from abc import abstractmethod
from multiprocessing import cpu_count, Pool
import numpy as np
from dipy.reconst.multi_voxel import MultiVoxelFit
from contextlib import contextmanager

_dipy_num_cpu = 1


def activate_multiprocessing(num_cpu=None):
    """
    Function to activate multiprocessing.

    Parameters
    ----------
    num_cpu : int
        Number of CPU's. 
        default: None
    """
    global _dipy_num_cpu
    if num_cpu is None:
        _dipy_num_cpu = cpu_count()
    elif num_cpu > 0:
        _dipy_num_cpu = num_cpu


def deactivate_multiprocessing():
    """
    Function to deactivate multiprocessing.
    """
    global _dipy_num_cpu
    _dipy_num_cpu = 1


def update_outputs(index, result, outputs):
    for key, array in outputs.items():
        array[index] = result[key]


class MultiVoxelFunction(object):
    """A function that can be applied to every voxel of a dMRI data set.
    Subclass MultiVoxelFunction and define the _main and _default_values
    methods to create a subclass. Both methods should return a dictionary of
    outputs, where the keys are the names of the outputs and the values are
    arrays.  _main will be used in voxels where mask is true, and the result
    from _default_values will be used in voxels where mask is False.
    """

    @abstractmethod
    def _main(self, single_voxel_data, *args, **kwargs):
        raise NotImplementedError("Implement this method in a subclass.")

    @abstractmethod
    def _default_values(self, data, mask, *args, **kwargs):
        raise NotImplementedError("Implement this method in a subclass.")

    def _setup_outputs(self, data, mask, *args, **kwargs):
        default_values = self._default_values(data, mask, *args, **kwargs)
        outputs = {}
        shape = mask.shape
        ndim = len(shape)
        for key, array in default_values.items():
            out_array = np.empty(shape + array.shape, array.dtype)
            out_array[...] = array
            outputs[key] = out_array
        return outputs

    def _serial(self, data, mask, *args, **kwargs):
        outputs = self._setup_outputs(data, mask, *args, **kwargs)
        for ijk in np.ndindex(mask.shape):
            if not mask[ijk]:
                continue
            vox = data[ijk]
            result = self._main(vox, *args, **kwargs)
            update_outputs(ijk, result, outputs)
        return outputs

    def __call__(self, data, mask, *args, **kwargs):
        self._serial(self, data, mask, *args, **kwargs)


class UpdateCallback(MultiVoxelFunction):

    def __init__(self, index, outputs, errors):
        self.index = index
        self.outputs = outputs
        self.errors = errors

    def __call__(self, result):
        update_outputs(self.index, result, self.outputs)
        # Remove from errors to indicate successful completion and free memory.
        self.errors.pop(repr(self.index))


def _array_split_points(mask, num_chunks):
    # TODO split input based on mask values so each thread gets about the same
    # number of voxels where mask is True.
    chunk_size = mask.size / num_chunks
    split_points = np.arange(1, num_chunks + 1) * chunk_size
    return np.round(split_points).astype(int)


def call_remotely(parallel_function, *args, **kwargs):
    return parallel_function._serial(*args, **kwargs)


class ParallelFunction(MultiVoxelFunction):

    def _parallel(self, data, mask, *args, **kwargs):
        ndim = mask.ndim
        shape = mask.shape
        if data.shape[:ndim] != shape:
            raise ValueError("mask and data shapes do not match")

        # Flatten the inputs
        data = data.reshape((-1,) + data.shape[ndim:])
        mask = mask.ravel()
        size = mask.size
        outputs = self._setup_outputs(data, mask, *args, **kwargs)

        num_cpu = _dipy_num_cpu
        num_chunks = min(10 * num_cpu, mask.size)
        end_points = _array_split_points(mask, num_chunks)

        pool = Pool(num_cpu)
        start = 0
        errors = {}
        for end in end_points:
            index = slice(start, end)
            chunk_args = (self, data[index], mask[index]) + args
            callback = UpdateCallback(index, outputs, errors)
            r = pool.apply_async(call_remotely, chunk_args, kwargs,
                                 callback=callback)
            # As of python 2.7, the async_result is the only way to track
            # errors, in python 3 an error callback can be used.
            errors[repr(index)] = r
            start = end

        del r
        pool.close()
        pool.join()

        if errors:
            index, r = errors.popitem()
            # If errors is not empty, callbacks did not all execute, r.get()
            # should raise an error.
            r.get()
            # If r.get() does not raise an error, something else went wrong.
            msg = "Parallel function did not execute successfully."
            raise RuntimeError(msg)

        # Un-ravel the outputs
        for key in outputs:
            array = outputs[key]
            outputs[key] = array.reshape(shape + array.shape[1:])

        return outputs

    def __call__(self, data, mask, *args, **kwargs):
        if _dipy_num_cpu == 1:
            return self._serial(data, mask, *args, **kwargs)
        else:
            return self._parallel(data, mask, *args, **kwargs)


class ParallelFit(ParallelFunction):

    def _main(self, single_voxel_data, model=None):
        """Fit method for every voxel in data"""
        fit = model.fit(single_voxel_data)
        return {"fit_array": fit}

    def _default_values(self, data, mask, model=None):
        return {"fit_array": np.array(None)}


parallel_fit = ParallelFit()


def parallel_voxel_fit(single_voxel_fit):
    """Wraps single_voxel_fit method to turn a model into a multi voxel model.
    Use this decorator on the fit method of your model to take advantage of the
    MultiVoxelFit and ParallelFit machinery of dipy.
    Parameters:
    -----------
    single_voxel_fit : callable
        Should have a signature like: model [self when a model method is being
        wrapped], data [single voxel data].
    Returns:
    --------
    multi_voxel_fit_function : callable
    Examples:
    ---------
    >>> import numpy as np
    >>> from dipy.reconst.base import ReconstModel, ReconstFit
    >>> class Model(ReconstModel):
    ...     @parallel_voxel_fit
    ...     def fit(self, single_voxel_data):
    ...         return ReconstFit(self, single_voxel_data.sum())
    >>> model = Model(None)
    >>> data = np.random.random((2, 3, 4, 5))
    >>> fit = model.fit(data)
    >>> np.allclose(fit.data, data.sum(-1))
    True
    """

    def new_fit(model, data, mask=None):
        if data.ndim == 1:
            return single_voxel_fit(model, data)
        if mask is None:
            mask = np.ones(data.shape[:-1], bool)
        fit = parallel_fit(data, mask, model=model)
        return MultiVoxelFit(model, fit["fit_array"], mask)

    return new_fit
