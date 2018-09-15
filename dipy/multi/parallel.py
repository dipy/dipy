"""Classes and functions to implement multiprocessing."""
from __future__ import division

from abc import abstractmethod
import numpy as np

from dipy.multi.config import _dipy_num_cpu, manager
from dipy.reconst.multi_voxel import MultiVoxelFit


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


class TrackProgress(object):

    def __init__(self, n):
        self.count = 0
        self.total = n
        self.reset()

    def reset(self):
        self.start = time.time()

    def task_done(self):
        duration = time.time() - self.start
        self.count += 1
        msg = "task %d out of %d done in %s sec"
        msg = msg % (self.count, self.total, duration)
        print(msg)


class UpdateCallback(MultiVoxelFunction):

    def __init__(self, index, outputs, errors, tracker):
        self.index = index
        self.outputs = outputs
        self.errors = errors
        self.tracker = tracker

    def __call__(self, result):
        update_outputs(self.index, result, self.outputs)
        # Remove from errors to indicate successful completion and free memory.
        self.errors.pop(repr(self.index))
        self.tracker.task_done()


def _array_split_points(mask, num_chunks):
    # TODO split input based on mask values so each thread gets about the same
    # number of voxels where mask is True.
    cumsum = mask.cumsum()
    num_chunks = min(num_chunks, cumsum[-1])
    even_spacing = np.linspace(1, cumsum[-1], num_chunks + 1)

    split_points = cumsum.searchsorted(even_spacing, 'left')
    split_points[-1] += 1
    assert (split_points[-1] == len(cumsum) or
            cumsum[split_points[-1]] == cumsum[-1])
    return split_points


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
        num_chunks = 100 * num_cpu
        split_points = _array_split_points(mask, num_chunks)
        start = split_points[0]
        end_points = split_points[1:]

        # pool = Pool(num_cpu)
        if manger is None:
            raise ValueError()
        pool = manager.pool
        if pool is None:
            raise ValueError()

        errors = {}
        tracker = TrackProgress(len(split_points))
        for end in end_points:
            index = slice(start, end)
            chunk_args = (self, data[index], mask[index]) + args
            callback = UpdateCallback(index, outputs, errors, tracker)
            r = pool.apply_async(call_remotely, chunk_args, kwargs,
                                 callback=callback)
            # As of python 2.7, the async_result is the only way to track
            # errors, in python 3 an error callback can be used.
            errors[repr(index)] = r
            start = end

        del r
        pool_done = False

        while not pool_done:
            time.sleep(3)
            result = list(errors.values())
            pool_done = all(r.ready() for r in result)

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
