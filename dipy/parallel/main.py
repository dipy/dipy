from __future__ import division

from abc import abstractmethod
from multiprocessing import cpu_count, Pool
import numpy as np

from numpy import ndindex

import dipy.parallel.config


def update_outputs(index, result, outputs):
    for key, array in outputs.items():
        array[index] = result[key]


class MultiVoxelFuntion(object):
    """A function that can be applied to every voxel of a dMRI data set.

    Subclass MultiVoxelFuntion and define the _main and _default_Values methods
    to create a subclass. Both methods should return a dictionary of outputs,
    where the keys are the names of the outputs and the values are arrays.
    _main will be used in voxels where mask is true, and the result from
    _default_values will be used in voxels where mask is False.
    """

    @abstractmethod
    def _main(self, single_voxel_data, *args, **kwargs):
        pass

    @abstractmethod
    def _defalut_values(self, data, mask, *args, **kwargs):
        pass

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
        for ijk in ndindex(mask.shape):
            if not mask[ijk]:
                continue
            vox = data[ijk]
            result = self._main(vox, *args, **kwargs)
            update_outputs(ijk, result, outputs)
        return outputs

    def __call__(self, data, mask, *args, **kwargs):
        self._serial(self, data, mask, *args, **kwargs)


class UpdateCallback(MultiVoxelFuntion):

    def __init__(self, index, outputs, errors):
        self.index = index
        self.outputs = outputs
        self.errors = errors

    def __call__(self, result):
        if isinstance(result, Exception):
            self.errors.append(result)
        else:
            update_outputs(self.index, result, self.outputs)


def _array_split_points(mask, num_chunks):
    # TODO split input based on mask values so each thread gets about the same
    # number of voxels where mask is True.
    chunk_size = mask.size / num_chunks
    split_points = np.arange(1, num_chunks + 1) * chunk_size
    return np.round(split_points).astype(int)


def call_remotly(parallel_function, *args, **kwargs):
    try:
        return parallel_function._serial(*args, **kwargs)
    except Exception as e:
        return e


class ParallelFunction(MultiVoxelFuntion):

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

        num_cpu = dipy.parallel.config._dipy_num_cpu
        num_chunks = min(10 * num_cpu, mask.size)
        end_points = _array_split_points(mask, num_chunks)

        pool = Pool(num_cpu)
        start = 0
        errors = []
        for end in end_points:
            index = slice(start, end)
            chunk_args = (self, data[index], mask[index]) + args
            callback = UpdateCallback(index, outputs, errors)
            pool.apply_async(call_remotly, chunk_args, kwargs,
                             callback=callback)
            start = end

        pool.close()
        pool.join()

        if errors:
            raise errors[0]

        # Un-ravel the outputs
        for key in outputs:
            array = outputs[key]
            outputs[key] = array.reshape(shape + array.shape[1:])

        return outputs

    def __call__(self, data, mask, *args, **kwargs):
        if dipy.parallel.config._dipy_num_cpu == 1:
            return self._serial(data, mask, *args, **kwargs)
        else:
            return self._parallel(data, mask, *args, **kwargs)
