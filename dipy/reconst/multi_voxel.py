"""Tools to easily make multi voxel models"""

from itertools import repeat
from multiprocessing import cpu_count, Pool
from os import path
from warnings import warn

import dill
from nibabel.tmpdirs import InTemporaryDirectory
import numpy as np
from numpy.lib.stride_tricks import as_strided

from ..core.ndindex import ndindex
from .quick_squash import quick_squash as _squash
from .base import ReconstModel, ReconstFit


def multi_voxel_fit(single_voxel_fit):
    """Method decorator to turn a single voxel model fit
    definition into a multi voxel model fit definition
    """
    def new_fit(self, data, mask=None, nbr_processes=2):
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

        # Lauch mutliprocessing is nbr_processes != 1
        if not nbr_processes == 1 and np.sum(mask > 0) > nbr_processes:
            return parallel_fit(self, data, mask, nbr_processes)

        # Fit data where mask is True
        fit_array = np.empty(data.shape[:-1], dtype=object)
        for ijk in ndindex(data.shape[:-1]):
            if mask[ijk]:
                fit_array[ijk] = single_voxel_fit(self, data[ijk])
        return MultiVoxelFit(self, fit_array, mask)

    def parallel_fit(self, data, mask, nbr_processes):

        if nbr_processes is None or nbr_processes == 0:
            try:
                nbr_processes = cpu_count()
            except NotImplementedError:
                warn("Cannot determine number of cpus. \
                     Uses a single process")
                return new_fit(self, data, mask=mask, nbr_processes=1)

        shape = mask.shape
        data = np.reshape(data, (-1, data.shape[-1]))
        n = np.prod(mask.shape)

        nbr_chunks = min(nbr_processes ** 2, n)
        chunk_size = int(np.ceil(n / nbr_chunks))
        indices = list(zip(np.arange(0, n, chunk_size),
                           np.arange(0, n, chunk_size) + chunk_size))

        with InTemporaryDirectory() as tmpdir:
            data_file_name = path.join(tmpdir, 'data.npy')
            np.save(data_file_name, data)
            if mask is not None:
                mask = mask.flatten()
                mask_file_name = path.join(tmpdir, 'mask.npy')
                np.save(mask_file_name, mask)
            else:
                mask_file_name = None

            pool = Pool(nbr_processes)
            results = pool.map(_fit_parallel_sub,
                               zip(repeat((data_file_name, mask_file_name)),
                                   indices,
                                   repeat(self),
                                   repeat(dill.dumps(single_voxel_fit))))
            pool.close()
        fit_array = np.empty(len(mask), dtype=object)
        for i, (start_pos, end_pos) in enumerate(indices):
            fit_array[start_pos: end_pos] = results[i]

        # Make sure all worker processes have exited before leaving context
        # manager in order to prevent temporary file deletion errors in windows
        pool.join()

        fit_array = np.reshape(fit_array, shape)
        mask = np.reshape(mask, shape)
        return MultiVoxelFit(self, fit_array, mask)

    return new_fit

def _fit_parallel_sub(args):
    (data_file_name, mask_file_name) = args[0]
    (start_pos, end_pos) = args[1]
    self = args[2]
    single_voxel_fit = dill.loads(args[3])
    data = np.load(data_file_name, mmap_mode='r')[start_pos:end_pos]
    mask = np.load(mask_file_name, mmap_mode='r')[start_pos:end_pos]

    fit_array = np.empty(len(mask), dtype=object)
    for i in range(len(mask)):
        if mask[i]:
            fit_array[i] = single_voxel_fit(self, data[i])

    return fit_array

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
        if not hasattr(self.model, 'predict'):
            msg = "This model does not have prediction implemented yet"
            raise NotImplementedError(msg)

        S0 = kwargs.get('S0', np.ones(self.fit_array.shape))
        idx = ndindex(self.fit_array.shape)
        ijk = next(idx)
        def gimme_S0(S0, ijk):
            if isinstance(S0, np.ndarray):
                return S0[ijk]
            else:
                return S0

        kwargs['S0'] = gimme_S0(S0, ijk)
        first_pred = self.fit_array[ijk].predict(*args, **kwargs)
        result = np.empty(self.fit_array.shape + (first_pred.shape[-1],))
        result[ijk] = first_pred
        for ijk in idx:
            kwargs['S0'] = gimme_S0(S0, ijk)
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
