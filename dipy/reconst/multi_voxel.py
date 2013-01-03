"""Tools to easily make multi voxel models"""
import numpy as np
from numpy.lib.stride_tricks import as_strided

from ..core.ndindex import ndindex
from .quick_squash import quick_squash as _squash


def multi_voxel_model(SingleVoxelModel):
    """Class decorator to turn a single voxel model into a multi voxel model

    See Also
    --------
    dipy/docs/examples/multiVoxelModel.py

    """
    class MultiVoxelModel(SingleVoxelModel):
        """A subclass of SingleVoxelModel that fits many voxels"""

        def fit(self, data, mask=None):
            """Fit model for every voxel in data"""
            # If only one voxel just return a normal fit
            if data.ndim == 1:
                return SingleVoxelModel.fit(self, data)

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
                    fit_array[ijk] = SingleVoxelModel.fit(self, data[ijk])
            return MultiVoxelFit(self, fit_array, mask)

    return MultiVoxelModel


class MultiVoxelFit(object):
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


class CallableArray(np.ndarray):
    """An array which can be called like a function"""
    def __call__(self, *args, **kwargs):
        result = np.empty(self.shape, dtype=object)
        for ijk in ndindex(self.shape):
            item = self[ijk]
            if item is not None:
                result[ijk] = item(*args, **kwargs)
        return _squash(result)
