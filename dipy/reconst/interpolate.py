"""Interpolators wrap arrays to allow the array to be indexed in continuous coordinates

This module uses the trackvis coordinate system, for more information about
this coordinate system please see dipy.tracking.utils
The following modules also use this coordinate system:
dipy.tracking.utils
dipy.tracking.integration
dipy.reconst.interpolate
"""
from numpy import asarray
from dipy.reconst.recspeed import trilinear_interp

class Interpolator(object):
    """Class to be subclassed by different interpolator types"""
    def __init__(self, data, voxel_size, mask=None):
        self._data = data
        self._voxel_size = asarray(voxel_size, 'float')
        if mask is not None:
            if mask.ndim != len(self._voxel_size):
                raise ValueError()
            self._mask = asarray(mask, 'bool')
        else:
            self._mask = None

class NearestNeighborInterpolator(Interpolator):
    """Interpolates data using nearest neighbor interpolation"""

    def __getitem__(self, index):
        index = tuple(index // self._voxel_size)
        if min(index) < 0:
            raise IndexError('Negative Index')
        if self._mask is not None:
            if not self._mask[index]:
                raise StopIteration('outside mask')
        return self._data[index]

class TriLinearInterpolator(Interpolator):
    def __init__(self, data, voxel_size, mask=None, thresh=0):
        self._data = data
        self._voxel_size = asarray(voxel_size, 'float')
        self.thresh = thresh
        if mask is not None:
            mask = asarray(mask, 'float')
            mask = mask.reshape(mask.shape + (1,))
            self._mask = mask
        else:
            self._mask = None

    def __getitem__(self, index):
        if mask is not None:
            mask_value = trilinear_interp(self._mask, index, self._voxel_size)
            if mask_value < 1:
                raise StopIteration('outside mask')
        return trilinear_interp(self._mask, index, self.voxel_size)
