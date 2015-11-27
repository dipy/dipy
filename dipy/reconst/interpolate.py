"""Interpolators wrap arrays to allow the array to be indexed in continuous coordinates

This module uses the trackvis coordinate system, for more information about
this coordinate system please see dipy.tracking.utils
The following modules also use this coordinate system:
dipy.tracking.utils
dipy.tracking.integration
dipy.reconst.interpolate
"""
from numpy import array
from dipy.reconst.recspeed import trilinear_interp

class OutsideImage(Exception):
    pass

class Interpolator(object):
    """Class to be subclassed by different interpolator types"""
    def __init__(self, data, voxel_size):
        self.data = data
        self.voxel_size = array(voxel_size, dtype=float, copy=True)

class NearestNeighborInterpolator(Interpolator):
    """Interpolates data using nearest neighbor interpolation"""

    def __getitem__(self, index):
        index = tuple(index / self.voxel_size)
        if min(index) < 0:
            raise OutsideImage('Negative Index')
        try:
            return self.data[tuple(array(index).astype(int))]
        except IndexError:
            raise OutsideImage

class TriLinearInterpolator(Interpolator):
    """Interpolates data using trilinear interpolation

    interpolate 4d diffusion volume using 3 indices, ie data[x, y, z]
    """
    def __init__(self, data, voxel_size):
        super(TriLinearInterpolator, self).__init__(data, voxel_size)
        if self.voxel_size.shape != (3,) or self.data.ndim != 4:
            raise ValueError("Data should be 4d volume of diffusion data and "
                             "voxel_size should have 3 values, ie the size "
                             "of a 3d voxel")

    def __getitem__(self, index):
        index = array(index, copy=False, dtype="float")
        try:
            return trilinear_interp(self.data, index, self.voxel_size)
        except IndexError:
            raise OutsideImage
