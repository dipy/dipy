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
        self.voxel_size = array(voxel_size, 'float')

class NearestNeighborInterpolator(Interpolator):
    """Interpolates data using nearest neighbor interpolation"""

    def __getitem__(self, index):
        index = tuple(index // self.voxel_size)
        if min(index) < 0:
            raise OutsideImage('Negative Index')
        try:
            return self.data[index]
        except IndexError:
            raise OutsideImage

class TriLinearInterpolator(Interpolator):
    """Interpolates data using trilinear interpolation"""
    def __getitem__(self, index):
        index = array(index, copy=False)
        try:
            return trilinear_interp(self.data, index, self.voxel_size)
        except IndexError:
            raise OutsideImage
