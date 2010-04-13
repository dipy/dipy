''' Classes to wrap DICOM objects and files '''

import numpy as np

from . import csareader as csar
from . import dicomreaders as dcr
from ..core.onetime import setattr_on_read as one_time


class WrapperError(Exception):
    pass


class WrapperCollection(object):
    pass


class Wrapper(object):
    ''' Class to wrap DIXOM files '''
    def __init__(self, dcm_data=None):
        if dcm_data is None:
            dcm_data = {}
        self.dcm_data = dcm_data

    @one_time
    def csa_header(self):
        return dcr.get_csa_header(self.dcm_data)

    @one_time
    def is_mosaic(self):
        return dcr.is_mosaic(self.dcm_data)

    @one_time
    def image_orient_patient(self):
        iop = self.get('ImageOrientationPatient')
        if iop is None:
            return None
        return np.array(iop).reshape(2,3).T

    @one_time
    def slice_normal(self):
        if not self.csa_header is None:
            slice_normal = csar.get_slice_normal(self.csa_header)
            if not slice_normal is None:
                return slice_normal
        iop = self.image_orient_patient
        if iop is None:
            return None
        return np.cross(*iop.T[:])

    @one_time
    def rotation_matrix(self):
        R = np.eye(3)
        R[:,:2] = self.image_orient_patient
        R[:,2] = self.slice_normal
        return R

    @one_time
    def voxel_sizes(self):
        pix_space = self.get('PixelSpacing')
        if pix_space is None:
            return None
        zs =  self.get('SpacingBetweenSlices')
        if zs is None:
            zs = 1
        return pix_space + [zs]

    @one_time
    def pixel_array(self):
        return self['pixel_array']
    
    @one_time
    def image_position(self):
        image_position = self.get('ImagePositionPatient')
        if image_position is None:
            return None
        if not self.is_mosaic:
            return image_position
        # size of mosaic array before rearranging to 3D
        md_xy = np.array(self.pixel_array.shape)
        # size of slice X, Y in array after reshaping to 3D
        n_o_m = csar.get_n_mosaic(self.csa_header)
        mosaic_size = np.ceil(np.sqrt(n_o_m))
        rd_xy = md_xy / mosaic_size
        # apply algorithm for undoing mosaic translation error - see
        # ``dicom_mosaic`` doc for details
        vox_trans_fixes = (md_xy - rd_xy) / 2
        M = self.rotation_matrix[:,:2] * self.voxel_sizes[:2]
        return image_position + np.dot(M, vox_trans_fixes[:,None]).ravel()

    @one_time
    def slice_indicator(self):
        if self.image_position is None or self.rotation_matrix is None:
            return None
        return np.dot(self.image_position, self.rotation_matrix[:,2])
                
    @one_time
    def affine(self):
        orient = self.rotation_matrix
        vox = self.voxel_sizes
        ipp = self.image_position
        if None in (orient, vox, ipp):
            raise WrapperError('Not enough information for affine')
        aff = np.eye(4)
        aff[:3,:3] = orient * np.array(vox)
        aff[:3,3] = ipp
        return aff

    def __getitem__(self, key):
        ''' Return values from DICOM object'''
        try:
            return getattr(self.dcm_data, key)
        except AttributeError:
            raise KeyError('%s not defined in dcm_data' % key)

    def get(self, key, default=None):
        return getattr(self.dcm_data, key, default)

    def get_data(self):
        data = self.pixel_array
        scale = self.get('RescaleSlope', 1)
        offset = self.get('RescaleIntercept', 0)
        # a little optimization.  If we are applying either the scale or
        # the offset, we need to allow upcasting to float.
        if scale != 1:
            if offset == 0:
                return data * scale
            return data * scale + offset
        if offset != 0:
            return data + offset
        return data
