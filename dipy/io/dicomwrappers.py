''' Classes to wrap DICOM objects and files '''

import numpy as np

from . import csareader as csar
from . import dicomreaders as dcr


class WrapperError(Exception):
    pass

DEFAULT_ATTR_KEYS = (
    'InstanceNumber',
    'AcquisitionNumber',
    'ImageOrientationPatient',
    'ImagePositionPatient',
    'Rows',
    'Columns',
    'PixelSpacing',
    'SpacingBetweenSlices',
    )


class DicomWrapper(object):
    ''' Class to wrap DIXOM files '''
    def __init__(self, dcm_data=None):
        self.dcm_data = dcm_data
        self.fields = {}
        fields = self.fields
        for key in DEFAULT_ATTR_KEYS:
            fields[key] = getattr(dcm_data, key, None)
        self.is_mosaic = False
        if not dcm_data is None:
            self.csa_header = dcr.get_csa_header(dcm_data)
            self.is_mosaic = dcr.is_mosaic(dcm_data)
        else:
            self.csa_header = None
        self.image_orient = None
        if not fields['ImageOrientationPatient'] is None:
            image_orient = np.eye(3)
            iop = np.array(dcm_data.ImageOrientationPatient)
            image_orient[:,:2] = iop.reshape(2,3).T
            slice_normal = None
            if not self.csa_header is None:
                slice_normal = csar.get_slice_normal(self.csa_header)
            if slice_normal is None:
                slice_normal = np.cross(
                    image_orient[:,0],
                    image_orient[:,1])
            image_orient[:,2] = slice_normal
            self.image_orient = image_orient
        self.voxel_sizes = None
        if not fields['PixelSpacing'] is None:
            zs =  fields['SpacingBetweenSlices']
            if zs is None:
                zs = 1
            self.voxel_sizes = fields['PixelSpacing'] + [zs]
        self.image_position = fields['ImagePositionPatient']
        if not self.image_position is None:
            if self.is_mosaic:
                # size of mosaic array before rearranging to 3D
                md_xy = np.array(dcm_data.PixelArray.shape)
                # size of slice X, Y in array after reshaping to 3D
                n_o_m = csar.get_n_mosaic(self.csa_header)
                mosaic_size = np.ceil(np.sqrt(n_o_m))
                rd_xy = md_xy / mosaic_size
                # apply algorithm for undoing mosaic translation error - see
                # ``dicom_mosaic`` doc for details
                vox_trans_fixes = (md_xy - rd_xy) / 2
                M = self.image_orient[:,:2] * self.voxel_sizes[:2]
                t = self.image_position + np.dot(M,
                                                 vox_trans_fixes[:,None]).ravel()
                self.image_position = t
        self.slice_number = None
        if not self.image_position is None and not self.image_orient is None:
            self.slice_number = np.dot(self.image_position,
                                       self.image_orient[:,2])
        
                
    @classmethod
    def from_file(klass, file_like, *args, **kwargs):
        import dicom
        dcm_data = dicom.read_file(file_like, *args, **kwargs)
        return klass(dcm_data)

    def __getitem__(self, key):
        return self.fields[key]

    def get_data(self):
        raw_data = self.dcm_data.pixel_array
        try:
            scale = self.dcm_data.RescaleSlope
        except AttributeError:
            scale = 1
        try:
            offset = self.dcm_data.RescaleIntercept
        except AttributeError:
            offset = 0
        if scale != 1:
            data *= scale
        if offset != 0:
            data += offset
        return data

    def get_affine(self):
        orient = self.image_orient
        aff = np.eye(4)
        aff[:3,:3] = self.image_orient * self.voxel_sizes
        aff[:3,3] = self.image_position
        return aff
