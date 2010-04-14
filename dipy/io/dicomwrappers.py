''' Classes to wrap DICOM objects and files

The wrappers encapsulate the capcbilities of the different DICOM
formats.

They also allow dictionary-like access to named fiields.


'''

import numpy as np

from . import csareader as csar
from .dwiparams import B2q
from ..core.geometry import nearest_pos_semi_def
from .utils import allopen
from ..core.onetime import setattr_on_read as one_time


class WrapperError(Exception):
    pass


def wrapper_from_file(file_like):
    import dicom
    fobj = allopen(file_like)
    dcm_data = dicom.read_file(fobj)
    return make_wrapper(dcm_data)


def make_wrapper(dcm_data):
    csa = csar.get_csa_header(dcm_data)
    if csa is None:
        return Wrapper(dcm_data)
    if not csar.is_mosaic(csa):
        return SiemensWrapper(dcm_data, csa)
    return MosaicWrapper(dcm_data, csa)


class Wrapper(object):
    ''' Class to wrap general DICOM files

    Attributes (or rather, things that at least look like attributes):
    
    '''
    def __init__(self, dcm_data=None):
        ''' Initialize wrapper

        Parameters
        ----------
        dcm_data : None or object, optional
           object should allow attribute access.  Usually this will be
           a ``dicom.dataset.Dataset`` object resulting from reading a
           DICOM file.   If None, we just make an empty dict. 
        '''
        if dcm_data is None:
            dcm_data = {}
        self.dcm_data = dcm_data

    @one_time
    def image_shape(self):
        return (self.get('Rows'), self.get('Columns'))

    @one_time
    def is_mosaic(self):
        return False

    @one_time
    def image_orient_patient(self):
        iop = self.get('ImageOrientationPatient')
        if iop is None:
            return None
        return np.array(iop).reshape(2,3).T

    @one_time
    def slice_normal(self):
        iop = self.image_orient_patient
        if iop is None:
            return None
        return np.cross(*iop.T[:])

    @one_time
    def rotation_matrix(self):
        iop = self.image_orient_patient
        s_norm = self.slice_normal
        if None in (iop, s_norm):
            raise WrapperError('Not enough information '
                               'for rotation matrix')
        R = np.eye(3)
        R[:,:2] = iop
        R[:,2] = s_norm
        # check this is in fact a rotation matrix
        assert np.allclose(np.eye(3),
                           np.dot(R, R.T),
                           atol=1e-6)
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
        try:
            return self['pixel_array']
        except KeyError:
            raise WrapperError('Image does not appear to have data')
    
    @one_time
    def image_position(self):
        return self.get('ImagePositionPatient')

    @one_time
    def slice_indicator(self):
        ''' See doc/theory/dicom_orientation for description '''
        if self.image_position is None or self.rotation_matrix is None:
            return None
        return np.mean(self.image_position / self.rotation_matrix[:,2])
                
    @one_time
    def affine(self):
        ''' Return mapping between voxel and DICOM coordinate system
        
        Parameters
        ----------
        None

        Returns
        -------
        aff : (4,4) affine
           Affine giving transformation between voxels in data array and
           the DICOM patient coordinate system.
        '''
        orient = self.rotation_matrix
        vox = self.voxel_sizes
        ipp = self.image_position
        if None in (orient, vox, ipp):
            raise WrapperError('Not enough information for affine')
        aff = np.eye(4)
        aff[:3,:3] = orient * np.array(vox)
        aff[:3,3] = ipp
        return aff

    @one_time
    def b_matrix(self):
        return None

    def q_vector(self):
        return None

    def __getitem__(self, key):
        ''' Return values from DICOM object'''
        try:
            return getattr(self.dcm_data, key)
        except AttributeError:
            raise KeyError('%s not defined in dcm_data' % key)

    def get(self, key, default=None):
        return getattr(self.dcm_data, key, default)

    def get_data(self):
        return self._scale_data(self.pixel_array)

    def _scale_data(self, data):
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


class SiemensWrapper(Wrapper):
    ''' Wrapper for Siemens format DICOMs '''
    def __init__(self, dcm_data=None, csa_header=None):
        if dcm_data is None:
            dcm_data = {}
        self.dcm_data = dcm_data
        if csa_header is None:
            csa_header = csar.get_csa_header(dcm_data)
        self.csa_header = csa_header

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
    def b_matrix(self):
        ''' Get DWI B matrix from Siemens DICOM referring to voxel space

        Parameters
        ----------
        None
        
        Returns
        -------
        B : (3,3) array or None
           B matrix in *voxel* orientation space.  Returns None if this is
           not a Siemens header with the required information.  We return
           None if this is a b0 acquisition
        '''
        hdr = self.csa_header
        # read B matrix as recorded in CSA header.  This matrix refers to
        # the space of the DICOM patient coordinate space.
        B = csar.get_b_matrix(hdr)
        if B is None: # may be not diffusion or B0 image
            bval_requested = csar.get_b_value(hdr)
            if bval_requested is None:
                return None
            if bval_requested != 0:
                raise csar.CSAError('No B matrix and b value != 0')
            return np.zeros((3,3))
        # rotation from voxels to DICOM PCS, inverted to give the rotation
        # from DPCS to voxels.  Because this is an orthonormal matrix, its
        # transpose is its inverse
        R = self.rotation_matrix.T
        # because B results from V dot V.T, the rotation B is given by R dot
        # V dot V.T dot R.T == R dot B dot R.T
        B_vox = np.dot(R, np.dot(B, R.T))
        # fix presumed rounding errors in the B matrix by making it positive
        # semi-definite. 
        return nearest_pos_semi_def(B_vox)

    @one_time
    def q_vector(self):
        ''' Get DWI q vector referring to voxel space

        Parameters
        ----------
        None

        Returns
        -------
        q: (3,) array
           Estimated DWI q vector in *voxel* orientation space.  Returns
           None if this is not (detectably) a DWI
        '''
        B = self.b_matrix
        if B is None:
            return None
        return B2q(B)


class MosaicWrapper(SiemensWrapper):

    @one_time
    def n_mosaic(self):
        if self.csa_header is None:
            raise WrapperError('No CSA information in data '
                               'is this really Siemans Mosiac?')
        return csar.get_n_mosaic(self.csa_header)

    @one_time
    def mosaic_size(self):
        n_o_m = self.n_mosaic
        return np.ceil(np.sqrt(n_o_m))

    @one_time
    def image_shape(self):
        # reshape pixel slice array back from mosaic
        rows = self.get('Rows')
        cols = self.get('Columns')
        mosaic_size = self.mosaic_size
        return (rows / mosaic_size,
                cols / mosaic_size,
                self.n_mosaic)
                
    @one_time
    def is_mosaic(self):
        return True

    @one_time
    def image_position(self):
        ''' Return position of first voxel in data block

        Parameters
        ----------
        None

        Returns
        -------
        img_pos : (3,) array
           position in mm of voxel (0,0,0) in Mosaic array
        '''
        image_position = self.get('ImagePositionPatient')
        if image_position is None:
            return None
        # size of mosaic array before rearranging to 3D
        md_xy = np.array([self.get('Rows'), self.get('Columns')])
        # size of slice X, Y in array after reshaping to 3D
        rd_xy = md_xy / self.mosaic_size
        # apply algorithm for undoing mosaic translation error - see
        # ``dicom_mosaic`` doc for details
        vox_trans_fixes = (md_xy - rd_xy) / 2
        M = self.rotation_matrix[:,:2] * self.voxel_sizes[:2]
        return image_position + np.dot(M, vox_trans_fixes[:,None]).ravel()
    
    def get_data(self):
        ''' Get scaled image data from DICOMs

        Resorts 

        Returns
        -------
        data : array
           array with data as scaled from any scaling in the DICOM
           fields. 
        '''
        n_rows, n_cols, n_mosaic = self.image_shape
        mosaic_size = self.mosaic_size
        data = self.pixel_array
        v4=data.reshape(mosaic_size,n_rows,
                        mosaic_size,n_cols)
        v4p=np.rollaxis(v4,2,1)
        v3=v4p.reshape(mosaic_size*mosaic_size,n_rows,n_cols)
        # delete any padding slices
        v3 = v3[:n_mosaic]
        return self._scale_data(v3)
