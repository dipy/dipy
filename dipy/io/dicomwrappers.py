''' Classes to wrap DICOM objects and files

The wrappers encapsulate the capcbilities of the different DICOM
formats.

They also allow dictionary-like access to named fiields.

For calculated attributes, we return None where needed data is missing.
It seemed strange to raise an error during attribute processing, other
than an AttributeError - breaking the 'properties manifesto'.   So, any
procesing that needs to raise an error, should be in a method, rather
than in a property, or property-like thing. 
'''

import operator

import numpy as np

from . import csareader as csar
from .dwiparams import B2q
from ..core.geometry import nearest_pos_semi_def
from .utils import allopen
from ..core.onetime import setattr_on_read as one_time


class WrapperError(Exception):
    pass


def wrapper_from_file(file_like):
    ''' Create DICOM wrapper from `file_like` object

    Parameters
    ----------
    file_like : object
       filename string or file-like object, pointing to a valid DICOM
       file readable by ``pydicom``

    Returns
    -------
    dcm_w : ``dicomwrappers.Wrapper`` or subclass
       DICOM wrapper corresponding to DICOM data type
    '''
    import dicom
    fobj = allopen(file_like)
    dcm_data = dicom.read_file(fobj)
    return wrapper_from_data(dcm_data)


def wrapper_from_data(dcm_data):
    ''' Create DICOM wrapper from DICOM data object

    Parameters
    ----------
    dcm_data : ``dicom.dataset.Dataset`` instance or similar
       Object allowing attribute access, with DICOM attributes.
       Probably a dataset as read by ``pydicom``.
       
    Returns
    -------
    dcm_w : ``dicomwrappers.Wrapper`` or subclass
       DICOM wrapper corresponding to DICOM data type
    '''
    csa = csar.get_csa_header(dcm_data)
    if csa is None:
        return Wrapper(dcm_data)
    if not csar.is_mosaic(csa):
        return SiemensWrapper(dcm_data, csa)
    return MosaicWrapper(dcm_data, csa)


class Wrapper(object):
    ''' Class to wrap general DICOM files

    Methods:

    * get_affine()
    * get_data()
    * get_pixel_array()
    * mabye_same_vol(other)
    * __getitem__ : return attributes from `dcm_data` 
    * get(key[, default]) - as usual given __getitem__ above

    Attributes and things that look like attributes:

    * dcm_data : object
    * image_shape : tuple
    * image_orient_patient : (3,2) array
    * slice_normal : (3,) array
    * rotation_matrix : (3,3) array
    * voxel_sizes : tuple length 3
    * image_position : sequence length 3
    * slice_indicator : float
    * vol_match_signature : tuple
    '''
    is_csa = False
    is_mosaic = False
    b_matrix = None
    q_vector = None
    
    def __init__(self, dcm_data=None):
        ''' Initialize wrapper

        Parameters
        ----------
        dcm_data : None or object, optional
           object should allow attribute access.  Usually this will be
           a ``dicom.dataset.Dataset`` object resulting from reading a
           DICOM file.   If None, just make an empty dict. 
        '''
        if dcm_data is None:
            dcm_data = {}
        self.dcm_data = dcm_data

    @one_time
    def image_shape(self):
        shape = (self.get('Rows'), self.get('Columns'))
        if None in shape:
            return None
        return shape
    
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
            return None
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
        return tuple(pix_space + [zs])

    @one_time
    def image_position(self):
        ''' Return position of first voxel in data block

        Parameters
        ----------
        None

        Returns
        -------
        img_pos : (3,) array
           position in mm of voxel (0,0) in image array
        '''
        ipp = self.get('ImagePositionPatient')
        if ipp is None:
            return None
        return np.array(ipp)

    @one_time
    def slice_indicator(self):
        ''' A number that is higher for higher slices in Z

        Comparing this number between two adjacent slices should give a
        difference equal to the voxel size in Z. 
        
        See doc/theory/dicom_orientation for description
        '''
        ipp = self.image_position
        s_norm = self.slice_normal
        if None in (ipp, s_norm):
            return None
        return np.inner(ipp, s_norm)

    @one_time
    def instance_number(self):
        ''' Just becase we use this a lot for sorting '''
        return self.get('InstanceNumber')

    @one_time
    def vol_match_signature(self):
        ''' Signature for matching slices into volumes

        We use `signature` in ``self.maybe_same_vol(other)``.  

        Returns
        -------
        signature : dict
           with values of 2-element sequences, where first element is
           value, and second element is function to compare this value
           with another.  This allows us to pass things like arrays,
           that might need to be ``allclose`` instead of equal
        '''
        # dictionary with value, comparison func tuple
        signature = {}
        eq = operator.eq
        for key in ('SeriesNumber',
                    'ImageType',
                    'SequenceName',
                    'SeriesInstanceID',
                    'EchoNumbers'):
            signature[key] = (self.get(key), eq)
        signature['image_shape'] = (self.image_shape, eq)
        signature['iop'] = (self.image_orient_patient, none_or_close)
        signature['vox'] = (self.voxel_sizes, none_or_close)
        return signature
    
    def __getitem__(self, key):
        ''' Return values from DICOM object'''
        try:
            return getattr(self.dcm_data, key)
        except AttributeError:
            raise KeyError('%s not defined in dcm_data' % key)

    def get(self, key, default=None):
        return getattr(self.dcm_data, key, default)

    def get_affine(self):
        ''' Return mapping between voxel and DICOM coordinate system
        
        Parameters
        ----------
        None

        Returns
        -------
        aff : (4,4) affine
           Affine giving transformation between voxels in data array and
           mm in the DICOM patient coordinate system.
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

    def get_pixel_array(self):
        ''' Return unscaled pixel array from DICOM '''
        try:
            return self['pixel_array']
        except KeyError:
            raise WrapperError('Cannot find data in DICOM')
    
    def get_data(self):
        ''' Get scaled image data from DICOMs

        Returns
        -------
        data : array
           array with data as scaled from any scaling in the DICOM
           fields. 
        '''
        return self._scale_data(self.get_pixel_array())

    def maybe_same_vol(self, other):
        ''' First pass at clustering into volumes check

        Parameters
        ----------
        other : object
           object with ``vol_match_signature`` attribute that is a
           mapping.  Usually it's a ``Wrapper`` or sub-class instance.

        Returns
        -------
        tf : bool
           True if `other` might be in the same volume as `self`, False
           otherwise. 
        '''
        # compare signature dictionaries.  The dictionaries each contain
        # comparison rules, we prefer our own when we have them.  If a
        # key is not present in either dictionary, assume the value is
        # None.
        my_sig = self.vol_match_signature
        your_sig = other.vol_match_signature
        my_keys = set(my_sig)
        your_keys = set(your_sig)
        # we have values in both signatures
        for key in my_keys.intersection(your_keys):
            v1, func = my_sig[key]
            v2, _ = your_sig[key]
            if not func(v1, v2):
                return False
        # values present in one or the other but not both
        for keys, sig in ((my_keys - your_keys, my_sig),
                          (your_keys - my_keys, your_sig)):
            for key in keys:
                v1, func = sig[key]
                if not func(v1, None):
                    return False
        return True

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
    ''' Wrapper for Siemens format DICOMs

    Adds attributes:

    * csa_header : mapping
    * b_matrix : (3,3) array
    * q_vector : (3,) array
    '''
    is_csa = True

    def __init__(self, dcm_data=None, csa_header=None):
        ''' Initialize Siemens wrapper

        The Siemens-specific information is in the `csa_header`, either
        passed in here, or read from the input `dcm_data`. 

        Parameters
        ----------
        dcm_data : None or object, optional
           object should allow attribute access.  If `csa_header` is
           None, it should also be possible to extract a CSA header from
           `dcm_data`. Usually this will be a ``dicom.dataset.Dataset``
           object resulting from reading a DICOM file.  If None, we just
           make an empty dict.
        csa_header : None or mapping, optional
           mapping giving values for Siemens CSA image sub-header.  If
           None, we try and read the CSA information from `dcm_data`.
           If this fails, we fall back to an empty dict.
        '''
        if dcm_data is None:
            dcm_data = {}
        self.dcm_data = dcm_data
        if csa_header is None:
            csa_header = csar.get_csa_header(dcm_data)
            if csa_header is None:
                csa_header = {}
        self.csa_header = csa_header

    @one_time
    def slice_normal(self):
        slice_normal = csar.get_slice_normal(self.csa_header)
        if not slice_normal is None:
            return np.array(slice_normal)
        iop = self.image_orient_patient
        if iop is None:
            return None
        return np.cross(*iop.T[:])

    @one_time
    def vol_match_signature(self):
        ''' Add ICE dims from CSA header to signature '''
        signature = super(SiemensWrapper, self).vol_match_signature
        ice = csar.get_ice_dims(self.csa_header)
        if not ice is None:
            ice = ice[:6] + ice[8:9]
        signature['ICE_Dims'] = (ice, lambda x, y: x == y)
        return signature
    
    @one_time
    def b_matrix(self):
        ''' Get DWI B matrix referring to voxel space

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
    ''' Class for Siemens mosaic format data

    Mosaic format is a way of storing a 3D image in a 2D slice - and
    it's as simple as you'd image it would be - just storing the slices
    in a mosaic similar to a light-box print.

    We need to allow for this when getting the data and (because of an
    idiosyncracy in the way Siemens stores the images) calculating the
    position of the first voxel.

    Adds attributes:

    * n_mosaic : int
    * mosaic_size : float
    '''
    is_mosaic = True
    
    def __init__(self, dcm_data=None, csa_header=None, n_mosaic=None):
        ''' Initialize Siemens Mosaic wrapper

        The Siemens-specific information is in the `csa_header`, either
        passed in here, or read from the input `dcm_data`. 

        Parameters
        ----------
        dcm_data : None or object, optional
           object should allow attribute access.  If `csa_header` is
           None, it should also be possible for to extract a CSA header
           from `dcm_data`. Usually this will be a
           ``dicom.dataset.Dataset`` object resulting from reading a
           DICOM file.  If None, just make an empty dict.
        csa_header : None or mapping, optional
           mapping giving values for Siemens CSA image sub-header.
        n_mosaic : None or int, optional
           number of images in mosaic.  If None, try to get this number
           fron `csa_header`.  If this fails, raise an error
        '''
        SiemensWrapper.__init__(self, dcm_data, csa_header)
        if n_mosaic is None:
            try:
                n_mosaic = csar.get_n_mosaic(self.csa_header)
            except KeyError:
                pass
            if n_mosaic is None or n_mosaic == 0:
                raise WrapperError('No valid mosaic number in CSA '
                                   'header; is this really '
                                   'Siemans mosiac data?')
        self.n_mosaic = n_mosaic
        self.mosaic_size = np.ceil(np.sqrt(n_mosaic))
        
    @one_time
    def image_shape(self):
        # reshape pixel slice array back from mosaic
        rows = self.get('Rows')
        cols = self.get('Columns')
        if None in (rows, cols):
            return None
        mosaic_size = self.mosaic_size
        return (int(rows / mosaic_size),
                int(cols / mosaic_size),
                self.n_mosaic)
                
    @one_time
    def image_position(self):
        ''' Return position of first voxel in data block

        Adjusts Siemans mosaic position vector for bug in mosaic format
        position.  See ``dicom_mosaic`` in doc/theory for details. 

        Parameters
        ----------
        None

        Returns
        -------
        img_pos : (3,) array
           position in mm of voxel (0,0,0) in Mosaic array
        '''
        ipp = self.get('ImagePositionPatient')
        o_rows, o_cols = (self.get('Rows'), self.get('Columns'))
        iop = self.image_orient_patient
        vox = self.voxel_sizes
        if None in (ipp, o_rows, o_cols, iop, vox):
            return None
        # size of mosaic array before rearranging to 3D
        md_xy = np.array([o_rows, o_cols])
        # size of slice X, Y in array after reshaping to 3D
        rd_xy = md_xy / self.mosaic_size
        # apply algorithm for undoing mosaic translation error - see
        # ``dicom_mosaic`` doc
        vox_trans_fixes = (md_xy - rd_xy) / 2
        M = iop * vox[:2]
        return ipp + np.dot(M, vox_trans_fixes[:,None]).ravel()
    
    def get_data(self):
        ''' Get scaled image data from DICOMs

        Resorts data block from mosaic to 3D

        Returns
        -------
        data : array
           array with data as scaled from any scaling in the DICOM
           fields. 
        '''
        shape = self.image_shape
        if shape is None:
            raise WrapperError('No valid information for image shape')
        n_rows, n_cols, n_mosaic = shape
        mosaic_size = self.mosaic_size
        data = self.get_pixel_array()
        v4=data.reshape(mosaic_size,n_rows,
                        mosaic_size,n_cols)
        v4p=np.rollaxis(v4,2,1)
        v3=v4p.reshape(mosaic_size*mosaic_size,n_rows,n_cols)
        # delete any padding slices
        v3 = v3[:n_mosaic]
        return self._scale_data(v3)


def none_or_close(val1, val2, rtol=1e-5, atol=1e-6):
    ''' Match if `val1` and `val2` are both None, or are close

    Parameters
    ----------
    val1 : None or array-like
    val2 : None or array-like
    rtol : float, optional
       Relative tolerance; see ``np.allclose``
    atol : float, optional
       Absolute tolerance; see ``np.allclose``
       
    Returns
    -------
    tf : bool
       True iff (both `val1` and `val2` are None) or (`val1` and `val2`
       are close arrays, as detected by ``np.allclose`` with parameters
       `rtol` and `atal`).

    Examples
    --------
    >>> none_or_close(None, None)
    True
    >>> none_or_close(1, None)
    False
    >>> none_or_close(None, 1)
    False
    >>> none_or_close([1,2], [1,2])
    True
    >>> none_or_close([0,1], [0,2])
    False
    '''
    if (val1, val2) == (None, None):
        return True
    if None in (val1, val2):
        return False
    return np.allclose(val1, val2, rtol, atol)
