import numpy as np

import dicom

import nibabel as nib

from . import csareader as csar
from .dwiparams import B2q


class CSAError(Exception):
    pass


class MosiacError(CSAError):
    pass


DPCS_TO_TAL = np.diag([-1, -1, 1, 1])


def _fairly_close(A, B):
    return np.allclose(A, B, atol=1e-6)


def mosaic_to_nii(dcm_data):
    ''' Get Nifti file from Siemens 

    Parameters
    ----------
    dcm_data : ``dicom.DataSet``
       DICOM header / image as read by ``dicom`` package

    Returns
    -------
    img : ``Nifti1Image``
       Nifti image object
    '''
    # read CSA headers
    hdr_info = get_csa_header(dcm_data)
    if hdr_info is None or not is_mosaic(dcm_data):
        raise MosaicError('data does not appear to be mosaic format')
    # get Mosaic size
    n_o_m = csar.get_n_mosaic(hdr_info)
    # reshape pixel slice array back from mosaic
    mosaic_size = np.ceil(np.sqrt(n_o_m))
    data = dcm_data.pixel_array
    n_rows = data.shape[0] / mosaic_size
    n_cols = data.shape[1] / mosaic_size
    v4=data.reshape(mosaic_size,n_rows,
                    mosaic_size,n_cols)
    v4p=np.rollaxis(v4,2,1)
    v3=v4p.reshape(mosaic_size*mosaic_size,n_rows,n_cols)
    # delete any padding slices
    v3 = v3[:n_o_m]
    # affine
    aff = get_vox_to_dpcs(dcm_data)
    aff = np.dot(DPCS_TO_TAL, aff)
    return nib.Nifti1Image(v3.T, aff)


def has_csa(dcm_data):
    return get_csa_header(dcm_data) is not None


def get_csa_header(dcm_data, csa_type='image'):
    ''' Get CSA header information from DICOM header

    Return None if the header does not contain CSA information of the
    specified `csa_type`

    Parameters
    ----------
    dcm_data : dicom.Dataset
       DICOM dataset object as read from DICOM file
    csa_type : {'image', 'series'}, optional
       Type of CSA field to read; default is 'image'

    Returns
    -------
    csa_info : None or dict
       Parsed CSA field of `csa_type` or None, if we cannot find the CSA
       information.
    '''
    if csa_type == 'image':
        element_no = 0x1010
        label = 'Image'
    elif csa_type == 'series':
        element_no = 0x1020
        label = 'Series'
    else:
        raise ValueError('Invalid CSA header type "%s"'
                         % csa_type)
    try:
        tag = dcm_data[0x29, element_no]
    except KeyError:
        return None
    if tag.name != '[CSA %s Header Info]' % label:
        return None
    return csar.read(tag.value)


def is_mosaic(dcm_data):
    ''' Return True if the data is of Siemens mosaic type

    Parameters
    ----------
    dcm_data : ``dicom.Dataset`
       DICOM dataset object as read from DICOM file

    Returns
    -------
    tf : bool
       True if the `dcm_data` appears to be of Siemens mosaic type,
       False otherwise
    '''
    try:
        dcm_data.ImageType
    except AttributeError:
        return False
    hdr = get_csa_header(dcm_data)
    if hdr is None:
        return False
    if csar.get_acq_mat_txt(hdr) is None:
        return False
    n_o_m = csar.get_n_mosaic(hdr)
    return not (n_o_m is None) and n_o_m != 0


def get_b_matrix(dcm_data):
    ''' Get DWI B matrix from Siemens DICOM referring to voxel space

    Parameters
    ----------
    dcm_data : ``dicom.Dataset``
       Read DICOM header

    Returns
    -------
    B : (3,3) array or None
       B matrix in *voxel* orientation space.  Returns None if this is
       not a Siemens header with the required information.  We return
       None if this is a b0 acquisition
    '''
    hdr = get_csa_header(dcm_data)
    if hdr is None:
        raise CSAError('data does not appear to be Siemens format')
    # read B matrix as recorded in CSA header.  This matrix is in DICOM
    # patient coordinate space. 
    B = csar.get_b_matrix(hdr)
    if B is None:
        return None
    # We need the rotations from the DICOM header and the Siemens header
    # in order to convert the B matrix to voxel space
    iop = np.array(dcm_data.ImageOrientationPatient)
    iop = iop.reshape(2,3).T
    snv = csar.get_slice_normal(hdr)
    # rotation from voxels to DICOM PCS. Because this is an orthogonal
    # matrix, its inverse is its transpose
    R = np.c_[iop, snv]
    assert _fairly_close(np.eye(3), np.dot(R, R.T))
    return np.dot(R.T, B)


def get_q_vector(dcm_data):
    ''' Get DWI q vector from Siemens DICOM referring to voxel space

    Parameters
    ----------
    dcm_data : ``dicom.Dataset``
       Read DICOM header

    Returns
    -------
    q: (3,) array
       Estimated DWI q vector in *voxel* orientation space.  Returns
       None if this is not a DWI
    '''
    B = get_b_matrix(dcm_data)
    if B is None:
        return None
    return B2q(B)


def get_vox_to_dpcs(dcm_data):
    ''' Return mapping between voxel and DICOM space for mosaic
    
    Parameters
    ----------
    dcm_data : ``dicom.Dataset`
       DICOM dataset object as read from DICOM file etc.  It should be
       in Siemens mosaic format

    Returns
    -------
    aff : (4,4) affine
       Affine giving transformation between voxels in mosaic data array
       after rearranging to 3D, and the DICOM patient coordinate
       system.  
    '''
    hdr = get_csa_header(dcm_data)
    if hdr is None:
        raise MosaicError('data does not appear to be mosaic format')
    # compile orthogonal component of matrix
    iop = np.array(dcm_data.ImageOrientationPatient)
    iop = iop.reshape(2,3).T
    snv = csar.get_slice_normal(hdr)
    R = np.c_[iop, snv]
    # check that the slice normal vector does in fact result in an
    # orthogonal matrix
    assert _fairly_close(np.eye(3), np.dot(R, R.T))
    # compile scaling part of matrix
    s = dcm_data.PixelSpacing + [dcm_data.SpacingBetweenSlices]
    aff = np.eye(4)
    aff[:3,:3] = R * s
    # get translation part of affine.  We have to fix an error in the
    # Siemens stated offset in the mosaic format
    i = dcm_data.ImagePositionPatient
    # size of mosaic array before rearranging to 3D
    md_xy = np.array(dcm_data.PixelArray.shape)
    # size of slice X, Y in array after reshaping to 3D
    n_o_m = csar.get_n_mosaic(hdr)
    mosaic_size = np.ceil(np.sqrt(n_o_m))
    rd_xy = md_xy / mosaic_size
    # apply algorithm for undoing mosaic translation error - see
    # ``dicom_mosaic`` doc for details
    vox_trans_fixes = (md_xy - rd_xy)/ 2
    M = iop * dcm_data.PixelSpacing
    t = i + np.dot(M, vox_trans_fixes[:,None]).ravel()
    aff[:3,3] = t
    return aff

    
