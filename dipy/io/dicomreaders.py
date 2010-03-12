import numpy as np

import dicom

import nibabel as nib

from . import csareader as csar


class CSAError(Exception):
    pass


class MosiacError(CSAError):
    pass


DPCS_TO_TAL = np.diag([-1, -1, 1, 1])


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


def has_csa(dicom_header):
    return get_csa_header(dicom_header) is not None


def get_csa_header(dicom_header, csa_type='image'):
    ''' Get CSA header information from DICOM header

    Return None if the header does not contain CSA information of the
    specified `csa_type`

    Parameters
    ----------
    dicom_header : dicom.Dataset
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
        tag = dicom_header[0x29, element_no]
    except KeyError:
        return None
    if tag.name != '[CSA %s Header Info]' % label:
        return None
    return csar.read(tag.value)


def is_mosaic(dicom_header):
    ''' Return True if the data is of Siemens mosaic type

    Parameters
    ----------
    dicom_header : ``dicom.Dataset`
       DICOM dataset object as read from DICOM file

    Returns
    -------
    tf : bool
       True if the `dicom_header` appears to be of Siemens mosaic type,
       False otherwise
    '''
    try:
        dicom_header.ImageType
    except AttributeError:
        return False
    hdr = get_csa_header(dicom_header)
    if hdr is None:
        return False
    if csar.get_acq_mat_txt(hdr) is None:
        return False
    n_o_m = csar.get_n_mosaic(hdr)
    return not (n_o_m is None) and n_o_m != 0


def get_b_matrix(dicom_header):
    ''' Get voxel B matrix from Siemens DICOM '''
    hdr = get_csa_header(dicom_header)
    if hdr_info is None:
        raise CSAError('data does not appear to be Siemens format')
    iop = np.array(dicom_header.ImageOrientationPatient)
    iop = iop.reshape(2,3).T
    snv = csar.get_slice_normal(hdr_info)
    R = np.c_[iop, snv] # vox to dpcs
    Rdash = npl.inv(R) # dpcs to vox
    B = csar.get_b_matrix(hdr_info)
    return np.dot(Rdash, B)


def get_q_vector(dicom_header):
    B = get_b_matrix(dicom_header)
    return B2q(B)


def get_vox_to_dpcs(dcm_data):
    ''' Return mapping between voxel and DICOM space for mosaic'''
    hdr = get_csa_header(dcm_data)
    if hdr is None:
        raise MosaicError('data does not appear to be mosaic format')
    iop = np.array(dcm_data.ImageOrientationPatient)
    iop = iop.reshape(2,3).T
    snv = csar.get_slice_normal(hdr)
    R = np.c_[iop, snv]
    s = dcm_data.PixelSpacing + [dcm_data.SpacingBetweenSlices]
    aff = np.eye(4)
    aff[:3,:3] = np.dot(R, np.diag(s))
    i = dcm_data.ImagePositionPatient
    md_xy = np.array(dcm_data.PixelArray.shape)
    n_o_m = csar.get_n_mosaic(hdr)
    mosaic_size = np.ceil(np.sqrt(n_o_m))
    rd_xy = md_xy / mosaic_size
    vox_trans_fixes = (md_xy - rd_xy)/ 2
    t = i + np.dot(iop, vox_trans_fixes[:,None]).ravel()
    aff[:3,3] = t
    return aff

    
