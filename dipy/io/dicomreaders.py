import numpy as np

import dicom

import nibabel as nib

from . import csareader as csar


class MosiacError(Exception):
    pass


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
    if not is_mosaic(dcm_data):
        raise MosaicError('data does not appear to be mosaic format')
    # read CSA headers
    hdr_info = get_csa_header(dcm_data)
    # get Mosaic size
    n_o_m = hdr_info['tags']['NumberOfImagesInMosaic']
    n_o_m = n_o_m['items'][0]
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
    return nib.Nifti1Image(v3.T, None)


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
    csa_info : None ordict
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
    tags = hdr['tags']
    if len(tags['AcquisitionMatrixText']['items']) == 0:
        return False
    items = tags['NumberOfImagesInMosaic']['items']
    return len(items) != 0 and items[0] != 0


def get_vox_to_dicom(dicom_header):
    ''' Return mapping between voxel and DICOM space '''
    pass
