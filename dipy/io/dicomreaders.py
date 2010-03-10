import dicom

import csareader as csar


class CSAHeaderError(Exception):
    pass


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
