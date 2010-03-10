import dicom

import csareader as csar

class CSAHeaderError(Exception):
    pass

def has_csa(dicom_header):
    return get_csa_header(dicom_header) is not None

def get_csa_header(dicom_header, type='image'):
    if type == 'image':
        minor = 0x1010
        label = 'Image'
    elif type == 'series':
        minor = 0x1020
        label = 'Series'
    else:
        raise ValueError('Invalid csa dictionary option')
    try:
        tag = dicom_header[0x29,minor]
    except KeyError:
        return None    
    if tag.name != '[CSA %s Header Info]' % label:
        return None
    return csar.read(tag.value)

def is_mosaic(dicom_image_header):
    try:
        dicom_image_header.ImageType
    except AttributeError:
        return False
    hdr = get_csa_header(dicom_image_header)
    if hdr is None:
        return False
    tags = hdr['tags']
    if len(tags['AcquisitionMatrixText']['items']) == 0:
        return False
    items = tags['NumberOfImagesInMosaic']['items']
    return len(items) != 0 and items[0] != 0
