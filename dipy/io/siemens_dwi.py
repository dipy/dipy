''' Function to read Siemens DWI using pydicom '''

import numpy as np

import dicom as dcm

import nibabel as nib

from . import csareader as csa


def read_dwi(filename):
    ''' Read Siemens Mosaic file

    Parameters
    ----------
    filename : str
       filename of Siemens Mosaic file

    Returns
    -------
    img : ``Nifti1Image``
       Nifti image object
    '''
    dcm_data = dcm.read_file(filename)
    # read CSA headers
    hdr_str = dcm_data[0x29, 0x1010].value
    hdr_info = csa.read(hdr_str)
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
