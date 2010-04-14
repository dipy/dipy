from os.path import join as pjoin
import glob

import numpy as np

from ..core.geometry import vector_norm

from . import csareader as csar


class MosaicError(csar.CSAError):
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
    from .dicomwrappers import make_wrapper
    import nibabel as nib
    dcm_w = make_wrapper(dcm_data)
    if not dcm_w.is_mosaic:
        raise MosaicError('data does not appear to be in mosaic format')
    data = dcm_w.get_data()
    aff = np.dot(DPCS_TO_TAL, dcm_w.affine)
    return nib.Nifti1Image(data.T, aff)


def read_mosaic_dwi_dir(dicom_path, globber='*.dcm'):
    ''' Read all Siemens DICOMs in directory, return arrays, params

    Parameters
    ----------
    dicom_path : str
       path containing mosaic DICOM images
    globber : str, optional
       glob to apply within `dicom_path` to select DICOM files.  Default
       is ``*.dcm``
       
    Returns
    -------
    data : 4D array
       data array with last dimension being acquisition. If there were N
       acquisitions, each of shape (X, Y, Z), `data` will be shape (X,
       Y, Z, N)
    affine : (4,4) array
       affine relating 3D voxel space in data to RAS world space
    b_values : (N,) array
       b values for each acquisition
    unit_gradients : (N, 3) array
       gradient directions of unit length for each acquisition
    '''
    from .dicomwrappers import wrapper_from_file
    full_globber = pjoin(dicom_path, globber)
    filenames = sorted(glob.glob(full_globber))
    b_values = []
    gradients = []
    arrays = []
    if len(filenames) == 0:
        raise IOError('Found no files with "%s"' % full_globber)
    for fname in filenames:
        dcm_w = wrapper_from_file(fname)
        arrays.append(dcm_w.get_data()[...,None])
        q = dcm_w.q_vector
        b = vector_norm(q)
        g = q / b
        b_values.append(b)
        gradients.append(g)
    affine = np.dot(DPCS_TO_TAL, dcm_w.affine)
    return (np.concatenate(arrays, -1),
            affine,
            np.array(b_values),
            np.array(gradients))

    
