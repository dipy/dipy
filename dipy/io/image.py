
import nibabel as nib
import numpy as np


def load_nifti_data(fname, as_ndarray=True):
    """Load only the data array from a nifti file.

    Parameters
    ----------
    fname : str
        Full path to the file.
    as_ndarray: bool, optional
        convert nibabel ArrayProxy to a numpy.ndarray.
        If you want to save memory and delay this casting, just turn this
        option to False (default: True)

    Returns
    -------
    data: np.ndarray or nib.ArrayProxy

    See Also
    --------
    load_nifti

    """
    img = nib.load(fname)
    return np.asanyarray(img.dataobj) if as_ndarray else img.dataobj


def load_nifti(fname, return_img=False, return_voxsize=False,
               return_coords=False, as_ndarray=True):
    """Load data and other information from a nifti file.

    Parameters
    ----------
    fname : str
        Full path to a nifti file.

    return_img : bool, optional
        Whether to return the nibabel nifti img object. Default: False

    return_voxsize: bool, optional
        Whether to return the nifti header zooms. Default: False

    return_coords : bool, optional
        Whether to return the nifti header aff2axcodes. Default: False

    as_ndarray: bool, optional
        convert nibabel ArrayProxy to a numpy.ndarray.
        If you want to save memory and delay this casting, just turn this
        option to False (default: True)

    Returns
    -------
    A tuple, with (at the most, if all keyword args are set to True):
    (data, img.affine, img, vox_size, nib.aff2axcodes(img.affine))

    See Also
    --------
    load_nifti_data

    """
    img = nib.load(fname)
    data = np.asanyarray(img.dataobj) if as_ndarray else img.dataobj
    vox_size = img.header.get_zooms()[:3]

    ret_val = [data, img.affine]

    if return_img:
        ret_val.append(img)
    if return_voxsize:
        ret_val.append(vox_size)
    if return_coords:
        ret_val.append(nib.aff2axcodes(img.affine))

    return tuple(ret_val)


def save_nifti(fname, data, affine, hdr=None):
    """Save a data array into a nifti file.

    Parameters
    ----------
    fname : str
        The full path to the file to be saved.

    data : ndarray
        The array with the data to save.

    affine : 4x4 array
        The affine transform associated with the file.

    hdr : nifti header, optional
        May contain additional information to store in the file header.

    Returns
    -------
    None

    """
    result_img = nib.Nifti1Image(data, affine, header=hdr)
    result_img.to_filename(fname)


def save_qa_metric(fname, xopt, fopt):
    """Save Quality Assurance metrics.

    Parameters
    ----------
    fname: string
        File name to save the metric values.
    xopt: numpy array
        The metric containing the
        optimal parameters for
        image registration.
    fopt: int
        The distance between the registered images.

    """
    np.savetxt(fname, xopt, header="Optimal Parameter metric")
    with open(fname, 'a') as f:
        f.write('# Distance after registration\n')
        f.write(str(fopt))
