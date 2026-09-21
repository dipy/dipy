import warnings

import nibabel as nib
import numpy as np
from packaging.version import Version

from dipy.io.utils import decfa, has_rgb_dtype, unpack_rgb_array
from dipy.utils.deprecator import warning_for_keywords


def _nifti_data(img, *, as_ndarray):
    """Extract numeric data or the original proxy from a NIfTI image.

    Parameters
    ----------
    img : nibabel.spatialimages.SpatialImage
        Image supplying the data object.
    as_ndarray : bool
        Whether to materialize and unpack structured color data.

    Returns
    -------
    data : ndarray or nibabel.arrayproxy.ArrayProxy
        Numeric array or the unchanged data proxy.
    """
    if not as_ndarray:
        return img.dataobj

    data = np.asanyarray(img.dataobj)
    if has_rgb_dtype(img.dataobj):
        warnings.warn(
            "Loaded NIfTI data has a structured RGB/RGBA dtype (DT_RGB24 or "
            "DT_RGBA32) and was auto-converted to a plain numeric array. "
            "Pass as_ndarray=False to get the raw structured dtype.",
            UserWarning,
            stacklevel=4,
        )
        data = unpack_rgb_array(data)
    return data


@warning_for_keywords()
def load_nifti_data(fname, *, as_ndarray=True):
    """Load only the data array from a nifti file.

    Parameters
    ----------
    fname : str or Path
        Full path to the file.
    as_ndarray: bool, optional
        convert nibabel ArrayProxy to a numpy.ndarray.
        If you want to save memory and delay this casting, just turn this
        option to False.

    Returns
    -------
    data: np.ndarray or nib.ArrayProxy

    Notes
    -----
    Structured RGB/RGBA data (NIfTI ``DT_RGB24``/``DT_RGBA32``) is
    auto-unpacked into a plain array with an added trailing color-channel
    axis, and a ``UserWarning`` is raised. Pass ``as_ndarray=False`` to skip
    this and receive the raw structured data.

    See Also
    --------
    load_nifti

    """
    img = nib.load(fname)
    return _nifti_data(img, as_ndarray=as_ndarray)


@warning_for_keywords()
def load_nifti(
    fname,
    *,
    return_img=False,
    return_voxsize=False,
    return_coords=False,
    as_ndarray=True,
):
    """Load data and other information from a nifti file.

    Parameters
    ----------
    fname : str or Path
        Full path to a nifti file.

    return_img : bool, optional
        Whether to return the nibabel nifti img object.

    return_voxsize: bool, optional
        Whether to return the nifti header zooms.

    return_coords : bool, optional
        Whether to return the nifti header aff2axcodes.

    as_ndarray: bool, optional
        convert nibabel ArrayProxy to a numpy.ndarray.
        If you want to save memory and delay this casting, just turn this
        option to False.

    Returns
    -------
    A tuple, with (at the most, if all keyword args are set to True):
    (data, img.affine, img, vox_size, nib.aff2axcodes(img.affine))

    Notes
    -----
    Like `load_nifti_data`, structured RGB/RGBA data is unpacked into a
    numeric array with a trailing color-channel axis and a ``UserWarning``
    when ``as_ndarray=True``. Pass ``as_ndarray=False`` to retain the raw
    proxy.

    The returned image is unchanged: ``np.asanyarray(img.dataobj)`` accesses
    its structured data, while ``img.get_fdata()`` cannot cast it to float.

    See Also
    --------
    load_nifti_data

    """
    img = nib.load(fname)
    data = _nifti_data(img, as_ndarray=as_ndarray)
    vox_size = img.header.get_zooms()[:3]

    ret_val = [data, img.affine]

    if return_img:
        ret_val.append(img)
    if return_voxsize:
        ret_val.append(vox_size)
    if return_coords:
        ret_val.append(nib.aff2axcodes(img.affine))

    return tuple(ret_val)


@warning_for_keywords()
def save_nifti(
    fname, data, affine, *, hdr=None, dtype=None, as_decfa=False, scale=None
):
    """Save a data array into a nifti file.

    Parameters
    ----------
    fname : str or Path
        The full path to the file to be saved.

    data : ndarray
        The array with the data to save.

    affine : 4x4 array
        The affine transform associated with the file.

    hdr : nifti header, optional
        May contain additional information to store in the file header.

    as_decfa : bool, optional
        If True, encode data as a DEC FA image via ``dipy.io.utils.decfa``;
        ``dtype`` is ignored in this case.

    scale : bool or None, optional
        When ``as_decfa`` is True, controls scaling from the 0-1 to the
        0-255 range. None enables scaling for float data. Scaling is ignored
        for integer data or when ``as_decfa`` is False.

    Returns
    -------
    None

    """
    NIBABEL_4_0_0_PLUS = Version(nib.__version__) >= Version("4.0.0")
    # See GitHub issues
    #  * https://github.com/nipy/nibabel/issues/1046
    #  * https://github.com/nipy/nibabel/issues/1089
    # This only applies to NIfTI because the parent Analyze formats did
    # not support 64-bit integer data, so `set_data_dtype(int64)` would
    # already fail.
    danger_dts = (np.dtype("int64"), np.dtype("uint64"))
    if (
        hdr is None
        and dtype is None
        and data.dtype in danger_dts
        and NIBABEL_4_0_0_PLUS
    ):
        msg = f"Image data has type {data.dtype}, which may cause "
        msg += "incompatibilities with other tools. Indeed, Analyze formats "
        msg += "did not support 64-bit integer data.\n\n"
        msg += "To silent this, please specify the `header` or `dtype` "
        msg += "You could also use `np.asarray(data, dtype=np.int32)`. "
        msg += "This cast will make sure that you data is compatible with "
        msg += "other software."

        raise ValueError(msg)

    kwargs = {"dtype": dtype} if NIBABEL_4_0_0_PLUS else {}
    result_img = nib.Nifti1Image(data, affine, header=hdr, **kwargs)

    if as_decfa:
        if scale is None:
            scale = np.issubdtype(data.dtype, np.floating)
        result_img = decfa(result_img, scale=scale)

    result_img.to_filename(fname)


def save_qa_metric(fname, xopt, fopt):
    """Save Quality Assurance metrics.

    Parameters
    ----------
    fname: string or Path
        File name to save the metric values.
    xopt: numpy array
        The metric containing the
        optimal parameters for
        image registration.
    fopt: int
        The distance between the registered images.

    """
    np.savetxt(fname, xopt, header="Optimal Parameter metric")
    with open(fname, "a") as f:
        f.write("# Distance after registration\n")
        f.write(str(fopt))
