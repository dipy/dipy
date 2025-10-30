"""Utility functions for file formats"""

import enum
import numbers
from pathlib import Path

import nibabel as nib
from nibabel import Nifti1Image
from nibabel.streamlines import detect_format
import numpy as np
from trx import trx_file_memmap

import dipy
from dipy.testing.decorators import warning_for_keywords
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

pd, have_pd, _ = optional_package("pandas")

if have_pd:
    import pandas as pd


class Space(enum.Enum):
    """Enum to simplify future change to convention"""

    VOX = "vox"
    VOXMM = "voxmm"
    RASMM = "rasmm"
    LPSMM = "lpsmm"


class Origin(enum.Enum):
    """Enum to simplify future change to convention"""

    # TODO: maybe gifti and vtk should be different origins?
    # Required to do mapping using numpy
    NIFTI = "center"
    TRACKVIS = "corner"


def nifti1_symmat(image_data, *args, **kwargs):
    """Returns a Nifti1Image with a symmetric matrix intent

    Parameters
    ----------
    image_data : array-like
        should have lower triangular elements of a symmetric matrix along the
        last dimension
    *args
        Passed to Nifti1Image
    **kwargs
        Passed to Nifti1Image

    Returns
    -------
    image : Nifti1Image
        5d, extra dimensions added before the last. Has symmetric matrix intent
        code

    """
    image_data = make5d(image_data)
    last_dim = image_data.shape[-1]
    n = (np.sqrt(1 + 8 * last_dim) - 1) / 2
    if (n % 1) != 0:
        raise ValueError("input_data does not seem to have matrix elements")

    image = Nifti1Image(image_data, *args, **kwargs)
    hdr = image.header
    hdr.set_intent("symmetric matrix", (n,))
    return image


def make5d(data):
    """reshapes the input to have 5 dimensions, adds extra dimensions just
    before the last dimension
    """
    data = np.asarray(data)
    if data.ndim > 5:
        raise ValueError("input is already more than 5d")
    shape = data.shape
    shape = shape[:-1] + (1,) * (5 - len(shape)) + shape[-1:]
    return data.reshape(shape)


@warning_for_keywords()
def decfa(img_orig, *, scale=False):
    """
    Create a nifti-compliant directional-encoded color FA image.

    Parameters
    ----------
    img_orig : Nifti1Image class instance.
        Contains encoding of the DEC FA image with a 4D volume of data, where
        the elements on the last dimension represent R, G and B components.

    scale: bool.
        Whether to scale the incoming data from the 0-1 to the 0-255 range
        expected in the output.

    Returns
    -------
    img : Nifti1Image class instance with dtype set to store tuples of
        uint8 in (R, G, B) order.


    Notes
    -----
    For a description of this format, see:

    https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/datatype.html
    """

    dest_dtype = np.dtype([("R", "uint8"), ("G", "uint8"), ("B", "uint8")])
    out_data = np.zeros(img_orig.shape[:3], dtype=dest_dtype)

    data_orig = np.asanyarray(img_orig.dataobj)

    if scale:
        data_orig = (data_orig * 255).astype("uint8")

    for ii in np.ndindex(img_orig.shape[:3]):
        val = data_orig[ii]
        out_data[ii] = (val[0], val[1], val[2])

    new_hdr = img_orig.header
    new_hdr["dim"][4] = 1
    new_hdr.set_intent(1001, name="Color FA")
    new_hdr.set_data_dtype(dest_dtype)

    return Nifti1Image(out_data, affine=img_orig.affine, header=new_hdr)


def decfa_to_float(img_orig):
    """
    Convert a nifti-compliant directional-encoded color FA image into a
    nifti image with RGB encoded in floating point resolution.

    Parameters
    ----------
    img_orig : Nifti1Image class instance.
        Contains encoding of the DEC FA image with a 3D volume of data, where
        each element is a (R, G, B) tuple in uint8.

    Returns
    -------
    img : Nifti1Image class instance with float dtype.

    Notes
    -----
    For a description of this format, see:

    https://nifti.nimh.nih.gov/nifti-1/documentation/nifti1fields/nifti1fields_pages/datatype.html
    """

    data_orig = np.asanyarray(img_orig.dataobj)
    out_data = np.zeros(data_orig.shape + (3,), dtype=np.uint8)

    for ii in np.ndindex(img_orig.shape[:3]):
        val = data_orig[ii]
        out_data[ii] = np.array([val[0], val[1], val[2]])

    new_hdr = img_orig.header
    new_hdr["dim"][4] = 3

    # Remove the original intent
    new_hdr.set_intent(0)
    new_hdr.set_data_dtype(float)

    return Nifti1Image(out_data, affine=img_orig.affine, header=new_hdr)


def is_reference_info_valid(affine, dimensions, voxel_sizes, voxel_order):
    """Validate basic data type and value of spatial attribute.

    Does not ensure that voxel_sizes and voxel_order are self-coherent with
    the affine.

    Only verifies the following:
        - affine is of the right type (float) and dimension (4,4)
        - affine contain values in the rotation part
        - dimensions is of right type (int) and length (3)
        - voxel_sizes is of right type (float) and length (3)
        - voxel_order is of right type (str) and length (3)

    The listed parameters are what is expected, provide something else and this
    function should fail (cover common mistakes).

    Parameters
    ----------
    affine: ndarray (4,4)
        Transformation of VOX to RASMM
    dimensions: ndarray (3,), int16
        Volume shape for each axis
    voxel_sizes:  ndarray (3,), float32
        Size of voxel for each axis
    voxel_order: string
        Typically 'RAS' or 'LPS'

    Returns
    -------
    output : bool
        Does the input represent a valid 'state' of spatial attribute

    """
    all_valid = True
    only_3d_warning = False

    if not affine.shape == (4, 4):
        all_valid = False
        logger.warning("Transformation matrix must be 4x4")

    if not affine[0:3, 0:3].any():
        all_valid = False
        logger.warning("Rotation matrix cannot be all zeros")

    if not len(dimensions) >= 3:
        all_valid = False
        only_3d_warning = True

    for i in dimensions:
        if not isinstance(i, numbers.Integral):
            all_valid = False
            logger.warning("Dimensions must be int.")
        if i <= 0:
            all_valid = False
            logger.warning("Dimensions must be above 0.")

    if not len(voxel_sizes) >= 3:
        all_valid = False
        only_3d_warning = True
    for i in voxel_sizes:
        if not isinstance(i, numbers.Number):
            all_valid = False
            logger.warning("Voxel size must be int/float.")
        if i <= 0:
            all_valid = False
            logger.warning("Voxel size must be above 0.")

    if not len(voxel_order) >= 3:
        all_valid = False
        only_3d_warning = True
    for i in voxel_order:
        if not isinstance(i, str):
            all_valid = False
            logger.warning("Voxel order must be string/char.")
        if i not in ["R", "A", "S", "L", "P", "I"]:
            all_valid = False
            logger.warning("Voxel order does not follow convention.")

    if only_3d_warning:
        logger.warning("Only 3D (and above) reference are considered valid.")

    return all_valid


def get_reference_info(reference):
    """Get the spatial attributes of the given data file.

    Parameters
    ----------
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict), TrxFile or trx.header (dict)
        Reference that provides the spatial attribute.

    Returns
    -------
    output : tuple
        - affine ndarray (4,4), np.float64, transformation of VOX to RASMM
        - dimensions ndarray (3,), int16, volume shape for each axis
        - voxel_sizes  ndarray (3,), float32, size of voxel for each axis
        - voxel_order, string, Typically 'RAS' or 'LPS'
    """
    is_nifti = False
    is_trk = False
    is_sft = False
    is_trx = False

    if isinstance(reference, (str, Path)):
        _, ext = split_filename_extension(reference)
        ext = ext.lower()
        if ext in [".nii", ".nii.gz"]:
            header = nib.load(reference).header
            is_nifti = True
        elif ext == ".trk":
            header = nib.streamlines.load(reference, lazy_load=True).header
            is_trk = True
        elif ext == ".trx":
            header = trx_file_memmap.load(reference).header
            is_trx = True
    elif isinstance(reference, trx_file_memmap.TrxFile):
        header = reference.header
        is_trx = True
    elif isinstance(reference, nib.nifti1.Nifti1Image):
        header = reference.header
        is_nifti = True
    elif isinstance(reference, nib.streamlines.trk.TrkFile):
        header = reference.header
        is_trk = True
    elif isinstance(reference, nib.nifti1.Nifti1Header):
        header = reference
        is_nifti = True
    elif isinstance(reference, dict) and "magic_number" in reference:
        header = reference
        is_trk = True
    elif isinstance(reference, dict) and "NB_VERTICES" in reference:
        header = reference
        is_trx = True
    elif isinstance(reference, dipy.io.stateful_tractogram.StatefulTractogram):
        is_sft = True
    elif isinstance(reference, dipy.io.stateful_surface.StatefulSurface):
        is_sft = True

    if is_nifti:
        affine = header.get_best_affine()
        dimensions = header["dim"][1:4]
        voxel_sizes = header["pixdim"][1:4]

        if not affine[0:3, 0:3].any():
            raise ValueError(
                "Invalid affine, contains only zeros."
                "Cannot determine voxel order from transformation"
            )
        voxel_order = "".join(nib.aff2axcodes(affine))
    elif is_trk:
        affine = header["voxel_to_rasmm"]
        dimensions = header["dimensions"]
        voxel_sizes = header["voxel_sizes"]
        voxel_order = header["voxel_order"]
    elif is_sft:
        affine, dimensions, voxel_sizes, voxel_order = reference.space_attributes
    elif is_trx:
        affine = header["VOXEL_TO_RASMM"]
        dimensions = header["DIMENSIONS"]
        voxel_sizes = nib.affines.voxel_sizes(affine)
        voxel_order = "".join(nib.aff2axcodes(affine))
    else:
        raise TypeError("Input reference is not one of the supported format")

    if isinstance(voxel_order, np.bytes_):
        voxel_order = voxel_order.decode("utf-8")

    is_reference_info_valid(affine, dimensions, voxel_sizes, voxel_order)

    return affine.astype(np.float64), dimensions, voxel_sizes, voxel_order


def is_header_compatible(reference_1, reference_2):
    """Compare the spatial attributes of the data in the input files.

    Parameters
    ----------
    reference_1 : Nifti or Trk filename, Nifti1Image or TrkFile,
        Nifti1Header or trk.header (dict)
        Reference that provides the spatial attribute.
    reference_2 : Nifti or Trk filename, Nifti1Image or TrkFile,
        Nifti1Header or trk.header (dict)
        Reference that provides the spatial attribute.

    Returns
    -------
    output : bool
        ``True`` if all spatial attributes match, ``False`` otherwise.
    """

    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = get_reference_info(
        reference_1
    )
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = get_reference_info(
        reference_2
    )

    identical_header = True
    if not np.allclose(affine_1, affine_2, rtol=1e-03, atol=1e-03):
        logger.error("Affine not equal")
        identical_header = False

    if not np.array_equal(dimensions_1, dimensions_2):
        logger.error("Dimensions not equal")
        identical_header = False

    if not np.allclose(voxel_sizes_1, voxel_sizes_2, rtol=1e-03, atol=1e-03):
        logger.error("Voxel_size not equal")
        identical_header = False

    if voxel_order_1 != voxel_order_2:
        logger.error("Voxel_order not equal")
        identical_header = False

    return identical_header


def create_tractogram_header(
    tractogram_type, affine, dimensions, voxel_sizes, voxel_order
):
    """Write a standard trk/tck header from spatial attribute"""
    if isinstance(tractogram_type, str):
        tractogram_type = detect_format(tractogram_type)

    new_header = tractogram_type.create_empty_header()
    new_header[nib.streamlines.Field.VOXEL_SIZES] = tuple(voxel_sizes)
    new_header[nib.streamlines.Field.DIMENSIONS] = tuple(dimensions)
    new_header[nib.streamlines.Field.VOXEL_TO_RASMM] = affine
    new_header[nib.streamlines.Field.VOXEL_ORDER] = voxel_order

    return new_header


def create_nifti_header(affine, dimensions, voxel_sizes):
    """Write a standard nifti header from spatial attribute"""
    new_header = nib.Nifti1Header()
    new_header.set_sform(affine)
    new_header["dim"][1:4] = dimensions
    new_header["pixdim"][1:4] = voxel_sizes

    new_header.affine = new_header.get_best_affine()

    return new_header


@warning_for_keywords()
def save_buan_profiles_hdf5(fname, dt, *, key=None):
    """Saves the given input dataframe to .h5 file

    Parameters
    ----------
    fname : string or Path
        file name for saving the hdf5 file
    dt : Pandas DataFrame
        DataFrame to be saved as .h5 file
    key : str, optional
        Key to retrieve the contents in the HDF5 file. The file rootname will
        be used if not provided.

    """

    df = pd.DataFrame(dt)
    filename_hdf5 = Path(fname).with_suffix(".h5")

    if key is None:
        key, _ = split_filename_extension(fname)

    store = pd.HDFStore(filename_hdf5, complevel=9)
    store.append(key, df, data_columns=True, complevel=9)
    store.close()


@warning_for_keywords()
def read_img_arr_or_path(data, *, affine=None):
    """
    Helper function that handles inputs that can be paths, nifti img or arrays

    Parameters
    ----------
    data : array or nib.Nifti1Image, str or Path.
        Either as a 3D/4D array or as a nifti image object, or as
        a string containing the full path to a nifti file.

    affine : 4x4 array, optional.
        Must be provided for `data` provided as an array. If provided together
        with Nifti1Image or str `data`, this input will over-ride the affine
        that is stored in the `data` input. Default: use the affine stored
        in `data`.

    Returns
    -------
    data, affine : ndarray and 4x4 array
    """
    if isinstance(data, np.ndarray) and affine is None:
        raise ValueError(
            "If data is provided as an array, an affine has ", "to be provided as well"
        )
    if isinstance(data, (str, Path)):
        data = nib.load(data)
    if isinstance(data, nib.Nifti1Image):
        if affine is None:
            affine = data.affine
        data = data.get_fdata()
    return data, affine


def recursive_compare(d1, d2, level="root"):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            s1 = set(d1.keys())
            s2 = set(d2.keys())
            common_keys = s1 & s2
            if s1 - s2:
                raise ValueError(f"Keys {s1 - s2} in d1 but not in d2")
        else:
            common_keys = set(d1.keys())

        for k in common_keys:
            recursive_compare(d1[k], d2[k], level=f"{level}.{k}")

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            raise ValueError(f"Lists do not have the same length at level {level}")
        common_len = min(len(d1), len(d2))

        for i in range(common_len):
            recursive_compare(d1[i], d2[i], level=f"{level}[{i}]")

    else:
        if np.dtype(d1).itemsize != np.dtype(d2).itemsize:
            raise ValueError(f"Values {d1}, {d2} do not match at level {level}")


def split_filename_extension(filename):
    """Split  the filename and its extension(s).

    In our field filename can have period in it (e.g. smoothwm.L.surf.gii)
    At the moment only one double extension is supported (.nii.gz, .gii.gz)

    Parameters
    ----------
    filename : str or Path
        The input filename.

    Returns
    -------
    name : str
        The filename without its extension(s).
    extension : str
        The extension(s) of the filename, including the dot(s).
    """
    filename_str = str(filename).lower()
    if (
        filename_str.count(".gii") >= 2
        or filename_str.count(".nii") >= 2
        or filename_str.count(".gz") >= 2
    ):
        logger.warning(
            "Filename contains more than two instances of .gii, .nii, or .gz."
            " This may be risky or bad practice."
        )

    filename = Path(filename)

    extensions = filename.suffixes
    if len(extensions) > 1 and extensions[-1] == ".gz":
        name = filename.with_suffix("").with_suffix("").name
        extension = "".join(extensions[-2:])  # e.g., .nii.gz
    elif len(extensions) >= 1:
        name = filename.with_suffix("").name
        extension = "".join(extensions[-1])
    else:
        name = filename.name
        extension = "".join(extensions)

    return str(name), extension
