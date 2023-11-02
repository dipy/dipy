""" Utility functions for file formats """
import logging
import numbers
import os
from dipy.utils.optpkg import optional_package
import dipy
import nibabel as nib
from nibabel.streamlines import detect_format
from nibabel import Nifti1Image
import numpy as np
from trx import trx_file_memmap

pd, have_pd, _ = optional_package("pandas")

if have_pd:
    import pandas as pd


def nifti1_symmat(image_data, *args, **kwargs):
    """Returns a Nifti1Image with a symmetric matrix intent

    Parameters
    ----------
    image_data : array-like
        should have lower triangular elements of a symmetric matrix along the
        last dimension
    all other arguments and keywords are passed to Nifti1Image

    Returns
    -------
    image : Nifti1Image
        5d, extra dimensions added before the last. Has symmetric matrix intent
        code

    """
    image_data = make5d(image_data)
    last_dim = image_data.shape[-1]
    n = (np.sqrt(1+8*last_dim) - 1)/2
    if (n % 1) != 0:
        raise ValueError("input_data does not seem to have matrix elements")

    image = Nifti1Image(image_data, *args, **kwargs)
    hdr = image.header
    hdr.set_intent('symmetric matrix', (n,))
    return image


def make5d(data):
    """reshapes the input to have 5 dimensions, adds extra dimensions just
    before the last dimension
    """
    data = np.asarray(data)
    if data.ndim > 5:
        raise ValueError("input is already more than 5d")
    shape = data.shape
    shape = shape[:-1] + (1,)*(5-len(shape)) + shape[-1:]
    return data.reshape(shape)


def decfa(img_orig, scale=False):
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

    dest_dtype = np.dtype([('R', 'uint8'), ('G', 'uint8'), ('B', 'uint8')])
    out_data = np.zeros(img_orig.shape[:3], dtype=dest_dtype)

    data_orig = np.asanyarray(img_orig.dataobj)

    if scale:
        data_orig = (data_orig * 255).astype('uint8')

    for ii in np.ndindex(img_orig.shape[:3]):
        val = data_orig[ii]
        out_data[ii] = (val[0], val[1], val[2])

    new_hdr = img_orig.header
    new_hdr['dim'][4] = 1
    new_hdr.set_intent(1001, name='Color FA')
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
    out_data = np.zeros(data_orig.shape + (3, ), dtype=np.uint8)

    for ii in np.ndindex(img_orig.shape[:3]):
        val = data_orig[ii]
        out_data[ii] = np.array([val[0], val[1], val[2]])

    new_hdr = img_orig.header
    new_hdr['dim'][4] = 3

    # Remove the original intent
    new_hdr.set_intent(0)
    new_hdr.set_data_dtype(float)

    return Nifti1Image(out_data, affine=img_orig.affine, header=new_hdr)


def is_reference_info_valid(affine, dimensions, voxel_sizes, voxel_order):
    """Validate basic data type and value of spatial attribute.

    Does not ensure that voxel_sizes and voxel_order are self-coherent with
    the affine.
    Only verify the following:
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
        logging.warning('Transformation matrix must be 4x4')

    if not affine[0:3, 0:3].any():
        all_valid = False
        logging.warning('Rotation matrix cannot be all zeros')

    if not len(dimensions) >= 3:
        all_valid = False
        only_3d_warning = True

    for i in dimensions:
        if not isinstance(i, numbers.Integral):
            all_valid = False
            logging.warning('Dimensions must be int.')
        if i <= 0:
            all_valid = False
            logging.warning('Dimensions must be above 0.')

    if not len(voxel_sizes) >= 3:
        all_valid = False
        only_3d_warning = True
    for i in voxel_sizes:
        if not isinstance(i, numbers.Number):
            all_valid = False
            logging.warning('Voxel size must be int/float.')
        if i <= 0:
            all_valid = False
            logging.warning('Voxel size must be above 0.')

    if not len(voxel_order) >= 3:
        all_valid = False
        only_3d_warning = True
    for i in voxel_order:
        if not isinstance(i, str):
            all_valid = False
            logging.warning('Voxel order must be string/char.')
        if i not in ['R', 'A', 'S', 'L', 'P', 'I']:
            all_valid = False
            logging.warning('Voxel order does not follow convention.')

    if only_3d_warning:
        logging.warning('Only 3D (and above) reference are considered valid.')

    return all_valid


def split_name_with_gz(filename):
    """
    Returns the clean basename and extension of a file.
    Means that this correctly manages the ".nii.gz" extensions.

    Parameters
    ----------
    filename: str
        The filename to clean

    Returns
    -------
        base, ext : tuple(str, str)
        Clean basename and the full extension
    """
    base, ext = os.path.splitext(filename)

    if ext.lower() == ".gz":
        # Test if we have a .nii additional extension
        temp_base, add_ext = os.path.splitext(base)

        if add_ext.lower() == ".nii" or add_ext.lower() == ".trk":
            ext = add_ext + ext
            base = temp_base

    return base, ext


def get_reference_info(reference):
    """ Will compare the spatial attribute of 2 references.

    Parameters
    ----------
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict), TrxFile or trx.header (dict)
        Reference that provides the spatial attribute.
    Returns
    -------
    output : tuple
        - affine ndarray (4,4), np.float32, transformation of VOX to RASMM
        - dimensions ndarray (3,), int16, volume shape for each axis
        - voxel_sizes  ndarray (3,), float32, size of voxel for each axis
        - voxel_order, string, Typically 'RAS' or 'LPS'
    """

    is_nifti = False
    is_trk = False
    is_sft = False
    is_trx = False
    if isinstance(reference, str):
        _, ext = split_name_with_gz(reference)
        ext = ext.lower()
        if ext in ['.nii', '.nii.gz']:
            header = nib.load(reference).header
            is_nifti = True
        elif ext == '.trk':
            header = nib.streamlines.load(reference, lazy_load=True).header
            is_trk = True
        elif ext == '.trx':
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
    elif isinstance(reference, dict) and 'magic_number' in reference:
        header = reference
        is_trk = True
    elif isinstance(reference, dict) and 'NB_VERTICES' in reference:
        header = reference
        is_trx = True
    elif isinstance(reference, dipy.io.stateful_tractogram.StatefulTractogram):
        is_sft = True

    if is_nifti:
        affine = header.get_best_affine()
        dimensions = header['dim'][1:4]
        voxel_sizes = header['pixdim'][1:4]

        if not affine[0:3, 0:3].any():
            raise ValueError(
                'Invalid affine, contains only zeros.'
                'Cannot determine voxel order from transformation')
        voxel_order = ''.join(nib.aff2axcodes(affine))
    elif is_trk:
        affine = header['voxel_to_rasmm']
        dimensions = header['dimensions']
        voxel_sizes = header['voxel_sizes']
        voxel_order = header['voxel_order']
    elif is_sft:
        affine, dimensions, voxel_sizes, voxel_order =\
            reference.space_attributes
    elif is_trx:
        affine = header['VOXEL_TO_RASMM']
        dimensions = header['DIMENSIONS']
        voxel_sizes = nib.affines.voxel_sizes(affine)
        voxel_order = ''.join(nib.aff2axcodes(affine))
    else:
        raise TypeError('Input reference is not one of the supported format')

    if isinstance(voxel_order, np.bytes_):
        voxel_order = voxel_order.decode('utf-8')

    is_reference_info_valid(affine, dimensions, voxel_sizes, voxel_order)

    return affine.astype(np.float32), dimensions, voxel_sizes, voxel_order


def is_header_compatible(reference_1, reference_2):
    """ Will compare the spatial attribute of 2 references

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
        Does all the spatial attribute match
    """

    affine_1, dimensions_1, voxel_sizes_1, voxel_order_1 = get_reference_info(
        reference_1)
    affine_2, dimensions_2, voxel_sizes_2, voxel_order_2 = get_reference_info(
        reference_2)

    identical_header = True
    if not np.allclose(affine_1, affine_2, rtol=1e-03, atol=1e-03):
        logging.error('Affine not equal')
        identical_header = False

    if not np.array_equal(dimensions_1, dimensions_2):
        logging.error('Dimensions not equal')
        identical_header = False

    if not np.allclose(voxel_sizes_1, voxel_sizes_2, rtol=1e-03, atol=1e-03):
        logging.error('Voxel_size not equal')
        identical_header = False

    if voxel_order_1 != voxel_order_2:
        logging.error('Voxel_order not equal')
        identical_header = False

    return identical_header


def create_tractogram_header(tractogram_type, affine, dimensions, voxel_sizes,
                             voxel_order):
    """ Write a standard trk/tck header from spatial attribute """
    if isinstance(tractogram_type, str):
        tractogram_type = detect_format(tractogram_type)

    new_header = tractogram_type.create_empty_header()
    new_header[nib.streamlines.Field.VOXEL_SIZES] = tuple(voxel_sizes)
    new_header[nib.streamlines.Field.DIMENSIONS] = tuple(dimensions)
    new_header[nib.streamlines.Field.VOXEL_TO_RASMM] = affine
    new_header[nib.streamlines.Field.VOXEL_ORDER] = voxel_order

    return new_header


def create_nifti_header(affine, dimensions, voxel_sizes):
    """ Write a standard nifti header from spatial attribute """
    new_header = nib.Nifti1Header()
    new_header.set_sform(affine)
    new_header['dim'][1:4] = dimensions
    new_header['pixdim'][1:4] = voxel_sizes

    new_header.affine = new_header.get_best_affine()

    return new_header


def save_buan_profiles_hdf5(fname, dt):
    """ Saves the given input dataframe to .h5 file

    Parameters
    ----------
    fname : string
        file name for saving the hdf5 file
    dt : Pandas DataFrame
        DataFrame to be saved as .h5 file

    """

    df = pd.DataFrame(dt)
    filename_hdf5 = fname + '.h5'

    store = pd.HDFStore(filename_hdf5, complevel=9)
    store.append(fname, df, data_columns=True, complevel=9)
    store.close()


def read_img_arr_or_path(data, affine=None):
    """
    Helper function that handles inputs that can be paths, nifti img or arrays

    Parameters
    ----------
    data : array or nib.Nifti1Image or str.
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
        raise ValueError("If data is provided as an array, an affine has ",
                         "to be provided as well")
    if isinstance(data, str):
        data = nib.load(data)
    if isinstance(data, nib.Nifti1Image):
        if affine is None:
            affine = data.affine
        data = data.get_fdata()
    return data, affine
