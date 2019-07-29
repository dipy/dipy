""" Utility functions for file formats """
import logging
import os

import dipy
import nibabel as nib
from nibabel.streamlines import detect_format
from nibabel import Nifti1Image
import numpy as np


def nifti1_symmat(image_data, *args, **kwargs):
    """Returns a Nifti1Image with a symmetric matrix intent

    Parameters
    -----------
    image_data : array-like
        should have lower triangular elements of a symmetric matrix along the
        last dimension
    all other arguments and keywords are passed to Nifti1Image

    Returns
    --------
    image : Nifti1Image
        5d, extra dimensions addes before the last. Has symmetric matrix intent
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


def make5d(input):
    """reshapes the input to have 5 dimensions, adds extra dimensions just
    before the last dimession
    """
    input = np.asarray(input)
    if input.ndim > 5:
        raise ValueError("input is already more than 5d")
    shape = input.shape
    shape = shape[:-1] + (1,)*(5-len(shape)) + shape[-1:]
    return input.reshape(shape)


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

    data_orig = img_orig.get_data()

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

    data_orig = img_orig.get_data()
    out_data = np.zeros(data_orig.shape + (3, ), dtype=np.uint8)

    for ii in np.ndindex(img_orig.shape[:3]):
        val = data_orig[ii]
        out_data[ii] = np.array([val[0], val[1], val[2]])

    new_hdr = img_orig.header
    new_hdr['dim'][4] = 3

    # Remove the original intent
    new_hdr.set_intent(0)
    new_hdr.set_data_dtype(np.float)

    return Nifti1Image(out_data, affine=img_orig.affine, header=new_hdr)


def get_reference_info(reference):
    """ Will compare the spatial attribute of 2 references

    Parameters
    ----------
    reference : Nifti or Trk filename, Nifti1Image or TrkFile, Nifti1Header or
        trk.header (dict)
        Reference that provides the spatial attribute.

    Returns
    -------
    output : tuple
        - affine ndarray (4,4), np.float32, tranformation of VOX to RASMM
        - dimensions list (3), int, volume shape for each axis
        - voxel_sizes  list (3), float, size of voxel for each axis
        - voxel_order, string, Typically 'RAS' or 'LPS'
    """

    is_nifti = False
    is_trk = False
    is_sft = False
    if isinstance(reference, str):
        try:
            header = nib.load(reference).header
            is_nifti = True
        except nib.filebasedimages.ImageFileError:
            pass
        try:
            header = nib.streamlines.load(reference, lazy_load=True).header
            _, extension = os.path.splitext(reference)
            if extension == '.trk':
                is_trk = True
        except ValueError:
            pass
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
    elif isinstance(reference, dipy.io.stateful_tractogram.StatefulTractogram):
        is_sft = True

    if is_nifti:
        affine = np.eye(4).astype(np.float32)
        affine[0, 0:4] = header['srow_x']
        affine[1, 0:4] = header['srow_y']
        affine[2, 0:4] = header['srow_z']
        dimensions = header['dim'][1:4]
        voxel_sizes = header['pixdim'][1:4]
        voxel_order = ''.join(nib.aff2axcodes(affine))
    elif is_trk:
        affine = header['voxel_to_rasmm']
        dimensions = header['dimensions']
        voxel_sizes = header['voxel_sizes']
        voxel_order = header['voxel_order']
    elif is_sft:
        affine, dimensions, voxel_sizes, voxel_order = reference.space_attribute
    else:
        raise TypeError('Input reference is not one of the supported format')

    if isinstance(voxel_order, np.bytes_):
        voxel_order = voxel_order.decode('utf-8')

    return affine, dimensions, voxel_sizes, voxel_order


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
    if not np.allclose(affine_1, affine_2):
        logging.error('Affine not equal')
        identical_header = False

    if not np.array_equal(dimensions_1, dimensions_2):
        logging.error('Dimensions not equal')
        identical_header = False

    if not np.allclose(voxel_sizes_1, voxel_sizes_2):
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
    new_header['srow_x'] = affine[0, 0:4]
    new_header['srow_y'] = affine[1, 0:4]
    new_header['srow_z'] = affine[2, 0:4]
    new_header['dim'][1:4] = dimensions
    new_header['pixdim'][1:4] = voxel_sizes

    return new_header
