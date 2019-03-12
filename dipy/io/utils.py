""" Utility functions for file formats """
from __future__ import division, print_function, absolute_import

import numpy as np
from nibabel import Nifti1Image


def nifti1_symmat(image_data, *args, **kwargs):
    """Returns a Nifti1Image with a symmetric matrix intent

    Parameters:
    -----------
    image_data : array-like
        should have lower triangular elements of a symmetric matrix along the
        last dimension
    all other arguments and keywords are passed to Nifti1Image

    Returns:
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
