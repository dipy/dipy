''' Utility functions for file formats '''
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
