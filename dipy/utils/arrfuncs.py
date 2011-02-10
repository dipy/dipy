""" Utilities to manipulate numpy arrays """

import sys

import numpy as np

from nibabel.volumeutils import endian_codes, native_code, swapped_code


def as_native_array(arr):
    """ Return `arr` as native byteordered array

    If arr is already native byte ordered, return unchanged.  If it is opposite
    endian, then make a native byte ordered copy and return that

    Parameters
    ----------
    arr : ndarray

    Returns
    -------
    native_arr : ndarray
        If `arr` was native order, this is just `arr`. Otherwise it's a new
        array such that ``np.all(native_arr == arr)``, with native byte
        ordering.
    """
    if endian_codes[arr.dtype.byteorder] == native_code:
        return arr
    return arr.byteswap().newbyteorder()

