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


def pinv_vec(a, rcond=1e-15):
    """Vectorized version of numpy.linalg.pinv"""

    a = np.asarray(a)
    swap = np.arange(a.ndim)
    swap[[-2, -1]] = swap[[-1, -2]]
    u, s, v = np.linalg.svd(a, full_matrices=False)
    cutoff = np.maximum.reduce(s, axis=-1, keepdims=True) * rcond
    mask = s > cutoff
    s[mask] = 1. / s[mask]
    s[~mask] = 0
    return np.einsum('...ij,...jk',
                     np.transpose(v, swap) * s[..., None, :],
                     np.transpose(u, swap))
