""" Utilities to manipulate numpy arrays """

import sys
from distutils.version import LooseVersion

import numpy as np

from nibabel.volumeutils import endian_codes, native_code, swapped_code


NUMPY_LESS_1_8 = LooseVersion(np.version.short_version) < '1.8'


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


def pinv(a, rcond=1e-15):
    """Vectorized version of numpy.linalg.pinv"""

    a = np.asarray(a)
    if NUMPY_LESS_1_8:
        # numpy 1.8.0 introduced a vectorized version of np.linalg.svd,
        # if using older numpy fall back to using non vectorized np.linalg.pinv
        shape = a.shape[:-2]
        a = a.reshape(-1, a.shape[-2], a.shape[-1])
        result = np.empty((a.shape[0], a.shape[2], a.shape[1]))
        for i, item in enumerate(a):
            result[i] = np.linalg.pinv(item, rcond)
        return result.reshape(shape + (a.shape[2], a.shape[1]))
    else:
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


def eigh(a, UPLO='L'):
    """Iterate over eigh if it doesn't support vectorized operation"""
    a = np.asarray(a)
    if a.ndim > 2 and NUMPY_LESS_1_8:
        # numpy 1.8.0 introduced a vectorized version of np.linalg.eigh,
        # if using older numpy fall back to iterating over np.linalg.eigh
        shape = a.shape[:-2]
        a = a.reshape(-1, a.shape[-2], a.shape[-1])
        evals = np.empty((a.shape[0], a.shape[1]))
        evecs = np.empty((a.shape[0], a.shape[1], a.shape[1]))
        for i, item in enumerate(a):
            evals[i], evecs[i] = np.linalg.eigh(item, UPLO)
        return (evals.reshape(shape + (a.shape[1], )),
                evecs.reshape(shape + (a.shape[1], a.shape[1])))
    return np.linalg.eigh(a, UPLO)
