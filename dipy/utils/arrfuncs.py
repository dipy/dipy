""" Utilities to manipulate numpy arrays """

import numpy as np
from nibabel.volumeutils import endian_codes, native_code


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
    """Vectorized version of `numpy.linalg.pinv`

    If numpy version is less than 1.8, it falls back to iterating over
    `np.linalg.pinv` since there isn't a vectorized version of `np.linalg.svd`
    available.

    Parameters
    ----------
    a : array_like (..., M, N)
        Matrix to be pseudo-inverted.
    rcond : float
        Cutoff for small singular values.

    Returns
    -------
    B : ndarray (..., N, M)
        The pseudo-inverse of `a`.

    Raises
    ------
    LinAlgError
        If the SVD computation does not converge.

    See Also
    --------
    np.linalg.pinv
    """
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


def eigh(a, UPLO='L'):
    """Iterate over `np.linalg.eigh` if it doesn't support vectorized operation

    Parameters
    ----------
    a : array_like (..., M, M)
        Hermitian/Symmetric matrices whose eigenvalues and
        eigenvectors are to be computed.
    UPLO : {'L', 'U'}, optional
        Specifies whether the calculation is done with the lower triangular
        part of `a` ('L', default) or the upper triangular part ('U').

    Returns
    -------
    w : ndarray (..., M)
        The eigenvalues in ascending order, each repeated according to
        its multiplicity.
    v : ndarray (..., M, M)
        The column ``v[..., :, i]`` is the normalized eigenvector corresponding
        to the eigenvalue ``w[..., i]``.

    Raises
    ------
    LinAlgError
        If the eigenvalue computation does not converge.

    See Also
    --------
    np.linalg.eigh
    """
    a = np.asarray(a)
    return np.linalg.eigh(a, UPLO)
