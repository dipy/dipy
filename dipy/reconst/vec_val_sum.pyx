import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def vec_val_vect(vecs, vals):
    """ Vectorize `vecs`.diag(`vals`).`vecs`.T for last 2 dimensions of `vecs`

    Parameters
    ----------
    vecs : shape (..., M, N) array
        containing tensor in last two dimensions; M, N usually equal to (3, 3)
    vals : shape (..., N) array
        diagonal values carried in last dimension, ``...`` shape above must
        match that for `vecs`

    Returns
    -------
    res : shape (..., M, M) array
        For all the dimensions ellided by ``...``, loops to get (M, N) ``vec``
        matrix, and (N,) ``vals`` vector, and calculates
        ``vec.dot(np.diag(val).dot(vec.T)``.

    Raises
    ------
    ValueError : non-matching ``...`` dimensions of `vecs`, `vals`
    ValueError : non-matching ``N`` dimensions of `vecs`, `vals`

    Examples
    --------
    Make a 3D array where the first dimension is only 1

    >>> vecs = np.arange(9).reshape((1, 3, 3))
    >>> vals = np.arange(3).reshape((1, 3))
    >>> vec_val_vect(vecs, vals)
    array([[[   9.,   24.,   39.],
            [  24.,   66.,  108.],
            [  39.,  108.,  177.]]])

    That's the same as the 2D case (apart from the float casting):

    >>> vecs = np.arange(9).reshape((3, 3))
    >>> vals = np.arange(3)
    >>> np.dot(vecs, np.dot(np.diag(vals), vecs.T))
    array([[  9,  24,  39],
           [ 24,  66, 108],
           [ 39, 108, 177]])
    """
    vecs = np.asarray(vecs)
    vals = np.asarray(vals)
    cdef:
        cnp.npy_intp t, N, ndim, rows, cols, r, c, in_r_out_c
        double [:, :, :] vecr
        double [:, :] valr
        double [:, :] vec
        double [:, :] out_vec
        double [:] val
        double [:, :, :] out
        double row_c
    # Avoid negative indexing to avoid errors with False boundscheck decorator
    # and Cython > 0.18
    ndim = vecs.ndim
    common_shape = vecs.shape[:(ndim-2)]
    rows, cols = vecs.shape[ndim-2], vecs.shape[ndim-1]
    if vals.shape != common_shape + (cols,):
        raise ValueError('dimensions do not match')
    N = np.prod(common_shape)
    vecr = np.array(vecs.reshape((N, rows, cols)), dtype=float)
    valr = np.array(vals.reshape((N, cols)), dtype=float)
    out = np.zeros((N, rows, rows))
    with nogil:
        for t in range(N): # loop over the early dimensions
            vec = vecr[t]
            val = valr[t]
            out_vec = out[t]
            for r in range(rows):
                for c in range(cols):
                    row_c = vec[r, c] * val[c]
                    for in_r_out_c in range(rows):
                        out_vec[r, in_r_out_c] += row_c * vec[in_r_out_c, c]
    return np.reshape(out, (common_shape + (rows, rows)))
