import numpy as np
cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def vec_val_vect(vecs, vals):
    """ Vectorize `vecs` . diag(`vals`) . `vecs`.T over N-2 dimensions of vecs

    Parameters
    ----------
    vecs : shape (..., P, P) array
        containing tensor in last two dimensions, P usually equal to 3
    vals : shape (..., P) array
        diagonal values carried in last dimension, ``...`` shape above must
        match that for `vecs`

    Returns
    -------
    res : shape (..., P, P) array
        For all the dimensions ellided by ``...`` loops to get (P, P) ``vec``
        matrix, and (P,) ``vals`` vector, and calculates
        ``vec.dot(np.diag(val).dot(vec.T)``.

    Raises
    ------
    ValueError : non-matching ``...`` dimensions of `vecs`, `vals`
    ValueError : non-matching ``P`` dimensions of `vecs`, `vals`

    Examples
    --------
    Make a 3D array where the first dimension is only 1

    >>> vecs = np.arange(9).reshape((1, 3, 3))
    >>> vals = np.arange(3).reshape((1, 3))
    >>> vec_val_vect(vecs, vals)
    array([[  9,  24,  39],
           [ 24,  66, 108],
           [ 39, 108, 177]])

    That's the same as the 2D case

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
        cnp.npy_intp t, N, rows, cols, r, c, out_c
        double [:, :, :] vecr
        double [:, :] valr
        double [:, :] vec
        double [:, :] out_vec
        double [:] val
        double [:, :, :] out
        double row_c
    common_shape = vecs.shape[:-2]
    rows, cols = vecs.shape[-2], vecs.shape[-1]
    if vals.shape != common_shape + (cols,):
        raise ValueError('dimensions do not match')
    if cols != rows:
        raise ValueError('Must have same number of rows, cols')
    N = np.prod(common_shape)
    vecr = np.array(vecs.reshape((N, rows, cols)), dtype=float)
    valr = np.array(vals.reshape((N, cols)), dtype=float)
    out = np.zeros((N, rows, cols))
    with nogil:
        for t in range(N): # loop over tensors
            vec = vecr[t]
            val = valr[t]
            out_vec = out[t]
            for r in range(rows):
                for c in range(cols):
                    row_c = vec[r, c] * val[c]
                    for out_c in range(cols):
                        out_vec[r, out_c] += row_c * vec[out_c, c]
    return np.reshape(out, (common_shape + (rows, cols)))
