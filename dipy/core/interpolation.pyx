cimport cython
cimport numpy as np

import numpy as np
import time

from libc.math cimport floor

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int trilinear_interpolate4d_c(
        double[:, :, :, :] data,
        double* point,
        double[:] result) nogil:
    """Tri-linear interpolation along the last dimension of a 4d array

    Parameters
    ----------
    point : 1d array (3,)
        3 doubles representing a 3d point in space. If point has integer values
        ``[i, j, k]``, the result will be the same as ``data[i, j, k]``.
    data : 4d array
        Data to be interpolated.
    result : 1d array
        The result of interpolation. Should have length equal to the
        ``data.shape[3]``.
    Returns
    -------
    err : int
         0 : successful interpolation.
        -1 : point is outside the data area, meaning round(point) is not a
             valid index to data.
        -2 : point has the wrong shape
        -3 : shape of data and result do not match

    """
    cdef:
        np.npy_intp flr, N
        double w, rem
        np.npy_intp index[3][2]
        double weight[3][2]

    if data.shape[3] != result.shape[0]:
        return -3

    for i in range(3):
        if point[i] < -.5 or point[i] >= (data.shape[i] - .5):
            return -1

        flr = <np.npy_intp> floor(point[i])
        rem = point[i] - flr

        index[i][0] = flr + (flr == -1)
        index[i][1] = flr + (flr != (data.shape[i] - 1))
        weight[i][0] = 1 - rem
        weight[i][1] = rem

    N = result.shape[0]
    for i in range(N):
        result[i] = 0

    for i in range(2):
        for j in range(2):
            for k in range(2):
                w = weight[0][i] * weight[1][j] * weight[2][k]
                for L in range(N):
                    result[L] += w * data[index[0][i], index[1][j],
                                          index[2][k], L]
    return 0


cpdef trilinear_interpolate4d(double[:, :, :, :] data, double[:] point,
                              np.ndarray out=None):
    """Tri-linear interpolation along the last dimension of a 4d array

    Parameters
    ----------
    point : 1d array (3,)
        3 doubles representing a 3d point in space. If point has integer values
        ``[i, j, k]``, the result will be the same as ``data[i, j, k]``.
    data : 4d array
        Data to be interpolated.
    out : 1d array, optional
        The output array for the result of the interpolation.

    Returns
    -------
    out : 1d array
        The result of interpolation.

    """
    cdef:
        int err
        double[::1] outview

    if out is None:
        out = np.empty(data.shape[3])
    outview = out

    err = trilinear_interpolate4d_c(data, &point[0], out)

    if err == 0:
        return out
    elif err == -1:
        raise IndexError("The point point is outside data")
    elif err == -2:
        raise ValueError("Point must be a 1d array with shape (3,).")
    elif err == -3:
        # This should only happen if the user passes an bad out array
        msg = "out array must have same size as the last dimension of data."
        raise ValueError(msg)


def nearestneighbor_interpolate(data, point):
    index = tuple(np.round(point).astype(np.int))
    return data[index]

