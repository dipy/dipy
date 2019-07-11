cimport cython
cimport numpy as np

import numpy as np
import time

from libc.math cimport floor

from dipy.align.fused_types cimport floating, number

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


def interpolate_vector_2d(floating[:, :, :] field, double[:, :] locations):
    r"""Bilinear interpolation of a 2D vector field

    Interpolates the 2D vector field at the given locations. This function is
    a wrapper for _interpolate_vector_2d for testing purposes, it is
    equivalent to using scipy.ndimage.interpolation.map_coordinates with
    bilinear interpolation at each vector component

    Parameters
    ----------
    field : array, shape (S, R, 2)
        the 2D vector field to be interpolated
    locations : array, shape (n, 2)
        (locations[i,0], locations[i,1]), 0<=i<n must contain the row and
        column coordinates to interpolate the vector field at

    Returns
    -------
    out : array, shape (n, 2)
        out[i,:], 0<=i<n will be the interpolated vector at coordinates
        locations[i,:], or (0,0) if locations[i,:] is outside the field
    inside : array, (n,)
        if (locations[i,0], locations[i,1]) is inside the vector field
        then inside[i]=1, else inside[i]=0
    """
    ftype = np.asarray(field).dtype
    cdef:
        cnp.npy_intp i, n = locations.shape[0]
        floating[:, :] out = np.zeros(shape=(n, 2), dtype=ftype)
        int[:] inside = np.empty(shape=(n,), dtype=np.int32)
    with nogil:
        for i in range(n):
            inside[i] = _interpolate_vector_2d[floating](field,
                locations[i, 0], locations[i, 1], &out[i, 0])
    return np.asarray(out), np.asarray(inside)


cdef inline int _interpolate_vector_2d(floating[:, :, :] field, double dii,
                                       double djj, floating *out) nogil:
    r"""Bilinear interpolation of a 2D vector field

    Interpolates the 2D displacement field at (dii, djj) and stores the
    result in out. If (dkk, dii, djj) is outside the vector field's domain, a
    zero vector is written to out instead.

    Parameters
    ----------
    field : array, shape (R, C)
        the input 2D displacement field
    dii : floating
        the first coordinate of the interpolating position
    djj : floating
        the second coordinate of the interpolating position
    out : array, shape (2,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dii, djj) is inside the domain of the displacement field,
        inside == 1, otherwise inside == 0
    """
    cdef:
        cnp.npy_intp nr = field.shape[0]
        cnp.npy_intp nc = field.shape[1]
        cnp.npy_intp ii, jj
        double alpha, beta, calpha, cbeta
        int inside
    if((dii <= -1) or (djj <= -1) or (dii >= nr) or (djj >= nc)):
        out[0] = 0
        out[1] = 0
        return 0
    # ---top-left
    ii = <int>floor(dii)
    jj = <int>floor(djj)

    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta

    inside = 0
    if (ii >= 0) and (jj >= 0):
        out[0] = alpha * beta * field[ii, jj, 0]
        out[1] = alpha * beta * field[ii, jj, 1]
        inside += 1
    else:
        out[0] = 0
        out[1] = 0
    # ---top-right
    jj += 1
    if (jj < nc) and (ii >= 0):
        out[0] += alpha * cbeta * field[ii, jj, 0]
        out[1] += alpha * cbeta * field[ii, jj, 1]
        inside += 1
    # ---bottom-right
    ii += 1
    if (jj < nc) and (ii < nr):
        out[0] += calpha * cbeta * field[ii, jj, 0]
        out[1] += calpha * cbeta * field[ii, jj, 1]
        inside += 1
    # ---bottom-left
    jj -= 1
    if (jj >= 0) and (ii < nr):
        out[0] += calpha * beta * field[ii, jj, 0]
        out[1] += calpha * beta * field[ii, jj, 1]
        inside += 1
    return 1 if inside == 4 else 0


def interpolate_scalar_2d(floating[:, :] image, double[:, :] locations):
    r"""Bilinear interpolation of a 2D scalar image

    Interpolates the 2D image at the given locations. This function is
    a wrapper for _interpolate_scalar_2d for testing purposes, it is
    equivalent to scipy.ndimage.interpolation.map_coordinates with
    bilinear interpolation

    Parameters
    ----------
    field : array, shape (S, R)
        the 2D image to be interpolated
    locations : array, shape (n, 2)
        (locations[i,0], locations[i,1]), 0<=i<n must contain the row and
        column coordinates to interpolate the image at

    Returns
    -------
    out : array, shape (n,)
        out[i], 0<=i<n will be the interpolated scalar at coordinates
        locations[i,:], or 0 if locations[i,:] is outside the image
    inside : array, (n,)
        if locations[i:] is inside the image then inside[i]=1, else
        inside[i]=0
    """
    ftype = np.asarray(image).dtype
    cdef:
        cnp.npy_intp i, n = locations.shape[0]
        floating[:] out = np.zeros(shape=(n,), dtype=ftype)
        int[:] inside = np.empty(shape=(n,), dtype=np.int32)
    with nogil:
        for i in range(n):
            inside[i] = _interpolate_scalar_2d[floating](image,
                locations[i, 0], locations[i, 1], &out[i])
    return np.asarray(out), np.asarray(inside)


cdef inline int _interpolate_scalar_2d(floating[:, :] image, double dii,
                                       double djj, floating *out) nogil:
    r"""Bilinear interpolation of a 2D scalar image

    Interpolates the 2D image at (dii, djj) and stores the
    result in out. If (dii, djj) is outside the image's domain,
    zero is written to out instead.

    Parameters
    ----------
    image : array, shape (R, C)
        the input 2D image
    dii : floating
        the first coordinate of the interpolating position
    djj : floating
        the second coordinate of the interpolating position
    out : array, shape (2,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dii, djj) is inside the domain of the image,
        inside == 1, otherwise inside == 0
    """
    cdef:
        cnp.npy_intp nr = image.shape[0]
        cnp.npy_intp nc = image.shape[1]
        cnp.npy_intp ii, jj
        int inside
        double alpha, beta, calpha, cbeta
    if((dii <= -1) or (djj <= -1) or (dii >= nr) or (djj >= nc)):
        out[0] = 0
        return 0
    # ---top-left
    ii = <int>floor(dii)
    jj = <int>floor(djj)

    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta

    inside = 0
    if (ii >= 0) and (jj >= 0):
        out[0] = alpha * beta * image[ii, jj]
        inside += 1
    else:
        out[0] = 0
    # ---top-right
    jj += 1
    if (jj < nc) and (ii >= 0):
        out[0] += alpha * cbeta * image[ii, jj]
        inside += 1
    # ---bottom-right
    ii += 1
    if (jj < nc) and (ii < nr):
        out[0] += calpha * cbeta * image[ii, jj]
        inside += 1
    # ---bottom-left
    jj -= 1
    if (jj >= 0) and (ii < nr):
        out[0] += calpha * beta * image[ii, jj]
        inside += 1
    return 1 if inside == 4 else 0


def interpolate_scalar_nn_2d(number[:, :] image, double[:, :] locations):
    r"""Nearest neighbor interpolation of a 2D scalar image

    Interpolates the 2D image at the given locations. This function is
    a wrapper for _interpolate_scalar_nn_2d for testing purposes, it is
    equivalent to scipy.ndimage.interpolation.map_coordinates with
    nearest neighbor interpolation

    Parameters
    ----------
    image : array, shape (S, R)
        the 2D image to be interpolated
    locations : array, shape (n, 2)
        (locations[i,0], locations[i,1]), 0<=i<n must contain the row and
        column coordinates to interpolate the image at

    Returns
    -------
    out : array, shape (n,)
        out[i], 0<=i<n will be the interpolated scalar at coordinates
        locations[i,:], or 0 if locations[i,:] is outside the image
    inside : array, (n,)
        if locations[i:] is inside the image then inside[i]=1, else
        inside[i]=0
    """
    ftype = np.asarray(image).dtype
    cdef:
        cnp.npy_intp i, n = locations.shape[0]
        number[:] out = np.zeros(shape=(n,), dtype=ftype)
        int[:] inside = np.empty(shape=(n,), dtype=np.int32)
    with nogil:
        for i in range(n):
            inside[i] = _interpolate_scalar_nn_2d[number](image,
                locations[i, 0], locations[i, 1], &out[i])
    return np.asarray(out), np.asarray(inside)


cdef inline int _interpolate_scalar_nn_2d(number[:, :] image, double dii,
                                          double djj, number *out) nogil:
    r"""Nearest-neighbor interpolation of a 2D scalar image

    Interpolates the 2D image at (dii, djj) using nearest neighbor
    interpolation and stores the result in out. If (dii, djj) is outside the
    image's domain, zero is written to out instead.

    Parameters
    ----------
    image : array, shape (R, C)
        the input 2D image
    dii : float
        the first coordinate of the interpolating position
    djj : float
        the second coordinate of the interpolating position
    out : array, shape (1,)
        the variable which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dii, djj) is inside the domain of the image,
        inside == 1, otherwise inside == 0
    """
    cdef:
        cnp.npy_intp nr = image.shape[0]
        cnp.npy_intp nc = image.shape[1]
        cnp.npy_intp ii, jj
        double alpha, beta, calpha, cbeta
    if((dii < 0) or (djj < 0) or (dii > nr - 1) or (djj > nc - 1)):
        out[0] = 0
        return 0
    # find the top left index and the interpolation coefficients
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    # no one is affected
    if((ii < 0) or (jj < 0) or (ii >= nr) or (jj >= nc)):
        out[0] = 0
        return 0
    calpha = dii - ii  # by definition these factors are nonnegative
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    if(alpha < calpha):
        ii += 1
    if(beta < cbeta):
        jj += 1
    # no one is affected
    if((ii < 0) or (jj < 0) or (ii >= nr) or (jj >= nc)):
        out[0] = 0
        return 0
    out[0] = image[ii, jj]
    return 1


def interpolate_scalar_nn_3d(number[:, :, :] image, double[:, :] locations):
    r"""Nearest neighbor interpolation of a 3D scalar image

    Interpolates the 3D image at the given locations. This function is
    a wrapper for _interpolate_scalar_nn_3d for testing purposes, it is
    equivalent to scipy.ndimage.interpolation.map_coordinates with
    nearest neighbor interpolation

    Parameters
    ----------
    image : array, shape (S, R, C)
        the 3D image to be interpolated
    locations : array, shape (n, 3)
        (locations[i,0], locations[i,1], locations[i,2), 0<=i<n must contain
        the coordinates to interpolate the image at

    Returns
    -------
    out : array, shape (n,)
        out[i], 0<=i<n will be the interpolated scalar at coordinates
        locations[i,:], or 0 if locations[i,:] is outside the image
    inside : array, (n,)
        if locations[i,:] is inside the image then inside[i]=1, else
        inside[i]=0
    """
    ftype = np.asarray(image).dtype
    cdef:
        cnp.npy_intp i, n = locations.shape[0]
        number[:] out = np.zeros(shape=(n,), dtype=ftype)
        int[:] inside = np.empty(shape=(n,), dtype=np.int32)
    with nogil:
        for i in range(n):
            inside[i] = _interpolate_scalar_nn_3d[number](image,
                locations[i, 0], locations[i, 1], locations[i, 2], &out[i])
    return np.asarray(out), np.asarray(inside)


cdef inline int _interpolate_scalar_nn_3d(number[:, :, :] volume, double dkk,
                                         double dii, double djj,
                                         number *out) nogil:
    r"""Nearest-neighbor interpolation of a 3D scalar image

    Interpolates the 3D image at (dkk, dii, djj) using nearest neighbor
    interpolation and stores the result in out. If (dkk, dii, djj) is outside
    the image's domain, zero is written to out instead.

    Parameters
    ----------
    image : array, shape (S, R, C)
        the input 2D image
    dkk : float
        the first coordinate of the interpolating position
    dii : float
        the second coordinate of the interpolating position
    djj : float
        the third coordinate of the interpolating position
    out : array, shape (1,)
        the variable which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dkk, dii, djj) is inside the domain of the image,
        inside == 1, otherwise inside == 0
    """
    cdef:
        cnp.npy_intp ns = volume.shape[0]
        cnp.npy_intp nr = volume.shape[1]
        cnp.npy_intp nc = volume.shape[2]
        cnp.npy_intp kk, ii, jj
        double alpha, beta, calpha, cbeta, gamma, cgamma
    if not (0 <= dkk <= ns - 1 and 0 <= dii <= nr - 1 and 0 <= djj <= nc - 1):
        out[0] = 0
        return 0
    # find the top left index and the interpolation coefficients
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    # no one is affected
    if not ((0 <= kk < ns) and (0 <= ii < nr) and (0 <= jj < nc)):
        out[0] = 0
        return 0
    cgamma = dkk - kk
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma
    if(gamma < cgamma):
        kk += 1
    if(alpha < calpha):
        ii += 1
    if(beta < cbeta):
        jj += 1
    # no one is affected
    if not ((0 <= kk < ns) and (0 <= ii < nr) and (0 <= jj < nc)):
        out[0] = 0
        return 0
    out[0] = volume[kk, ii, jj]
    return 1


def interpolate_scalar_3d(floating[:, :, :] image, locations):
    r"""Trilinear interpolation of a 3D scalar image

    Interpolates the 3D image at the given locations. This function is
    a wrapper for _interpolate_scalar_3d for testing purposes, it is
    equivalent to scipy.ndimage.interpolation.map_coordinates with
    trilinear interpolation

    Parameters
    ----------
    field : array, shape (S, R, C)
        the 3D image to be interpolated
    locations : array, shape (n, 3)
        (locations[i,0], locations[i,1], locations[i,2), 0<=i<n must contain
        the coordinates to interpolate the image at

    Returns
    -------
    out : array, shape (n,)
        out[i], 0<=i<n will be the interpolated scalar at coordinates
        locations[i,:], or 0 if locations[i,:] is outside the image
    inside : array, (n,)
        if locations[i,:] is inside the image then inside[i]=1, else
        inside[i]=0
    """
    ftype = np.asarray(image).dtype
    cdef:
        cnp.npy_intp i, n = locations.shape[0]
        floating[:] out = np.zeros(shape=(n,), dtype=ftype)
        int[:] inside = np.empty(shape=(n,), dtype=np.int32)
        double[:,:] _locations = np.array(locations, dtype=np.float64)
    with nogil:
        for i in range(n):
            inside[i] = _interpolate_scalar_3d[floating](image,
                _locations[i, 0], _locations[i, 1], _locations[i, 2], &out[i])
    return np.asarray(out), np.asarray(inside)


cdef inline int _interpolate_scalar_3d(floating[:, :, :] volume,
                                       double dkk, double dii, double djj,
                                       floating *out) nogil:
    r"""Trilinear interpolation of a 3D scalar image

    Interpolates the 3D image at (dkk, dii, djj) and stores the
    result in out. If (dkk, dii, djj) is outside the image's domain,
    zero is written to out instead.

    Parameters
    ----------
    image : array, shape (R, C)
        the input 2D image
    dkk : floating
        the first coordinate of the interpolating position
    dii : floating
        the second coordinate of the interpolating position
    djj : floating
        the third coordinate of the interpolating position
    out : array, shape (2,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dkk, dii, djj) is inside the domain of the image,
        inside == 1, otherwise inside == 0
    """
    cdef:
        cnp.npy_intp ns = volume.shape[0]
        cnp.npy_intp nr = volume.shape[1]
        cnp.npy_intp nc = volume.shape[2]
        cnp.npy_intp kk, ii, jj
        int inside
        double alpha, beta, calpha, cbeta, gamma, cgamma
    if not (-1 < dkk < ns and -1 < dii < nr and -1 < djj < nc):
        out[0] = 0
        return 0
    # find the top left index and the interpolation coefficients
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)
    # no one is affected

    cgamma = dkk - kk
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma

    inside = 0
    # ---top-left
    if (ii >= 0) and (jj >= 0) and (kk >= 0):
        out[0] = alpha * beta * gamma * volume[kk, ii, jj]
        inside += 1
    else:
        out[0] = 0
    # ---top-right
    jj += 1
    if (ii >= 0) and (jj < nc) and (kk >= 0):
        out[0] += alpha * cbeta * gamma * volume[kk, ii, jj]
        inside += 1
    # ---bottom-right
    ii += 1
    if (ii < nr) and (jj < nc) and (kk >= 0):
        out[0] += calpha * cbeta * gamma * volume[kk, ii, jj]
        inside += 1
    # ---bottom-left
    jj -= 1
    if (ii < nr) and (jj >= 0) and (kk >= 0):
        out[0] += calpha * beta * gamma * volume[kk, ii, jj]
        inside += 1
    kk += 1
    if(kk < ns):
        ii -= 1
        if (ii >= 0) and (jj >= 0):
            out[0] += alpha * beta * cgamma * volume[kk, ii, jj]
            inside += 1
        jj += 1
        if (ii >= 0) and (jj < nc):
            out[0] += alpha * cbeta * cgamma * volume[kk, ii, jj]
            inside += 1
        # ---bottom-right
        ii += 1
        if (ii < nr) and (jj < nc):
            out[0] += calpha * cbeta * cgamma * volume[kk, ii, jj]
            inside += 1
        # ---bottom-left
        jj -= 1
        if (ii < nr) and (jj >= 0):
            out[0] += calpha * beta * cgamma * volume[kk, ii, jj]
            inside += 1
    return 1 if inside == 8 else 0


def interpolate_vector_3d(floating[:, :, :, :] field, double[:, :] locations):
    r"""Trilinear interpolation of a 3D vector field

    Interpolates the 3D vector field at the given locations. This function is
    a wrapper for _interpolate_vector_3d for testing purposes, it is
    equivalent to using scipy.ndimage.interpolation.map_coordinates with
    trilinear interpolation at each vector component

    Parameters
    ----------
    field : array, shape (S, R, C, 3)
        the 3D vector field to be interpolated
    locations : array, shape (n, 3)
        (locations[i,0], locations[i,1], locations[i,2), 0<=i<n must contain
        the coordinates to interpolate the vector field at

    Returns
    -------
    out : array, shape (n, 3)
        out[i,:], 0<=i<n will be the interpolated vector at coordinates
        locations[i,:], or (0,0,0) if locations[i,:] is outside the field
    inside : array, (n,)
        if locations[i,:] is inside the vector field then inside[i]=1, else
        inside[i]=0
    """
    ftype = np.asarray(field).dtype
    cdef:
        cnp.npy_intp i, n = locations.shape[0]
        floating[:, :] out = np.zeros(shape=(n, 3), dtype=ftype)
        int[:] inside = np.empty(shape=(n,), dtype=np.int32)
    with nogil:
        for i in range(n):
            inside[i] = _interpolate_vector_3d[floating](field,
                locations[i, 0], locations[i, 1], locations[i, 2], &out[i, 0])
    return np.asarray(out), np.asarray(inside)


cdef inline int _interpolate_vector_3d(floating[:, :, :, :] field, double dkk,
                                       double dii, double djj,
                                       floating* out) nogil:
    r"""Trilinear interpolation of a 3D vector field

    Interpolates the 3D displacement field at (dkk, dii, djj) and stores the
    result in out. If (dkk, dii, djj) is outside the vector field's domain, a
    zero vector is written to out instead.

    Parameters
    ----------
    field : array, shape (S, R, C)
        the input 3D displacement field
    dkk : floating
        the first coordinate of the interpolating position
    dii : floating
        the second coordinate of the interpolating position
    djj : floating
        the third coordinate of the interpolating position
    out : array, shape (3,)
        the array which the interpolation result will be written to

    Returns
    -------
    inside : int
        if (dkk, dii, djj) is inside the domain of the displacement field,
        inside == 1, otherwise inside == 0
    """
    cdef:
        cnp.npy_intp ns = field.shape[0]
        cnp.npy_intp nr = field.shape[1]
        cnp.npy_intp nc = field.shape[2]
        cnp.npy_intp kk, ii, jj
        int inside
        double alpha, beta, gamma, calpha, cbeta, cgamma
    if not (-1 < dkk < ns and -1 < dii < nr and -1 < djj < nc):
        out[0] = 0
        out[1] = 0
        out[2] = 0
        return 0
    #---top-left
    kk = <int>floor(dkk)
    ii = <int>floor(dii)
    jj = <int>floor(djj)

    cgamma = dkk - kk
    calpha = dii - ii
    cbeta = djj - jj
    alpha = 1 - calpha
    beta = 1 - cbeta
    gamma = 1 - cgamma

    inside = 0
    if (ii >= 0) and (jj >= 0) and (kk >= 0):
        out[0] = alpha * beta * gamma * field[kk, ii, jj, 0]
        out[1] = alpha * beta * gamma * field[kk, ii, jj, 1]
        out[2] = alpha * beta * gamma * field[kk, ii, jj, 2]
        inside += 1
    else:
        out[0] = 0
        out[1] = 0
        out[2] = 0
    # ---top-right
    jj += 1
    if (jj < nc) and (ii >= 0) and (kk >= 0):
        out[0] += alpha * cbeta * gamma * field[kk, ii, jj, 0]
        out[1] += alpha * cbeta * gamma * field[kk, ii, jj, 1]
        out[2] += alpha * cbeta * gamma * field[kk, ii, jj, 2]
        inside += 1
    # ---bottom-right
    ii += 1
    if (jj < nc) and (ii < nr) and (kk >= 0):
        out[0] += calpha * cbeta * gamma * field[kk, ii, jj, 0]
        out[1] += calpha * cbeta * gamma * field[kk, ii, jj, 1]
        out[2] += calpha * cbeta * gamma * field[kk, ii, jj, 2]
        inside += 1
    # ---bottom-left
    jj -= 1
    if (jj >= 0) and (ii < nr) and (kk >= 0):
        out[0] += calpha * beta * gamma * field[kk, ii, jj, 0]
        out[1] += calpha * beta * gamma * field[kk, ii, jj, 1]
        out[2] += calpha * beta * gamma * field[kk, ii, jj, 2]
        inside += 1
    kk += 1
    if (kk < ns):
        ii -= 1
        if (jj >= 0) and (ii >= 0):
            out[0] += alpha * beta * cgamma * field[kk, ii, jj, 0]
            out[1] += alpha * beta * cgamma * field[kk, ii, jj, 1]
            out[2] += alpha * beta * cgamma * field[kk, ii, jj, 2]
            inside += 1
        jj += 1
        if (jj < nc) and (ii >= 0):
            out[0] += alpha * cbeta * cgamma * field[kk, ii, jj, 0]
            out[1] += alpha * cbeta * cgamma * field[kk, ii, jj, 1]
            out[2] += alpha * cbeta * cgamma * field[kk, ii, jj, 2]
            inside += 1
        # ---bottom-right
        ii += 1
        if (jj < nc) and (ii < nr):
            out[0] += calpha * cbeta * cgamma * field[kk, ii, jj, 0]
            out[1] += calpha * cbeta * cgamma * field[kk, ii, jj, 1]
            out[2] += calpha * cbeta * cgamma * field[kk, ii, jj, 2]
            inside += 1
        # ---bottom-left
        jj -= 1
        if (jj >= 0) and (ii < nr):
            out[0] += calpha * beta * cgamma * field[kk, ii, jj, 0]
            out[1] += calpha * beta * cgamma * field[kk, ii, jj, 1]
            out[2] += calpha * beta * cgamma * field[kk, ii, jj, 2]
            inside += 1
    return 1 if inside == 8 else 0
