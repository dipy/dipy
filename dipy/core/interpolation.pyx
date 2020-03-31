# cython: boundscheck=False, wraparound=False, cdivision=True

cimport cython
cimport numpy as np
cimport numpy as cnp

import numpy as np
import time
import warnings

from libc.math cimport floor

from dipy.align.fused_types cimport floating, number


def interp_rbf(data, sphere_origin, sphere_target,
               function='multiquadric', epsilon=None, smooth=0.1,
               norm="angle"):
    """Interpolate data on the sphere, using radial basis functions.

    Parameters
    ----------
    data : (N,) ndarray
        Function values on the unit sphere.
    sphere_origin : Sphere
        Positions of data values.
    sphere_target : Sphere
        M target positions for which to interpolate.

    function : {'multiquadric', 'inverse', 'gaussian'}
        Radial basis function.
    epsilon : float
        Radial basis function spread parameter. Defaults to approximate average
        distance between nodes.
    a good start
    smooth : float
        values greater than zero increase the smoothness of the
        approximation with 0 as pure interpolation. Default: 0.1
    norm : str
        A string indicating the function that returns the
        "distance" between two points.
        'angle' - The angle between two vectors
        'euclidean_norm' - The Euclidean distance

    Returns
    -------
    v : (M,) ndarray
        Interpolated values.

    See Also
    --------
    scipy.interpolate.Rbf

    """
    from scipy.interpolate import Rbf

    def angle(x1, x2):
        xx = np.arccos(np.clip((x1 * x2).sum(axis=0), -1, 1))
        return np.nan_to_num(xx)

    def euclidean_norm(x1, x2):
        return np.sqrt(((x1 - x2)**2).sum(axis=0))

    if norm == "angle":
        norm = angle
    elif norm == "euclidean_norm":
        w_s = "The Eucldian norm used for interpolation is inaccurate "
        w_s += "and will be deprecated in future versions. Please consider "
        w_s += "using the 'angle' norm instead"
        warnings.warn(w_s, PendingDeprecationWarning)
        norm = euclidean_norm

    # Workaround for bug in older versions of SciPy that don't allow
    # specification of epsilon None:
    if epsilon is not None:
        kwargs = {'function': function,
                  'epsilon': epsilon,
                  'smooth': smooth,
                  'norm': norm}
    else:
        kwargs = {'function': function,
                  'smooth': smooth,
                  'norm': norm}

    rbfi = Rbf(sphere_origin.x, sphere_origin.y, sphere_origin.z, data,
               **kwargs)
    return rbfi(sphere_target.x, sphere_target.y, sphere_target.z)


@cython.cdivision(True)
cdef cnp.npy_intp offset(cnp.npy_intp *indices,
                         cnp.npy_intp *strides,
                         int lenind,
                         int typesize) nogil:
    """ Access any element of any ndimensional numpy array using cython.

    Parameters
    ------------
    indices : cnp.npy_intp * (int64 *)
        Indices of the array for which we want to find the offset.
    strides : cnp.npy_intp * strides
    lenind : int, len(indices)
    typesize : int
        Number of bytes for data type e.g. if 8 for double, 4 for int32

    Returns
    ----------
    offset : integer
        Element position in array
    """
    cdef int i
    cdef cnp.npy_intp summ = 0
    for i from 0 <= i < lenind:
        summ += strides[i] * indices[i]
    summ /= <cnp.npy_intp>typesize
    return summ


cdef void splitoffset(float *offset, size_t *index, size_t shape) nogil:
    """Splits a global offset into an integer index and a relative offset"""
    offset[0] -= .5
    if offset[0] <= 0:
        index[0] = 0
        offset[0] = 0.
    elif offset[0] >= (shape - 1):
        index[0] = shape - 2
        offset[0] = 1.
    else:
        index[0] = <size_t> offset[0]
        offset[0] = offset[0] - index[0]


@cython.profile(False)
cdef inline float wght(int i, float r) nogil:
    if i:
        return r
    else:
        return 1.-r


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def trilinear_interp(np.ndarray[np.float32_t, ndim=4, mode='strided'] data,
                     np.ndarray[np.float_t, ndim=1, mode='strided'] index,
                     np.ndarray[np.float_t, ndim=1, mode='c'] voxel_size):
    """Interpolates vector from 4D `data` at 3D point given by `index`

    Interpolates a vector of length T from a 4D volume of shape (I, J, K, T),
    given point (x, y, z) where (x, y, z) are the coordinates of the point in
    real units (not yet adjusted for voxel size).
    """
    cdef:
        float x = index[0] / voxel_size[0]
        float y = index[1] / voxel_size[1]
        float z = index[2] / voxel_size[2]
        float weight
        size_t x_ind, y_ind, z_ind, ii, jj, kk, LL
        size_t last_d = data.shape[3]
        bint bounds_check
        np.ndarray[cnp.float32_t, ndim=1, mode='c'] result
    bounds_check = (x < 0 or y < 0 or z < 0 or
                    x > data.shape[0] or
                    y > data.shape[1] or
                    z > data.shape[2])
    if bounds_check:
        raise IndexError

    splitoffset(&x, &x_ind, data.shape[0])
    splitoffset(&y, &y_ind, data.shape[1])
    splitoffset(&z, &z_ind, data.shape[2])

    result = np.zeros(last_d, dtype='float32')
    for ii from 0 <= ii <= 1:
        for jj from 0 <= jj <= 1:
            for kk from 0 <= kk <= 1:
                weight = wght(ii, x)*wght(jj, y)*wght(kk, z)
                for LL from 0 <= LL < last_d:
                    result[LL] += data[x_ind+ii,y_ind+jj,z_ind+kk,LL]*weight
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def map_coordinates_trilinear_iso(cnp.ndarray[double, ndim=3] data,
                                  cnp.ndarray[double, ndim=2] points,
                                  cnp.ndarray[cnp.npy_intp, ndim=1] data_strides,
                                  cnp.npy_intp len_points,
                                  cnp.ndarray[double, ndim=1] result):
    """ Trilinear interpolation (isotropic voxel size)

    Has similar behavior to ``map_coordinates`` from ``scipy.ndimage``

    Parameters
    ----------
    data : array, float64 shape (X, Y, Z)
    points : array, float64 shape(N, 3)
    data_strides : array npy_intp shape (3,)
        Strides sequence for `data` array
    len_points : cnp.npy_intp
        Number of points to interpolate
    result : array, float64 shape(N)
        The output array. This array should be initialized before you call
        this function.  On exit it will contain the interpolated values from
        `data` at points given by `points`.

    Returns
    -------
    None

    Notes
    -----
    The output array `result` is filled in-place.
    """
    cdef:
        double w[8]
        double values[24]
        cnp.npy_intp index[24]
        cnp.npy_intp off, i, j
        double *ds=<double *> cnp.PyArray_DATA(data)
        double *ps=<double *> cnp.PyArray_DATA(points)
        cnp.npy_intp *strides = <cnp.npy_intp *> cnp.PyArray_DATA(data_strides)
        double *rs=<double *> cnp.PyArray_DATA(result)

    if not cnp.PyArray_CHKFLAGS(data, cnp.NPY_C_CONTIGUOUS):
        raise ValueError(u"data is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(points, cnp.NPY_C_CONTIGUOUS):
        raise ValueError(u"points is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(data_strides, cnp.NPY_C_CONTIGUOUS):
        raise ValueError(u"data_strides is not C contiguous")
    if not cnp.PyArray_CHKFLAGS(result, cnp.NPY_C_CONTIGUOUS):
        raise ValueError(u"result is not C contiguous")
    with nogil:
        for i in range(len_points):
            _trilinear_interpolation_iso(&ps[i * 3],
                                         <double *> w,
                                         <cnp.npy_intp *> index)
            rs[i] = 0
            for j in range(8):
                weight = w[j]
                off = offset(&index[j * 3], <cnp.npy_intp *> strides, 3, 8)
                value = ds[off]
                rs[i] += weight * value
    return


cdef void _trilinear_interpolation_iso(double *X,
                                       double *W,
                                       cnp.npy_intp *IN) nogil:
    """ Interpolate in 3d volumes given point X

    Returns
    -------
    W : weights
    IN : indices of the volume
    """

    cdef double Xf[3]
    cdef double d[3]
    cdef double nd[3]
    cdef cnp.npy_intp i
    # define the rectangular box where every corner is a neighboring voxel
    # (assuming center) !!! this needs to change for the affine case
    for i from 0 <= i < 3:
        Xf[i] = floor(X[i])
        d[i] = X[i] - Xf[i]
        nd[i] = 1 - d[i]
    # weights
    # the weights are actually the volumes of the 8 smaller boxes that define
    # the initial rectangular box for more on trilinear have a look here
    # http://en.wikipedia.org/wiki/Trilinear_interpolation
    # http://local.wasp.uwa.edu.au/~pbourke/miscellaneous/interpolation/index.html
    W[0]=nd[0] * nd[1] * nd[2]
    W[1]= d[0] * nd[1] * nd[2]
    W[2]=nd[0] *  d[1] * nd[2]
    W[3]=nd[0] * nd[1] *  d[2]
    W[4]= d[0] *  d[1] * nd[2]
    W[5]=nd[0] *  d[1] *  d[2]
    W[6]= d[0] * nd[1] *  d[2]
    W[7]= d[0] *  d[1] *  d[2]
    # indices
    # the indices give you the indices of the neighboring voxels (the corners
    # of the box) e.g. the qa coordinates
    IN[0] =<cnp.npy_intp>Xf[0];   IN[1] =<cnp.npy_intp>Xf[1];    IN[2] =<cnp.npy_intp>Xf[2]
    IN[3] =<cnp.npy_intp>Xf[0]+1; IN[4] =<cnp.npy_intp>Xf[1];    IN[5] =<cnp.npy_intp>Xf[2]
    IN[6] =<cnp.npy_intp>Xf[0];   IN[7] =<cnp.npy_intp>Xf[1]+1;  IN[8] =<cnp.npy_intp>Xf[2]
    IN[9] =<cnp.npy_intp>Xf[0];   IN[10]=<cnp.npy_intp>Xf[1];    IN[11]=<cnp.npy_intp>Xf[2]+1
    IN[12]=<cnp.npy_intp>Xf[0]+1; IN[13]=<cnp.npy_intp>Xf[1]+1;  IN[14]=<cnp.npy_intp>Xf[2]
    IN[15]=<cnp.npy_intp>Xf[0];   IN[16]=<cnp.npy_intp>Xf[1]+1;  IN[17]=<cnp.npy_intp>Xf[2]+1
    IN[18]=<cnp.npy_intp>Xf[0]+1; IN[19]=<cnp.npy_intp>Xf[1];    IN[20]=<cnp.npy_intp>Xf[2]+1
    IN[21]=<cnp.npy_intp>Xf[0]+1; IN[22]=<cnp.npy_intp>Xf[1]+1;  IN[23]=<cnp.npy_intp>Xf[2]+1
    return


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


class OutsideImage(Exception):
    pass


class Interpolator(object):
    """Class to be subclassed by different interpolator types"""
    def __init__(self, data, voxel_size):
        self.data = data
        self.voxel_size = np.array(voxel_size, dtype=float, copy=True)


class NearestNeighborInterpolator(Interpolator):
    """Interpolates data using nearest neighbor interpolation"""

    def __getitem__(self, index):
        index = tuple(index / self.voxel_size)
        if min(index) < 0:
            raise OutsideImage('Negative Index')
        try:
            return self.data[tuple(np.array(index).astype(int))]
        except IndexError:
            raise OutsideImage


class TriLinearInterpolator(Interpolator):
    """Interpolates data using trilinear interpolation

    interpolate 4d diffusion volume using 3 indices, ie data[x, y, z]
    """
    def __init__(self, data, voxel_size):
        super(TriLinearInterpolator, self).__init__(data, voxel_size)
        if self.voxel_size.shape != (3,) or self.data.ndim != 4:
            raise ValueError("Data should be 4d volume of diffusion data and "
                             "voxel_size should have 3 values, ie the size "
                             "of a 3d voxel")

    def __getitem__(self, index):
        index = np.array(index, copy=False, dtype="float")
        try:
            return trilinear_interp(self.data, index, self.voxel_size)
        except IndexError:
            raise OutsideImage