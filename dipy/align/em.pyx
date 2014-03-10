#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True


import numpy as np
cimport cython
from fused_types cimport floating, number

cdef extern from "math.h":
    double floor(double x) nogil
    int isinf(double) nogil

cdef inline int ifloor(double x) nogil:
    return int(floor(x))


def quantize_positive_image(floating[:, :] v, int num_levels):
    r"""
    Quantizes the input image at the given number of intensity levels, 
    considering 0 as a special value: at the end, only those voxels with zero
    intensity have label zero

    Parameters
    ----------
    v : array, shape (R, C)
        the image to be quantized
    num_levels : int
        the number of levels
    """
    cdef int nrows = v.shape[0]
    cdef int ncols = v.shape[1]
    cdef int npix = nrows * ncols
    cdef int i, j, l
    cdef double epsilon, delta
    cdef int[:] hist = np.zeros(shape=(num_levels,), dtype=np.int32)
    cdef int[:, :] out = np.zeros(shape=(nrows, ncols,), dtype=np.int32)
    cdef floating[:] levels = np.zeros(shape=(num_levels,), dtype=np.asarray(v).dtype)
    num_levels -= 1  # zero is one of the levels
    if(num_levels < 1):
        return None, None, None
    cdef double min_val = -1
    cdef double max_val = -1

    with nogil:

        for i in range(nrows):
            for j in range(ncols):
                if(v[i, j] > 0):
                    if((min_val < 0) or (v[i, j] < min_val)):
                        min_val = v[i, j]
                    if(v[i, j] > max_val):
                        max_val = v[i, j]
        epsilon = 1e-8
        delta = (max_val - min_val + epsilon) / num_levels
        # notice that we decreased num_levels, so levels[0..num_levels] are well
        # defined
        if((num_levels < 2) or (delta < epsilon)):
            for i in range(nrows):
                for j in range(ncols):
                    if(v[i, j] > 0):
                        out[i, j] = 1
                    else:
                        out[i, j] = 0
                        hist[0] += 1
            levels[0] = 0
            levels[1] = 0.5 * (min_val + max_val)
            hist[1] = npix - hist[0]
            with gil:
                return out, levels, hist

        levels[0] = 0
        levels[1] = delta * 0.5
        for i in range(2, 1 + num_levels):
            levels[i] = levels[i - 1] + delta
        for i in range(nrows):
            for j in range(ncols):
                if(v[i, j] > 0):
                    l = ifloor((v[i, j] - min_val) / delta)
                    out[i, j] = l + 1
                    hist[l + 1] += 1
                else:
                    out[i, j] = 0
                    hist[0] += 1

    return out, levels, hist


def quantize_positive_volume(floating[:, :, :] v, int num_levels):
    r"""
    Quantizes the input volume at the given number of intensity levels,
    considering 0 as a special value: at the end, only those voxels with zero
    intensity have label zero

    Parameters
    ----------
    v : array, shape (S, R, C)
        the volume to be quantized
    num_levels : int
        the number of levels
    """
    cdef int nslices = v.shape[0]
    cdef int nrows = v.shape[1]
    cdef int ncols = v.shape[2]
    cdef int nvox = nrows * ncols * nslices
    cdef int i, j, k, l
    cdef double epsilon, delta
    cdef int[:] hist = np.zeros(shape=(num_levels,), dtype=np.int32)
    cdef int[:, :, :] out = np.zeros(shape=(nslices, nrows, ncols), dtype=np.int32)
    cdef floating[:] levels = np.zeros(shape=(num_levels,), dtype=np.asarray(v).dtype)
    num_levels -= 1  # zero is one of the levels
    if(num_levels < 1):
        return None, None, None
    cdef double min_val = -1
    cdef double max_val = -1

    with nogil:

        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if(v[k, i, j] > 0):
                        if((min_val < 0) or (v[k, i, j] < min_val)):
                            min_val = v[k, i, j]
                        if(v[k, i, j] > max_val):
                            max_val = v[k, i, j]
        epsilon = 1e-8
        delta = (max_val - min_val + epsilon) / num_levels
        # notice that we decreased num_levels, so levels[0..num_levels] are well
        # defined
        if((num_levels < 2) or (delta < epsilon)):
            for k in range(nslices):
                for i in range(nrows):
                    for j in range(ncols):
                        if(v[k, i, j] > 0):
                            out[k, i, j] = 1
                        else:
                            out[k, i, j] = 0
                            hist[0] += 1
            levels[0] = 0
            levels[1] = 0.5 * (min_val + max_val)
            hist[1] = nvox - hist[0]
            with gil:
                return out, levels, hist
        levels[0] = 0
        levels[1] = delta * 0.5
        for i in range(2, 1 + num_levels):
            levels[i] = levels[i - 1] + delta
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if(v[k, i, j] > 0):
                        l = ifloor((v[k, i, j] - min_val) / delta)
                        out[k, i, j] = l + 1
                        hist[l + 1] += 1
                    else:
                        out[k, i, j] = 0
                        hist[0] += 1
    return out, levels, hist


def compute_masked_image_class_stats(int[:, :] mask, floating[:, :] v,
                                     int numLabels, int[:, :] labels):
    r"""
    Computes the mean and standard deviation of the intensities in 'v' for
    each corresponding label in 'labels'. In other words, for each label
    L, it computes the mean and standard deviation of the intensities in 'v'
    at pixels whose label in 'labels' is L. This is used by the EM metric
    to compute statistics for each hidden variable represented by the labels.

    Parameters
    ----------
    mask : array, shape (R, C)
        the mask of pixels that will be taken into account for computing the 
        statistics. All zero pixels in mask will be ignored
    v : array, shape (R, C)
        the image wich the statistics will be computed from
    numLabels : int 
        the number of diferent labels in 'labels' (equal to the
        number of hidden variables in the EM metric)
    labels : array, shape (R, C) 
        the label assigned to each pixel
    """
    cdef int nrows = v.shape[0]
    cdef int ncols = v.shape[1]
    cdef int i, j
    cdef double INF64 = np.inf
    cdef int[:] counts = np.zeros(shape=(numLabels,), dtype=np.int32)
    cdef floating[:] means = np.zeros(shape=(numLabels,), dtype=np.asarray(v).dtype)
    cdef floating[:] variances = np.zeros(shape=(numLabels, ), dtype=np.asarray(v).dtype)

    with nogil:
        for i in range(nrows):
            for j in range(ncols):
                if(mask[i, j] != 0):
                    means[labels[i, j]] += v[i, j]
                    variances[labels[i, j]] += v[i, j] ** 2
                    counts[labels[i, j]] += 1
        for i in range(numLabels):
            if(counts[i] > 0):
                means[i] /= counts[i]
            if(counts[i] > 1):
                variances[i] = variances[i] / counts[i] - means[i] ** 2
            else:
                variances[i] = INF64
    return means, variances


def compute_masked_volume_class_stats(int[:, :, :] mask, floating[:, :, :] v,
                                      int numLabels, int[:, :, :] labels):
    r"""
    Computes the mean and standard deviation of the intensities in 'v' for
    each corresponding label in 'labels'. In other words, for each label
    L, it computes the mean and standard deviation of the intensities in 'v'
    at voxels whose label in 'labels' is L. This is used by the EM metric
    to compute statistics for each hidden variable represented by the labels.

    Parameters
    ----------
    mask : array, shape (S, R, C)
        the mask of voxels that will be taken into account for computing the 
        statistics. All zero voxels in mask will be ignored
    v : array, shape (R, C)
        the volume wich the statistics will be computed from
    numLabels : int 
        the number of diferent labels in 'labels' (equal to the
        number of hidden variables in the EM metric)
    labels : array, shape (R, C) 
        the label assigned to each pixel
    """
    cdef int nslices = v.shape[0]
    cdef int nrows = v.shape[1]
    cdef int ncols = v.shape[2]
    cdef int i, j, k
    cdef double INF64 = np.inf
    cdef int[:] counts = np.zeros(shape=(numLabels,), dtype=np.int32)
    cdef floating[:] means = np.zeros(shape=(numLabels,), dtype=np.asarray(v).dtype)
    cdef floating[:] variances = np.zeros(shape=(numLabels, ), dtype=np.asarray(v).dtype)

    with nogil:
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if(mask[k, i, j] != 0):
                        means[labels[k, i, j]] += v[k, i, j]
                        variances[labels[k, i, j]] += v[k, i, j] ** 2
                        counts[labels[k, i, j]] += 1
        for i in range(numLabels):
            if(counts[i] > 0):
                means[i] /= counts[i]
            if(counts[i] > 1):
                variances[i] = variances[i] / counts[i] - means[i] ** 2
            else:
                variances[i] = INF64
    return means, variances

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_em_demons_step_2d(floating[:,:] delta_field,
                              floating[:,:] sigma_field,
                              floating[:,:,:] gradient_moving,
                              double sigma_reg,
                              floating[:,:,:] out):
    r"""
    Parameters
    ----------
    delta_field : array, shape(R, C)
        contains, at each pixel, the difference between the moving image (warped 
        under the current deformation s ) J and the static image I:
        delta_field[i,j] = J(s(i,j)) - I(i,j). The order is important, changing
        to delta_field[i,j] = I(i,j) - J(s(i,j)) yields the backward demoms step
        warping the static image towards the moving, which may not be the
        intended behavior unless the 'gradient_moving' passed corresponds to
        the gradient of the static image
    sigma_field : array, shape(R, C)
        contains, at each pixel (i, j), the estimated variance (not std) of the
        hidden variable associated to the intensity at static[i,j] (which must 
        have been previously quantized)
    gradient_moving : array, shape(R, C, 2)
        the gradient of the moving image
    sigma_reg : float
        parameter controlling the amount of reguarization (under the Ridge 
        regression model: \min_{x} ||Ax - y||^2 + \frac{1}{'sigmadiff'}||x||^2)
    out : array, shape(R, C, 2)
        the resulting demons step will be writen to this array
    """

    cdef int nr = delta_field.shape[0]
    cdef int nc = delta_field.shape[1]
    cdef int i, j
    cdef double neg_delta, sigma, nrm2, energy

    if out is None:
        out = np.zeros((nr, nc, 2), dtype=np.asarray(delta_field).dtype)

    with nogil:

        energy = 0
        for i in range(nr):
            for j in range(nc):
                sigma = sigma_field[i,j]
                neg_delta = -1 * delta_field[i,j]
                energy += (neg_delta**2)
                if(isinf(sigma)):
                    out[i, j, 0], out[i, j, 1] = 0, 0 
                else:
                    nrm2 = gradient_moving[i, j, 0]**2 + gradient_moving[i, j, 1]**2
                    if(sigma == 0):
                        if nrm2 == 0:
                            out[i, j, 0], out[i, j, 1] = 0, 0 
                        else:
                            out[i, j, 0] = neg_delta * gradient_moving[i, j, 0] / nrm2
                            out[i, j, 1] = neg_delta * gradient_moving[i, j, 1] / nrm2
                    else: 
                        out[i, j, 0] = sigma_reg * neg_delta * gradient_moving[i, j, 0]/(sigma_reg * nrm2 + sigma)
                        out[i, j, 1] = sigma_reg * neg_delta * gradient_moving[i, j, 1]/(sigma_reg * nrm2 + sigma)
    return out, energy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_em_demons_step_3d(floating[:,:,:] delta_field,
                              floating[:,:,:] sigma_field,
                              floating[:,:,:,:] gradient_moving,
                              double sigma_reg,
                              floating[:,:,:,:] out):
    r"""
    Parameters
    ----------
    delta_field : array, shape(S, R, C)
        contains, at each pixel, the difference between the moving image (warped 
        under the current deformation s ) J and the static image I:
        delta_field[k,i,j] = J(s(k,i,j)) - I(k,i,j). The order is important, changing
        to delta_field[k,i,j] = I(k,i,j) - J(s(k,i,j)) yields the backward demoms step
        warping the static image towards the moving, which may not be the
        intended behavior unless the 'gradient_moving' passed corresponds to
        the gradient of the static image
    sigma_field : array, shape(S, R, C)
        contains, at each pixel (k, i, j), the estimated variance (not std) of the
        hidden variable associated to the intensity at static[k,i,j] (which must 
        have been previously quantized)
    gradient_moving : array, shape(S, R, C, 2)
        the gradient of the moving image
    sigma_reg : float
        parameter controlling the amount of reguarization (under the Ridge 
        regression model: \min_{x} ||Ax - y||^2 + \frac{1}{'sigmadiff'}||x||^2)
    out : array, shape(S, R, C, 2)
        the resulting demons step will be writen to this array
    """

    cdef int ns = delta_field.shape[0]
    cdef int nr = delta_field.shape[1]
    cdef int nc = delta_field.shape[2]
    cdef int i, j, k
    cdef double neg_delta, sigma, nrm2, energy

    if out is None:
        out = np.zeros((ns, nr, nc, 3), dtype=np.asarray(delta_field).dtype)

    with nogil:

        energy = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    sigma = sigma_field[k,i,j]
                    neg_delta = -1 * delta_field[k,i,j]
                    energy += (neg_delta**2)
                    if(isinf(sigma)):
                        out[k, i, j, 0], out[k, i, j, 1], out[k, i, j, 2]  = 0, 0, 0
                    else:
                        nrm2 = gradient_moving[k, i, j, 0]**2 + gradient_moving[k, i, j, 1]**2 + gradient_moving[k, i, j, 2]**2
                        if(sigma == 0):
                            if nrm2 == 0:
                                out[k, i, j, 0], out[k, i, j, 1], out[k, i, j, 2] = 0, 0, 0 
                            else:
                                out[k, i, j, 0] = neg_delta * gradient_moving[k, i, j, 0] / nrm2
                                out[k, i, j, 1] = neg_delta * gradient_moving[k, i, j, 1] / nrm2
                                out[k, i, j, 2] = neg_delta * gradient_moving[k, i, j, 2] / nrm2
                        else: 
                            out[k, i, j, 0] = sigma_reg * neg_delta * gradient_moving[k, i, j, 0]/(sigma_reg * nrm2 + sigma)
                            out[k, i, j, 1] = sigma_reg * neg_delta * gradient_moving[k, i, j, 1]/(sigma_reg * nrm2 + sigma)
                            out[k, i, j, 2] = sigma_reg * neg_delta * gradient_moving[k, i, j, 2]/(sigma_reg * nrm2 + sigma)
    return out, energy
