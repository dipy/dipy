#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True


import numpy as np
cimport cython
cimport numpy as cnp
from dipy.align.fused_types cimport floating, number
cdef extern from "dpy_math.h" nogil:
    int dpy_isinf(double)
    double floor(double)

cdef inline int ifloor(double x) nogil:
    return int(floor(x))

def quantize_positive_2d(floating[:, :] v, int num_levels):
    r"""Quantizes a 2D image to num_levels quantization levels

    Quantizes the input image at num_levels intensity levels considering <=0
    as a special value. Those input pixels <=0, and only those, will be
    assigned a quantization level of 0. The positive values are divided into
    the remaining num_levels-1 uniform quanization levels.

    The following are undefined, and raise a ValueError:
    * Quantizing at zero levels because at least one level must be assigned
    * Quantizing at one level because positive values should be assigned a
      level different from the secial level 0 (at least 2 levels are needed)

    Parameters
    ----------
    v : array, shape (R, C)
        the image to be quantized
    num_levels : int
        the number of levels

    Returns
    -------
    out : array, shape (R, C), same shape as v
        the quantized image
    levels: array, shape (num_levels,)
        the quantization values: levels[0]=0, and levels[i] is the mid-point
        of the interval of intensities that are assigned to quantization
        level i, i=1, ..., num_levels-1.
    hist: array, shape (num_levels,)
        histogram: the number of pixels that were assigned to each quantization
        level
    """
    ftype = np.asarray(v).dtype
    cdef:
        cnp.npy_intp nrows = v.shape[0]
        cnp.npy_intp ncols = v.shape[1]
        cnp.npy_intp npix = nrows * ncols
        cnp.npy_intp i, j, l
        double epsilon, delta
        double min_val = -1
        double max_val = -1
        int[:] hist = np.zeros(shape=(num_levels,), dtype=np.int32)
        int[:, :] out = np.zeros(shape=(nrows, ncols,), dtype=np.int32)
        floating[:] levels = np.zeros(shape=(num_levels,), dtype=ftype)

    #Quantizing at zero levels is undefined
    #Quantizing at one level is not supported because we want to make sure the
    #maximum level in the quantization is never greater than num_levels-1
    if(num_levels < 2):
        raise ValueError('Quantization levels must be at least 2')

    num_levels -= 1  # zero is one of the levels

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
        levels[1] = min_val + delta * 0.5
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

    return np.asarray(out), np.array(levels), np.array(hist)


def quantize_positive_3d(floating[:, :, :] v, int num_levels):
    r"""Quantizes a 3D volume to num_levels quantization levels

    Quantizes the input volume at num_levels intensity levels considering <=0
    as a special value. Those input voxels <=0, and only those, will be
    assigned a quantization level of 0. The positive values are divided into
    the remaining num_levels-1 uniform quanization levels.

    The following are undefined, and raise a ValueError:
    * Quantizing at zero levels because at least one level must be assigned
    * Quantizing at one level because positive values should be assigned a
      level different from the secial level 0 (at least 2 levels are needed)

    Parameters
    ----------
    v : array, shape (S, R, C)
        the volume to be quantized
    num_levels : int
        the number of levels

    Returns
    -------
    out : array, shape (S, R, C), same shape as v
        the quantized volume
    levels: array, shape (num_levels,)
        the quantization values: levels[0]=0, and levels[i] is the mid-point
        of the interval of intensities that are assigned to quantization
        level i, i=1, ..., num_levels-1.
    hist: array, shape (num_levels,)
        histogram: the number of voxels that were assigned to each quantization
        level
    """
    ftype = np.asarray(v).dtype
    cdef:
        cnp.npy_intp nslices = v.shape[0]
        cnp.npy_intp nrows = v.shape[1]
        cnp.npy_intp ncols = v.shape[2]
        cnp.npy_intp nvox = nrows * ncols * nslices
        cnp.npy_intp i, j, k, l
        double epsilon, delta
        double min_val = -1
        double max_val = -1
        int[:] hist = np.zeros(shape=(num_levels,), dtype=np.int32)
        int[:, :, :] out = np.zeros(shape=(nslices, nrows, ncols),
                                    dtype=np.int32)
        floating[:] levels = np.zeros(shape=(num_levels,), dtype=ftype)

    #Quantizing at zero levels is undefined
    #Quantizing at one level is not supported because we want to make sure the
    #maximum level in the quantization is never greater than num_levels-1
    if(num_levels < 2):
        raise ValueError('Quantization levels must be at least 2')

    num_levels -= 1  # zero is one of the levels

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
        levels[1] = min_val + delta * 0.5
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
    return np.asarray(out), np.asarray(levels), np.asarray(hist)


def compute_masked_class_stats_2d(int[:, :] mask, floating[:, :] v,
                                     int num_labels, int[:, :] labels):
    r"""Computes the mean and std. for each quantization level.

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
        the image which the statistics will be computed from
    num_labels : int
        the number of different labels in 'labels' (equal to the
        number of hidden variables in the EM metric)
    labels : array, shape (R, C)
        the label assigned to each pixel

    Returns
    -------
    means : array, shape (num_labels,)
        means[i], 0<=i<num_labels will be the mean intensity in v of all
        voxels labeled i, or 0 if no voxels are labeled i
    variances : array, shape (num_labels,)
        variances[i], 0<=i<num_labels will be the standard deviation of the
        intensities in v of all voxels labeled i, or infinite if less than 2
        voxels are labeld i.
    """
    ftype=np.asarray(v).dtype
    cdef:
        cnp.npy_intp nrows = v.shape[0]
        cnp.npy_intp ncols = v.shape[1]
        cnp.npy_intp i, j
        double INF64 = np.inf
        int[:] counts = np.zeros(shape=(num_labels,), dtype=np.int32)
        floating diff
        floating[:] means = np.zeros(shape=(num_labels,), dtype=ftype)
        floating[:] variances = np.zeros(shape=(num_labels, ), dtype=ftype)

    with nogil:
        for i in range(nrows):
            for j in range(ncols):
                if(mask[i, j] != 0):
                    means[labels[i, j]] += v[i, j]
                    counts[labels[i, j]] += 1
        for i in range(num_labels):
            if(counts[i] > 0):
                means[i] /= counts[i]
        for i in range(nrows):
            for j in range(ncols):
                if(mask[i, j] != 0):
                    diff = v[i, j] - means[labels[i, j]]
                    variances[labels[i, j]] += diff ** 2

        for i in range(num_labels):
            if(counts[i] > 1):
                variances[i] /= counts[i]
            else:
                variances[i] = INF64
    return np.asarray(means), np.asarray(variances)


def compute_masked_class_stats_3d(int[:, :, :] mask, floating[:, :, :] v,
                                      int num_labels, int[:, :, :] labels):
    r"""Computes the mean and std. for each quantization level.

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
    v : array, shape (S, R, C)
        the volume which the statistics will be computed from
    num_labels : int
        the number of different labels in 'labels' (equal to the
        number of hidden variables in the EM metric)
    labels : array, shape (S, R, C)
        the label assigned to each pixel

    Returns
    -------
    means : array, shape (num_labels,)
        means[i], 0<=i<num_labels will be the mean intensity in v of all
        voxels labeled i, or 0 if no voxels are labeled i
    variances : array, shape (num_labels,)
        variances[i], 0<=i<num_labels will be the standard deviation of the
        intensities in v of all voxels labeled i, or infinite if less than 2
        voxels are labeld i.
    """
    ftype=np.asarray(v).dtype
    cdef:
        cnp.npy_intp nslices = v.shape[0]
        cnp.npy_intp nrows = v.shape[1]
        cnp.npy_intp ncols = v.shape[2]
        cnp.npy_intp i, j, k
        double INF64 = np.inf
        floating diff
        int[:] counts = np.zeros(shape=(num_labels,), dtype=np.int32)
        floating[:] means = np.zeros(shape=(num_labels,), dtype=ftype)
        floating[:] variances = np.zeros(shape=(num_labels, ), dtype=ftype)

    with nogil:
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if(mask[k, i, j] != 0):
                        means[labels[k, i, j]] += v[k, i, j]
                        counts[labels[k, i, j]] += 1
        for i in range(num_labels):
            if(counts[i] > 0):
                means[i] /= counts[i]
        for k in range(nslices):
            for i in range(nrows):
                for j in range(ncols):
                    if(mask[k, i, j] != 0):
                        diff = means[labels[k, i, j]] - v[k, i, j]
                        variances[labels[k, i, j]] += diff ** 2
        for i in range(num_labels):
            if(counts[i] > 1):
                variances[i] /= counts[i]
            else:
                variances[i] = INF64
    return np.asarray(means), np.asarray(variances)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_em_demons_step_2d(floating[:,:] delta_field,
                              floating[:,:] sigma_sq_field,
                              floating[:,:,:] gradient_moving,
                              double sigma_sq_x,
                              floating[:,:,:] out):
    r"""Demons step for EM metric in 2D

    Computes the demons step [Vercauteren09] for SSD-driven registration
    ( eq. 4 in [Vercauteren09] ) using the EM algorithm [Arce14] to handle
    multi-modality images.

    In this case, $\sigma_i$ in eq. 4 of [Vercauteren] is estimated using the EM
    algorithm, while in the original version of diffeomorphic demons it is
    estimated by the difference between the image values at each pixel.

    Parameters
    ----------
    delta_field : array, shape (R, C)
        contains, at each pixel, the difference between the moving image (warped
        under the current deformation s(. , .) ) J and the static image I:
        delta_field[i,j] = J(s(i,j)) - I(i,j). The order is important, changing
        to delta_field[i,j] = I(i,j) - J(s(i,j)) yields the backward demons step
        warping the static image towards the moving, which may not be the
        intended behavior unless the 'gradient_moving' passed corresponds to
        the gradient of the static image
    sigma_sq_field : array, shape (R, C)
        contains, at each pixel (i, j), the estimated variance (not std) of the
        hidden variable associated to the intensity at static[i,j] (which must
        have been previously quantized)
    gradient_moving : array, shape (R, C, 2)
        the gradient of the moving image
    sigma_sq_x : float
        parameter controlling the amount of regularization. It corresponds to
        $\sigma_x^2$ in algorithm 1 of Vercauteren et al.[2]
    out : array, shape (R, C, 2)
        the resulting demons step will be written to this array

    Returns
    -------
    demons_step : array, shape (R, C, 2)
        the demons step to be applied for updating the current displacement
        field
    energy : float
        the current em energy (before applying the returned demons_step)

    References
    ----------
    [Arce14] Arce-santana, E., Campos-delgado, D. U., & Vigueras-g, F. (2014).
             Non-rigid Multimodal Image Registration Based on the
             Expectation-Maximization Algorithm, (168140), 36-47.

    [Vercauteren09] Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N.
                    (2009). Diffeomorphic demons: efficient non-parametric
                    image registration. NeuroImage, 45(1 Suppl), S61-72.
                    doi:10.1016/j.neuroimage.2008.10.040
    """
    cdef:
        cnp.npy_intp nr = delta_field.shape[0]
        cnp.npy_intp nc = delta_field.shape[1]
        cnp.npy_intp i, j
        double delta, sigma_sq_i, nrm2, energy, den, prod

    if out is None:
        out = np.zeros((nr, nc, 2), dtype=np.asarray(delta_field).dtype)

    with nogil:

        energy = 0
        for i in range(nr):
            for j in range(nc):
                sigma_sq_i = sigma_sq_field[i,j]
                delta = delta_field[i,j]
                energy += (delta**2)
                if dpy_isinf(sigma_sq_i) != 0:
                    out[i, j, 0], out[i, j, 1] = 0, 0
                else:
                    nrm2 = (gradient_moving[i, j, 0]**2 +
                            gradient_moving[i, j, 1]**2)
                    if(sigma_sq_i == 0):
                        if nrm2 == 0:
                            out[i, j, 0], out[i, j, 1] = 0, 0
                        else:
                            out[i, j, 0] = (delta *
                                            gradient_moving[i, j, 0] / nrm2)
                            out[i, j, 1] = (delta *
                                            gradient_moving[i, j, 1] / nrm2)
                    else:
                        den = (sigma_sq_x * nrm2 + sigma_sq_i)
                        prod = sigma_sq_x * delta
                        out[i, j, 0] = prod * gradient_moving[i, j, 0] / den
                        out[i, j, 1] = prod * gradient_moving[i, j, 1] / den
    return np.asarray(out), energy

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_em_demons_step_3d(floating[:,:,:] delta_field,
                              floating[:,:,:] sigma_sq_field,
                              floating[:,:,:,:] gradient_moving,
                              double sigma_sq_x,
                              floating[:,:,:,:] out):
    r"""Demons step for EM metric in 3D

    Computes the demons step [Vercauteren09] for SSD-driven registration
    ( eq. 4 in [Vercauteren09] ) using the EM algorithm [Arce14] to handle
    multi-modality images.

    In this case, $\sigma_i$ in eq. 4 of [Vercauteren09] is estimated using
    the EM algorithm, while in the original version of diffeomorphic demons
    it is estimated by the difference between the image values at each pixel.

    Parameters
    ----------
    delta_field : array, shape (S, R, C)
        contains, at each pixel, the difference between the moving image (warped
        under the current deformation s ) J and the static image I:
        delta_field[k,i,j] = J(s(k,i,j)) - I(k,i,j). The order is important,
        changing to delta_field[k,i,j] = I(k,i,j) - J(s(k,i,j)) yields the
        backward demons step warping the static image towards the moving, which
        may not be the intended behavior unless the 'gradient_moving' passed
        corresponds to the gradient of the static image
    sigma_sq_field : array, shape (S, R, C)
        contains, at each pixel (k, i, j), the estimated variance (not std) of
        the hidden variable associated to the intensity at static[k,i,j] (which
        must have been previously quantized)
    gradient_moving : array, shape (S, R, C, 2)
        the gradient of the moving image
    sigma_sq_x : float
        parameter controlling the amount of regularization. It corresponds to
        $\sigma_x^2$ in algorithm 1 of Vercauteren et al.[2].
    out : array, shape (S, R, C, 2)
        the resulting demons step will be written to this array

    Returns
    -------
    demons_step : array, shape (S, R, C, 3)
        the demons step to be applied for updating the current displacement
        field
    energy : float
        the current em energy (before applying the returned demons_step)

    References
    ----------
    [Arce14] Arce-santana, E., Campos-delgado, D. U., & Vigueras-g, F. (2014).
             Non-rigid Multimodal Image Registration Based on the
             Expectation-Maximization Algorithm, (168140), 36-47.

    [Vercauteren09] Vercauteren, T., Pennec, X., Perchant, A., & Ayache, N.
                    (2009). Diffeomorphic demons: efficient non-parametric
                    image registration. NeuroImage, 45(1 Suppl), S61-72.
                    doi:10.1016/j.neuroimage.2008.10.040
    """
    cdef:
        cnp.npy_intp ns = delta_field.shape[0]
        cnp.npy_intp nr = delta_field.shape[1]
        cnp.npy_intp nc = delta_field.shape[2]
        cnp.npy_intp i, j, k
        double delta, sigma_sq_i, nrm2, energy, den

    if out is None:
        out = np.zeros((ns, nr, nc, 3), dtype=np.asarray(delta_field).dtype)

    with nogil:

        energy = 0
        for k in range(ns):
            for i in range(nr):
                for j in range(nc):
                    sigma_sq_i = sigma_sq_field[k,i,j]
                    delta = delta_field[k,i,j]
                    energy += (delta**2)
                    if dpy_isinf(sigma_sq_i) != 0:
                        out[k, i, j, 0] = 0
                        out[k, i, j, 1] = 0
                        out[k, i, j, 2] = 0
                    else:
                        nrm2 = (gradient_moving[k, i, j, 0]**2 +
                                gradient_moving[k, i, j, 1]**2 +
                                gradient_moving[k, i, j, 2]**2)
                        if(sigma_sq_i == 0):
                            if nrm2 == 0:
                                out[k, i, j, 0] = 0
                                out[k, i, j, 1] = 0
                                out[k, i, j, 2] = 0
                            else:
                                out[k, i, j, 0] = (delta *
                                    gradient_moving[k, i, j, 0] / nrm2)
                                out[k, i, j, 1] = (delta *
                                    gradient_moving[k, i, j, 1] / nrm2)
                                out[k, i, j, 2] = (delta *
                                    gradient_moving[k, i, j, 2] / nrm2)
                        else:
                            den = (sigma_sq_x * nrm2 + sigma_sq_i)
                            out[k, i, j, 0] = (sigma_sq_x * delta *
                                gradient_moving[k, i, j, 0] / den)
                            out[k, i, j, 1] = (sigma_sq_x * delta *
                                gradient_moving[k, i, j, 1] / den)
                            out[k, i, j, 2] = (sigma_sq_x * delta *
                                gradient_moving[k, i, j, 2] / den)
    return np.asarray(out), energy
