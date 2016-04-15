from __future__ import division

import numpy as np
cimport numpy as cnp

cimport cython
cimport safe_openmp as openmp
from safe_openmp cimport have_openmp

from cython.parallel import parallel, prange
from multiprocessing import cpu_count

from libc.math cimport sqrt, exp
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


def nlmeans_3d(arr, mask=None, sigma=None, patch_radius=1,
               block_radius=5, rician=True, num_threads=None):
    """ Non-local means for denoising 3D images

    Parameters
    ----------
    arr : 3D ndarray
        The array to be denoised
    mask : 3D ndarray
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.
    num_threads : int
        Number of threads. If None (default) then all available threads
        will be used.

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.
    """

    if arr.ndim != 3:
        raise ValueError('data needs to be a 3D ndarray', arr.shape)

    if mask is None:
        mask = np.ones(arr.shape, dtype='f8')
    else:
        mask = np.ascontiguousarray(mask, dtype='f8')

    if mask.ndim != 3:
        raise ValueError('mask needs to be a 3D ndarray', mask.shape)

    if sigma.ndim != 3:
        raise ValueError('sigma needs to be a 3D ndarray', sigma.shape)

    arr = np.ascontiguousarray(arr, dtype='f8')
    arr = add_padding_reflection(arr, block_radius)
    mask = add_padding_reflection(mask.astype('f8'), block_radius)
    sigma = np.ascontiguousarray(sigma, dtype='f8')
    sigma = add_padding_reflection(sigma.astype('f8'), block_radius)
    
    if num_threads == 1:
        arrnlm = _nlmeans_3d_conv(arr, mask, sigma, patch_radius, block_radius,
                                  rician, num_threads)
        return arrnlm
    else:
        arrnlm = _nlmeans_3d(arr, mask, sigma, patch_radius, block_radius,
                             rician, num_threads)

        return remove_padding(arrnlm, block_radius)


@cython.wraparound(False)
@cython.boundscheck(False)
def _nlmeans_3d_conv(double [:, :, ::1] arr, double [:, :, ::1] mask, double [:, :, ::1] sigma,
                       patch_radius=1, block_radius=5, rician=True):
    """ This algorithm denoises the value of every voxel (i, j, k) by
    calculating a weight between a moving 3D patch and a static 3D patch
    centered at (i, j, k). The moving patch can only move inside a
    3D block.
    """

    cdef:
        cnp.npy_intp i, j, k, I, J, K, l, m, n, L, M, N, x, y, z
        double[:, :, ::1] out
        cnp.npy_intp P = patch_radius
        cnp.npy_intp B = block_radius

    I = arr.shape[0]-2*B
    J = arr.shape[1]-2*B
    K = arr.shape[2]-2*B

    out = np.empty([I, J, K])

    out = process_block_conv(arr, mask, I, J, K, B, P, sigma)

    new = np.asarray(out)

    if rician:
        new -= 2 * np.asarray(remove_padding(sigma, block_radius))**2
        new[new < 0] = 0

    return np.sqrt(new)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double[:, :, ::1] process_block_conv(double [:, :, ::1] arr,
                                 double [:, :, ::1] mask, cnp.npy_intp I,
                                 cnp.npy_intp J, cnp.npy_intp K, cnp.npy_intp B,
                                 cnp.npy_intp P, double [:, :, ::1] sigma):
    """ Process the block; optimized with convolution
    Parameters
    ----------
    arr : 3D array
        C contiguous array of doubles
    mask: 3D array
        C contiguous array of doubles
    I, J, K : int
        shape of block
    B : int
        block radius
    P : int
        patch radius
    sigma : 3D array
        local noise standard deviation
    Returns
    -------
    new_value : double
    """

    cdef:
        cnp.npy_intp m, n, o, a, b, c, x, y, z, i, step, thread_id
        double patch_vol_size
        double v
        double * cache
        double * mask_cache
        double * sigma_block
        double * denom_block
        double [:, :, ::1] buffer_out
        double * sumw
        double * sum_out
        double * d
        double denom
        double single_w
        double single_d
        double d_conv
        double s_conv
        cnp.npy_intp * BS
        cnp.npy_intp * PS

    BS = <cnp.npy_intp *> malloc(3 * sizeof(cnp.npy_intp))
    BS[0] = B * 2 + I
    BS[1] = B * 2 + J
    BS[2] = B * 2 + K
    PS = <cnp.npy_intp *> malloc(3 * sizeof(cnp.npy_intp))
    PS[0] = P * 2 + I
    PS[1] = P * 2 + J
    PS[2] = P * 2 + K

    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1)

    sumw = <double *> malloc(I * J * K * sizeof(double))
    sum_out = <double *> malloc(I * J * K * sizeof(double))

    d = <double *> malloc(PS[0] * PS[1]  * PS[2] * sizeof(double))
    cache = <double *> malloc(BS[0] * BS[1] * BS[2] * sizeof(double))
    mask_cache = <double *> malloc(BS[0] * BS[1] * BS[2] * sizeof(double))
    sigma_block = <double *> malloc(BS[0] * BS[1] * BS[2] * sizeof(double))
    denom_block = <double *> malloc((BS[0]-P) * (BS[1]-P) * (BS[2]-P) * sizeof(double))
    copy_block_3d(cache, BS[0], BS[1], BS[2], arr, 0, 0, 0)
    copy_block_3d(mask_cache, BS[0], BS[1], BS[2], mask, 0, 0, 0)
    copy_block_3d(sigma_block, BS[0], BS[1], BS[2], sigma, 0, 0, 0)

    for a in range(I * J * K):
        sumw[a] = 0
        sum_out[a] = 0

    # Calculate denom upfront
    for a in range(P, BS[0]-P):
        for b in range(P, BS[1]-P):
            for c in range(P, BS[0]-P):
                s_conv = 0
                for x in range(-P, P + 1):
                    for y in range(-P, P + 1):
                        for z in range(-P, P + 1):
                            s_conv += sigma_block[get_flattened_index(a+x, b+y, c+z, BS[0], BS[1], BS[2])]

                denom_block[get_flattened_index(a-P, b-P, c-P, BS[0]-P, BS[1]-P, BS[2]-P)] = \
                        sqrt(2) * s_conv**2 / patch_vol_size

    for m in range(P, 2 * B + 1 - P):
        for n in range(P, 2 * B + 1 - P):
            for o in range(P, 2 * B + 1 - P):

                # Calculate square distance batch
                for a in range(-P, P + I):
                    for b in range(-P, P + J):
                        for c in range(-P, P + K):
                            single_d =  \
                                cache[get_flattened_index(B+a, B+b, B+c, BS[0], BS[1], BS[2])] - \
                                cache[get_flattened_index(m+a, n+b, o+c, BS[0], BS[1], BS[2])]
                            # d[P+a, P+b, P+c] = single_d*single_d
                            d[get_flattened_index(P+a, P+b, P+c, PS[0], PS[1], PS[2])] = single_d * single_d

                # Convolution
                for a in range(0, I):                    
                    for b in range(0, J):
                        for c in range(0, K):
                            if mask_cache[get_flattened_index(a+B, b+B, c+B, BS[0], BS[1], BS[2])] != 0:
                                d_conv = 0.
                                for x in range(-P, P + 1):
                                    for y in range(-P, P + 1):
                                        for z in range(-P, P + 1):
                                            d_conv += d[get_flattened_index(P+a+x, P+b+y, P+c+z, PS[0], PS[1], PS[2])]
                                single_w = exp(- d_conv / denom_block[get_flattened_index(a+m-P, b+n-P, c+o-P, BS[0]-P, BS[1]-P, BS[2]-P)])
                                sumw[get_flattened_index(a, b, c, I, J, K)] += single_w
                                v = cache[get_flattened_index(a+m, b+n, c+o, BS[0], BS[1], BS[2])]
                                sum_out[get_flattened_index(a, b, c, I, J, K)] += single_w * v * v


    for a in range(0, I * J * K):
        if sumw[a] == 0:
            sum_out[a] = 0
        else:
            sum_out[a] /= sumw[a]

    buffer_out = np.empty((I, J, K))
    copy_block_3d_back(buffer_out, 0, 0, 0, sum_out, I, J, K)    
    
    free(cache)
    free(mask_cache)
    free(d)
    free(sumw)
    free(sum_out)
    free(sigma_block)
    return buffer_out


@cython.wraparound(False)
@cython.boundscheck(False)
def _nlmeans_3d(double [:, :, ::1] arr, double [:, :, ::1] mask,
                double [:, :, ::1] sigma, patch_radius=1, block_radius=5,
                rician=True, num_threads=None):
    """ This algorithm denoises the value of every voxel (i, j, k) by
    calculating a weight between a moving 3D patch and a static 3D patch
    centered at (i, j, k). The moving patch can only move inside a
    3D block.
    """

    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double [:, :, ::1] out = np.zeros_like(arr)
        double summ = 0
        cnp.npy_intp P = patch_radius
        cnp.npy_intp B = block_radius
        int all_cores = openmp.omp_get_num_procs()
        int threads_to_use = -1

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]

    if num_threads is not None:
        threads_to_use = num_threads
    else:
        threads_to_use = all_cores

    if have_openmp:
        openmp.omp_set_dynamic(0)
        openmp.omp_set_num_threads(threads_to_use)

    # move the block
    with nogil, parallel():

        for i in prange(B, I - B, schedule="guided"):
            for j in range(B, J - B):
                for k in range(B, K - B):

                    if mask[i, j, k] == 0:
                        continue

                    out[i, j, k] = process_block(arr, i, j, k, B, P, sigma)

    if have_openmp and num_threads is not None:
        openmp.omp_set_num_threads(all_cores)

    new = np.asarray(out)

    if rician:
        new -= 2 * np.asarray(sigma)**2
        new[new < 0] = 0

    return np.sqrt(new)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double process_block(double [:, :, ::1] arr,
                          cnp.npy_intp i, cnp.npy_intp j, cnp.npy_intp k,
                          cnp.npy_intp B, cnp.npy_intp P, double [:, :, ::1] sigma) nogil:
    """ Process the block with center at (i, j, k)

    Parameters
    ----------
    arr : 3D array
        C contiguous array of doubles
    i, j, k : int
        center of block
    B : int
        block radius
    P : int
        patch radius
    sigma : 3D array
        local noise standard deviation

    Returns
    -------
    new_value : double
    """

    cdef:
        cnp.npy_intp m, n, o, M, N, O, a, b, c, cnt, step
        double patch_vol_size
        double summ, d, w, sumw, sum_out, x, sigm
        double * W
        double * cache
        double * sigma_block
        double denom
        cnp.npy_intp BS = B * 2 + 1

    cnt = 0
    sumw = 0
    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1)

    W = <double *> malloc(BS * BS * BS * sizeof(double))
    cache = <double *> malloc(BS * BS * BS * sizeof(double))
    sigma_block = <double *> malloc(BS * BS * BS * sizeof(double))

    # (i, j, k) coordinates are the center of the static patch
    # copy block in cache
    copy_block_3d(cache, BS, BS, BS, arr, i - B, j - B, k - B)
    copy_block_3d(sigma_block, BS, BS, BS, sigma, i - B, j - B, k - B)

    # calculate weights between the central patch and the moving patch in block
    # (m, n, o) coordinates are the center of the moving patch
    # (a, b, c) run inside both patches
    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):

                summ = 0
                sigm = 0

                # calculate square distance
                for a in range(-P, P + 1):
                    for b in range(-P, P + 1):
                        for c in range(-P, P + 1):

                            # this line takes most of the time! mem access
                            d = cache[(B + a) * BS * BS + (B + b) * BS + (B + c)] - cache[(m + a) * BS * BS + (n + b) * BS + (o + c)]
                            summ += d * d
                            sigm += sigma_block[(m + a) * BS * BS + (n + b) * BS + (o + c)]

                denom = sqrt(2) * (sigm / patch_vol_size)**2
                w = exp(-(summ / patch_vol_size) / denom)
                sumw += w
                W[cnt] = w
                cnt += 1

    cnt = 0
    sum_out = 0

    # calculate normalized weights and sums of the weights with the positions
    # of the patches
    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):

                if sumw > 0:
                    w = W[cnt] / sumw
                else:
                    w = 0

                x = cache[m * BS * BS + n * BS + o]
                sum_out += w * x * x
                cnt += 1

    free(W)
    free(cache)
    free(sigma_block)

    return sum_out


def add_padding_reflection(double [:, :, ::1] arr, padding):
    cdef:
        double [:, :, ::1] final
        cnp.npy_intp i, j, k
        cnp.npy_intp B = padding
        cnp.npy_intp [::1] indices_i = correspond_indices(arr.shape[0], padding)
        cnp.npy_intp [::1] indices_j = correspond_indices(arr.shape[1], padding)
        cnp.npy_intp [::1] indices_k = correspond_indices(arr.shape[2], padding)

    final = np.zeros(np.array((arr.shape[0], arr.shape[1], arr.shape[2])) + 2*padding)

    for i in range(final.shape[0]):
        for j in range(final.shape[1]):
            for k in range(final.shape[2]):
                final[i, j, k] = arr[indices_i[i], indices_j[j], indices_k[k]]

    return final


def correspond_indices(dim_size, padding):
    return np.ascontiguousarray(np.hstack((np.arange(1, padding + 1)[::-1],
                                np.arange(dim_size),
                                np.arange(dim_size - padding - 1, dim_size - 1)[::-1])),
                                dtype=np.intp)


def remove_padding(arr, padding):
    shape = arr.shape
    return arr[padding:shape[0] - padding,
               padding:shape[1] - padding,
               padding:shape[2] - padding]


@cython.wraparound(False)
@cython.boundscheck(False)
cdef cnp.npy_intp copy_block_3d(double * dest,
                                cnp.npy_intp I,
                                cnp.npy_intp J,
                                cnp.npy_intp K,
                                double [:, :, ::1] source,
                                cnp.npy_intp min_i,
                                cnp.npy_intp min_j,
                                cnp.npy_intp min_k) nogil:

    cdef cnp.npy_intp i, j

    for i in range(I):
        for j in range(J):
            memcpy(&dest[i * J * K  + j * K], &source[i + min_i, j + min_j, min_k], K * sizeof(double))

    return 1


def cpu_count():
    if have_openmp:
        return openmp.omp_get_num_procs()
    else:
        return 1

