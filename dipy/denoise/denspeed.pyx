from __future__ import division

import numpy as np
cimport numpy as cnp
cimport cython
from cython.parallel import parallel, prange

from libc.math cimport sqrt, exp
from libc.stdlib cimport malloc, free
from libc.string cimport memcpy


def nlmeans_3d(arr, mask=None, patch_radius=1, block_radius=5, sigma=None, rician=True):

    arr = np.ascontiguousarray(arr, dtype='f8')

    arr = add_border(arr, block_radius)

    if mask is None:
        mask = np.ones_like(arr, dtype=np.uint8)
    else:
        mask = np.ascontiguousarray(mask, dtype=np.uint8)

    arrnlm = _nlmeans_3d(arr, mask, patch_radius, block_radius, sigma, rician)

    return remove_border(arrnlm, block_radius)


@cython.wraparound(False)
@cython.boundscheck(False)
def _nlmeans_3d(double [:, :, ::1] arr, cnp.npy_uint8 [:, :, ::1] mask,
                patch_radius=1, block_radius=5,
                sigma=None, rician=True):

    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double [:, :, ::1] out = np.zeros_like(arr)
        double summ = 0
        double sigm = 0
        cnp.npy_intp P = patch_radius
        cnp.npy_intp B = block_radius

    if sigma is None:
        sigm = 5 # call piesno
    else:
        sigm = sigma

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]

    #move the block
    with nogil, parallel(num_threads=I):
        for i in prange(B, I - B):
            for j in range(B , J - B):
                for k in range(B, K - B):

                    if mask[i, j, k] == 0:
                        continue

                    out[i, j, k] = process_block(arr, i, j, k, B, P, sigm)

    new = np.asarray(out)

    if rician:
        new -= 2 * sigm ** 2
        new[new < 0] = 0

    return np.sqrt(new)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double process_block(double [:, :, ::1] arr,
                          cnp.npy_intp i, cnp.npy_intp j, cnp.npy_intp k,
                          cnp.npy_intp B, cnp.npy_intp P, double sigma) nogil:

    cdef:
        cnp.npy_intp m, n, o, M, N, O, a, b, c, cnt, step
        double patch_vol_size
        double summ, d, w, sumw, sum_out, x
        double * W
        double * cache
        double denom
        cnp.npy_intp BS = B * 2 + 1


    cnt = 0
    sumw = 0
    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1)
    denom = sigma * sigma

    W = <double *> malloc(BS * BS * BS * sizeof(double))
    cache = <double *> malloc(BS * BS * BS * sizeof(double))

    # (i, j, k) coordinates are the center of the static patch
    # copy block in cache
    copy_sub_array(cache, BS, BS, BS, arr, i - B, j - B, k - B)

    # calculate weights between the central patch and the moving patch in block
    # (m, n, o) coordinates are the center of the moving patch
    # (a, b, c) run incide both patches
    for m in range(P, BS - P):
        for n in range(P, BS - P):
            for o in range(P, BS - P):

                summ = 0

                # calculate square distance
                for a in range(- P, P + 1):
                    for b in range(- P, P + 1):
                        for c in range(- P, P + 1):

                            # this line takes most of the time! mem access
                            d = cache[(B + a) * BS * BS + (B + b) * BS + (B + c)] - cache[(m + a) * BS * BS + (n + b) * BS + (o + c)]
                            summ += d * d

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

    return sum_out


def add_border(arr, padding):
    arr = np.pad(arr, (padding, padding,), mode='reflect')
    return arr


def remove_border(arr, padding):
    shape = arr.shape
    return arr[padding:shape[0] - padding,
               padding:shape[1] - padding,
               padding:shape[2] - padding]

def test_copy_sub_array():

    source = np.ones((10, 10, 10))
    source[2, 2, 2] = 2
    source[6, 6, 7] = 3
    dest = np.zeros((5, 5, 6))

    copy_sub_memview(dest, source, 2, 2, 2)
    print(dest)

    source = np.ones((10, 10, 10))
    source[2, 2, 2] = 2
    source[6, 6, 7] = 3

    cdef cnp.ndarray[double, ndim=3, mode ='c'] dest2 = np.zeros((5, 5, 6))
    copy_sub_array(<double *>dest2.data, 5, 5, 6, source, 2, 2, 2)
    print(dest2)

@cython.wraparound(False)
@cython.boundscheck(False)
cdef void copy_sub_memview(double [:, :, ::1] dest,
                         double [:, :, ::1] source,
                         cnp.npy_intp min_i,
                         cnp.npy_intp min_j,
                         cnp.npy_intp min_k) nogil:

    cdef cnp.npy_intp I, J, K, i, j

    I = dest.shape[0]
    J = dest.shape[1]
    K = dest.shape[2]

    for i in range(I):
        for j in range(J):
            memcpy(&dest[i, j, 0], &source[i + min_i, j + min_j, min_k], K * sizeof(double))

    return


@cython.wraparound(False)
@cython.boundscheck(False)
cdef void copy_sub_array(double * dest,
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

    return




