import numpy as np
cimport numpy as cnp
cimport cython

from libc.math cimport sqrt, exp


@cython.wraparound(False)
@cython.boundscheck(False)
def _nlmeans_3d(double [:, :, ::1] arr, patch_size=3, block_size=11, sigma=None, rician=True):

    cdef:
        cnp.npy_intp i, j, k, I, J, K
        double [:, :, ::1] out = np.zeros_like(arr)
        double [::1] W = np.zeros(block_size ** 3)
        double summ = 0
        double sigm = 0
        cnp.npy_intp P = patch_size / 2
        cnp.npy_intp B = block_size / 2

    if sigma is None:
        sigm = 5 # call piesno

    I = arr.shape[0]
    J = arr.shape[1]
    K = arr.shape[2]

    with nogil:
        for i in range(B, I - B - 1):
            for j in range(B, J - B - 1):
                for k in range(B, K - B - 1):

                    out[i, j, k] = process_block(arr, W, i, j, k, B, P, sigm)

    new = np.asarray(out)
    if rician:
        new -= 2 * sigm ** 2
        new[new < 0] = 0

    return np.sqrt(new)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef double process_block(double [:, :, ::1] arr, double [::1] W,
                          int i, int j, int k, int B, int P, double sigma) nogil:

    cdef:
        cnp.npy_intp m, n, o, M, N, O, a, b, c, cnt,
        double patch_vol_size
        double summ, x, d, w, sumw, sum_out


    cnt = 0
    patch_vol_size = (P + P + 1) * (P + P + 1) * (P + P + 1)
    sumw = 0


    # calculate weights between the central patch and the moving patch in block
    for m in range(i - B, i + B + 1):
        for n in range(j - B, j + B + 1):
            for o in range(k - B, k + B + 1):

                summ = 0

                # calculate square distance
                for a in range(- P, P + 1):
                    for b in range(- P, P + 1):
                        for c in range(- P, P + 1):

                            x = arr[i + a, j + b, k + c]
                            d = x - arr[m + a, n + b , o + c]
                            d = d * d

                            summ += d

                w = exp( - (summ / patch_vol_size) / sigma)
                sumw += w
                W[cnt] = w
                cnt += 1

    cnt = 0

    sum_out = 0

    for m in range(i - B, i + B + 1):
        for n in range(j - B, j + B + 1):
            for o in range(k - B, k + B + 1):

                w = W[cnt] / sumw

                x = arr[m, n, o]

                sum_out += w * x * x

                cnt += 1

    return sum_out

