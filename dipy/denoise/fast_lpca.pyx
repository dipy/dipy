import numpy as np
cimport cython
cimport numpy as cnp
from time import time

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void fast_vecmat_mul(double[:] x, double[:, :] A, double[:] out) nogil:
    '''
    Fast vector multiplication with matrix
    out = xA
    '''
    cdef:
        cnp.npy_intp i, j
        double s
        cnp.npy_intp n0 = A.shape[0]
        cnp.npy_intp An1 = A.shape[1]

    for i in range(An1):
        s = 0
        for j in range(n0):
            s += x[j] * A[j, i]

        out[i] = s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_outer_prod(double[:, :] out, double[:] x, int utype)nogil:
    cdef:
        cnp.npy_intp i, j, n
    n = x.shape[0]
    if utype == 0:
        for i in range(n):
            for j in range(n):
                out[i, j] = x[i] * x[j]
    elif utype == 1:
        for i in range(n):
            for j in range(n):
                out[i, j] += x[i] * x[j]
    elif utype == -1:
        for i in range(n):
            for j in range(n):
                out[i, j] -= x[i] * x[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_vector(double[:] out, double[:] x, int utype)nogil:
    cdef:
        cnp.npy_intp i, n
    n = x.shape[0]
    if utype == 0:
        for i in range(n):
            out[i] = x[i]
    elif utype == 1:
        for i in range(n):
            out[i] += x[i]
    elif utype == -1:
        for i in range(n):
            out[i] -= x[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_matrix(double[:, :] out, double[:, :] cov, int utype)nogil:
    cdef:
        cnp.npy_intp i, j, n
    n = cov.shape[0]
    if utype == 0:
        for i in range(n):
            for j in range(n):
                out[i, j] = cov[i, j]
    elif utype == 1:
        for i in range(n):
            for j in range(n):
                out[i, j] += cov[i, j]
    elif utype == -1:
        for i in range(n):
            for j in range(n):
                out[i, j] -= cov[i, j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fast_lpca(double[:, :, :, :] I, int radius, double[:, :, :] sigma):
    '''
    Local PCA Based Denoising of Diffusion Datasets

    Parameters
    ----------
    I : A 4D array which is to be denoised

    sigma : 4D array
        standard deviation of the noise estimated from the data
        If the sigma is a float or a 3D array one muct convert it
        into a 4D array before passing

    radius : The radius of the local patch to
             be taken around each voxel
    Returns
    -------

    denoised_arr : 4D array
        this is the denoised array of the same size as that of
        arr (input data)

    References
    ----------
    [1] Diffusion Weighted Image Denoising Using Overcomplete Local PCA
        Manjon JV, Coupe P, Concha L, Buades A, Collins DL

    [2] On the computation of integrals over fixed-size rectangles of
        arbitrary dimension
        Omar Ocegueda, Oscar Dalmau, Eleftherios Garyfallidis,
        Maxime Descoteaux, Mariano Rivera

    '''

    cdef:
        cnp.npy_intp n0 = I.shape[0]
        cnp.npy_intp n1 = I.shape[1]
        cnp.npy_intp n2 = I.shape[2]
        cnp.npy_intp ndiff = I.shape[3]
        cnp.npy_intp m = 1 + (2 * radius)
        cnp.npy_intp nsamples = m * m * m
        cnp.npy_intp cur_i, prev_i, i, j, k, ii
        cnp.npy_intp i0, j0, k0, p, q, r, s
        double l0norm
        double[:, :, :, :, :] T = np.zeros((2, n1, n2, ndiff, ndiff))
        double[:, :, :, :] S = np.zeros((2, n1, n2, ndiff))
        double[:, :] cov = np.empty((ndiff, ndiff))
        double[:] mu = np.empty(ndiff)
        double[:] x
        double[:, :, :] theta = np.zeros((n0, n1, n2))
        double[:, :, :, :] thetax = np.zeros((n0, n1, n2, ndiff))
        double[:, :, :, :] out = np.zeros((n0, n1, n2, ndiff))
        double[:] temp = np.zeros(ndiff)
        double[:] temp1 = np.zeros(ndiff)
        double[:, :] P = np.zeros((ndiff, ndiff))
        double[:] d = np.zeros(ndiff)
        double[:, :] W_hat = np.zeros((ndiff, ndiff))
        double[:, :] W_hatt = np.zeros((ndiff, ndiff))

    
    with nogil:
        cur_i = 1
        for i in range(n0):
            cur_i = 1 - cur_i
            prev_i = 1 - cur_i
            for j in range(n1):
                for k in range(n2):
                    x = I[i, j, k, :]
                    # Start with last corner
                    update_outer_prod(cov, x, 0)  # q=(0,0,0)
                    update_vector(mu, x, 0)  # q=(0,0,0)

                    # Add signed rectangles
                    if i > 0:
                        update_matrix(cov, T[prev_i, j, k], 1)  # q=(1, 0, 0)
                        update_vector(mu, S[prev_i, j, k], 1)  # q=(1, 0, 0)
                        if j > 0:
                            # q=(1, 1, 0)
                            update_matrix(cov, T[prev_i, j - 1, k], -1)
                            # q=(1, 1, 0)
                            update_vector(mu, S[prev_i, j - 1, k], -1)
                            if k > 0:
                                # q=(1, 1, 1)
                                update_matrix(cov, T[prev_i, j - 1, k - 1], 1)
                                # q=(1, 1, 1)
                                update_vector(mu, S[prev_i, j - 1, k - 1], 1)
                        if k > 0:
                            # q=(1, 0, 1)
                            update_matrix(cov, T[prev_i, j, k - 1], -1)
                            # q=(1, 0, 1)
                            update_vector(mu, S[prev_i, j, k - 1], -1)
                    if j > 0:
                        # q=(0, 1, 0)
                        update_matrix(cov, T[cur_i, j - 1, k], 1)
                        update_vector(mu, S[cur_i, j - 1, k], 1)  # q=(0, 1, 0)
                        if k > 0:
                            # q=(0, 1, 1)
                            update_matrix(cov, T[cur_i, j - 1, k - 1], -1)
                            # q=(0, 1, 1)
                            update_vector(mu, S[cur_i, j - 1, k - 1], -1)
                    if k > 0:
                        # q=(0, 0, 1)
                        update_matrix(cov, T[cur_i, j, k - 1], 1)
                        update_vector(mu, S[cur_i, j, k - 1], 1)  # q=(0, 0, 1)

                    # Add displaced signed corners
                    if i >= m:
                        x = I[i - m, j, k]  # q=(1, 0, 0)
                        update_outer_prod(cov, x, -1)
                        update_vector(mu, x, -1)
                        if j >= m:
                            x = I[i - m, j - m, k]  # q=(1, 1, 0)
                            update_outer_prod(cov, x, 1)
                            update_vector(mu, x, 1)
                            if k >= m:
                                x = I[i - m, j - m, k - m]  # q=(1, 1, 1)
                                update_outer_prod(cov, x, -1)
                                update_vector(mu, x, -1)
                        if k >= m:
                            x = I[i - m, j, k - m]  # q=(1, 0, 1)
                            update_outer_prod(cov, x, 1)
                            update_vector(mu, x, 1)
                    if j >= m:
                        x = I[i, j - m, k]  # q=(0, 1, 0)
                        update_outer_prod(cov, x, -1)
                        update_vector(mu, x, -1)
                        if k >= m:
                            x = I[i, j - m, k - m]  # q=(0, 1, 1)
                            update_outer_prod(cov, x, 1)
                            update_vector(mu, x, 1)
                    if k >= m:
                        x = I[i, j, k - m]  # q=(0, 0, 1)
                        update_outer_prod(cov, x, -1)
                        update_vector(mu, x, -1)

                    # Save current integrals for future reference
                    update_matrix(T[cur_i, j, k], cov, 0)
                    update_vector(S[cur_i, j, k], mu, 0)

                    # Use integral of current rectangle
                    # Last corner is (i, j, k) => center is (i-radius,
                    # j-radius, k-radius)
                    if (i >= m - 1) and (j >= m - 1) and (k >= m - 1):
                        # Divide by the number of samples
                        for ii in range(ndiff):
                            mu[ii] /= nsamples
                            for jj in range(ndiff):
                                cov[ii, jj] /= nsamples
                        # Subtract self-outer product of mu from cov
                        update_outer_prod(cov, mu, -1)

                        i0 = i - radius
                        j0 = j - radius
                        k0 = k - radius

                        with gil:
                            # eigen value decomposition
                            P[...] = 0
                            d, W_hat = np.linalg.eigh(cov)

                        l0norm = 0
                        for p in range(ndiff):
                            if d[p] >= sigma[i0, j0, k0] * 2.3 * 2.3:
                                l0norm += 1
                                update_outer_prod(P, W_hat[:, p], 1)

                        l0norm = 1.0 / (1 + l0norm)

                        # precompute the W_hat.W_hat_transpose for PCA
                        # projection plus reconstruct
                        
                        # make the projection and reconstruct

                        for p in range(i0 - radius, i0 + radius + 1):
                            for q in range(j0 - radius, j0 + radius + 1):
                                for r in range(k0 - radius, k0 + radius + 1):

                                    for s in range(ndiff):
                                        temp[s] = I[p, q, r, s] - mu[s]

                                    fast_vecmat_mul(temp, P, temp1)
                                    update_vector(temp1, mu, 1)
                                    theta[p, q, r] += l0norm

                                    for s in range(ndiff):
                                        thetax[p, q, r, s] += temp1[s] * l0norm


        for i in range(n0):
            for j in range(n1):
                for k in range(n2):
                    for s in range(ndiff):
                        out[i, j, k, s] = thetax[i, j, k, s] / theta[i, j, k]

    return out
