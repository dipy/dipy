import numpy as np
cimport cython
cimport numpy as cnp


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void fast_mat_mul(double[:, :] A, double[:, :] B, double[:, :] out) nogil:
    # we are computing A*B
    cdef:
        cnp.npy_intp i, j, k
        double s
        cnp.npy_intp An0 = A.shape[0]
        cnp.npy_intp An1 = A.shape[1]
        cnp.npy_intp Bn1 = B.shape[1]

    for i in range(An0):
        for j in range(Bn1):
            s = 0
            for k in range(An1):
                s += A[i, k] * B[k, j]

            out[i, j] = s


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void fast_vecmat_mul(double[:] x, double[:, :] A, double[:] out) nogil:
    # we are computing x*A
    cdef:
        cnp.npy_intp i, j
        double s
        cnp.npy_intp n0 = x.shape[0]
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
def fast_lpca(double[:, :, :, :] I, int radius, double[:, :, :, :] sigma):
    '''
    Local PCA Based Denoising of Diffusion Datasets

    References
    ----------
    Diffusion Weighted Image Denoising Using Overcomplete Local PCA
    Manjon JV, Coupe P, Concha L, Buades A, Collins DL

    Parameters
    ----------
    I : A 4D array which is to be denoised
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    radius : The radius of the local patch to
             be taken around each voxel
    tou : float or 3D array
        threshold parameter for diagonal matrix thresholding,
        default value = (2.3 * sigma * sigma)
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.

    Returns
    -------

    denoised_arr : 4D array
        this is the denoised array of the same size as that of
        arr (input data)

    '''
    cdef:
        cnp.npy_intp n0 = I.shape[0]
        cnp.npy_intp n1 = I.shape[1]
        cnp.npy_intp n2 = I.shape[2]
        cnp.npy_intp ndiff = I.shape[3]
        cnp.npy_intp m = 1 + (2 * radius)
        cnp.npy_intp nsamples = m * m * m
        cnp.npy_intp cur_i, prev_i, i, j, k, ii
        cnp.npy_intp i0, j0, k0, p0, p, q, r, s
        double l0norm
        double[:, :, :, :, :] T = np.zeros((2, n1, n2, ndiff, ndiff))
        double[:, :, :, :] S = np.zeros((2, n1, n2, ndiff))
        double[:, :] cov = np.empty((ndiff, ndiff))
        double[:] mu = np.empty(ndiff)
        double[:] x
        double[:, :, :, :] theta = np.zeros((n0, n1, n2, ndiff))
        double[:, :, :, :] thetax = np.zeros((n0, n1, n2, ndiff))
        double[:] temp = np.zeros(ndiff)
        double[:, :] P = np.zeros((ndiff, ndiff))
        double[:] d = np.zeros(ndiff)
        double[:, :] W_hat = np.zeros((ndiff, ndiff))
        double[:, :] W_hatt = np.zeros((ndiff, ndiff))

    nsamples = m * m * m
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
                            d, W_hat = np.linalg.eigh(cov)

                        l0norm = 0
                        for p in range(ndiff):

                            if d[p] < sigma[i0, j0, k0, p] * 2.3 * 2.3:
                                d[p] = 0
                                for q in range(ndiff):
                                    W_hat[q, p] = 0
                                    W_hatt[p, q] = 0
                            else:
                                l0norm += 1
                                for q in range(ndiff):
                                    W_hatt[p, q] = W_hat[q, p]

                        l0norm = 1.0 / (1 + l0norm)

                        # precompute the W_hat.W_hat_transpose for PCA
                        # projection plus reconstruct
                        fast_mat_mul(W_hat, W_hatt, P)

                        # make the projection and reconstruct
                        for p in range(i0 - radius, i0 + radius + 1):
                            for q in range(j0 - radius, j0 + radius + 1):
                                for r in range(k0 - radius, k0 + radius + 1):

                                    p0 = p + q * m + r * m
                                    for s in range(ndiff):
                                        temp[s] = I[p, q, r, s] - mu[s]

                                    fast_vecmat_mul(temp, P, temp)

                                    for s in range(ndiff):
                                        temp[s] = temp[s] + mu[s]
                                        theta[p, q, r, s] += l0norm
                                        thetax[p, q, r, s] += temp[s] * l0norm

    out = np.divide(thetax, theta)

    return out
