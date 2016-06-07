import numpy as np
cimport cython
cimport numpy as cnp

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_outer_prod(double[:,:] out, double[:] x, int utype)nogil:
    cdef:
        cnp.npy_intp i, j, n
    n = x.shape[0]
    if utype==0:
        for i in range(n):
            for j in range(n):
                out[i,j] = x[i] * x[j]
    elif utype==1:
        for i in range(n):
            for j in range(n):
                out[i,j] += x[i] * x[j]
    elif utype==-1:
        for i in range(n):
            for j in range(n):
                out[i,j] -= x[i] * x[j]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_vector(double[:] out, double[:] x, int utype)nogil:
    cdef:
        cnp.npy_intp i, n
    n = x.shape[0]
    if utype==0:
        for i in range(n):
            out[i] = x[i]
    elif utype==1:
        for i in range(n):
            out[i] += x[i]
    elif utype==-1:
        for i in range(n):
            out[i] -= x[i]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef void update_matrix(double[:,:] out, double[:,:] cov, int utype)nogil:
    cdef:
        cnp.npy_intp i, j, n
    n = cov.shape[0]
    if utype==0:
        for i in range(n):
            for j in range(n):
                out[i,j] = cov[i, j]
    elif utype==1:
        for i in range(n):
            for j in range(n):
                out[i,j] += cov[i, j]
    elif utype==-1:
        for i in range(n):
            for j in range(n):
                out[i,j] -= cov[i, j]



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def fast_local_covariance(double[:,:,:,:] I, int radius, double[:,:,:,:,:] out, double[:,:,:,:] meanout):
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
        cnp.npy_intp cur_i, prev_i, i, j, k, ii, nsamples
        double[:,:,:,:,:] T = np.zeros((2, n1, n2, ndiff, ndiff))
        double[:,:,:,:] S = np.zeros((2, n1, n2, ndiff))
        double[:,:] cov = np.empty((ndiff, ndiff))
        double[:] mu = np.empty(ndiff)
        double[:] x

    out[...] = 0
    meanout[...] = 0

    with nogil:
        nsamples = m * m * m
        cur_i = 1
        for i in range(n0):
            cur_i = 1 - cur_i
            prev_i = 1 - cur_i
            for j in range(n1):
                for k in range(n2):
                    x = I[i,j,k, :]
                    # Start with last corner
                    update_outer_prod(cov, x, 0)# q=(0,0,0)
                    update_vector(mu, x, 0)# q=(0,0,0)

                    # Add signed rectangles
                    if i>0:
                        update_matrix(cov, T[prev_i,j,k], 1)# q=(1, 0, 0)
                        update_vector(mu, S[prev_i,j,k], 1)# q=(1, 0, 0)
                        if j>0:
                            update_matrix(cov, T[prev_i,j-1,k], -1)# q=(1, 1, 0)
                            update_vector(mu, S[prev_i,j-1,k], -1)# q=(1, 1, 0)
                            if k>0:
                                update_matrix(cov, T[prev_i,j-1,k-1], 1)# q=(1, 1, 1)
                                update_vector(mu, S[prev_i,j-1,k-1], 1)# q=(1, 1, 1)
                        if k>0:
                            update_matrix(cov, T[prev_i,j,k-1], -1)# q=(1, 0, 1)
                            update_vector(mu, S[prev_i,j,k-1], -1)# q=(1, 0, 1)
                    if j>0:
                        update_matrix(cov, T[cur_i,j-1,k], 1)# q=(0, 1, 0)
                        update_vector(mu, S[cur_i,j-1,k], 1)# q=(0, 1, 0)
                        if k>0:
                            update_matrix(cov, T[cur_i,j-1,k-1], -1)# q=(0, 1, 1)
                            update_vector(mu, S[cur_i,j-1,k-1], -1)# q=(0, 1, 1)
                    if k>0:
                        update_matrix(cov, T[cur_i,j,k-1], 1)# q=(0, 0, 1)
                        update_vector(mu, S[cur_i,j,k-1], 1)# q=(0, 0, 1)

                    # Add displaced signed corners
                    if i>=m:
                        x = I[i-m,j,k]# q=(1, 0, 0)
                        update_outer_prod(cov, x, -1)
                        update_vector(mu, x, -1)
                        if j>=m:
                            x = I[i-m,j-m,k]# q=(1, 1, 0)
                            update_outer_prod(cov, x, 1)
                            update_vector(mu, x, 1)
                            if k>=m:
                                x = I[i-m,j-m,k-m]# q=(1, 1, 1)
                                update_outer_prod(cov, x, -1)
                                update_vector(mu, x, -1)
                        if k>=m:
                            x = I[i-m,j,k-m]# q=(1, 0, 1)
                            update_outer_prod(cov, x, 1)
                            update_vector(mu, x, 1)
                    if j>=m:
                        x = I[i,j-m,k]# q=(0, 1, 0)
                        update_outer_prod(cov, x, -1)
                        update_vector(mu, x, -1)
                        if k>=m:
                            x = I[i,j-m,k-m]# q=(0, 1, 1)
                            update_outer_prod(cov, x, 1)
                            update_vector(mu, x, 1)
                    if k>=m:
                        x = I[i,j,k-m]# q=(0, 0, 1)
                        update_outer_prod(cov, x, -1)
                        update_vector(mu, x, -1)

                    # Save current integrals for future reference
                    update_matrix(T[cur_i, j, k], cov, 0)
                    update_vector(S[cur_i, j, k], mu, 0)

                    # Use integral of current rectangle
                    # Last corner is (i, j, k) => center is (i-radius, j-radius, k-radius)
                    if (i>=m-1) and (j>=m-1) and (k>=m-1):
                        # Divide by the number of samples
                        for ii in range(ndiff):
                            mu[ii]/= nsamples
                            for jj in range(ndiff):
                                cov[ii,jj] /= nsamples
                        # Subtract self-outer product of mu from cov
                        update_outer_prod(cov, mu, -1)

                        #Local covariance ready
                        update_matrix(out[i-radius, j-radius, k-radius], cov, 0)
                        # Update the local mean
                        update_vector(meanout[i-radius, j-radius, k-radius], mu, 0)
