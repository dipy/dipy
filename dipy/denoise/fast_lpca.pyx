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
def fast_lpca(double[:,:,:,:] I, int radius, double[:,:,:,:] sigma):
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

    theta = np.zeros((n0, n1, n2, ndiff))
    thetax = np.zeros((n0, n1, n2, ndiff))
    tou = np.array(sigma) * 2.3 * 2.3
    # out[...] = 0
    # meanout[...] = 0
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
                        # update_matrix(out[i-radius, j-radius, k-radius], cov, 0)
                        # update_vector(meanout[i-radius, j-radius, k-radius], mu, 0)

                    with gil:
                        # eigen value decomposition
                        [d,W] = np.linalg.eigh(cov)
                        
                        if((i >= radius and i < n0-radius) and (j >= radius and j < n1-radius) and (k >= radius and k < n2-radius)):
                        # make the projection and reconstruct
                            temp = I[i - radius: i + radius + 1,
                                       j - radius: j + radius + 1,
                                       k - radius: k + radius + 1, :]
                            temp = np.array(temp)
                            
                            X = temp.reshape(m * m * m, ndiff)
                            X = X - np.array([mu, ] * X.shape[0],
                                             dtype=np.float64)

                            d[d < tou[i, j, k, :]] = 0
                            # Y = X.dot(W)
                            W_hat = W * 0
                            W_hat[:, d > 0] = W[:, d > 0]

                            Y = X.dot(W_hat)
                            X_est = Y.dot(np.transpose(W_hat))

                            # add the mean
                            temp = X_est + \
                                np.array([mu, ] * X_est.shape[0], dtype=np.float64)
                            # Get the block 
                            temp = temp.reshape(m, m, m, ndiff)
                            
                            # Also update the estimate matrix which is X_est * theta
                            # average using the blockwise method

                            theta[i - radius: i + radius + 1,
                                  j - radius: j + radius + 1,
                                  k - radius: k + radius + 1,
                                  :] = theta[i - radius: i + radius + 1,
                                             j - radius: j + radius + 1,
                                             k - radius: k + radius + 1,
                                             :] +  1.0 / (1.0 + np.linalg.norm(d,
                                                                  ord=0))

                            thetax[i - radius: i + radius + 1,
                                   j - radius: j + radius + 1,
                                   k - radius: k + radius + 1,
                                   :] = thetax[i - radius: i + radius + 1,
                                               j - radius: j + radius + 1,
                                               k - radius: k + radius + 1,
                                               :] + temp / (1 + np.linalg.norm(d,ord=0))
                        

    out = thetax / theta

    return out