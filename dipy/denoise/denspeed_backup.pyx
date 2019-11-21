
cimport cython

cimport safe_openmp as openmp
from safe_openmp cimport have_openmp
from cython.parallel import prange
from libc.math cimport sqrt, round
from libc.stdio cimport printf
from dipy.core.linalg cimport fast_matvec, fast_eig, fast_dgemm
import numpy as np

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def fast_mp_pca(arr, mask=None, patch_radius=2, return_sigma=False,
                out_dtype=None, num_threads=None):
    r"""Local PCA-based denoising of diffusion datasets.

    Parameters
    ----------
    arr : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.
    patch_radius : int (optional)
        The radius of the local patch to be taken around each voxel (in
        voxels). Default: 2 (denoise in blocks of 5x5x5 voxels).
    return_sigma : bool
        If true, a noise standard deviation estimate based on the
        Marcenko-Pastur distribution is returned [2]_.
        Default: False.
    out_dtype : str or dtype, optional
        The dtype for the output array. Default: output has the same dtype as
        the input.

    Returns
    -------
    denoised_arr : 4D array
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values
    noise_arr : 3D array
        Voxelwise standard deviation of the noise estimated from the data.
    sigma : float
        Mean value of noise standard deviations over all voxels (mean of
        noise_arr).

    References
    ----------
    .. [Veraart16] Veraart J, Fieremans E, Novikov DS (2016)
                  Diffusion MRI noise mapping using random matrix theory.
                  Magnetic resonance in Medicine 76(5), p1582-1593.
                  https://doi.org/10.1002/mrm.26059

    """
    if out_dtype is None:
        out_dtype = arr.dtype
    # We perform all computations in float64 precision but return the
    # results in the original input's precision
    arr = arr.astype(np.double)

    if not arr.ndim == 4:
        raise ValueError("PCA denoising can only be performed on 4D arrays.",
                         arr.shape)

    cdef:
        # OPENMP variables
        int all_cores = openmp.omp_get_num_procs()
        int threads_to_use = -1

        # looping and index variables
        int i,j,k,ii,jj,kk,v,vol,z, p,cnt, tmp, tmp0
        int sizes[3]                       # == arr.shape . Used because of nogil.

        # Denoising array dimension variables
        int mm = arr.shape[3]              # n_DWIs
        int c_patch_radius = patch_radius
        int nn = (2 * c_patch_radius + 1) ** 3   # number of voxels in local kernel
        int half_nn = int(nn/2.)
        int rr = mm if (mm<nn) else nn     # dimension of the square PCA matrix


        # for memory usage in OPENMP loops. We create a temp memory for each slice for each OPENMP thread.
        # There should be a better way to do this.

        # data matrix
        double[:, :, ::1] X = np.zeros([arr.shape[2], nn, mm], dtype=np.double)
        # eigenvector matrix
        double[:, :, ::1] C = np.zeros([arr.shape[2], rr, rr], dtype=np.double)
        # diagonal eigenvalue matrix
        double[:, :, ::1] diag_eigmat = np.zeros([arr.shape[2], rr, rr],
                                                 dtype=np.double)
        # eigenvalue array
        double[:, ::1] W = np.zeros([arr.shape[2], rr], dtype=np.double)
        # eigenvalue cummulative
        double[::1] cum_W = np.zeros([rr], dtype = np.double)
        double[::1] eigenV = np.zeros([rr], dtype = np.double)

        # temporary variables used in LAPACK calls
        double[:, ::1] temp0 = np.zeros([arr.shape[2], rr], dtype=np.double)
        double[:, ::1] temp1 = np.zeros([arr.shape[2], rr], dtype=np.double)
        double[:, ::1] temp2 = np.zeros([arr.shape[2], mm], dtype=np.double)
        double[:, ::1] temp3 = np.zeros([arr.shape[2], mm], dtype=np.double)
        int LWORK=2*(1 + 6 * rr + 2 * rr * rr)
        double[:, ::1] WORK = np.zeros([arr.shape[2], LWORK], dtype=np.double)
        int LIWORK=2*(3+5*rr)
        int[:, ::1] IWORK = np.zeros([arr.shape[2], LIWORK], dtype=np.int32)

        # variables for noise cutoff eigenvalue computations
        double gamma, sigma_hat, sigma2, rhs, r0
        int p_hat

        # for memoryviewing the original images
        double[:,:,:,:] arr_view = arr

    threads_to_use = num_threads or all_cores
    patch_size = 2 * patch_radius + 1
    if patch_size **3 < mm:
        return arr
    print("cy patch_size", patch_size, nn)

    if have_openmp:
        openmp.omp_set_dynamic(0)
        openmp.omp_set_num_threads(threads_to_use)

    sizes[0] = arr.shape[0]
    sizes[1] = arr.shape[1]
    sizes[2] = arr.shape[2]

    noise_arr = np.zeros([arr.shape[0],arr.shape[1],arr.shape[2]],dtype=np.float64)
    denoised_arr = np.zeros(arr.shape, dtype=np.float64)

    if mask is None:
        # If mask is not specified, use the whole volume
        mask = np.ones_like(arr, dtype=int)[..., 0]


    cdef double [:, :, :, :] denoised_arr_view = denoised_arr
    cdef double [:, :, :] noise_arr_view = noise_arr
    cdef int [:, :, :] mask_arr_view = mask.astype(np.int32)

    cdef int e = 0
    cdef double tamp = 0.
    cdef int m2 = 0

    # OPENMP loop over slices
    for k in prange(0, sizes[2], nogil=True, schedule=static):
        for j in range(0,sizes[1]):
            for i in range(0,sizes[0]):
                cum_W[:] = 0.

                if mask_arr_view[i, j, k] == 0:
                    printf(" %d, %d, %d", i,j,k)
                    continue
                # copy the local patch into array X
                cnt =- 1
                for ii in range(i - c_patch_radius, i + c_patch_radius + 1):
                    for jj in range(j - c_patch_radius, j + c_patch_radius + 1):
                         for kk in range(k - c_patch_radius, k + c_patch_radius + 1):
                            cnt=cnt+1
                            if((ii>=0) & (ii<sizes[0]) & (jj>=0) & (jj<sizes[1]) & (kk>=0) & (kk<sizes[2])):
                                for vol in range(mm):
                                    with cython.boundscheck(False):
                                        X[k,cnt,vol]=arr_view[ii,jj,kk,vol]


                # Matrix Multiplication X^TX or XX^T
                fast_dgemm(X[k,:, :], C[k,:, :])

                # Compute Eigen Value and Eigen Vector
                fast_eig(C[k,:, :], W[k,:], WORK[k,:], LWORK,IWORK[k,:],LIWORK)
                eigenV[:] = W[k,:]

                # Rearrange into Descending order
                eigenV[:] = eigenV[::-1]

                for p in range(rr):
                    eigenV[p] = eigenV[p] / nn

                # Initializing variables
                gamma = 0.
                sigma_hat = 0.
                rhs = 0.
                p_hat = 0

                # noise eigenvalue cutoff computation
                # Find non-positive eigen value index
                z = rr
                # Find cut-off index for non-positive eigen values
                for p in range(rr):
                    tamp = eigenV[p]
                    if eigenV[p] < 1e-18:
                        z = p
                        break

                m2 = z
                cum_W[m2-1]=0

                ########
                for p in range(m2-2):
                    print("%d", p)

                for p in range(m2-2):
                    v = m2 - 2 - p

                    cum_W[v] = cum_W[v+1]+ eigenV[v+1]


                # Find p_hat, cut-off index for noise.
                p_hat = z
                for p in range(m2-1):
                    r0 = m2-p-1
                    gamma = r0 / nn
                    sigma_hat = (eigenV[p+1] - eigenV[m2-1])/(4.)/ (sqrt(gamma))
                    rhs = r0 * sigma_hat
                    if cum_W[p] >= rhs:
                        p_hat = p
                        break

                print("p_hat:%d \n",p_hat)
                if m2 <= p_hat+ 1:
                    sigma2 = 0.
                else:
                    r0 = m2 - p_hat - 1
                    sigma2 = cum_W[p_hat + 1] / r0

                # Check if we have negative value
                if sigma2 < 0:
                    printf("r0 %f Sigma %f\n",r0, sigma2)

                # Reconstruct the images, by finding positive lambda (tmp0)
                tmp0 = rr - p_hat - 1
                v = rr
                # Nultify all lamda <= positive lamda,
                for p in range(tmp0):
                    diag_eigmat[k, p, p] = 0
                for p in range(tmp0, v):
                    diag_eigmat[k, p, p] = 1

                # Equivalent to equation [6] in the reference
                if mm <= nn:
                    fast_matvec(b't', C[k, :, :], X[k, half_nn, :], temp2[k,:])
                    fast_matvec(b'n', diag_eigmat[k, :, :], temp2[k, :], temp3[k, :])
                    fast_matvec(b'n', C[k, :, :], temp3[k, :], temp2[k, :])

                else:
                    for ii in range(rr):
                        temp1[k,ii]= C[k, ii, half_nn]

                    fast_matvec(b'n', diag_eigmat[k, :, :], temp1[k, :], temp0[k, :])
                    fast_matvec(b'n', C[k, :, :], temp0[k, :], temp1[k, :])
                    fast_matvec(b'n', X[k, :, :], temp1[k, :], temp2[k, :])

                denoised_arr_view[i, j, k, :] = temp2[k, :]
                noise_arr_view[i, j, k] = sqrt(sigma2)

    #sigma = np.mean(noise_arr[noise_arr != 0])
    if return_sigma:
        return denoised_arr.astype(out_dtype), noise_arr.astype(out_dtype)  # , sigma
    else:
        return denoised_arr.astype(out_dtype)


