
import numpy as np
cimport numpy as cnp
cimport cython

cimport safe_openmp as openmp
from safe_openmp cimport have_openmp
from cython.parallel import prange
from multiprocessing import cpu_count
from libc.math cimport sqrt
from math import floor
from scipy.linalg.cython_blas cimport dgemm
from scipy.linalg.cython_blas cimport dgemv
from scipy.linalg.cython_lapack cimport dsyevd
from scipy.linalg.cython_lapack cimport dlasrt


# Fast Matrix-Vector Multiplications
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void fast_matvec(char ta, double[:,::1] A, double[:] b,
                      double[:] y, double alpha=1.0, double beta=0.0,
                      int incx=1) nogil:
    r"""Performing Matrix - Vector Multiplication A*x or A.T*x.

    This function dgemv() function from LAPACK, originally function can perform
    y := alpha*A*x + beta*y,   or   y := alpha*A.T*x + beta*y

    Parameters
    ----------
    ta : string
        Apply transpose to input matrix
        'n' = no transpose, 't' = is transpose
    A : ndarray
        Matrix A
    b : ndarray (N, 3)
        vector
    y : ndarray,
        Matrix y (zeros)
    alpha : float
        (default=1.0)
    beta : float
        (default=0.0)

    Returns
    -------
    y : int
        y = A*x or A.T*x

    Notes
    -----
    For more info: Look up LAPACK dgemv() function

    """
    cdef:
        char transa
        int m,n
        double *a0=&A[0,0]
        double *b0=&b[0]
        double *y0=&y[0]

    if ta == b'n':
        transa=b'n'
        n= A.shape[0]
        m= A.shape[1]

        dgemv(&transa, &m , &n, &alpha, a0, &m,  b0, &incx, &beta, y0, &incx)
    else:
        transa=b't'
        n= A.shape[0]
        m= A.shape[1]
        dgemv(&transa, &m , &n, &alpha, a0, &m,  b0, &incx, &beta, y0, &incx)


# Fast Computing Eigen Values
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cdef void fast_eig(double[:,::1] arr, double[::1] out_w, double[::1] out_work,
                   int lwork, int[::1] out_iwork, int liwork) nogil:
    """Computes all eigenvalues and, optionally, eigenvectors of a real symmetric matrix A.

    If eigenvectors are desired, it uses a divide and conquer algorithm. Using
    method dsyevd() from LAPACK return eigen value in ascending order.

    Parameters
    ----------
    arr : array
        Matrix A to compute eigen val and eigen vec (in). We reuse this array to write
        the Orthogonal Eigen vector (out)
    out_w : array,
        Eigen Value
    out_work : arrray
        return the optimal LWORK
    lwork : int
        if JOBZ = V,and N > 1 then  int LWORK > 1 + 6*N + 2*N**2.
    out_iwork : array
        return the optimal LIWORK
    liwork : int
         if JOBZ = V,and N > 1 then  int LIWORK > 3+5*N.

    Notes
    -----
    - JOBZ :"V" for compute eigen values and eigen vectors (default)
           :"N" for compute eigen values only
    - UPLO :'U' for Upper triangle of A is stored; (default)
           :'L' for Lower triangle of A is stored.
    """
    cdef:
        char JOBA=b'D'
        char JOBZ=b'V'
        char UPLO=b'U'
        int incx=1
        # Matrix Order
        int N = arr.shape[0]
        double *a0=&arr[0,0]
        double *w0=&out_w[0]
        double *work0=&out_work[0]
        int *iwork0=&out_iwork[0]

        int lda=N
        int lw=lwork
        int liw= liwork
        int info

    # Output compute is in Ascending Order
    dsyevd( &JOBZ, &UPLO, &N, a0, &lda, w0,work0,&lwork, iwork0,&liw,&info)
    # Using dlasrt to turn sort data into Descending Order
    dlasrt ( &JOBA, &N, w0, &info)


# Fast Matrix-Matrix Multiplication
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef void fast_dgemm(double[:,::1] in_arr, double[:,::1] out_arr) nogil:
    r"""Performs matrix multiplication (a*a.T or a.T*a).

    Parameters
    ----------
    in_arr: array
        2D Matrix
    out_arr: array
        2D Matrix to store result

"""
    cdef:
        char transa
        char transb
        int m2, n2, m,n,k
        double alpha=1.0
        double beta=0.0
        double *a0=&in_arr[0,0]
        double *c0=&out_arr[0,0]

    m2= in_arr.shape[0]
    n2= in_arr.shape[1]
    if m2 <= n2:           #a*a.T
        transa=b't'
        transb=b'n'
        m=m2
        n=m2
        k=n2
        dgemm(&transa, &transb, &m , &n, &k, &alpha, a0, &k, a0,
               &k, &beta, c0, &m)
    else:                # a.T*a
        transa=b'n'
        transb=b't'
        m=n2
        n=n2
        k=m2
        dgemm(&transa, &transb, &m , &n, &k, &alpha, a0, &m, a0,
               &n, &beta, c0, &m)


@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def randomlpca_denoise(arr, patch_extent=0, out_dtype=None,num_threads=None):
    r"""Local PCA-based denoising of diffusion datasets.

    Parameters
    ----------
    arr : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.
    patch_extent : int, optional
        The diameter of the local patch to be taken around each voxel (in
        voxels). The radius will be half of this value. If not provided,
        the default will be automatically computed as:

        .. math ::

                patch_extent = max(5,\lfloor N^{1/3} \rfloor)

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
    if arr.dtype == np.float32:
        calc_dtype = np.double
    else:
        calc_dtype = np.double

    arr=arr.astype(calc_dtype)

    if not arr.ndim == 4:
        raise ValueError("PCA denoising can only be performed on 4D arrays.",
                         arr.shape)

    if patch_extent <= 0:
        Nvols = arr.shape[3]
        patch_extent = max(5,Nvols ** (1. / 3.))

    cdef:
        #OPENMP variables
        int all_cores = openmp.omp_get_num_procs()
        int threads_to_use = -1

        #looping and index variables
        int i,j,k,ii,jj,kk,v,vol,z, p,cnt, tmp, tmp0

        int sizes[3]                       # == arr.shape . Used because of nogil.

        #denoising array dimension variables
        int patch_radius = int(floor(patch_extent/2.))
        int mm = arr.shape[3]              # n_DWIs
        int nn = (2*patch_radius+1) ** 3   # number of voxels in local kernel
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
        double[:,:,:,:] arr_view =arr

    if num_threads is not None:
        threads_to_use = num_threads
    else:
        threads_to_use = all_cores

    if have_openmp:
        openmp.omp_set_dynamic(0)
        openmp.omp_set_num_threads(threads_to_use)

    sizes[0] = arr.shape[0]
    sizes[1] = arr.shape[1]
    sizes[2] = arr.shape[2]

    noise_arr = np.zeros([arr.shape[0],arr.shape[1],arr.shape[2]],dtype=np.float64)
    denoised_arr = np.zeros(arr.shape, dtype=np.float64)


    cdef double [:,:,:,:] denoised_arr_view =denoised_arr
    cdef double [:,:,:] noise_arr_view =noise_arr

    # OPENMP loop over slices
    for k in prange(0, sizes[2], nogil=True, schedule=static):
        for j in range(0,sizes[1]):
            for i in range(0,sizes[0]):
                cum_W[:] = 0.

                # copy the local patch into array X
                cnt=-1
                for ii in range(i-patch_radius, i + patch_radius + 1):
                    for jj in range(j-patch_radius, j + patch_radius + 1):
                         for kk in range(k-patch_radius, k + patch_radius + 1):
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


                # Initializing variables
                gamma = 0.
                sigma_hat = 0.
                rhs = 0.
                p_hat = 0

                # noise eigenvalue cutoff computation
                # Find non-positive eigen value index
                z = rr
                for p in range(rr):
                    eigenV[p] = eigenV[p] / nn

                # Find cut-off index for non-positive eigen values
                for p in range(rr):
                    if eigenV[p] <= 0:
                        z = p
                        break

                for p in range(z-2):
                    v = z - 2 - p
                    cum_W[v] = cum_W[v+1]+ eigenV[v+1]

                # Find p_hat, cut-off index for noise.
                p_hat = z
                for p in range(z-1):
                    r0 = z-p-1
                    gamma = r0 / nn
                    sigma_hat = (eigenV[p+1] - eigenV[z-1])/(4. * sqrt(gamma))
                    rhs = r0 * sigma_hat
                    if cum_W[p] >= rhs:
                        p_hat = p
                        break

                # Noise images
                if p_hat == p - 1:
                    sigma2 = 0.
                else:
                    r0 = p - p_hat - 1
                    sigma2 = cum_W[p_hat] / r0
                noise_arr_view[i, j, k] = sqrt(sigma2)

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

    sigma = np.mean(noise_arr[noise_arr != 0])
    return denoised_arr.astype(out_dtype), noise_arr.astype(out_dtype), sigma


