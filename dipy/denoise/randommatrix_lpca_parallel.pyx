#!python

import numpy as np
import time
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

# Fast Matrix-Vector Multiplications from Lapack
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef int fast_matvec(char ta, double[:,::1] A, double[:] b, double[:] y, double alpha=1.0, double beta=0.0,int incx=1) nogil except -1:

    cdef:
        char transa
        int m,n
        double *a0=&A[0,0]
        double *b0=&b[0]
        double *y0=&y[0]

    if ta == 'n':
        transa='n'
        n= A.shape[0]
        m= A.shape[1]

        dgemv(&transa, &m , &n, &alpha, a0, &m,  b0, &incx, &beta, y0, &incx)
    else:
        transa='t'
        n= A.shape[0]
        m= A.shape[1]
        dgemv(&transa, &m , &n, &alpha, a0, &m,  b0, &incx, &beta, y0, &incx)

    return 0

# Fast Computing Eigen Values Multiplication from Lapack
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
cpdef int fast_eig(double[:,::1] a, double[::1] W, double[::1] WORK, int LWORK, int[::1] IWORK, int LIWORK) nogil except -1:

    cdef:
        char JOBA='D'
        char JOBZ='V'
        char UPLO='U'
        int incx=1
        int N = a.shape[0]
        double *a0=&a[0,0]
        double *w0=&W[0]
        double *work0=&WORK[0]
        int *iwork0=&IWORK[0]

        int lda=N
        int lw=LWORK
        int liw= LIWORK
        int info

    # Output compute is in Ascending Order
    dsyevd( &JOBZ, &UPLO, &N, a0, &lda, w0,work0,&LWORK, iwork0,&liw,&info)
    # Inverse the result to Descending Order
    dlasrt ( &JOBA, &N, w0, &info)
    return 0

# Fast Matrix-Matrix Multiplication
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
@cython.nonecheck(False)
cpdef int fast_dgemm(double[:,::1] a, double[:,::1] c) nogil except -1:

    cdef:
        char transa
        char transb
        int m2, n2, m,n,k
        double alpha=1.0
        double beta=0.0
        double *a0=&a[0,0]
        double *c0=&c[0,0]

    m2= a.shape[0]
    n2= a.shape[1]

    if m2 <= n2:           # XXt
        transa='t'
        transb='n'
        m=m2
        n=m2
        k=n2

        dgemm(&transa, &transb, &m , &n, &k, &alpha, a0, &k, a0,
               &k, &beta, c0, &m)
    else:                # XtX
        transa='n'
        transb='t'
        m=n2
        n=n2
        k=m2

        dgemm(&transa, &transb, &m , &n, &k, &alpha, a0, &m, a0,
               &n, &beta, c0, &m)

    return 0



@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
def randommatrix_lpca_parallel(arr, patch_extent=0, out_dtype=None,num_threads=None):
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
    start_time = time.time()

    if out_dtype is None:
        out_dtype = arr.dtype

    # We perform all computations in float64 precision but return the
    # results in the original input's precision
    if arr.dtype == np.float64:
        calc_dtype = np.float64
    else:
        calc_dtype = np.float32
        arr=arr.astype(np.float64)

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
        int rr = mm if (mm<nn) else nn     # dimension of the square PCA matrix

        # for memory usage in OPENMP loops. We create a temp memory for each slice for each OPENMP thread.
        # There should be a better way to do this.
        double[:,:,::1] X = np.zeros([arr.shape[2],nn,mm], dtype=np.double)    # data matrix
        double[:,:,::1] C = np.zeros([arr.shape[2],rr,rr], dtype=np.double)    # eigenvector matrix
        double[:,:,::1] diag_eigmat = np.zeros([arr.shape[2],rr,rr], dtype=np.double)  # diagonal eigenvalue matrix
        double[:,::1] W = np.zeros([arr.shape[2],rr], dtype=np.double)         # eigenvalue array

        double[::1] cum_W = np.zeros([rr], dtype = np.double)                  # eigenvalue cummulative
        double[::1] eigenV = np.zeros([rr], dtype = np.double)

        # temporary variables used in LAPACK calls
        double[:,::1] temp0 = np.zeros([arr.shape[2],rr], dtype=np.double)
        double[:,::1] temp1 = np.zeros([arr.shape[2],rr], dtype=np.double)
        double[:,::1] temp2 = np.zeros([arr.shape[2],mm], dtype=np.double)
        double[:,::1] temp3 = np.zeros([arr.shape[2],mm], dtype=np.double)
        int LWORK=2*(1+6*rr+2*rr*rr)
        double[:,::1] WORK = np.zeros([arr.shape[2],LWORK], dtype=np.double)
        int LIWORK=2*(3+5*rr)
        int[:,::1] IWORK = np.zeros([arr.shape[2],LIWORK], dtype=np.int32)

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


                # X^TX or XX^T
                fast_dgemm(X[k,:, :], C[k,:, :])
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
                    eigenV[p] = eigenV[p]/nn

                for p in range(rr):
                    if eigenV[p] <= 0:
                        z = p
                        break
		
		# Cummulatively sum up eigen values.
                for p in range(z-2):
                    v = z - 2 - p
                    cum_W[v] = cum_W[v+1]+ eigenV[v+1]

		# Finding p_hat and sigma_hat_sq in eq [10-12]
                p_hat = z
                for p in range(z-1):
                    r0 = z-p-1
                    gamma = r0 /nn
                    sigma_hat = (eigenV[p+1] - eigenV[z-1])/(4. * sqrt(gamma))
                    rhs = r0 * sigma_hat
                    if cum_W[p] >= rhs:
                        p_hat = p
                        break

		# If p_hat equal to p-1, there is no noise signal
                if p_hat == p - 1:
                    sigma2 = 0.
                else:
                    r0 = p - p_hat - 1
                    sigma2 = cum_W[p_hat] / r0

                noise_arr_view[i,j,k]=sqrt(sigma2)
                
		# Reconstruct the denoised image
                tmp0 = rr - p_hat - 1
                v = rr
		
		# Creating diagonal eigen matrix.
                for p in range(tmp0):
                    diag_eigmat[k,p,p]=0
                for p in range(tmp0,v):
                    diag_eigmat[k,p,p]=1
		
		# Equation [6], perform matrix multiplication
                if mm<=nn:
                    fast_matvec('t',C[k,:,:], X[k,nn/2,:], temp2[k,:])
                    fast_matvec('n',diag_eigmat[k,:,:],temp2[k,:],temp3[k,:])
                    fast_matvec('n',C[k,:,:],temp3[k,:],temp2[k,:])

                else:
                    for ii in range(rr):
                        temp1[k,ii]= C[k,ii,nn/2]

                    fast_matvec('n',diag_eigmat[k,:,:], temp1[k,:] ,  temp0[k,:])
                    fast_matvec('n',C[k,:,:], temp0[k,:],   temp1[k,:])
                    fast_matvec('n',X[k,:,:], temp1[k,:],   temp2[k,:])

                denoised_arr_view[i,j,k,:] = temp2[k,:]
                noise_arr_view[i,j,k] = sqrt(sigma2)

    sigma = np.mean(noise_arr[noise_arr != 0])
    return denoised_arr.astype(calc_dtype), noise_arr.astype(calc_dtype), sigma


