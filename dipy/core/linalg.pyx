

cimport cython

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
cdef void fast_dgemm(double[:,::1] in_arr, double[:,::1] out_arr) nogil:
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
