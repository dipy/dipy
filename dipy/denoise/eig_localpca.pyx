# cython: boundscheck=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: wraparound=False
"""
Optimized Cython implementation of the local PCA denoising core used in DIPY.

This module provides Cython-accelerated routines for the computationally
intensive parts of local PCA denoising, in particular a triple nested loop
that applies eigenvalue-based PCA to 3D patches of 4D input data.
"""

import numpy as np
cimport numpy as cnp
from numpy cimport ndarray
import cython

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

from scipy.linalg.cython_lapack cimport ssyev, dsyev
from scipy.linalg.cython_blas cimport sgemm, dgemm

ctypedef cython.floating f_t
ctypedef cnp.float32_t f32
ctypedef cnp.float64_t f64
ctypedef cnp.uint8_t  u8


cnp.import_array()


cdef inline f_t mean_from_start(const f_t* a, int n) noexcept nogil:
    """
    Return the mean of the first ``n`` values in ``a``.
    """
    
    cdef int i
    cdef f_t s = 0
    for i in range(n):
        s += a[i]
    return s / n


cdef inline void pca_classifier(const f_t[:] evals, int n_evals, int nvoxels,
                                   f_t* out_var) noexcept nogil:
    """
    Classify which PCA eigenvalues are related to noise and estimate the
    noise variance.

    Parameters
    ----------
    evals : 1D typed memoryview
        Array containing the PCA eigenvalues in ascending order.
    n_evals : int
        Number of available eigenvalues.
    nvoxels : int
        Number of voxels used to compute the eigenvalues.
    out_var : pointer to f_t
        Output location for the estimated noise variance.

    Notes
    -----
    This is based on the algorithm described in :footcite:p:`Veraart2016c`.

    References
    ----------
    .. footbibliography::
    """

    cdef int start = 0
    if n_evals > nvoxels - 1:
        start = n_evals - (nvoxels - 1)
        n_evals = nvoxels - 1

    cdef f_t var = mean_from_start(&evals[start], n_evals)
    cdef int c = n_evals - 1
    cdef f_t r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    while r > 0.0:
        var = mean_from_start(&evals[start], c)
        c = c - 1
        r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    out_var[0] = var


cdef inline void compute_mean_center(f_t[:, :] X, f_t[:] M, int n_samples, int N) noexcept nogil:
    """
    Compute the column-wise mean of ``X`` and subtract it in place.

    Parameters
    ----------
    X : 2D typed memoryview
        Input patch matrix of shape ``(n_samples, N)``. Modified in place.
    M : 1D typed memoryview
        Output buffer for the column means.
    n_samples : int
        Number of rows in ``X``.
    N : int
        Number of columns in ``X``.
    """

    cdef int s, t
    for t in range(N):
        M[t] = 0.0
    for s in range(n_samples):
        for t in range(N):
            M[t] += X[s, t]
    for t in range(N):
        M[t] /= n_samples
    for s in range(n_samples):
        for t in range(N):
            X[s, t] -= M[t]


cdef inline void build_cov(f_t[:, :] X, f_t[:, :] C, int ns, int N) noexcept nogil:
    """
    Build the covariance matrix ``C = X^T X / ns``.

    Parameters
    ----------
    X : 2D typed memoryview
        Mean-centered patch matrix of shape ``(ns, N)``.
    C : 2D typed memoryview
        Output covariance matrix of shape ``(N, N)``.
    ns : int
        Number of samples in the patch.
    N : int
        Number of signal dimensions.
    """

    cdef float alpha32, beta32
    cdef double alpha64, beta64

    # This is the low-level equivalent of: C = X^T X / ns
    if f_t is float:
        alpha32 = <float>(1.0 / ns)
        beta32 = 0.0
        sgemm(b'N', b'T',
              &N, &N, &ns,
              &alpha32,
              &X[0, 0], &N,
              &X[0, 0], &N,
              &beta32,
              &C[0, 0], &N)
    else:
        alpha64 = 1.0 / ns
        beta64 = 0.0
        dgemm(b'N', b'T',
              &N, &N, &ns,
              &alpha64,
              &X[0, 0], &N,
              &X[0, 0], &N,
              &beta64,
              &C[0, 0], &N)


cdef inline void reconstruct(
    f_t[:, :] X,
    const f_t[:] M,
    const f_t* W,
    int ns, int N,
    int n_signal,
    f_t[:, :] Yt
) noexcept nogil:
    """
    Reconstruct ``X`` from the retained PCA signal components.

    Parameters
    ----------
    X : 2D typed memoryview
        Mean-centered patch matrix of shape ``(ns, N)``. Overwritten in place
        with the reconstructed patch.
    M : 1D typed memoryview
        Mean vector of length ``N``.
    W : pointer
        Pointer to the retained eigenvectors.
    ns : int
        Number of samples in the patch.
    N : int
        Number of signal dimensions.
    n_signal : int
        Number of retained signal components.
    Yt : 2D typed memoryview
        Temporary buffer.
    """

    cdef int t, s
    cdef float alpha32 = 1.0
    cdef float beta32 = 0.0
    cdef double alpha64 = 1.0
    cdef double beta64 = 0.0

    if n_signal <= 0:
        for s in range(ns):
            for t in range(N):
                X[s, t] = M[t]
        return

    if f_t is float:
        # Equivalent to Y = W^T X^T = (X W)^T. Necessary since sgemm assumes column-major.
        # W is already column major, but X isn't.
        sgemm(b'T', b'N',
              &n_signal, &ns, &N,
              &alpha32,
              <float*>W, &N,
              <float*>&X[0, 0], &N,
              &beta32,
              <float*>&Yt[0, 0], &n_signal)

        # Equivalent to doing W Y = W (X W)^T = W W^T X^T
        # Writing the output back to row-major (C-order), we have X W W^T
        sgemm(b'N', b'N',
              &N, &ns, &n_signal,
              &alpha32,
              <float*>W, &N,
              <float*>&Yt[0, 0], &n_signal,
              &beta32,
              <float*>&X[0, 0], &N)
    else:
        # Equivalent to Y = W^T X^T = (X W)^T. Necessary since dgemm assumes column-major.
        # W is already column major, but X isn't.
        dgemm(b'T', b'N',
              &n_signal, &ns, &N,
              &alpha64,
              <double*>W, &N,
              <double*>&X[0, 0], &N,
              &beta64,
              <double*>&Yt[0, 0], &n_signal)

        # Equivalent to doing W Y = W (X W)^T = W W^T X^T
        # Writing the output back to row-major (C-order), we have X W W^T
        dgemm(b'N', b'N',
              &N, &ns, &n_signal,
              &alpha64,
              <double*>W, &N,
              <double*>&Yt[0, 0], &n_signal,
              &beta64,
              <double*>&X[0, 0], &N)

    for s in range(ns):
        for t in range(N):
            X[s, t] += M[t]


# Main loop (float32/float64)
cdef void genpca_loop(
    f_t[:, :, :, :] data,
    u8[:, :, :] mask,
    f_t[:, :, :, :] theta,
    f_t[:, :, :, :] thetax,
    bint estimate_sigma,
    f_t[:, :, :] var_map,         # only valid if estimate_sigma==False
    bint return_sigma,
    f_t[:, :, :] var_acc,         # only valid if return_sigma and estimate_sigma
    f_t[:, :, :] thetavar,        # only valid if return_sigma and estimate_sigma
    int patch_radius_x, int patch_radius_y, int patch_radius_z,
    f_t tau_factor
) noexcept nogil:
    """
    Main nogil loop for local PCA denoising.

    Parameters
    ----------
    data : 4D typed memoryview
        Array of data to be denoised.
    mask : 3D typed memoryview
        A mask with voxels that are true inside the brain and false outside of
        it.
    theta : 4D memoryview
        Accumulator for weights.
    thetax : 4D memoryview
        Accumulator for weighted reconstructed signal.
    estimate_sigma : bool
        Whether to estimate local noise variance from the eigenvalue spectrum.
    var_map : 3D memoryview
        Input noise variance map if ``estimate_sigma`` is False.
    return_sigma : bool
        Whether variance estimates should also be accumulated for output.
    var_acc : 3D memoryview
        Weighted variance accumulator.
    thetavar : 3D memoryview
        Variance weight accumulator.
    patch_radius_x, patch_radius_y, patch_radius_z : int
        Patch radius along each spatial dimension.
    tau_factor : float
        Thresholding of PCA eigenvalues.
    """
    
    cdef Py_ssize_t Xdim = data.shape[0]
    cdef Py_ssize_t Ydim = data.shape[1]
    cdef Py_ssize_t Zdim = data.shape[2]
    cdef int N = <int>data.shape[3]

    cdef int size_x = 2*patch_radius_x + 1
    cdef int size_y = 2*patch_radius_y + 1
    cdef int size_z = 2*patch_radius_z + 1
    cdef int n_samples = size_x*size_y*size_z

    cdef f_t[:, ::1] X
    cdef f_t[::1] M
    cdef f_t[::1, :] C
    cdef f_t[::1] d
    cdef f_t[:, ::1] proj_buf
    cdef f_t[::1] work

    cdef int info
    cdef int lwork = -1 # query optimal workspace first

    cdef float work_query32[1]
    cdef double work_query64[1]

    # Memory layout:
    # - X/M/proj_buf/work are C-order.
    # - C is F-order because ssyev/dsyev expect column-major.
    with gil:
        if f_t is float:
            X = np.empty((n_samples, N), dtype=np.float32)
            M = np.empty((N,), dtype=np.float32)
            C = np.empty((N, N), dtype=np.float32, order='F')
            d = np.empty((N,), dtype=np.float32)
            proj_buf = np.empty((n_samples, N), dtype=np.float32)
            
            ssyev(b'V', b'L',
              &N,
              &C[0, 0], &N,
              &d[0],
              &work_query32[0], &lwork,
              &info)

            lwork = <int>work_query32[0]
            work = np.empty((lwork,), dtype=np.float32)

        else:
            X = np.empty((n_samples, N), dtype=np.float64)
            M = np.empty((N,), dtype=np.float64)
            C = np.empty((N, N), dtype=np.float64, order='F')
            d = np.empty((N,), dtype=np.float64)
            proj_buf = np.empty((n_samples, N), dtype=np.float64)

            dsyev(b'V', b'L',
                &N,
                &C[0, 0], &N,
                &d[0],
                &work_query64[0], &lwork,
                &info)

            lwork = <int>work_query64[0]
            work = np.empty((lwork,), dtype=np.float64)
            
    cdef Py_ssize_t i, j, k
    cdef int dx, dy, dz
    cdef Py_ssize_t ii, jj, kk
    cdef int s, t
    cdef f_t this_var, tau, this_theta
    cdef int ncomps, n_signal
    cdef const f_t* W

    for k in range(patch_radius_z, Zdim - patch_radius_z):
        for j in range(patch_radius_y, Ydim - patch_radius_y):
            for i in range(patch_radius_x, Xdim - patch_radius_x):
                if mask[i, j, k] == 0:
                    continue

                s = 0
                for dx in range(-patch_radius_x, patch_radius_x + 1):
                    ii = i + dx
                    for dy in range(-patch_radius_y, patch_radius_y + 1):
                        jj = j + dy
                        for dz in range(-patch_radius_z, patch_radius_z + 1):
                            kk = k + dz
                            for t in range(N):
                                X[s, t] = data[ii, jj, kk, t]
                            s += 1

                compute_mean_center[f_t](X, M, n_samples, N)
                build_cov[f_t](X, C, n_samples, N)

                # LAPACK eigendecomposition (column-major)
                if f_t is float:
                    ssyev(b'V', b'L',
                          &N,
                          &C[0, 0], &N,
                          &d[0],
                          &work[0], &lwork,
                          &info)
                else:
                    dsyev(b'V', b'L',
                          &N,
                          &C[0, 0], &N,
                          &d[0],
                          &work[0], &lwork,
                          &info)
                
                if info != 0:
                    with gil:
                        if info < 0:
                            raise np.linalg.LinAlgError(
                                f"Illegal value in argument {-info} of internal syev"
                            )
                        else:
                            raise np.linalg.LinAlgError(
                                "The algorithm failed to converge; "
                                f"{info} off-diagonal elements of an intermediate "
                                "tridiagonal form did not converge to zero."
                            )

                if estimate_sigma:
                    pca_classifier[f_t](d, N, n_samples, &this_var)
                else:
                    this_var = var_map[i, j, k]

                tau = (tau_factor * tau_factor) * this_var

                ncomps = 0
                for t in range(N):
                    if d[t] < tau:
                        ncomps += 1
                    else:
                        break

                n_signal = N - ncomps
                W = &C[0, 0] + ncomps * N

                reconstruct[f_t](X,
                                 M,
                                 W,
                                 n_samples, N,
                            n_signal,
                            proj_buf)

                this_theta = <f_t>(1.0 / (1.0 + N - ncomps))

                s = 0
                for dx in range(-patch_radius_x, patch_radius_x + 1):
                    ii = i + dx
                    for dy in range(-patch_radius_y, patch_radius_y + 1):
                        jj = j + dy
                        for dz in range(-patch_radius_z, patch_radius_z + 1):
                            kk = k + dz

                            for t in range(N):
                                theta[ii, jj, kk, t] += this_theta
                                thetax[ii, jj, kk, t] += X[s, t] * this_theta

                            if return_sigma and estimate_sigma:
                                var_acc[ii, jj, kk] += this_var * this_theta
                                thetavar[ii, jj, kk] += this_theta

                            s += 1


cdef tuple run_core_f32(
    ndarray data,
    ndarray mask,
    bint estimate_sigma,
    ndarray var_map,
    int patch_radius_x, int patch_radius_y, int patch_radius_z,
    double tau_factor,
    bint return_sigma,
    ndarray var_acc,
    ndarray thetavar
):
    """
    Execute the float32 implementation of the local PCA core.

    This helper allocates float32 accumulation buffers and dispatches the main
    Cython loop. 

    Parameters
    ----------
    data : 4D array
        Array of data to be denoised.
    mask : 3D boolean array
        A mask with voxels that are true inside the brain and false outside of
        it.
    estimate_sigma : bool
        Whether the local noise variance should be estimated from the
        eigenvalue spectrum.
    var_map : 3D array
        Input variance map if ``estimate_sigma`` is False; dummy array
        otherwise.
    patch_radius_x, patch_radius_y, patch_radius_z : int
        Patch radius along each spatial dimension.
    tau_factor : float
        Thresholding factor for PCA eigenvalues.
    return_sigma : bool
        Whether noise variance accumulation buffers should be updated.
    var_acc : ndarray
        Accumulator for weighted local variance estimates.
    thetavar : ndarray
        Accumulator for variance weights.

    Returns
    -------
    theta : ndarray
        Accumulated weights.
    thetax : ndarray
        Weighted reconstructed signal.
    """

    cdef f32[:, :, :, :] cdata = data
    cdef u8[:, :, :] cmask = mask
    cdef tuple data_shape = (<object>data).shape

    cdef ndarray theta = np.zeros(data_shape, dtype=np.float32)
    cdef ndarray thetax = np.zeros(data_shape, dtype=np.float32)

    cdef f32[:, :, :, :] ctheta = theta
    cdef f32[:, :, :, :] cthetax = thetax
    cdef f32[:, :, :] cvar_map = var_map
    cdef f32[:, :, :] cvar_acc = var_acc
    cdef f32[:, :, :] cthetavar = thetavar
    cdef f32 ctau_factor = tau_factor

    with nogil:
        genpca_loop[float](cdata, cmask, ctheta, cthetax,
                        estimate_sigma, cvar_map,
                        return_sigma, cvar_acc, cthetavar,
                        patch_radius_x, patch_radius_y, patch_radius_z,
                        ctau_factor)

    return theta, thetax


cdef tuple run_core_f64(
    ndarray data,
    ndarray mask,
    bint estimate_sigma,
    ndarray var_map,
    int patch_radius_x, int patch_radius_y, int patch_radius_z,
    double tau_factor,
    bint return_sigma,
    ndarray var_acc,
    ndarray thetavar
):
    """
    Execute the float64 implementation of the local PCA core.

    This helper allocates float64 accumulation buffers and dispatches the main
    Cython loop. 

    Parameters
    ----------
    data : 4D array
        Array of data to be denoised.
    mask : 3D boolean array
        A mask with voxels that are true inside the brain and false outside of
        it.
    estimate_sigma : bool
        Whether the local noise variance should be estimated from the
        eigenvalue spectrum.
    var_map : 3D array
        Input variance map if ``estimate_sigma`` is False; dummy array
        otherwise.
    patch_radius_x, patch_radius_y, patch_radius_z : int
        Patch radius along each spatial dimension.
    tau_factor : float
        Thresholding factor for PCA eigenvalues.
    return_sigma : bool
        Whether noise variance accumulation buffers should be updated.
    var_acc : ndarray
        Accumulator for weighted local variance estimates.
    thetavar : ndarray
        Accumulator for variance weights.

    Returns
    -------
    theta : ndarray
        Accumulated weights.
    thetax : ndarray
        Weighted reconstructed signal.
    """

    cdef f64[:, :, :, :] cdata = data
    cdef u8[:, :, :] cmask = mask
    cdef tuple data_shape = (<object>data).shape

    cdef ndarray theta = np.zeros(data_shape, dtype=np.float64)
    cdef ndarray thetax = np.zeros(data_shape, dtype=np.float64)

    cdef f64[:, :, :, :] ctheta = theta
    cdef f64[:, :, :, :] cthetax = thetax
    cdef f64[:, :, :] cvar_map = var_map
    cdef f64[:, :, :] cvar_acc = var_acc
    cdef f64[:, :, :] cthetavar = thetavar
    cdef f64 ctau_factor = tau_factor

    with nogil:
        genpca_loop[double](cdata, cmask, ctheta, cthetax,
            estimate_sigma, cvar_map,
            return_sigma, cvar_acc, cthetavar,
            patch_radius_x, patch_radius_y, patch_radius_z,
                        ctau_factor)

    return theta, thetax


# Python wrapper: genpca_core
def genpca_core(
    data,
    *,
    mask=None,
    var_map=None,
    return_sigma=False,
    patch_radius_x=2,
    patch_radius_y=2,
    patch_radius_z=2,
    tau_factor=None,
    out_dtype=None,
):
    """
    Perform PCA-based denoising on a 4D volume. This is a Python wrapper 
    around the optimized Cython core function `genpca_loop`. The code only supports
    eigenvalue decomposition.

    Parameters
    ----------
    data : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions. The first 3 dimensions must have
        size >= 2 * patch_radius + 1 or size = 1.
    mask : 3D boolean array
        A mask with voxels that are true inside the brain and false outside of
        it. The function denoises within the true part and returns zeros
        outside of those voxels.
    var_map : 3D array, optional
        Voxelwise noise variance map estimated from the data. If it is not given, it
        will be estimated based on random matrix theory 
        :footcite:p:`Veraart2016b`, :footcite:p:`Veraart2016c`.
    return_sigma : bool, optional
        If true, the Standard deviation of the noise will be returned.
    patch_radius_x : int, optional
        The radius of the local patch to be taken around each voxel (in
        voxels) along the x-axis.
    patch_radius_y : int, optional
        The radius of the local patch to be taken around each voxel (in
        voxels) along the y-axis.
    patch_radius_z : int, optional
       The radius of the local patch to be taken around each voxel (in
        voxels) along the z-axis.
    tau_factor : float, optional
        Thresholding of PCA eigenvalues is done by nulling out eigenvalues that
        are smaller than:

        .. math::

                \tau = (\tau_{factor} \sigma)^2

        $\tau_{factor}$ can be set to a predefined values (e.g. $\tau_{factor} =
        2.3$ :footcite:p:`Manjon2013`), or automatically calculated using random
        matrix theory (in case that $\tau_{factor}$ is set to None).
    out_dtype : dtype, optional
        The dtype for the output array. Default: output has the same dtype as
        the input.

    Returns
    -------
    denoised : 4D array
        This is the denoised array of the same size as that of the input data.

    sigma : 3D array, optional
        Estimated noise standard deviation, returned only if
        ``return_sigma=True``.
    """

    in_dtype = data.dtype
    if out_dtype is None:
        out_dtype = in_dtype

    estimate_sigma = (var_map is None)
    calc_dtype = np.float64 if data.dtype == np.float64 else np.float32

    data = np.ascontiguousarray(data, dtype=calc_dtype)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)

    if estimate_sigma:
        var_map = np.zeros((1, 1, 1), dtype=calc_dtype)
    else:
        var_map = np.ascontiguousarray(var_map, dtype=calc_dtype)

    shape = data.shape
    var_acc = np.zeros((shape[0], shape[1], shape[2]), dtype=calc_dtype)
    thetavar = np.zeros((shape[0], shape[1], shape[2]), dtype=calc_dtype)

    if calc_dtype == np.float32:
        theta, thetax = run_core_f32(
            data,
            mask,
            estimate_sigma,
            var_map,
            patch_radius_x, patch_radius_y, patch_radius_z,
            tau_factor,
            return_sigma,
            var_acc,
            thetavar,
        )
    else:
        theta, thetax = run_core_f64(
            data,
            mask,
            estimate_sigma,
            var_map,
            patch_radius_x, patch_radius_y, patch_radius_z,
            tau_factor,
            return_sigma,
            var_acc,
            thetavar,
        )

    den = thetax / theta
    den = np.clip(den, 0, None)
    den[mask == 0] = 0

    if return_sigma:
        if estimate_sigma:
            var_out = var_acc / thetavar
            var_out[mask == 0] = 0
            return den.astype(out_dtype), np.sqrt(var_out)
        else:
            return den.astype(out_dtype), np.sqrt(var_map)
    else:
        return den.astype(out_dtype)