import numpy as np
cimport numpy as cnp
from numpy cimport ndarray

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

from scipy.linalg.cython_lapack cimport ssyev, dsyev
from scipy.linalg.cython_blas cimport sgemm, dgemm

ctypedef cnp.float32_t f32
ctypedef cnp.float64_t f64
ctypedef cnp.uint8_t  u8


cnp.import_array()


cdef inline f32 mean_from_start_f32(const f32* a, int n) noexcept nogil:
    cdef int i
    cdef f32 s = 0.0
    for i in range(n):
        s += a[i]
    return s / n


cdef inline f64 mean_from_start_f64(const f64* a, int n) noexcept nogil:
    cdef int i
    cdef f64 s = 0.0
    for i in range(n):
        s += a[i]
    return s / n


cdef inline void pca_classifier_f32(const f32[:] evals, int n_evals, int nvoxels,
                                   f32* out_var, int* out_ncomps) noexcept nogil:
    cdef int start = 0
    if n_evals > nvoxels - 1:
        start = n_evals - (nvoxels - 1)
        n_evals = nvoxels - 1

    cdef f32 var = mean_from_start_f32(&evals[start], n_evals)
    cdef int c = n_evals - 1
    cdef f32 r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    while r > 0.0:
        var = mean_from_start_f32(&evals[start], c)
        c = c - 1
        r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    out_var[0] = var
    out_ncomps[0] = c + 1


cdef inline void pca_classifier_f64(const f64[:] evals, int n_evals, int nvoxels,
                                   f64* out_var, int* out_ncomps) noexcept nogil:
    cdef int start = 0
    cdef int m = n_evals
    if m > nvoxels - 1:
        start = m - (nvoxels - 1)
        m = nvoxels - 1

    if m <= 0:
        out_var[0] = 0.0
        out_ncomps[0] = 0
        return

    cdef f64 var = mean_from_start_f64(&evals[start], m)
    cdef int c = m - 1
    cdef f64 r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    while r > 0.0 and c > 0:
        var = mean_from_start_f64(&evals[start], c)
        c = c - 1
        r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    out_var[0] = var
    out_ncomps[0] = c + 1


# -----------------------------
# Helpers: mean, center, cov
# -----------------------------
cdef inline void compute_mean_center_f32(f32[:, :] X, f32[:] M, int n_samples, int N) noexcept nogil:
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


cdef inline void compute_mean_center_f64(f64[:, :] X, f64[:] M, int ns, int N) noexcept nogil:
    cdef int s, t
    cdef f64 inv_ns = 1.0 / ns
    for t in range(N):
        M[t] = 0.0
    for s in range(ns):
        for t in range(N):
            M[t] += X[s, t]
    for t in range(N):
        M[t] *= inv_ns
    for s in range(ns):
        for t in range(N):
            X[s, t] -= M[t]


cdef inline void build_cov_f32(f32[:, :] X, f32[:, :] C, int ns, int N) noexcept nogil:
    cdef float alpha = <float>(1.0 / ns)
    cdef float beta = 0.0

    # This is the low-level equivalent of: C = X^T X / ns
    # NOTE: could be changed for something more readable, such as plain matrix
    # multiplication with for loop. 
    sgemm(b'N', b'T',
          &N, &N, &ns,
          &alpha,
            &X[0, 0], &N,
            &X[0, 0], &N,
          &beta,
            &C[0, 0], &N)


cdef inline void build_cov_f64(f64[:, :] X, f64[:, :] C, int ns, int N) noexcept nogil:
    cdef double alpha = 1.0 / ns
    cdef double beta = 0.0

    # BLAS covariance step: C <- (1/ns) * X * X^T
    # We call GEMM as ('N','T') so the second operand is treated as transposed.
    # This is the low-level equivalent of: C = X.T.dot(X) / ns

    dgemm(b'N', b'T',
          &N, &N, &ns,
          &alpha,
            &X[0, 0], &N,
            &X[0, 0], &N,
          &beta,
            &C[0, 0], &N)


cdef inline void reconstruct_f32(
    f32[:, :] X,
    const f32[:] M,
    const f32* W,
    int ns, int N,
    int n_signal,
    f32[:, :] Yt
) noexcept nogil:

    cdef float alpha = 1.0
    cdef float beta0 = 0.0
    cdef int t, s

    if n_signal <= 0:
        for s in range(ns):
            for t in range(N):
                X[s, t] = M[t]
        return

    # Equivalent to Y = W^T X^T = (X W)^T. Necessary since sgemm assuumes column-major.
    # W is already column major, but X isn't. 
    sgemm(b'T', b'N',
        &n_signal, &ns, &N,
        &alpha,
        <float*>W, &N,
        <float*>&X[0, 0],  &N,
        &beta0,
        <float*>&Yt[0, 0], &n_signal)

    # Equivalent to doing W Y = W (X W) ^T = W W^T X^T
    # Writing the output back to row-major (C-order), we have X W W^T
    sgemm(b'N', b'N',
            &N, &ns, &n_signal,
            &alpha,
            <float*>W, &N,
            <float*>&Yt[0, 0], &n_signal,
            &beta0,
            <float*>&X[0, 0],  &N)

    for s in range(ns):
        for t in range(N):
            X[s, t] += M[t]


cdef inline void reconstruct_f64(
    f64[:, :] X,
    const f64[:] M,
    const f64* W,
    int ns, int N,
    int n_signal,
    f64[:, :] Yt
) noexcept nogil:

    cdef double alpha = 1.0
    cdef double beta0 = 0.0
    cdef int t, s

    if n_signal <= 0:
        for s in range(ns):
            for t in range(N):
                X[s, t] = M[t]
        return

    dgemm(b'T', b'N',
            &n_signal, &ns, &N,
          &alpha,
                    <double*>W, &N,
                    <double*>&X[0, 0],  &N,
          &beta0,
                    <double*>&Yt[0, 0], &n_signal)

    dgemm(b'N', b'N',
            &N, &ns, &n_signal,
          &alpha,
                        <double*>W, &N,
                    <double*>&Yt[0, 0], &n_signal,
          &beta0,
                    <double*>&X[0, 0],  &N)

    for s in range(ns):
        for t in range(N):
            X[s, t] += M[t]


# -----------------------------
# Main loop (float32/float64)
# -----------------------------
cdef void genpca_loop_f32(
    f32[:, :, :, :] data,
    u8[:, :, :] mask,
    f32[:, :, :, :] theta,
    f32[:, :, :, :] thetax,
    bint estimate_sigma,
    f32[:, :, :] var_map,         # only valid if estimate_sigma==False
    bint return_sigma,
    f32[:, :, :] var_acc,         # only valid if return_sigma and estimate_sigma
    f32[:, :, :] thetavar,        # only valid if return_sigma and estimate_sigma
    int patch_radius_x, int patch_radius_y, int patch_radius_z,
    f32 tau_factor
) noexcept nogil:

    cdef Py_ssize_t Xdim = data.shape[0]
    cdef Py_ssize_t Ydim = data.shape[1]
    cdef Py_ssize_t Zdim = data.shape[2]
    cdef int N = <int>data.shape[3]

    cdef int size_x = 2*patch_radius_x + 1
    cdef int size_y = 2*patch_radius_y + 1
    cdef int size_z = 2*patch_radius_z + 1
    cdef int n_samples = size_x*size_y*size_z

    cdef f32[:, :] X
    cdef f32[:] M
    cdef f32[:, :] C
    cdef f32[:] d
    cdef f32[:, :] proj_buf
    cdef f32[:] work

    cdef int info
    cdef int lwork = 3 * N   # sufficient workspace for ssyev

    # Memory layout:
    # - X/M/proj_buf/work are C-order.
    # - C is F-order because ssyev/dsyev expect column-major.
    # Buffers (must allocate with GIL)
    with gil:
        X = np.empty((n_samples, N), dtype=np.float32)
        M = np.empty((N,), dtype=np.float32)
        C = np.empty((N, N), dtype=np.float32, order='F')
        d = np.empty((N,), dtype=np.float32)
        proj_buf = np.empty((n_samples, N), dtype=np.float32)
        work = np.empty((lwork,), dtype=np.float32)

    cdef Py_ssize_t i, j, k
    cdef int dx, dy, dz
    cdef Py_ssize_t ii, jj, kk
    cdef int s, t
    cdef f32 this_var, tau, this_theta
    cdef int ncomps, n_signal
    cdef const f32* W

    for k in range(patch_radius_z, Zdim - patch_radius_z):
        for j in range(patch_radius_y, Ydim - patch_radius_y):
            for i in range(patch_radius_x, Xdim - patch_radius_x):
                if mask[i, j, k] == 0:
                    continue
                s = 0
                for dx in range(-patch_radius_x, patch_radius_x+1):
                    ii = i + dx
                    for dy in range(-patch_radius_y, patch_radius_y+1):
                        jj = j + dy
                        for dz in range(-patch_radius_z, patch_radius_z+1):
                            kk = k + dz
                            for t in range(N):
                                X[s, t] = data[ii, jj, kk, t]
                            s += 1

                compute_mean_center_f32(X, M, n_samples, N)
                build_cov_f32(X, C, n_samples, N)
                
                # TODO: implement also svd path.
                # LAPACK eigendecomposition (column-major)
                # NOTE: could be changed for something more readable, implementing the math manually.
                # Scipy would require GIL and extra costs. 
                ssyev(b'V', b'L',
                    &N,
                    &C[0, 0], &N,
                    &d[0],
                    &work[0], &lwork,
                    &info)

                if info != 0:
                    # if LAPACK fails, skip this voxel (rare)
                    continue

                if estimate_sigma:
                    # Random matrix theory
                    pca_classifier_f32(d, N, n_samples, &this_var, &ncomps)
                else:
                    # Predefined variance
                    this_var = var_map[i, j, k]

                # Threshold by tau
                tau = (tau_factor * tau_factor) * this_var

                # Update ncomps according to tau_factor
                ncomps = 0
                for t in range(N):              
                    if d[t] < tau:
                        ncomps += 1
                    else:
                        break

                n_signal = N - ncomps
                W = &C[0, 0] + ncomps * N

                reconstruct_f32(X,
                                M,
                                W,
                                n_samples, N,
                                n_signal,
                                proj_buf)

                this_theta = <f32>(1.0 / (1.0 + N - ncomps))

                s = 0
                for dx in range(-patch_radius_x, patch_radius_x + 1):
                    ii = i + dx
                    for dy in range(-patch_radius_y, patch_radius_y + 1):
                        jj = j + dy
                        for dz in range(-patch_radius_z, patch_radius_z + 1):
                            kk = k + dz

                            for t in range(N):
                                theta[ii, jj, kk, t]  += this_theta
                                thetax[ii, jj, kk, t] += X[s, t] * this_theta

                            if return_sigma and estimate_sigma:
                                var_acc[ii, jj, kk] += this_var * this_theta
                                thetavar[ii, jj, kk] += this_theta

                            s += 1


cdef void genpca_loop_f64(
    f64[:, :, :, :] arr,
    u8[:, :, :] mask,
    f64[:, :, :, :] theta,
    f64[:, :, :, :] thetax,
    bint estimate_sigma,
    f64[:, :, :] var_map,
    bint return_sigma,
    f64[:, :, :] var_acc,
    f64[:, :, :] thetavar,
    int patch_radius_x, int patch_radius_y, int patch_radius_z,
    f64 tau_factor
) noexcept nogil:

    cdef Py_ssize_t Xdim = arr.shape[0]
    cdef Py_ssize_t Ydim = arr.shape[1]
    cdef Py_ssize_t Zdim = arr.shape[2]
    cdef int N = <int>arr.shape[3]

    cdef int size_x = 2*patch_radius_x + 1
    cdef int size_y = 2*patch_radius_y + 1
    cdef int size_z = 2*patch_radius_z + 1
    cdef int n_samples = size_x*size_y*size_z

    # Buffers (must allocate with GIL)
    cdef f64[:, :] patch_data
    cdef f64[:] mean_vec
    cdef f64[:, :] cov_mat
    cdef f64[:] eigenvalues
    cdef f64[:, :] proj_buf
    cdef f64[:] work

    cdef int info
    cdef int lwork = 3 * N

    # Memory layout:
    # - patch_data/proj_buf/work are C-order for sample-major gather/scatter locality.
    # - cov_mat is F-order because ssyev/dsyev expect column-major storage and
    #   overwrite cov_mat in-place with eigenvectors in that layout.
    # This mixed layout is intentional to avoid per-voxel transpose/copy costs.

    with gil:
        patch_data = np.empty((n_samples, N), dtype=np.float64)
        mean_vec = np.empty((N,), dtype=np.float64)
        cov_mat = np.empty((N, N), dtype=np.float64, order='F')
        eigenvalues = np.empty((N,), dtype=np.float64)
        proj_buf = np.empty((n_samples, N), dtype=np.float64)
        work = np.empty((lwork,), dtype=np.float64)

    cdef Py_ssize_t i, j, k
    cdef int dx, dy, dz
    cdef Py_ssize_t ii, jj, kk
    cdef int s, t
    cdef f64 this_var, tau, this_theta
    cdef int ncomps, n_signal
    cdef const f64* W

    for k in range(patch_radius_z, Zdim - patch_radius_z):
        for j in range(patch_radius_y, Ydim - patch_radius_y):
            for i in range(patch_radius_x, Xdim - patch_radius_x):

                if mask[i, j, k] == 0:
                    continue

                # gather patch
                s = 0
                for dx in range(-patch_radius_x, patch_radius_x + 1):
                    ii = i + dx
                    for dy in range(-patch_radius_y, patch_radius_y + 1):
                        jj = j + dy
                        for dz in range(-patch_radius_z, patch_radius_z + 1):
                            kk = k + dz
                            for t in range(N):
                                patch_data[s, t] = arr[ii, jj, kk, t]
                            s += 1

                # center + covariance
                compute_mean_center_f64(patch_data, mean_vec, n_samples, N)
                build_cov_f64(patch_data, cov_mat, n_samples, N)

                    # LAPACK eigendecomposition (column-major):
                    # - jobz='V': compute eigenvalues + eigenvectors in-place in cov_mat
                    # - uplo='L': use lower triangle of cov_mat
                    # More readable alternative: call scipy.linalg.eigh(cov_mat) from
                    # Python, but that requires the GIL and usually costs extra overhead.
                dsyev(b'V', b'L',
                      &N,
                      &cov_mat[0, 0], &N,
                      &eigenvalues[0],
                      &work[0], &lwork,
                      &info)

                if info != 0:
                    continue

                if estimate_sigma:
                    pca_classifier_f64(eigenvalues, N, n_samples, &this_var, &ncomps)
                else:
                    this_var = var_map[i, j, k]

                tau = (tau_factor * tau_factor) * this_var

                # count eigenvalues below threshold (noise comps)
                ncomps = 0
                for t in range(N):
                    if eigenvalues[t] < tau:
                        ncomps += 1
                    else:
                        break

                n_signal = N - ncomps
                W = &cov_mat[0, 0] + ncomps * N

                # reconstruct patch in-place
                reconstruct_f64(patch_data,
                                mean_vec,
                                W,
                                n_samples, N,
                                n_signal,
                                proj_buf)

                this_theta = 1.0 / (1.0 + N - ncomps)

                # scatter back
                s = 0
                for dx in range(-patch_radius_x, patch_radius_x + 1):
                    ii = i + dx
                    for dy in range(-patch_radius_y, patch_radius_y + 1):
                        jj = j + dy
                        for dz in range(-patch_radius_z, patch_radius_z + 1):
                            kk = k + dz

                            for t in range(N):
                                theta[ii, jj, kk, t]  += this_theta
                                thetax[ii, jj, kk, t] += patch_data[s, t] * this_theta

                            if return_sigma and estimate_sigma:
                                var_acc[ii, jj, kk] += this_var * this_theta
                                thetavar[ii, jj, kk] += this_theta

                            s += 1


cdef tuple _run_core_f32(
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
    cdef f32[:, :, :, :] cdata = data
    cdef u8[:, :, :] cmask = mask

    cdef ndarray theta = np.zeros(data.shape, dtype=np.float32)
    cdef ndarray thetax = np.zeros(data.shape, dtype=np.float32)

    cdef f32[:, :, :, :] ctheta = theta
    cdef f32[:, :, :, :] cthetax = thetax
    cdef f32[:, :, :] cvar_map = var_map
    cdef f32[:, :, :] cvar_acc = var_acc
    cdef f32[:, :, :] cthetavar = thetavar
    cdef f32 ctau_factor = tau_factor

    with nogil:
        genpca_loop_f32(cdata, cmask, ctheta, cthetax,
                        estimate_sigma, cvar_map,
                        return_sigma, cvar_acc, cthetavar,
                        patch_radius_x, patch_radius_y, patch_radius_z,
                        ctau_factor)

    return theta, thetax


cdef tuple _run_core_f64(
    ndarray arr_np,
    ndarray m_u8,
    bint estimate_sigma,
    ndarray vmap_np,
    int patch_radius_x, int patch_radius_y, int patch_radius_z,
    double tau_factor,
    bint return_sigma,
    ndarray var_acc_np,
    ndarray thetavar_np
):
    cdef f64[:, :, :, :] a = arr_np
    cdef u8[:, :, :] m = m_u8

    cdef int X = arr_np.shape[0]
    cdef int Y = arr_np.shape[1]
    cdef int Z = arr_np.shape[2]
    cdef int N = arr_np.shape[3]

    cdef ndarray theta = np.zeros((X, Y, Z, N), dtype=np.float64)
    cdef ndarray thetax = np.zeros((X, Y, Z, N), dtype=np.float64)

    cdef f64[:, :, :, :] th = theta
    cdef f64[:, :, :, :] tx = thetax
    cdef f64[:, :, :] vmap = vmap_np
    cdef f64[:, :, :] vacc = var_acc_np
    cdef f64[:, :, :] tv = thetavar_np

    with nogil:
        genpca_loop_f64(a, m, th, tx,
                        estimate_sigma, vmap,
                        return_sigma, vacc, tv,
                        patch_radius_x, patch_radius_y, patch_radius_z,
                        <f64>tau_factor)

    return theta, thetax


# -----------------------------
# Python wrapper: genpca_core
# -----------------------------
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
):
    """
    Full Cython implementation of the expensive triple-loop body.
    """

    estimate_sigma = (var_map is None)
    calc_dtype = np.float64 if data.dtype == np.float64 else np.float32
    
    data = np.ascontiguousarray(data, dtype=calc_dtype)
    mask = np.ascontiguousarray(mask, dtype=np.uint8)
    
    if estimate_sigma:
        var_map = np.zeros((1, 1, 1), dtype=calc_dtype)
    else:
        var_map = np.ascontiguousarray(var_map, dtype=calc_dtype)

    var_acc = np.zeros(data.shape[:-1], dtype=calc_dtype)
    thetavar = np.zeros(data.shape[:-1], dtype=calc_dtype)

    if calc_dtype == np.float32:
        theta, thetax = _run_core_f32(
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
        theta, thetax = _run_core_f64(
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
    den[mask_np == 0] = 0

    if return_sigma:
        if estimate_sigma:
            var_out = var / thetavar
            var_out[mask_np == 0] = 0
            return den.astype(out_dtype), np.sqrt(var_out)
        else:
            return den.astype(out_dtype), sigma
    else:
        return den.astype(out_dtype)