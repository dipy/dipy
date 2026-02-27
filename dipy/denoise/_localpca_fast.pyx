import numpy as np
cimport numpy as cnp
from numpy cimport ndarray
from cython cimport view

from libc.stdlib cimport malloc, free
from libc.math cimport sqrt

from scipy.linalg.cython_lapack cimport ssyev, dsyev
from scipy.linalg.cython_blas cimport sgemm, dgemm

ctypedef cnp.float32_t f32
ctypedef cnp.float64_t f64
ctypedef cnp.uint8_t  u8


cnp.import_array()


cdef inline f32 mean_prefix_f32(const f32* a, int n) noexcept nogil:
    cdef int i
    cdef f32 s = 0.0
    for i in range(n):
        s += a[i]
    return s / n


cdef inline f64 mean_prefix_f64(const f64* a, int n) noexcept nogil:
    cdef int i
    cdef f64 s = 0.0
    for i in range(n):
        s += a[i]
    return s / n


cdef inline void pca_classifier_f32(const f32* evals, int n_evals, int nvoxels,
                                   f32* out_var, int* out_ncomps) noexcept nogil:
    cdef int start = 0
    cdef int m = n_evals
    if m > nvoxels - 1:
        start = m - (nvoxels - 1)
        m = nvoxels - 1

    if m <= 0:
        out_var[0] = 0.0
        out_ncomps[0] = 0
        return

    cdef f32 var = mean_prefix_f32(evals + start, m)
    cdef int c = m - 1
    cdef f32 r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    while r > 0.0 and c > 0:
        var = mean_prefix_f32(evals + start, c)  # mean of first c elements (0..c-1)
        c = c - 1
        r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    out_var[0] = var
    out_ncomps[0] = c + 1


cdef inline void pca_classifier_f64(const f64* evals, int n_evals, int nvoxels,
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

    cdef f64 var = mean_prefix_f64(evals + start, m)
    cdef int c = m - 1
    cdef f64 r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    while r > 0.0 and c > 0:
        var = mean_prefix_f64(evals + start, c)
        c = c - 1
        r = evals[start + c] - evals[start] - 4.0 * sqrt((c + 1.0) / nvoxels) * var

    out_var[0] = var
    out_ncomps[0] = c + 1


# -----------------------------
# Helpers: mean, center, cov
# -----------------------------
cdef inline void compute_mean_center_f32(f32* X, f32* M, int ns, int N) noexcept nogil:
    cdef int s, t
    cdef f32 inv_ns = <f32>(1.0 / ns)
    for t in range(N):
        M[t] = 0.0
    for s in range(ns):
        for t in range(N):
            M[t] += X[s*N + t]
    for t in range(N):
        M[t] *= inv_ns
    for s in range(ns):
        for t in range(N):
            X[s*N + t] -= M[t]


cdef inline void compute_mean_center_f64(f64* X, f64* M, int ns, int N) noexcept nogil:
    cdef int s, t
    cdef f64 inv_ns = 1.0 / ns
    for t in range(N):
        M[t] = 0.0
    for s in range(ns):
        for t in range(N):
            M[t] += X[s*N + t]
    for t in range(N):
        M[t] *= inv_ns
    for s in range(ns):
        for t in range(N):
            X[s*N + t] -= M[t]


cdef inline void build_cov_f32(const f32* X, f32* C, int ns, int N) noexcept nogil:
    cdef float alpha = <float>(1.0 / ns)
    cdef float beta = 0.0

    sgemm(b'N', b'T',
          &N, &N, &ns,
          &alpha,
          <float*>X, &N,
          <float*>X, &N,
          &beta,
          <float*>C, &N)


cdef inline void build_cov_f64(const f64* X, f64* C, int ns, int N) noexcept nogil:
    cdef double alpha = 1.0 / ns
    cdef double beta = 0.0

    dgemm(b'N', b'T',
          &N, &N, &ns,
          &alpha,
          <double*>X, &N,
          <double*>X, &N,
          &beta,
          <double*>C, &N)


cdef inline void reconstruct_f32(
    f32* X,
    const f32* M,
    const f32* V,
    int ns, int N,
    int n_noise,
    f32* Yt
) noexcept nogil:

    cdef int rdim = N - n_noise
    cdef float alpha = 1.0
    cdef float beta0 = 0.0
    cdef int t, s

    if rdim <= 0:
        for s in range(ns):
            for t in range(N):
                X[s*N + t] = M[t]
        return

    cdef f32* A = X
    cdef int lda = N

    cdef const f32* Vs = V + n_noise * N
    cdef int ldv = N

    cdef int ldy = rdim

    sgemm(b'T', b'N',
          &rdim, &ns, &N,
          &alpha,
          <float*>Vs, &ldv,
          <float*>A,  &lda,
          &beta0,
          <float*>Yt, &ldy)

    sgemm(b'N', b'N',
          &N, &ns, &rdim,
          &alpha,
          <float*>Vs, &ldv,
          <float*>Yt, &ldy,
          &beta0,
          <float*>A,  &lda)

    for s in range(ns):
        for t in range(N):
            X[s*N + t] += M[t]


cdef inline void reconstruct_f64(
    f64* X,
    const f64* M,
    const f64* V,
    int ns, int N,
    int n_noise,
    f64* Yt
) noexcept nogil:

    cdef int rdim = N - n_noise
    cdef double alpha = 1.0
    cdef double beta0 = 0.0
    cdef int t, s

    if rdim <= 0:
        for s in range(ns):
            for t in range(N):
                X[s*N + t] = M[t]
        return

    cdef f64* A = X
    cdef int lda = N

    cdef const f64* Vs = V + n_noise * N
    cdef int ldv = N

    cdef int ldy = rdim

    dgemm(b'T', b'N',
          &rdim, &ns, &N,
          &alpha,
          <double*>Vs, &ldv,
          <double*>A,  &lda,
          &beta0,
          <double*>Yt, &ldy)

    dgemm(b'N', b'N',
          &N, &ns, &rdim,
          &alpha,
          <double*>Vs, &ldv,
          <double*>Yt, &ldy,
          &beta0,
          <double*>A,  &lda)

    for s in range(ns):
        for t in range(N):
            X[s*N + t] += M[t]


# -----------------------------
# Main loop (float32/float64)
# -----------------------------
cdef void genpca_loop_f32(
    f32[:, :, :, :] arr,
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

    cdef Py_ssize_t Xdim = arr.shape[0]
    cdef Py_ssize_t Ydim = arr.shape[1]
    cdef Py_ssize_t Zdim = arr.shape[2]
    cdef int N = <int>arr.shape[3]

    cdef int size_x = 2*patch_radius_x + 1
    cdef int size_y = 2*patch_radius_y + 1
    cdef int size_z = 2*patch_radius_z + 1
    cdef int n_samples = size_x*size_y*size_z

    # Buffers (must allocate with GIL)
    cdef f32[:, :] patch_data
    cdef f32[:] mean_vec
    cdef f32[:, :] cov_mat
    cdef f32[:] eigenvalues
    cdef f32[:, :] proj_buf
    cdef f32[:] work

    cdef int info
    cdef int lwork = 3 * N   # sufficient workspace for ssyev

    with gil:
        patch_data = view.array(
            shape=(n_samples, N),
            itemsize=sizeof(f32),
            format="f",
            mode="c"
        )

        mean_vec = view.array(
            shape=(N,),
            itemsize=sizeof(f32),
            format="f",
            mode="c"
        )

        cov_mat = view.array(
            shape=(N, N),
            itemsize=sizeof(f32),
            format="f",
            mode="fortran"
        )

        eigenvalues = view.array(
            shape=(N,),
            itemsize=sizeof(f32),
            format="f",
            mode="c"
        )

        proj_buf = view.array(
            shape=(n_samples, N),
            itemsize=sizeof(f32),
            format="f",
            mode="c"
        )

        work = view.array(
            shape=(lwork,),
            itemsize=sizeof(f32),
            format="f",
            mode="c"
        )

    cdef Py_ssize_t i, j, k
    cdef int dx, dy, dz
    cdef Py_ssize_t ii, jj, kk
    cdef int s, t
    cdef f32 this_var, tau, this_theta
    cdef int ncomps_noise

    # centers
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
                                patch_data[s, t] = arr[ii, jj, kk, t]
                            s += 1

                compute_mean_center_f32(&patch_data[0, 0], &mean_vec[0], n_samples, N)
                build_cov_f32(&patch_data[0,0], &cov_mat[0,0], n_samples, N)
                
                ssyev(b'V', b'L',
                    &N,
                    &cov_mat[0, 0], &N,
                    &eigenvalues[0],
                    &work[0], &lwork,
                    &info)

                if info != 0:
                    # if LAPACK fails, skip this voxel (rare)
                    continue

                if estimate_sigma:
                    # Estimate noise variance sigma^2 from the eigenvalue spectrum (MP / Veraart)
                    pca_classifier_f32(&eigenvalues[0], N, n_samples, &this_var, &ncomps_noise)
                else:
                    # Use user-provided variance map (sigma^2)
                    this_var = var_map[i, j, k]

                # Threshold in variance-units:
                tau = (tau_factor * tau_factor) * this_var

                # Count how many eigenvalues fall below tau -> "noise components"
                ncomps_noise = 0
                for t in range(N):              
                    if eigenvalues[t] < tau:
                        ncomps_noise += 1
                    else:
                        break

                reconstruct_f32(&patch_data[0, 0],
                                &mean_vec[0],
                                &cov_mat[0, 0],
                                n_samples, N,
                                ncomps_noise,
                                &proj_buf[0, 0])

                this_theta = <f32>(1.0 / (1.0 + N - ncomps_noise))

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

    with gil:
        patch_data = view.array(
            shape=(n_samples, N),
            itemsize=sizeof(f64),
            format="d",
            mode="c"
        )

        mean_vec = view.array(
            shape=(N,),
            itemsize=sizeof(f64),
            format="d",
            mode="c"
        )

        cov_mat = view.array(
            shape=(N, N),
            itemsize=sizeof(f64),
            format="d",
            mode="fortran"
        )

        eigenvalues = view.array(
            shape=(N,),
            itemsize=sizeof(f64),
            format="d",
            mode="c"
        )

        proj_buf = view.array(
            shape=(n_samples, N),
            itemsize=sizeof(f64),
            format="d",
            mode="c"
        )

        work = view.array(
            shape=(lwork,),
            itemsize=sizeof(f64),
            format="d",
            mode="c"
        )

    cdef Py_ssize_t i, j, k
    cdef int dx, dy, dz
    cdef Py_ssize_t ii, jj, kk
    cdef int s, t
    cdef f64 this_var, tau, this_theta
    cdef int ncomps_noise

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
                compute_mean_center_f64(&patch_data[0, 0], &mean_vec[0], n_samples, N)
                build_cov_f64(&patch_data[0, 0], &cov_mat[0, 0], n_samples, N)

                # eigendecomposition (cov_mat overwritten with eigenvectors)
                dsyev(b'V', b'L',
                      &N,
                      &cov_mat[0, 0], &N,
                      &eigenvalues[0],
                      &work[0], &lwork,
                      &info)

                if info != 0:
                    continue

                if estimate_sigma:
                    pca_classifier_f64(&eigenvalues[0], N, n_samples, &this_var, &ncomps_noise)
                else:
                    this_var = var_map[i, j, k]

                tau = (tau_factor * tau_factor) * this_var

                # count eigenvalues below threshold (noise comps)
                ncomps_noise = 0
                for t in range(N):
                    if eigenvalues[t] < tau:
                        ncomps_noise += 1
                    else:
                        break

                # reconstruct patch in-place
                reconstruct_f64(&patch_data[0, 0],
                                &mean_vec[0],
                                &cov_mat[0, 0],
                                n_samples, N,
                                ncomps_noise,
                                &proj_buf[0, 0])

                this_theta = 1.0 / (1.0 + N - ncomps_noise)

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
    cdef f32[:, :, :, :] a = arr_np
    cdef u8[:, :, :] m = m_u8

    cdef int X = arr_np.shape[0]
    cdef int Y = arr_np.shape[1]
    cdef int Z = arr_np.shape[2]
    cdef int N = arr_np.shape[3]

    cdef ndarray theta = np.zeros((X, Y, Z, N), dtype=np.float32)
    cdef ndarray thetax = np.zeros((X, Y, Z, N), dtype=np.float32)

    cdef f32[:, :, :, :] th = theta
    cdef f32[:, :, :, :] tx = thetax
    cdef f32[:, :, :] vmap = vmap_np
    cdef f32[:, :, :] vacc = var_acc_np
    cdef f32[:, :, :] tv = thetavar_np

    with nogil:
        genpca_loop_f32(a, m, th, tx,
                        estimate_sigma, vmap,
                        return_sigma, vacc, tv,
                        patch_radius_x, patch_radius_y, patch_radius_z,
                        <f32>tau_factor)
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
    arr,
    *,
    mask=None,
    sigma=None,
    patch_radius_arr_x=2,
    patch_radius_arr_y=2,
    patch_radius_arr_z=2,
    tau_factor=None,
    return_sigma=False,
    out_dtype=None,
):
    """
    Full Cython implementation of the expensive triple-loop body (eig path).
    - Matches DIPY genpca math for pca_method='eig'
    - Uses SciPy LAPACK ssyev/dsyev for eigen-decomposition
    """

    arr_np = np.ascontiguousarray(arr)
    if arr_np.ndim != 4:
        raise ValueError("arr must be 4D (X,Y,Z,N)")

    X, Y, Z, N = arr_np.shape

    if mask is None:
        mask_np = np.ones((X, Y, Z), dtype=np.bool_)
    else:
        mask_np = np.ascontiguousarray(mask, dtype=np.bool_)
        if mask_np.shape != (X, Y, Z):
            raise ValueError("mask shape must match arr spatial dims (X,Y,Z)")

    if out_dtype is None:
        out_dtype = arr_np.dtype

    patch_radius_x = int(patch_radius_arr_x)
    patch_radius_y = int(patch_radius_arr_y)
    patch_radius_z = int(patch_radius_arr_z)

    if X == 1: patch_radius_x = 0
    if Y == 1: patch_radius_y = 0
    if Z == 1: patch_radius_z = 0

    sx = 2*patch_radius_x + 1
    sy = 2*patch_radius_y + 1
    sz = 2*patch_radius_z + 1
    ns = sx*sy*sz
    if ns <= 1:
        raise ValueError("Cannot have only 1 sample; increase patch radius.")

    if tau_factor is None:
        tau_factor = 1.0 + np.sqrt(N / ns)

    estimate_sigma = (sigma is None)
    var_map = None

    if not estimate_sigma:
        if isinstance(sigma, np.ndarray):
            sig = np.ascontiguousarray(sigma)
            if sig.shape != (X, Y, Z):
                raise ValueError("sigma array must have shape (X,Y,Z)")
            var_map = (sig.astype(np.float64) ** 2)
        elif isinstance(sigma, (int, float)):
            var_map = (float(sigma) ** 2) * np.ones((X, Y, Z), dtype=np.float64)
        else:
            raise ValueError("sigma must be None, scalar, or ndarray")

    if arr_np.dtype == np.float64:
        calc_dtype = np.float64
    else:
        calc_dtype = np.float32
        arr_np = arr_np.astype(np.float32, copy=False)

    theta = np.zeros((X, Y, Z, N), dtype=calc_dtype)
    thetax = np.zeros((X, Y, Z, N), dtype=calc_dtype)

    m_u8 = np.ascontiguousarray(mask_np, dtype=np.uint8)

    if return_sigma and estimate_sigma:
        var_acc = np.zeros((X, Y, Z), dtype=calc_dtype)
        thetavar = np.zeros((X, Y, Z), dtype=calc_dtype)
    else:
        var_acc = np.zeros((1, 1, 1), dtype=calc_dtype)  
        thetavar = np.zeros((1, 1, 1), dtype=calc_dtype) 

    if calc_dtype == np.float32:
        arr_np = np.ascontiguousarray(arr_np, dtype=np.float32)
        vmap_np = np.zeros((1, 1, 1), dtype=np.float32) if estimate_sigma else np.ascontiguousarray(var_map, dtype=np.float32)

        theta, thetax = _run_core_f32(
            arr_np,
            m_u8,
            estimate_sigma,
            vmap_np,
            patch_radius_x, patch_radius_y, patch_radius_z,
            float(tau_factor),
            return_sigma,
            var_acc,
            thetavar,
        )
    else:
        arr_np = np.ascontiguousarray(arr_np, dtype=np.float64)
        vmap_np = np.zeros((1, 1, 1), dtype=np.float64) if estimate_sigma else np.ascontiguousarray(var_map, dtype=np.float64)

        theta, thetax = _run_core_f64(
            arr_np,
            m_u8,
            estimate_sigma,
            vmap_np,
            patch_radius_x, patch_radius_y, patch_radius_z,
            float(tau_factor),
            return_sigma,
            var_acc,
            thetavar,
        )

    den = thetax / theta
    den = np.clip(den, 0, None)
    den[mask_np == 0] = 0

    if return_sigma:
        if estimate_sigma:
            var_out = var_acc / thetavar
            var_out[mask_np == 0] = 0
            return den.astype(out_dtype), np.sqrt(var_out)
        else:
            return den.astype(out_dtype), sigma
    else:
        return den.astype(out_dtype)