# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False
"""
Inner loops of the FORCE fiber-alignment stage.

The matched library entry of each voxel is kept frozen; only the axes of its
fiber populations move.  This module holds the per-voxel kernels behind
``FORCEModel.fit`` and :func:`dipy.reconst.force.postprocess_peaks`:

- :func:`alignment_system` : Gauss-Newton system of the data term,
- :func:`kernel_signal` : signal of the fiber kernels,
- :func:`fiber_edges` : fiber-to-fiber links between neighboring voxels,
- :func:`coherence_sweeps` : colored Gauss-Seidel sweeps of the spatial stage.

All loops run without the GIL and are parallelized over voxels with OpenMP.
"""

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport cos, exp, fabs, pow, sin, sqrt

cdef enum:
    MAX_FIBERS = 3
    MAX_DOF = 6
    MAX_DOF_SQ = 36
    N_NEIGHBORS = 26

cdef double TINY = 1e-300


cdef inline double _dot3(const double* a, const double* b) noexcept nogil:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


cdef inline void _tangent_frame(
    const double* d, double* e1, double* e2
) noexcept nogil:
    """Two orthonormal vectors spanning the tangent plane of the unit vector d."""
    cdef double n
    if fabs(d[2]) < 0.9:
        e1[0] = d[1]
        e1[1] = -d[0]
        e1[2] = 0.0
    else:
        e1[0] = 0.0
        e1[1] = d[2]
        e1[2] = -d[1]
    n = sqrt(_dot3(e1, e1))
    if n > 0.0:
        e1[0] /= n
        e1[1] /= n
        e1[2] /= n
    e2[0] = d[1] * e1[2] - d[2] * e1[1]
    e2[1] = d[2] * e1[0] - d[0] * e1[2]
    e2[2] = d[0] * e1[1] - d[1] * e1[0]


cdef void _voxel_system(
    const float* y,
    const float* rest,
    const double* dirs,
    const unsigned char* valid,
    const double* poly,
    const int* shell,
    const double* gdirs,
    Py_ssize_t n_meas,
    Py_ssize_t n_fib,
    Py_ssize_t n_shell,
    Py_ssize_t n_term,
    double* hess,
    double* grad,
    double* cost,
    double* frames,
) noexcept nogil:
    """Gauss-Newton system of ``1 - rho^2`` for one voxel.

    ``rho`` is the cosine similarity between the measured signal ``y`` and the
    model ``rest + sum_k K_k(g . d_k)``.  The unknowns are the two tangent
    coordinates of every valid fiber axis; the model amplitude is profiled out
    (variable projection), which leaves ``hess = a^2 J^T P J`` and
    ``grad = a J^T r`` with ``P`` the projector orthogonal to the model.
    """
    cdef:
        double jj[MAX_DOF][MAX_DOF]
        double js[MAX_DOF]
        double jy[MAX_DOF]
        double col[MAX_DOF]
        double ss = 0.0, ys = 0.0, yy = 0.0
        double sig, yi, t, u, val, der, w, gx, gy, gz, alpha, ynorm
        const double* p
        const double* d
        const double* g
        Py_ssize_t i, k, q, a, b, s
        Py_ssize_t ndof = 2 * n_fib

    for a in range(ndof):
        js[a] = 0.0
        jy[a] = 0.0
        for b in range(ndof):
            jj[a][b] = 0.0

    for k in range(n_fib):
        if valid[k]:
            _tangent_frame(dirs + 3 * k, frames + 6 * k, frames + 6 * k + 3)
        else:
            for q in range(6):
                frames[6 * k + q] = 0.0

    for i in range(n_meas):
        s = shell[i]
        sig = rest[i]
        for a in range(ndof):
            col[a] = 0.0
        if s >= 0:
            g = gdirs + 3 * i
            gx = g[0]
            gy = g[1]
            gz = g[2]
            for k in range(n_fib):
                if not valid[k]:
                    continue
                d = dirs + 3 * k
                t = gx * d[0] + gy * d[1] + gz * d[2]
                u = t * t
                p = poly + (k * n_shell + s) * n_term
                val = p[n_term - 1]
                der = 0.0
                for q in range(n_term - 2, -1, -1):
                    der = der * u + val
                    val = val * u + p[q]
                sig += val
                w = 2.0 * der * t
                col[2 * k] = w * _dot3(g, frames + 6 * k)
                col[2 * k + 1] = w * _dot3(g, frames + 6 * k + 3)
        yi = y[i]
        ss += sig * sig
        ys += yi * sig
        yy += yi * yi
        for a in range(ndof):
            if col[a] != 0.0:
                js[a] += col[a] * sig
                jy[a] += col[a] * yi
                for b in range(a, ndof):
                    jj[a][b] += col[a] * col[b]

    if ss <= 0.0 or yy <= 0.0:
        cost[0] = 1.0
        for a in range(ndof):
            grad[a] = 0.0
            for b in range(ndof):
                hess[a * ndof + b] = 0.0
        return

    ynorm = sqrt(yy)
    alpha = ys / (ss * ynorm)
    cost[0] = 1.0 - ys * ys / (ss * yy)
    if cost[0] < 0.0:
        cost[0] = 0.0
    for a in range(ndof):
        grad[a] = alpha * (jy[a] / ynorm - alpha * js[a])
        for b in range(a, ndof):
            val = alpha * alpha * (jj[a][b] - js[a] * js[b] / ss)
            hess[a * ndof + b] = val
            hess[b * ndof + a] = val


def alignment_system(
    const float[:, ::1] signal not None,
    const float[:, ::1] rest not None,
    const double[:, :, ::1] dirs not None,
    const unsigned char[:, ::1] valid not None,
    const double[:, :, :, ::1] poly not None,
    const int[::1] shell not None,
    const double[:, ::1] gdirs not None,
):
    """Gauss-Newton system of the alignment data term for a batch of voxels.

    Parameters
    ----------
    signal : float32 memoryview (N, M)
        Measured signals.
    rest : float32 memoryview (N, M)
        Frozen part of each voxel's matched library signal.
    dirs : float64 memoryview (N, F, 3)
        Current unit fiber axes.
    valid : uint8 memoryview (N, F)
        Non-zero where a fiber exists.
    poly : float64 memoryview (N, F, S, P)
        Zonal kernel of every fiber on every shell, as coefficients of a
        polynomial in ``(g . d)^2``.
    shell : int32 memoryview (M,)
        Shell index of every measurement, ``-1`` where the measurement carries
        no orientation information.
    gdirs : float64 memoryview (M, 3)
        Gradient directions.

    Returns
    -------
    hess : ndarray (N, 2F, 2F)
        Gauss-Newton Hessian of ``(1 - rho^2) / 2`` in tangent coordinates.
    grad : ndarray (N, 2F)
        Descent direction ``-d/dx (1 - rho^2) / 2``.
    cost : ndarray (N,)
        ``1 - rho^2`` with ``rho`` the cosine similarity of signal and model.
    frames : ndarray (N, F, 2, 3)
        Tangent frame the coordinates refer to.
    """
    cdef:
        Py_ssize_t n_vox = signal.shape[0]
        Py_ssize_t n_meas = signal.shape[1]
        Py_ssize_t n_fib = dirs.shape[1]
        Py_ssize_t n_shell = poly.shape[2]
        Py_ssize_t n_term = poly.shape[3]
        Py_ssize_t n

    if n_fib > MAX_FIBERS:
        raise ValueError(f"At most {MAX_FIBERS} fibers per voxel are supported")
    if (
        rest.shape[0] != n_vox
        or rest.shape[1] != n_meas
        or dirs.shape[0] != n_vox
        or dirs.shape[2] != 3
        or valid.shape[0] != n_vox
        or valid.shape[1] != n_fib
        or poly.shape[0] != n_vox
        or poly.shape[1] != n_fib
        or shell.shape[0] != n_meas
        or gdirs.shape[0] != n_meas
        or gdirs.shape[1] != 3
    ):
        raise ValueError("Inconsistent array shapes")
    if n_shell < 1 or n_term < 1:
        raise ValueError("poly needs at least one shell and one term")

    cdef cnp.ndarray[cnp.float64_t, ndim=3, mode="c"] hess = np.zeros(
        (n_vox, 2 * n_fib, 2 * n_fib), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=2, mode="c"] grad = np.zeros(
        (n_vox, 2 * n_fib), dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] cost = np.ones(
        n_vox, dtype=np.float64
    )
    cdef cnp.ndarray[cnp.float64_t, ndim=4, mode="c"] frames = np.zeros(
        (n_vox, n_fib, 2, 3), dtype=np.float64
    )
    cdef double[:, :, ::1] hess_v = hess
    cdef double[:, ::1] grad_v = grad
    cdef double[::1] cost_v = cost
    cdef double[:, :, :, ::1] frames_v = frames

    if n_vox == 0:
        return hess, grad, cost, frames

    for n in prange(n_vox, schedule="static", nogil=True):
        _voxel_system(
            &signal[n, 0],
            &rest[n, 0],
            &dirs[n, 0, 0],
            &valid[n, 0],
            &poly[n, 0, 0, 0],
            &shell[0],
            &gdirs[0, 0],
            n_meas,
            n_fib,
            n_shell,
            n_term,
            &hess_v[n, 0, 0],
            &grad_v[n, 0],
            &cost_v[n],
            &frames_v[n, 0, 0, 0],
        )
    return hess, grad, cost, frames


def kernel_signal(
    const double[:, :, ::1] dirs not None,
    const unsigned char[:, ::1] valid not None,
    const double[:, :, :, ::1] poly not None,
    const int[::1] shell not None,
    const double[:, ::1] gdirs not None,
):
    """Signal of the fiber kernels, ``sum_k K_k(g . d_k)``, for a batch of voxels.

    Parameters
    ----------
    dirs : float64 memoryview (N, F, 3)
        Unit fiber axes.
    valid : uint8 memoryview (N, F)
        Non-zero where a fiber exists.
    poly : float64 memoryview (N, F, S, P)
        Zonal kernels as polynomial coefficients in ``(g . d)^2``.
    shell : int32 memoryview (M,)
        Shell index of every measurement, ``-1`` where unused.
    gdirs : float64 memoryview (M, 3)
        Gradient directions.

    Returns
    -------
    signal : ndarray of float32 (N, M)
        Kernel signal; zero for measurements without a shell.
    """
    cdef:
        Py_ssize_t n_vox = dirs.shape[0]
        Py_ssize_t n_fib = dirs.shape[1]
        Py_ssize_t n_meas = shell.shape[0]
        Py_ssize_t n_shell = poly.shape[2]
        Py_ssize_t n_term = poly.shape[3]
        Py_ssize_t n, i, k, q, s
        double t, u, val, acc

    if (
        dirs.shape[2] != 3
        or valid.shape[0] != n_vox
        or valid.shape[1] != n_fib
        or poly.shape[0] != n_vox
        or poly.shape[1] != n_fib
        or gdirs.shape[0] != n_meas
        or gdirs.shape[1] != 3
    ):
        raise ValueError("Inconsistent array shapes")
    if n_shell < 1 or n_term < 1:
        raise ValueError("poly needs at least one shell and one term")

    cdef cnp.ndarray[cnp.float32_t, ndim=2, mode="c"] out = np.zeros(
        (n_vox, n_meas), dtype=np.float32
    )
    cdef float[:, ::1] out_v = out

    for n in prange(n_vox, schedule="static", nogil=True):
        for i in range(n_meas):
            s = shell[i]
            if s < 0:
                continue
            acc = 0.0
            for k in range(n_fib):
                if not valid[n, k]:
                    continue
                t = (
                    gdirs[i, 0] * dirs[n, k, 0]
                    + gdirs[i, 1] * dirs[n, k, 1]
                    + gdirs[i, 2] * dirs[n, k, 2]
                )
                u = t * t
                val = poly[n, k, s, n_term - 1]
                for q in range(n_term - 2, -1, -1):
                    val = val * u + poly[n, k, s, q]
                acc = acc + val
            out_v[n, i] = <float>acc
    return out


def fiber_edges(
    const double[:, :, ::1] dirs not None,
    const unsigned char[:, ::1] keep not None,
    const cnp.int64_t[:, ::1] nbr not None,
    const double[:, ::1] unit not None,
    const double[::1] gweight not None,
    double alpha,
    double power,
    double cos_min,
):
    """Link every fiber to its mutually nearest fiber in each neighboring voxel.

    Parameters
    ----------
    dirs : float64 memoryview (V, F, 3)
        Unit fiber axes.
    keep : uint8 memoryview (V, F)
        Non-zero for fibers that may be linked.
    nbr : int64 memoryview (V, 26)
        Index of each neighboring voxel, ``-1`` where there is none.
    unit : float64 memoryview (26, 3)
        Unit offset to each neighbor, in the frame of the fiber axes.
    gweight : float64 memoryview (26,)
        Distance weight of each neighbor.
    alpha, power : float
        Fiber-continuity kernel ``alpha + (1 - alpha) |offset . axis|^power``.
    cos_min : float
        Links between fibers further apart than ``arccos(cos_min)`` are dropped.

    Returns
    -------
    partner : ndarray of int8 (V, 26, F)
        Index of the linked fiber in the neighbor, ``-1`` where there is none.
    weight : ndarray of float32 (V, 26, F)
        Link weight (symmetric between the two linked fibers).
    """
    cdef:
        Py_ssize_t n_vox = dirs.shape[0]
        Py_ssize_t n_fib = dirs.shape[1]
        Py_ssize_t v, o, k, j, kk, n, best_j, best_k
        double c, best, kv, kn

    if n_fib > MAX_FIBERS:
        raise ValueError(f"At most {MAX_FIBERS} fibers per voxel are supported")
    if nbr.shape[0] != n_vox or nbr.shape[1] != N_NEIGHBORS:
        raise ValueError("nbr must have shape (n_voxels, 26)")
    if unit.shape[0] != N_NEIGHBORS or gweight.shape[0] != N_NEIGHBORS:
        raise ValueError("unit and gweight must describe 26 neighbors")

    cdef cnp.ndarray[cnp.int8_t, ndim=3, mode="c"] partner = np.full(
        (n_vox, N_NEIGHBORS, n_fib), -1, dtype=np.int8
    )
    cdef cnp.ndarray[cnp.float32_t, ndim=3, mode="c"] weight = np.zeros(
        (n_vox, N_NEIGHBORS, n_fib), dtype=np.float32
    )
    cdef signed char[:, :, ::1] partner_v = partner
    cdef float[:, :, ::1] weight_v = weight

    for v in prange(n_vox, schedule="static", nogil=True):
        for o in range(N_NEIGHBORS):
            n = nbr[v, o]
            if n < 0:
                continue
            for k in range(n_fib):
                if not keep[v, k]:
                    continue
                best = -1.0
                best_j = -1
                for j in range(n_fib):
                    if keep[n, j]:
                        c = fabs(_dot3(&dirs[v, k, 0], &dirs[n, j, 0]))
                        if c > best:
                            best = c
                            best_j = j
                if best_j < 0 or best < cos_min:
                    continue
                # mutual: fiber k must also be the nearest of best_j
                best_k = k
                for kk in range(n_fib):
                    if keep[v, kk] and kk != k:
                        c = fabs(_dot3(&dirs[v, kk, 0], &dirs[n, best_j, 0]))
                        if c > best or (c == best and kk < k):
                            best_k = kk
                            break
                if best_k != k:
                    continue
                kv = alpha + (1.0 - alpha) * pow(
                    fabs(_dot3(&unit[o, 0], &dirs[v, k, 0])), power
                )
                kn = alpha + (1.0 - alpha) * pow(
                    fabs(_dot3(&unit[o, 0], &dirs[n, best_j, 0])), power
                )
                partner_v[v, o, k] = <signed char>best_j
                weight_v[v, o, k] = <float>(gweight[o] * sqrt(kv * kn))
    return partner, weight


cdef int _solve_spd(double* h, double* b, Py_ssize_t n) noexcept nogil:
    """Solve ``h x = b`` in place by Cholesky (h symmetric, n <= MAX_DOF)."""
    cdef Py_ssize_t i, j, k
    cdef double s
    for j in range(n):
        s = h[j * n + j]
        for k in range(j):
            s -= h[j * n + k] * h[j * n + k]
        if s <= TINY:
            return 1
        s = sqrt(s)
        h[j * n + j] = s
        for i in range(j + 1, n):
            for k in range(j):
                h[i * n + j] -= h[i * n + k] * h[j * n + k]
            h[i * n + j] /= s
    for i in range(n):
        s = b[i]
        for k in range(i):
            s -= h[i * n + k] * b[k]
        b[i] = s / h[i * n + i]
    for i in range(n - 1, -1, -1):
        s = b[i]
        for k in range(i + 1, n):
            s -= h[k * n + i] * b[k]
        b[i] = s / h[i * n + i]
    return 0


cdef double _update_voxel(
    Py_ssize_t v,
    double[:, :, ::1] dirs,
    const double[:, :, ::1] base,
    const double[:, :, :, ::1] frames,
    const double[:, :, ::1] hess,
    const double[:, ::1] lin,
    const unsigned char[:, ::1] valid,
    const double[:, :, ::1] ref,
    const cnp.int64_t[:, ::1] nbr,
    const signed char[:, :, ::1] partner,
    const float[:, :, ::1] weight,
    double two_mu,
    double inv_2s2,
    double cos_cone,
    double sin_cone,
) noexcept nogil:
    """Exact minimizer of the majorized objective for the fibers of voxel ``v``.

    Returns the largest ``1 - |cos|`` between old and new axes.
    """
    cdef:
        double h[MAX_DOF_SQ]
        double h0[MAX_DOF_SQ]
        double b[MAX_DOF]
        double b0[MAX_DOF]
        double new[3]
        double perp[3]
        double diag, r1, r2, c0, cur, sgn, wt, nrm, c, move = 0.0, ridge
        Py_ssize_t n_fib = dirs.shape[1]
        Py_ssize_t ndof = 2 * n_fib
        Py_ssize_t k, o, j, n, a, q, tries
        const double* p

    for a in range(ndof):
        b[a] = lin[v, a]
        for q in range(ndof):
            h[a * ndof + q] = hess[v, a, q]

    for k in range(n_fib):
        if not valid[v, k]:
            continue
        diag = 0.0
        r1 = 0.0
        r2 = 0.0
        for o in range(N_NEIGHBORS):
            j = partner[v, o, k]
            if j < 0:
                continue
            n = nbr[v, o]
            p = &dirs[n, j, 0]
            c0 = _dot3(p, &base[v, k, 0])
            sgn = 1.0
            if c0 < 0.0:
                sgn = -1.0
                c0 = -c0
            cur = _dot3(p, &dirs[v, k, 0])
            wt = two_mu * weight[v, o, k] * exp(-(1.0 - cur * cur) * inv_2s2)
            diag += wt * c0 * c0
            r1 += wt * c0 * sgn * _dot3(p, &frames[v, k, 0, 0])
            r2 += wt * c0 * sgn * _dot3(p, &frames[v, k, 1, 0])
        h[(2 * k) * ndof + 2 * k] += diag
        h[(2 * k + 1) * ndof + 2 * k + 1] += diag
        b[2 * k] += r1
        b[2 * k + 1] += r2

    for a in range(ndof * ndof):
        h0[a] = h[a]
    for a in range(ndof):
        b0[a] = b[a]
    ridge = 0.0
    tries = 0
    while _solve_spd(h, b, ndof) != 0:
        # Flat direction (a fiber the data cannot orient and no neighbor
        # constrains): damp it instead of moving it.
        tries += 1
        if ridge == 0.0:
            ridge = 1e-12
            for a in range(ndof):
                ridge += 1e-9 * fabs(h0[a * ndof + a])
        else:
            ridge *= 100.0
        for a in range(ndof * ndof):
            h[a] = h0[a]
        for a in range(ndof):
            h[a * ndof + a] += ridge
            b[a] = b0[a]
        if tries > 30:
            # Not a usable system (non-finite input): leave the voxel alone.
            for a in range(ndof):
                b[a] = 0.0
            break

    for k in range(n_fib):
        if not valid[v, k]:
            continue
        nrm = 0.0
        for q in range(3):
            new[q] = (
                base[v, k, q]
                + b[2 * k] * frames[v, k, 0, q]
                + b[2 * k + 1] * frames[v, k, 1, q]
            )
            nrm += new[q] * new[q]
        nrm = sqrt(nrm)
        c = 0.0
        for q in range(3):
            new[q] /= nrm
            c += new[q] * ref[v, k, q]
        if c < 0.0:
            c = -c
            for q in range(3):
                new[q] = -new[q]
        if c < cos_cone:
            nrm = 0.0
            for q in range(3):
                perp[q] = new[q] - c * ref[v, k, q]
                nrm += perp[q] * perp[q]
            nrm = sqrt(nrm)
            for q in range(3):
                new[q] = cos_cone * ref[v, k, q] + sin_cone * perp[q] / nrm
        c = 1.0 - fabs(_dot3(new, &dirs[v, k, 0]))
        if c > move:
            move = c
        for q in range(3):
            dirs[v, k, q] = new[q]
    return move


def coherence_sweeps(
    double[:, :, ::1] dirs not None,
    const double[:, :, ::1] base not None,
    const double[:, :, :, ::1] frames not None,
    const double[:, :, ::1] hess not None,
    const double[:, ::1] lin not None,
    const unsigned char[:, ::1] valid not None,
    const double[:, :, ::1] ref not None,
    const cnp.int64_t[:, ::1] nbr not None,
    const signed char[:, :, ::1] partner not None,
    const float[:, :, ::1] weight not None,
    const cnp.int64_t[::1] order not None,
    const cnp.int64_t[::1] color_start not None,
    double mu,
    double theta_r,
    double max_angle,
    int n_sweeps,
    double tol,
):
    """Colored Gauss-Seidel sweeps of the spatially coupled alignment.

    Minimizes, over the fiber axes of all voxels,

    ``sum_v (x_v^T H_v x_v / 2 - l_v^T x_v)
    + mu * sum_edges w_e * welsch(sin^2 angle(d_vk, d_nj))``

    where ``x_v`` are the tangent coordinates of voxel ``v``'s axes around
    ``base`` and ``welsch(r) = s^2 (1 - exp(-r / (2 s^2)))``,
    ``s = sin(theta_r)``.  The Welsch term is majorized at the current axes,
    so each voxel update is an exact linear solve and the objective never
    increases.  Voxels of one color share no edge and are updated in parallel.

    Parameters
    ----------
    dirs : float64 memoryview (V, F, 3)
        Fiber axes, updated in place.
    base : float64 memoryview (V, F, 3)
        Axes at which the data term was linearized.
    frames : float64 memoryview (V, F, 2, 3)
        Tangent frames at ``base``.
    hess : float64 memoryview (V, 2F, 2F)
        Data Hessian (identity on the coordinates of missing fibers).
    lin : float64 memoryview (V, 2F)
        Linear term of the data model.
    valid : uint8 memoryview (V, F)
        Non-zero where a fiber exists.
    ref : float64 memoryview (V, F, 3)
        Center of each fiber's trust region.
    nbr : int64 memoryview (V, 26)
        Neighbor voxel indices, ``-1`` where missing.
    partner : int8 memoryview (V, 26, F)
        Linked fiber in each neighbor (see :func:`fiber_edges`).
    weight : float32 memoryview (V, 26, F)
        Link weights.
    order : int64 memoryview (V,)
        Voxel indices sorted by color.
    color_start : int64 memoryview (n_colors + 1,)
        Offsets of each color in ``order``.
    mu : float
        Strength of the continuity term.
    theta_r : float
        Welsch scale, in radians.
    max_angle : float
        Radius of the trust region around ``ref``, in radians.
    n_sweeps : int
        Maximum number of sweeps.
    tol : float
        Stop when no axis moved by more than ``tol`` radians in a sweep.

    Returns
    -------
    moves : list of float
        Largest axis displacement (radians) of every sweep performed.
    """
    cdef:
        Py_ssize_t n_vox = dirs.shape[0]
        Py_ssize_t n_colors = color_start.shape[0] - 1
        Py_ssize_t c, i, start, stop
        double two_mu = 2.0 * mu
        double s = sin(theta_r)
        double inv_2s2 = 1.0 / (2.0 * s * s)
        double cos_cone = cos(max_angle)
        double sin_cone = sin(max_angle)
        double worst

    if dirs.shape[1] > MAX_FIBERS:
        raise ValueError(f"At most {MAX_FIBERS} fibers per voxel are supported")
    if order.shape[0] != n_vox:
        raise ValueError("order must list every voxel once")

    cdef cnp.ndarray[cnp.float64_t, ndim=1, mode="c"] moved = np.zeros(
        max(n_vox, 1), dtype=np.float64
    )
    cdef double[::1] moved_v = moved
    moves = []
    for _ in range(n_sweeps):
        for c in range(n_colors):
            start = color_start[c]
            stop = color_start[c + 1]
            for i in prange(start, stop, schedule="static", nogil=True):
                moved_v[i] = _update_voxel(
                    order[i],
                    dirs,
                    base,
                    frames,
                    hess,
                    lin,
                    valid,
                    ref,
                    nbr,
                    partner,
                    weight,
                    two_mu,
                    inv_2s2,
                    cos_cone,
                    sin_cone,
                )
        worst = float(np.arccos(np.clip(1.0 - moved.max(), -1.0, 1.0)))
        moves.append(worst)
        if worst < tol:
            break
    return moves
