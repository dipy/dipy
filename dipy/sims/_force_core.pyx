# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True
"""
FORCE Core Cython Module

Low-level signal generation for multi-compartment diffusion simulations.
"""

import numpy as np
cimport numpy as cnp

from libc.math cimport exp

ctypedef cnp.float64_t DTYPE_t
ctypedef cnp.uint8_t LABEL_t

# Diffusivity ranges (configurable via set_diffusivity_ranges)
cdef double WM_D_PAR_MIN = 2.0e-3
cdef double WM_D_PAR_MAX = 3.0e-3
cdef double WM_D_PERP_MIN = 0.3e-3
cdef double WM_D_PERP_MAX = 1.5e-3
cdef double GM_D_MIN = 0.7e-3
cdef double GM_D_MAX = 1.2e-3
cdef double CSF_D = 3.0e-3


def set_diffusivity_ranges(
    wm_d_par_range=(2.0e-3, 3.0e-3),
    wm_d_perp_range=(0.3e-3, 1.5e-3),
    gm_d_iso_range=(0.7e-3, 1.2e-3),
    csf_d=3.0e-3,
):
    """
    Update the diffusivity ranges used by the simulators.

    Parameters
    ----------
    wm_d_par_range : tuple or float
        White matter parallel diffusivity range (min, max) in mm^2/s.
        If a single float, uses fixed value.
    wm_d_perp_range : tuple or float
        White matter perpendicular diffusivity range in mm^2/s.
    gm_d_iso_range : tuple or float
        Gray matter isotropic diffusivity range in mm^2/s.
    csf_d : float
        CSF isotropic diffusivity in mm^2/s.
    """
    global WM_D_PAR_MIN, WM_D_PAR_MAX, WM_D_PERP_MIN, WM_D_PERP_MAX
    global GM_D_MIN, GM_D_MAX, CSF_D

    def _validate_val_or_pair(name, val):
        try:
            if hasattr(val, "__len__") and len(val) == 2:
                lo, hi = float(val[0]), float(val[1])
            else:
                lo = hi = float(val)
        except Exception as e:
            raise ValueError(f"{name} must be a float or (min, max) pair") from e
        if lo > hi:
            raise ValueError(f"{name} min must be <= max")
        return lo, hi

    WM_D_PAR_MIN, WM_D_PAR_MAX = _validate_val_or_pair("wm_d_par_range", wm_d_par_range)
    WM_D_PERP_MIN, WM_D_PERP_MAX = _validate_val_or_pair("wm_d_perp_range", wm_d_perp_range)
    GM_D_MIN, GM_D_MAX = _validate_val_or_pair("gm_d_iso_range", gm_d_iso_range)
    CSF_D = float(csf_d)


cdef inline double get_dperp_extra(double d_par, double f_intra) noexcept:
    """
    Compute extra-axonal perpendicular diffusivity using tortuosity model.

    Parameters
    ----------
    d_par : double
        Parallel diffusivity.
    f_intra : double
        Intra-axonal volume fraction.

    Returns
    -------
    d_perp : double
        Extra-axonal perpendicular diffusivity.
    """
    return d_par * (1.0 - f_intra) / (1.0 + f_intra)


cdef inline double fa_stick_zeppelin(double d_par, double d_perp, double f_intra) noexcept:
    """
    Compute microFA for stick-zeppelin model.

    Calculates microscopic fractional anisotropy for a mixture of
    sticks (Da, 0, 0) and zeppelins (Da, dperp_ex, dperp_ex).

    Parameters
    ----------
    d_par : double
        Parallel diffusivity.
    d_perp : double
        Perpendicular diffusivity.
    f_intra : double
        Intra-axonal volume fraction.

    Returns
    -------
    ufa : double
        Microscopic fractional anisotropy.
    """
    cdef double e = 1.0 - f_intra
    cdef double num = d_par - e * d_perp
    cdef double den = (d_par * d_par + 2.0 * (e * e) * d_perp * d_perp) ** 0.5
    return num / den


cdef inline double _sample_uniform_or_fixed(double lo, double hi) noexcept:
    """Sample uniformly from [lo, hi] or return fixed value if lo == hi."""
    if lo == hi:
        return lo
    else:
        return float(np.random.uniform(lo, hi))


cdef inline double sample_wm_d_par() noexcept:
    """Sample white matter parallel diffusivity."""
    return _sample_uniform_or_fixed(WM_D_PAR_MIN, WM_D_PAR_MAX)


cdef inline double sample_wm_d_perp() noexcept:
    """Sample white matter perpendicular diffusivity."""
    return _sample_uniform_or_fixed(WM_D_PERP_MIN, WM_D_PERP_MAX)


cdef inline double sample_gm_d_iso() noexcept:
    """Sample gray matter isotropic diffusivity."""
    return _sample_uniform_or_fixed(GM_D_MIN, GM_D_MAX)


cdef Py_ssize_t _closest_direction(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=1] vec
) noexcept:
    """
    Find the closest direction on a sphere to a given vector.

    Parameters
    ----------
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    vec : ndarray (3,)
        Target direction vector.

    Returns
    -------
    idx : int
        Index of the closest direction.
    """
    cdef Py_ssize_t i, n = target_sphere.shape[0]
    cdef double best = 1e300
    cdef double dist, dx, dy, dz
    cdef Py_ssize_t best_idx = 0
    cdef const double[:, :] ts_mv = target_sphere

    for i in range(n):
        dx = ts_mv[i, 0] - vec[0]
        dy = ts_mv[i, 1] - vec[1]
        dz = ts_mv[i, 2] - vec[2]
        dist = dx * dx + dy * dy + dz * dz
        if dist < best:
            best = dist
            best_idx = i

    return best_idx


def angle_between(v1, v2):
    """
    Compute the angle between two vectors.

    Parameters
    ----------
    v1 : ndarray (3,)
        First vector.
    v2 : ndarray (3,)
        Second vector.

    Returns
    -------
    angle : float
        Angle between the two vectors in radians.
    """
    cos_theta = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return np.arccos(cos_theta)


def is_angle_valid(angle, *, threshold=30):
    """
    Check if an angle satisfies minimum separation constraint.

    Parameters
    ----------
    angle : float
        Angle to check in radians.
    threshold : float, optional
        Minimum separation threshold in degrees.

    Returns
    -------
    is_valid : bool
        True if the angle satisfies the separation constraint.
    """
    angle_degrees = np.degrees(angle)
    return not (
        (-1 * threshold <= angle_degrees <= threshold) or
        (180 - threshold <= angle_degrees <= 180 + threshold)
    )


cpdef tuple generate_single_fiber(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    object multi_tensor_func,
    bint tortuosity
):
    """
    Generate diffusion signal for a single fiber population.

    Parameters
    ----------
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    evecs : ndarray (N, 3, 3)
        Eigenvector matrices for each sphere direction.
    bingham_sf : dict
        Pre-computed Bingham spherical functions.
    odi_list : ndarray
        List of orientation dispersion index values.
    bvals : ndarray
        B-values.
    bvecs : ndarray (M, 3)
        Gradient directions.
    multi_tensor_func : callable
        Function to compute multi-tensor signal.
    tortuosity : bool
        Whether to use tortuosity constraint.

    Returns
    -------
    tuple
        (signal, labels, num_fibers, dispersion, placeholder,
         neurite_density, odf, d_par, d_perp, fractions, f_ins)
    """
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double f_intra = float(np.random.uniform(0.6, 0.9))
        double f_extra = 1.0 - f_intra
        double d_par = sample_wm_d_par()
        double d_perp_extra
        double S0 = 100.0
        int idx

    if tortuosity:
        d_perp_extra = get_dperp_extra(d_par, f_intra)
    else:
        d_perp_extra = sample_wm_d_perp()

    labels = np.zeros(n_dirs, dtype=np.uint8)

    mevals_ex = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_ex[:, 0] = d_par
    mevals_ex[:, 1] = d_perp_extra
    mevals_ex[:, 2] = d_perp_extra

    mevals_in = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_in[:, 0] = d_par
    mevals_in[:, 1] = 0.0
    mevals_in[:, 2] = 0.0

    idx = int(np.random.randint(0, n_dirs))
    true_stick = target_sphere[idx]

    factor = float(np.random.choice(odi_list))

    fodf_gt = bingham_sf[idx][factor]
    fodf_gt = np.ascontiguousarray(fodf_gt, dtype=np.float64)
    fodf_gt = fodf_gt / np.sum(fodf_gt)

    S_in = multi_tensor_func(mevals_in, evecs, fodf_gt * 100.0, bvals, bvecs)
    S_ex = multi_tensor_func(mevals_ex, evecs, fodf_gt * 100.0, bvals, bvecs)

    S = f_intra * S_in + f_extra * S_ex

    nearest = _closest_direction(target_sphere, true_stick)
    labels[nearest] = 1

    return (
        S * S0,
        labels,
        1,
        factor,
        0.0,
        1.0 * f_intra,
        fodf_gt,
        d_par,
        f_extra * d_perp_extra,
        [1.0, 0.0, 0.0],
        [f_intra],
    )


cpdef tuple generate_two_fibers(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    object multi_tensor_func,
    bint tortuosity
):
    """
    Generate diffusion signal for two crossing fiber populations.

    Parameters
    ----------
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    evecs : ndarray (N, 3, 3)
        Eigenvector matrices for each sphere direction.
    bingham_sf : dict
        Pre-computed Bingham spherical functions.
    odi_list : ndarray
        List of orientation dispersion index values.
    bvals : ndarray
        B-values.
    bvecs : ndarray (M, 3)
        Gradient directions.
    multi_tensor_func : callable
        Function to compute multi-tensor signal.
    tortuosity : bool
        Whether to use tortuosity constraint.

    Returns
    -------
    tuple
        Signal and associated parameters.
    """
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double S0 = 100.0
        double d_par, d_perp_extra
        double fiber_frac1
        double wm_nd, wm_d_perp
        int i
        int idx0, idx1

    f_in = np.random.uniform(0.6, 0.9, 2).astype(np.float64)
    fiber_frac1 = float(np.random.uniform(0.2, 0.8))
    fiber_fractions = [fiber_frac1, 1.0 - fiber_frac1]

    d_par = sample_wm_d_par()
    if tortuosity:
        d_perp_extra = get_dperp_extra(d_par, float(f_in[0]))
    else:
        d_perp_extra = sample_wm_d_perp()

    labels = np.zeros(n_dirs, dtype=np.uint8)

    mevals_ex = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_ex[:, 0] = d_par
    mevals_ex[:, 1] = d_perp_extra
    mevals_ex[:, 2] = d_perp_extra

    mevals_in = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_in[:, 0] = d_par
    mevals_in[:, 1] = 0.0
    mevals_in[:, 2] = 0.0

    _MAX_ANGLE_TRIES = 10000
    index = np.random.randint(0, n_dirs, 2)
    _angle_tries = 0
    while not is_angle_valid(
        angle_between(target_sphere[index[0]], target_sphere[index[1]])
    ):
        index = np.random.randint(0, n_dirs, 2)
        _angle_tries += 1
        if _angle_tries >= _MAX_ANGLE_TRIES:
            raise RuntimeError(
                "Could not find two fiber directions with sufficient angular "
                "separation after %d attempts. Consider using a finer sphere."
                % _MAX_ANGLE_TRIES
            )

    idx0 = int(index[0])
    idx1 = int(index[1])
    true_stick1 = target_sphere[idx0]
    true_stick2 = target_sphere[idx1]

    factor = float(np.random.choice(odi_list))

    S = np.zeros(bvals.shape[0], dtype=np.float64)
    fodf = np.zeros(n_dirs, dtype=np.float64)

    for i in range(2):
        fodf_gt = bingham_sf[int(index[i])][factor]
        fodf_gt = np.ascontiguousarray(fodf_gt, dtype=np.float64)
        fodf_gt = fodf_gt / np.sum(fodf_gt)

        fodf += fiber_fractions[i] * fodf_gt

        S_in = multi_tensor_func(mevals_in, evecs, fodf_gt * 100.0, bvals, bvecs)
        S_ex = multi_tensor_func(mevals_ex, evecs, fodf_gt * 100.0, bvals, bvecs)

        f_intra = float(f_in[i])
        f_extra = 1.0 - f_intra

        S += fiber_fractions[i] * (f_intra * S_in + f_extra * S_ex)

    labels[_closest_direction(target_sphere, true_stick1)] = 1
    labels[_closest_direction(target_sphere, true_stick2)] = 1

    wm_nd = fiber_fractions[0] * float(f_in[0]) + fiber_fractions[1] * float(f_in[1])
    wm_d_perp = (
        fiber_fractions[0] * (1.0 - float(f_in[0])) * d_perp_extra
        + fiber_fractions[1] * (1.0 - float(f_in[1])) * d_perp_extra
    )

    return (
        S * S0,
        labels,
        2,
        factor,
        0.0,
        wm_nd,
        fodf,
        d_par,
        wm_d_perp,
        [fiber_frac1, 1.0 - fiber_frac1, 0.0],
        f_in.tolist(),
    )


cpdef tuple generate_three_fibers(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    object multi_tensor_func,
    bint tortuosity
):
    """
    Generate diffusion signal for three crossing fiber populations.

    Parameters
    ----------
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    evecs : ndarray (N, 3, 3)
        Eigenvector matrices for each sphere direction.
    bingham_sf : dict
        Pre-computed Bingham spherical functions.
    odi_list : ndarray
        List of orientation dispersion index values.
    bvals : ndarray
        B-values.
    bvecs : ndarray (M, 3)
        Gradient directions.
    multi_tensor_func : callable
        Function to compute multi-tensor signal.
    tortuosity : bool
        Whether to use tortuosity constraint.

    Returns
    -------
    tuple
        Signal and associated parameters.
    """
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double S0 = 100.0
        double d_par, d_perp_extra
        double wm_nd = 0.0
        double wm_d_perp = 0.0
        int k
        int idx0, idx1, idx2

    f_in = np.random.uniform(0.6, 0.9, 3).astype(np.float64)
    fiber_fracs = np.random.dirichlet([1.0, 1.0, 1.0]).astype(np.float64)
    while np.any(fiber_fracs < 0.2):
        fiber_fracs = np.random.dirichlet([1.0, 1.0, 1.0]).astype(np.float64)

    d_par = sample_wm_d_par()
    if tortuosity:
        d_perp_extra = get_dperp_extra(d_par, float(f_in[0]))
    else:
        d_perp_extra = sample_wm_d_perp()

    labels = np.zeros(n_dirs, dtype=np.uint8)

    mevals_ex = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_ex[:, 0] = d_par
    mevals_ex[:, 1] = d_perp_extra
    mevals_ex[:, 2] = d_perp_extra

    mevals_in = np.zeros_like(target_sphere, dtype=np.float64)
    mevals_in[:, 0] = d_par
    mevals_in[:, 1] = 0.0
    mevals_in[:, 2] = 0.0

    _MAX_ANGLE_TRIES = 10000
    index = np.random.randint(0, n_dirs, 3)
    _angle_tries = 0
    while (
        not is_angle_valid(
            angle_between(target_sphere[index[0]], target_sphere[index[1]]),
            threshold=60
        )
        or not is_angle_valid(
            angle_between(target_sphere[index[0]], target_sphere[index[2]]),
            threshold=60
        )
        or not is_angle_valid(
            angle_between(target_sphere[index[1]], target_sphere[index[2]]),
            threshold=60
        )
    ):
        index = np.random.randint(0, n_dirs, 3)
        _angle_tries += 1
        if _angle_tries >= _MAX_ANGLE_TRIES:
            raise RuntimeError(
                "Could not find three fiber directions with sufficient angular "
                "separation after %d attempts. Consider using a finer sphere."
                % _MAX_ANGLE_TRIES
            )

    idx0 = int(index[0])
    idx1 = int(index[1])
    idx2 = int(index[2])

    true_stick1 = target_sphere[idx0]
    true_stick2 = target_sphere[idx1]
    true_stick3 = target_sphere[idx2]

    factor = float(np.random.choice(odi_list))

    fodf = np.zeros(n_dirs, dtype=np.float64)
    S = np.zeros(bvals.shape[0], dtype=np.float64)

    for k in range(3):
        fodf_gt = bingham_sf[int(index[k])][factor]
        fodf_gt = np.ascontiguousarray(fodf_gt, dtype=np.float64)
        fodf_gt = fodf_gt / np.sum(fodf_gt)

        fodf += fiber_fracs[k] * fodf_gt

        S_in = multi_tensor_func(mevals_in, evecs, fodf_gt * 100.0, bvals, bvecs)
        S_ex = multi_tensor_func(mevals_ex, evecs, fodf_gt * 100.0, bvals, bvecs)

        S += fiber_fracs[k] * (float(f_in[k]) * S_in + (1.0 - float(f_in[k])) * S_ex)

    labels[_closest_direction(target_sphere, true_stick1)] = 1
    labels[_closest_direction(target_sphere, true_stick2)] = 1
    labels[_closest_direction(target_sphere, true_stick3)] = 1

    for k in range(3):
        wm_nd += fiber_fracs[k] * float(f_in[k])
        wm_d_perp += fiber_fracs[k] * (1.0 - float(f_in[k])) * d_perp_extra

    return (
        S * S0,
        labels,
        3,
        factor,
        0.0,
        wm_nd,
        fodf,
        d_par,
        wm_d_perp,
        fiber_fracs.tolist(),
        f_in.tolist(),
    )


cpdef tuple create_wm_signal(
    int num_fib,
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    object multi_tensor_func,
    bint tortuosity
):
    """
    Create white matter signal with specified number of fibers.

    Parameters
    ----------
    num_fib : int
        Number of fiber populations (1, 2, or 3).
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    evecs : ndarray (N, 3, 3)
        Eigenvector matrices for each sphere direction.
    bingham_sf : dict
        Pre-computed Bingham spherical functions.
    odi_list : ndarray
        List of orientation dispersion index values.
    bvals : ndarray
        B-values.
    bvecs : ndarray (M, 3)
        Gradient directions.
    multi_tensor_func : callable
        Function to compute multi-tensor signal.
    tortuosity : bool
        Whether to use tortuosity constraint.

    Returns
    -------
    tuple
        Signal and associated parameters.
    """
    if num_fib == 1:
        return generate_single_fiber(
            target_sphere, evecs, bingham_sf, odi_list,
            bvals, bvecs, multi_tensor_func, tortuosity
        )
    elif num_fib == 2:
        return generate_two_fibers(
            target_sphere, evecs, bingham_sf, odi_list,
            bvals, bvecs, multi_tensor_func, tortuosity
        )
    elif num_fib == 3:
        return generate_three_fibers(
            target_sphere, evecs, bingham_sf, odi_list,
            bvals, bvecs, multi_tensor_func, tortuosity
        )
    else:
        raise ValueError("num_fib must be 1, 2 or 3")


cpdef tuple create_gm_signal(
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere
):
    """
    Create gray matter isotropic diffusion signal.

    Parameters
    ----------
    bvals : ndarray
        B-values.
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.

    Returns
    -------
    tuple
        (signal, labels, num_fibers, dispersion, placeholder,
         neurite_density, odf, d_par, d_perp)
    """
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double d = sample_gm_d_iso()

    signal = np.exp(-bvals * d) * 100.0
    labels = np.zeros(n_dirs, dtype=np.uint8)
    gm_odf = np.ones(n_dirs, dtype=np.float64) / float(n_dirs)

    return (
        signal,
        labels,
        0,
        1.0,
        0.0,
        0.0,
        gm_odf,
        d,
        d,
    )


cpdef tuple create_csf_signal(
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere
):
    """
    Create CSF free water diffusion signal.

    Parameters
    ----------
    bvals : ndarray
        B-values.
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.

    Returns
    -------
    tuple
        (signal, labels, num_fibers, placeholder, free_water_fraction,
         placeholder, odf, d_par, d_perp)
    """
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double d = CSF_D

    signal = np.exp(-bvals * d) * 100.0
    labels = np.zeros(n_dirs, dtype=np.uint8)
    csf_odf = np.zeros(n_dirs, dtype=np.float64)

    return (
        signal,
        labels,
        0,
        1.0,
        1.0,
        0.0,
        csf_odf,
        d,
        d,
    )


cpdef tuple create_mixed_signal(
    cnp.ndarray[DTYPE_t, ndim=2] target_sphere,
    cnp.ndarray[DTYPE_t, ndim=3] evecs,
    object bingham_sf,
    cnp.ndarray[DTYPE_t, ndim=1] odi_list,
    cnp.ndarray[DTYPE_t, ndim=1] bvals,
    cnp.ndarray[DTYPE_t, ndim=2] bvecs,
    object multi_tensor_func,
    double wm_threshold,
    bint tortuosity
):
    """
    Create mixed WM/GM/CSF tissue signal.

    This is the main simulation function that generates realistic
    voxel signals with mixed tissue contributions.

    Parameters
    ----------
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    evecs : ndarray (N, 3, 3)
        Eigenvector matrices for each sphere direction.
    bingham_sf : dict
        Pre-computed Bingham spherical functions.
    odi_list : ndarray
        List of orientation dispersion index values.
    bvals : ndarray
        B-values.
    bvecs : ndarray (M, 3)
        Gradient directions.
    multi_tensor_func : callable
        Function to compute multi-tensor signal.
    wm_threshold : float
        Minimum WM fraction to include fiber labels.
    tortuosity : bool
        Whether to use tortuosity constraint.

    Returns
    -------
    tuple
        (signal, labels, num_fibers, dispersion, wm_fraction,
         gm_fraction, csf_fraction, neurite_density, odf,
         ufa_wm, ufa_voxel, fiber_fractions, wm_disp,
         wm_d_par, wm_d_perp, gm_d_par, csf_d_par, f_ins)
    """
    cdef:
        Py_ssize_t n_dirs = target_sphere.shape[0]
        double wm_fraction
        double gm_fraction
        double csf_fraction
        int num_fiber
        double odi
        double nd
        double ufa_wm = 0.0
        double ufa_voxel
        int k

    fractions = np.random.dirichlet([2.0, 1.0, 1.0]).astype(np.float64)
    wm_fraction = float(fractions[0])
    gm_fraction = float(fractions[1])
    csf_fraction = float(fractions[2])

    num_fiber = int(np.random.choice([1, 2, 3], p=[0.1, 0.2, 0.7]))

    wm_result = create_wm_signal(
        num_fiber,
        target_sphere,
        evecs,
        bingham_sf,
        odi_list,
        bvals,
        bvecs,
        multi_tensor_func,
        tortuosity,
    )
    wm_signal = wm_result[0]
    wm_label = wm_result[1]
    wm_num_fib = wm_result[2]
    wm_disp = wm_result[3]
    wm_nd = wm_result[5]
    wm_odf = wm_result[6]
    wm_d_par = wm_result[7]
    wm_d_perp = wm_result[8]
    fracs = wm_result[9]
    f_ins = wm_result[10]

    gm_result = create_gm_signal(bvals, target_sphere)
    gm_signal = gm_result[0]
    gm_disp = gm_result[3]
    gm_nd = gm_result[5]
    gm_d_par = gm_result[7]

    csf_result = create_csf_signal(bvals, target_sphere)
    csf_signal = csf_result[0]
    csf_d_par = csf_result[7]

    odi = wm_fraction * float(wm_disp) + gm_fraction * float(gm_disp) + csf_fraction * 1.0
    nd = wm_fraction * float(wm_nd) + gm_fraction * float(gm_nd)

    combined_signal = (
        wm_fraction * wm_signal
        + gm_fraction * gm_signal
        + csf_fraction * csf_signal
    )

    if wm_fraction > wm_threshold:
        combined_odf = 50.0 * wm_fraction * wm_odf
    else:
        wm_label = np.zeros(n_dirs, dtype=np.uint8)
        combined_odf = np.zeros(n_dirs, dtype=np.float16)

    for k in range(wm_num_fib):
        ufa_wm += fa_stick_zeppelin(
            float(wm_d_par),
            float(wm_d_perp),
            float(f_ins[k]),
        ) * float(fracs[k])

    ufa_voxel = ufa_wm * wm_fraction

    frac_arr = np.zeros(3, dtype=np.float32)
    for k in range(min(3, len(fracs))):
        frac_arr[k] = float(fracs[k])

    f_ins_arr = np.zeros(3, dtype=np.float32)
    for k in range(min(3, len(f_ins))):
        f_ins_arr[k] = float(f_ins[k])

    return (
        combined_signal,
        wm_label,
        wm_num_fib,
        odi,
        wm_fraction,
        gm_fraction,
        csf_fraction,
        nd,
        combined_odf.astype(np.float16),
        float(ufa_wm),
        float(ufa_voxel),
        frac_arr,
        float(wm_disp),
        float(wm_d_par),
        float(wm_d_perp),
        float(gm_d_par),
        float(csf_d_par),
        f_ins_arr,
    )
