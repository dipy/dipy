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


def is_angle_valid(angle, threshold=30):
    """
    Check if an angle satisfies minimum separation constraint.

    Parameters
    ----------
    angle : float
        Angle to check in radians.
    threshold : float, optional
        Minimum separation threshold in degrees. Default is 30.

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

    index = np.random.randint(0, n_dirs, 2)
    while not is_angle_valid(
        angle_between(target_sphere[index[0]], target_sphere[index[1]])
    ):
        index = np.random.randint(0, n_dirs, 2)

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
