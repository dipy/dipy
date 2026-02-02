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
