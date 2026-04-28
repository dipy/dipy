# cython: profile=True
# cython: embedsignature=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
"""
Optimized Cython routines for Diffusion Kurtosis Imaging (DKI).

This module contains Cython-optimized versions of computationally intensive
DKI functions, particularly the mean kurtosis calculation.
"""

cimport cython
import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt, fabs, atan, log

cnp.import_array()


cdef inline double arctanh_c(double x) noexcept nogil:
    """Compute inverse hyperbolic tangent."""
    return 0.5 * log((1.0 + x) / (1.0 - x))

# Defining keys to select a kurtosis tensor element with indexes (i, j, k, l)
# on a kt vector that contains only the 15 independent elements of the kurtosis
# tensor.
cdef int[16] ind_ele_arr = [0, 0, 3, 4, 9, 5, 12, 7, 8, 10, 9, 11, 13, 13, 14, 11]

# Map from key = (i+1)*(j+1)*(k+1)*(l+1) to index in 15-element kt array
cdef inline int get_kt_index(int key) noexcept nogil:
    """Get index into 15-element kt array from key."""
    if key == 1:
        return 0
    elif key == 16:
        return 1
    elif key == 81:
        return 2
    elif key == 2:
        return 3
    elif key == 3:
        return 4
    elif key == 8:
        return 5
    elif key == 24:
        return 6
    elif key == 27:
        return 7
    elif key == 54:
        return 8
    elif key == 4:
        return 9
    elif key == 9:
        return 10
    elif key == 36:
        return 11
    elif key == 6:
        return 12
    elif key == 12:
        return 13
    elif key == 18:
        return 14
    else:
        return 0



cdef inline bint positive_evals_single(double L1, double L2, double L3, 
                                        double er=2e-7) noexcept nogil:
    """Check if all eigenvalues are significantly larger than zero.

    Parameters
    ----------
    L1 : double
        First eigenvalue of the diffusion tensor.
    L2 : double
        Second eigenvalue of the diffusion tensor.
    L3 : double
        Third eigenvalue of the diffusion tensor.
    er : double, optional
        Threshold below which an eigenvalue is considered non-positive.

    Returns
    -------
    result : bint
        True if all three eigenvalues exceed ``er``, False otherwise.
    """
    return L1 > er and L2 > er and L3 > er



cdef double carlson_rf_single(double x, double y, double z, 
                               double errtol=3e-4) noexcept nogil:
    """Compute Carlson's incomplete elliptic integral of the first kind
    for a single set of values.
    
    Parameters
    ----------
    x, y, z : double
        Independent variables of the integral.
    errtol : double
        Error tolerance.
    
    Returns
    -------
    RF : double
        Value of the incomplete first order elliptic integral.
    """
    cdef:
        double xn = x
        double yn = y
        double zn = z
        double An = (xn + yn + zn) / 3.0
        double Q, xnroot, ynroot, znroot, lamda
        double X, Y, Z, E2, E3, RF
        double max_diff
        int n = 0
    
    # Compute Q
    max_diff = fabs(An - xn)
    if fabs(An - yn) > max_diff:
        max_diff = fabs(An - yn)
    if fabs(An - zn) > max_diff:
        max_diff = fabs(An - zn)
    Q = (3.0 * errtol) ** (-1.0 / 6.0) * max_diff
    
    # Convergence loop
    while (4.0 ** (-n)) * Q > fabs(An):
        xnroot = sqrt(xn)
        ynroot = sqrt(yn)
        znroot = sqrt(zn)
        lamda = xnroot * (ynroot + znroot) + ynroot * znroot
        n = n + 1
        xn = (xn + lamda) * 0.25
        yn = (yn + lamda) * 0.25
        zn = (zn + lamda) * 0.25
        An = (An + lamda) * 0.25
    
    # Post convergence calculation
    X = 1.0 - xn / An
    Y = 1.0 - yn / An
    Z = -X - Y
    E2 = X * Y - Z * Z
    E3 = X * Y * Z
    RF = (An ** (-0.5)) * (1.0 - E2 / 10.0 + E3 / 14.0 + 
                           (E2 * E2) / 24.0 - 3.0 / 44.0 * E2 * E3)
    
    return RF



cdef double carlson_rd_single(double x, double y, double z, 
                               double errtol=1e-4) noexcept nogil:
    """Compute Carlson's incomplete elliptic integral of the second kind
    for a single set of values.
    
    Parameters
    ----------
    x, y, z : double
        Independent variables of the integral.
    errtol : double
        Error tolerance.
    
    Returns
    -------
    RD : double
        Value of the incomplete second order elliptic integral.
    """
    cdef:
        double xn = x
        double yn = y
        double zn = z
        double A0 = (xn + yn + 3.0 * zn) / 5.0
        double An = A0
        double Q, xnroot, ynroot, znroot, lamda
        double sum_term = 0.0
        double X, Y, Z, E2, E3, E4, E5, RD
        double max_diff
        double power_4n
        int n = 0
    
    # Compute Q
    max_diff = fabs(An - xn)
    if fabs(An - yn) > max_diff:
        max_diff = fabs(An - yn)
    if fabs(An - zn) > max_diff:
        max_diff = fabs(An - zn)
    Q = (errtol / 4.0) ** (-1.0 / 6.0) * max_diff
    
    # Convergence loop
    while (4.0 ** (-n)) * Q > fabs(An):
        xnroot = sqrt(xn)
        ynroot = sqrt(yn)
        znroot = sqrt(zn)
        lamda = xnroot * (ynroot + znroot) + ynroot * znroot
        sum_term = sum_term + (4.0 ** (-n)) / (znroot * (zn + lamda))
        n = n + 1
        xn = (xn + lamda) * 0.25
        yn = (yn + lamda) * 0.25
        zn = (zn + lamda) * 0.25
        An = (An + lamda) * 0.25
    
    # Post convergence calculation
    power_4n = 4.0 ** n
    X = (A0 - x) / (power_4n * An)
    Y = (A0 - y) / (power_4n * An)
    Z = -(X + Y) / 3.0
    E2 = X * Y - 6.0 * Z * Z
    E3 = (3.0 * X * Y - 8.0 * Z * Z) * Z
    E4 = 3.0 * (X * Y - Z * Z) * Z * Z
    E5 = X * Y * Z * Z * Z
    
    RD = ((4.0 ** (-n)) * (An ** (-1.5)) * 
          (1.0 - 3.0 / 14.0 * E2 + 1.0 / 6.0 * E3 + 
           9.0 / 88.0 * (E2 * E2) - 3.0 / 22.0 * E4 - 
           9.0 / 52.0 * E2 * E3 + 3.0 / 26.0 * E5) + 
          3.0 * sum_term)
    
    return RD



cdef double F2m_single(double a, double b, double c, double er=2.5e-2) noexcept nogil:
    """Compute function F2 for a single voxel.

    Helper function required to compute the analytical solution of Mean
    Kurtosis.

    Parameters
    ----------
    a : double
        First diffusion tensor eigenvalue.
    b : double
        Second diffusion tensor eigenvalue.
    c : double
        Third diffusion tensor eigenvalue.
    er : double, optional
        Relative tolerance for detecting degenerate eigenvalue pairs.

    Returns
    -------
    F2 : double
        Value of the F2 function. Returns 0.0 when eigenvalues are
        non-positive.
    """
    cdef:
        double L1, L2, L3, RF, RD, F2
        double x, alpha
    
    # Check if eigenvalues are positive
    if not positive_evals_single(a, b, c):
        return 0.0
    
    # Check for singularity b==c
    if fabs(b - c) < b * er:
        # Check for singularity a==b and a==c (isotropic case)
        if fabs(a - b) < b * er:
            return 6.0 / 15.0
        
        # Singularity b==c but a!=b
        L1 = a
        L3 = (c + b) / 2.0
        
        # Compute alpha
        x = 1.0 - (L1 / L3)
        if x > 0:
            alpha = 1.0 / sqrt(x) * arctanh_c(sqrt(x))
        else:
            alpha = 1.0 / sqrt(-x) * atan(sqrt(-x))
        
        F2 = (6.0 * ((L1 + 2.0 * L3) ** 2) / 
              (144.0 * L3 * L3 * (L1 - L3) * (L1 - L3)) * 
              (L3 * (L1 + 2.0 * L3) + L1 * (L1 - 4.0 * L3) * alpha))
        return F2
    
    # Non-singular case b!=c
    L1 = a
    L2 = b
    L3 = c
    RF = carlson_rf_single(L1 / L2, L1 / L3, 1.0)
    RD = carlson_rd_single(L1 / L2, L1 / L3, 1.0)
    
    F2 = (((L1 + L2 + L3) ** 2) / (3.0 * (L2 - L3) * (L2 - L3)) * 
          (((L2 + L3) / (sqrt(L2 * L3))) * RF + 
           ((2.0 * L1 - L2 - L3) / (3.0 * sqrt(L2 * L3))) * RD - 2.0))
    
    return F2



cdef double F1m_single(double a, double b, double c, double er=2.5e-2) noexcept nogil:
    """Compute function F1 for a single voxel.

    Helper function required to compute the analytical solution of Mean
    Kurtosis.

    Parameters
    ----------
    a : double
        First diffusion tensor eigenvalue.
    b : double
        Second diffusion tensor eigenvalue.
    c : double
        Third diffusion tensor eigenvalue.
    er : double, optional
        Relative tolerance for detecting degenerate eigenvalue pairs.

    Returns
    -------
    F1 : double
        Value of the F1 function. Returns 0.0 when eigenvalues are
        non-positive, or 1/5 in the fully isotropic case.
    """
    cdef:
        double L1, L2, L3, RF, RD, F1
    
    # Check if eigenvalues are positive
    if not positive_evals_single(a, b, c):
        return 0.0
    
    # Check for singularity a==b and a==c (isotropic case)
    if fabs(a - b) < a * er and fabs(a - c) < a * er:
        return 1.0 / 5.0
    
    # Check for singularity a==b
    if fabs(a - b) < a * er and fabs(a - c) >= a * er:
        L1 = (a + b) / 2.0
        L3 = c
        return F2m_single(L3, L1, L1) / 2.0
    
    # Check for singularity a==c
    if fabs(a - c) < a * er and fabs(a - b) >= a * er:
        L1 = (a + c) / 2.0
        L2 = b
        return F2m_single(L2, L1, L1) / 2.0
    
    # Non-singular case a!=b and a!=c
    L1 = a
    L2 = b
    L3 = c
    RF = carlson_rf_single(L1 / L2, L1 / L3, 1.0)
    RD = carlson_rd_single(L1 / L2, L1 / L3, 1.0)
    
    F1 = (((L1 + L2 + L3) ** 2) / (18.0 * (L1 - L2) * (L1 - L3)) * 
          ((sqrt(L2 * L3) / L1) * RF + 
           ((3.0 * L1 * L1 - L1 * L2 - L1 * L3 - L2 * L3) / 
            (3.0 * L1 * sqrt(L2 * L3))) * RD - 1.0))
    
    return F1



cdef double G1m_single(double a, double b, double c, double er=2.5e-2) noexcept nogil:
    """Compute function G1 for a single voxel.

    Helper function required to compute the analytical solution of Radial
    Kurtosis.

    Parameters
    ----------
    a : double
        First diffusion tensor eigenvalue (axial).
    b : double
        Second diffusion tensor eigenvalue (radial).
    c : double
        Third diffusion tensor eigenvalue (radial).
    er : double, optional
        Relative tolerance for detecting the singularity b == c.

    Returns
    -------
    G1 : double
        Value of the G1 function. Returns 0.0 when eigenvalues are
        non-positive.
    """
    cdef double L1, L2, G1

    if not positive_evals_single(a, b, c):
        return 0.0

    if fabs(b - c) >= b * er:
        G1 = ((a + b + c) * (a + b + c) /
              (18.0 * b * (b - c) * (b - c)) *
              (2.0 * b + (c * c - 3.0 * b * c) / sqrt(b * c)))
        return G1

    # Singularity b==c
    L1 = a
    L2 = b
    G1 = (L1 + 2.0 * L2) * (L1 + 2.0 * L2) / (24.0 * L2 * L2)
    return G1



cdef double G2m_single(double a, double b, double c, double er=2.5e-2) noexcept nogil:
    """Compute function G2 for a single voxel.

    Helper function required to compute the analytical solution of Radial
    Kurtosis.

    Parameters
    ----------
    a : double
        First diffusion tensor eigenvalue (axial).
    b : double
        Second diffusion tensor eigenvalue (radial).
    c : double
        Third diffusion tensor eigenvalue (radial).
    er : double, optional
        Relative tolerance for detecting the singularity b == c.

    Returns
    -------
    G2 : double
        Value of the G2 function. Returns 0.0 when eigenvalues are
        non-positive.
    """
    cdef double L1, L2, G2

    if not positive_evals_single(a, b, c):
        return 0.0

    if fabs(b - c) >= b * er:
        G2 = ((a + b + c) * (a + b + c) /
              (3.0 * (b - c) * (b - c)) *
              ((b + c) / sqrt(b * c) - 2.0))
        return G2

    # Singularity b==c
    L1 = a
    L2 = b
    G2 = (L1 + 2.0 * L2) * (L1 + 2.0 * L2) / (12.0 * L2 * L2)
    return G2



cdef double Wrotate_element_single(double[:] kt, int indi, int indj, 
                                    int indk, int indl, 
                                    double[:, :] B) noexcept nogil:
    """Compute a single rotated kurtosis tensor element.
    
    Parameters
    ----------
    kt : array (15,)
        15 independent elements of the kurtosis tensor.
    indi, indj, indk, indl : int
        Indices of the rotated tensor element (0, 1, or 2 for x, y, z).
    B : array (3, 3)
        Eigenvector matrix (columns are eigenvectors).
    
    Returns
    -------
    Wre : double
        Rotated kurtosis tensor element.
    """
    cdef:
        double Wre = 0.0
        int il, jl, kl, ll, key, idx
        double multiplyB
    
    for il in range(3):
        for jl in range(3):
            for kl in range(3):
                for ll in range(3):
                    key = (il + 1) * (jl + 1) * (kl + 1) * (ll + 1)
                    idx = get_kt_index(key)
                    multiplyB = (B[il, indi] * B[jl, indj] * 
                                 B[kl, indk] * B[ll, indl])
                    Wre = Wre + multiplyB * kt[idx]
    
    return Wre



def mean_kurtosis_analytical(double[:, :] dki_params_flat,
                              double min_kurtosis=-3.0/7.0,
                              double max_kurtosis=3.0):
    """Compute mean kurtosis using the analytical solution.

    This is the Cython-optimized version of the analytical mean kurtosis
    calculation.

    Parameters
    ----------
    dki_params_flat : ndarray (n, 27)
        Flattened DKI parameters array.
    min_kurtosis : float, optional
        Minimum kurtosis value for clipping.
    max_kurtosis : float, optional
        Maximum kurtosis value for clipping.

    Returns
    -------
    MK : ndarray (n,)
        Mean kurtosis values.
    """
    cdef:
        cnp.npy_intp n_voxels = dki_params_flat.shape[0]
        cnp.npy_intp v
        double[:] MK = np.zeros(n_voxels, dtype=np.float64)
        double[:] evals = np.zeros(3, dtype=np.float64)
        double[:, :] evecs = np.zeros((3, 3), dtype=np.float64)
        double[:] kt = np.zeros(15, dtype=np.float64)
        double L1, L2, L3
        double Wxxxx, Wyyyy, Wzzzz, Wxxyy, Wxxzz, Wyyzz
        double F1_1, F1_2, F1_3, F2_1, F2_2, F2_3
        double mk_val
        int i, j
    
    with nogil:
        for v in range(n_voxels):
            # Extract eigenvalues
            L1 = dki_params_flat[v, 0]
            L2 = dki_params_flat[v, 1]
            L3 = dki_params_flat[v, 2]
            
            # Check if eigenvalues are valid
            if not positive_evals_single(L1, L2, L3):
                MK[v] = 0.0
                continue
            
            # Extract eigenvectors (stored as 3 rows of 3 elements each)
            for i in range(3):
                for j in range(3):
                    evecs[i, j] = dki_params_flat[v, 3 + i * 3 + j]
            
            # Extract kurtosis tensor elements
            for i in range(15):
                kt[i] = dki_params_flat[v, 12 + i]
            
            # Rotate kurtosis tensor elements
            Wxxxx = Wrotate_element_single(kt, 0, 0, 0, 0, evecs)
            Wyyyy = Wrotate_element_single(kt, 1, 1, 1, 1, evecs)
            Wzzzz = Wrotate_element_single(kt, 2, 2, 2, 2, evecs)
            Wxxyy = Wrotate_element_single(kt, 0, 0, 1, 1, evecs)
            Wxxzz = Wrotate_element_single(kt, 0, 0, 2, 2, evecs)
            Wyyzz = Wrotate_element_single(kt, 1, 1, 2, 2, evecs)
            
            # Compute F1 and F2 functions
            F1_1 = F1m_single(L1, L2, L3)
            F1_2 = F1m_single(L2, L1, L3)
            F1_3 = F1m_single(L3, L2, L1)
            F2_1 = F2m_single(L1, L2, L3)
            F2_2 = F2m_single(L2, L1, L3)
            F2_3 = F2m_single(L3, L2, L1)
            
            # Compute MK
            mk_val = (F1_1 * Wxxxx + F1_2 * Wyyyy + F1_3 * Wzzzz +
                      F2_1 * Wyyzz + F2_2 * Wxxzz + F2_3 * Wxxyy)
            
            # Apply clipping
            if mk_val < min_kurtosis:
                mk_val = min_kurtosis
            if mk_val > max_kurtosis:
                mk_val = max_kurtosis
            
            MK[v] = mk_val
    
    return np.asarray(MK)



def axial_kurtosis_analytical(double[:, :] dki_params_flat,
                               double min_kurtosis=-3.0/7.0,
                               double max_kurtosis=10.0):
    """Compute axial kurtosis using the analytical solution.

    Parameters
    ----------
    dki_params_flat : ndarray (n, 27)
        Flattened DKI parameters array.
    min_kurtosis : float, optional
        Minimum kurtosis value for clipping.
    max_kurtosis : float, optional
        Maximum kurtosis value for clipping.

    Returns
    -------
    AK : ndarray (n,)
        Axial kurtosis values.
    """
    cdef:
        cnp.npy_intp n_voxels = dki_params_flat.shape[0]
        cnp.npy_intp v
        double[:] AK = np.zeros(n_voxels, dtype=np.float64)
        double[:, :] evecs = np.zeros((3, 3), dtype=np.float64)
        double[:] kt = np.zeros(15, dtype=np.float64)
        double L1, L2, L3, md, Wxxxx, ak_val
        int i, j

    with nogil:
        for v in range(n_voxels):
            L1 = dki_params_flat[v, 0]
            L2 = dki_params_flat[v, 1]
            L3 = dki_params_flat[v, 2]

            if not positive_evals_single(L1, L2, L3):
                AK[v] = 0.0
                continue

            for i in range(3):
                for j in range(3):
                    evecs[i, j] = dki_params_flat[v, 3 + i * 3 + j]

            for i in range(15):
                kt[i] = dki_params_flat[v, 12 + i]

            Wxxxx = Wrotate_element_single(kt, 0, 0, 0, 0, evecs)

            md = (L1 + L2 + L3) / 3.0
            ak_val = Wxxxx * (md * md) / (L1 * L1)

            if ak_val < min_kurtosis:
                ak_val = min_kurtosis
            if ak_val > max_kurtosis:
                ak_val = max_kurtosis

            AK[v] = ak_val

    return np.asarray(AK)



def radial_kurtosis_analytical(double[:, :] dki_params_flat,
                                double min_kurtosis=-3.0/7.0,
                                double max_kurtosis=10.0):
    """Compute radial kurtosis using the analytical solution.

    Parameters
    ----------
    dki_params_flat : ndarray (n, 27)
        Flattened DKI parameters array.
    min_kurtosis : float, optional
        Minimum kurtosis value for clipping.
    max_kurtosis : float, optional
        Maximum kurtosis value for clipping.

    Returns
    -------
    RK : ndarray (n,)
        Radial kurtosis values.
    """
    cdef:
        cnp.npy_intp n_voxels = dki_params_flat.shape[0]
        cnp.npy_intp v
        double[:] RK = np.zeros(n_voxels, dtype=np.float64)
        double[:, :] evecs = np.zeros((3, 3), dtype=np.float64)
        double[:] kt = np.zeros(15, dtype=np.float64)
        double L1, L2, L3
        double Wyyyy, Wzzzz, Wyyzz
        double rk_val
        int i, j

    with nogil:
        for v in range(n_voxels):
            L1 = dki_params_flat[v, 0]
            L2 = dki_params_flat[v, 1]
            L3 = dki_params_flat[v, 2]

            if not positive_evals_single(L1, L2, L3):
                RK[v] = 0.0
                continue

            for i in range(3):
                for j in range(3):
                    evecs[i, j] = dki_params_flat[v, 3 + i * 3 + j]

            for i in range(15):
                kt[i] = dki_params_flat[v, 12 + i]

            Wyyyy = Wrotate_element_single(kt, 1, 1, 1, 1, evecs)
            Wzzzz = Wrotate_element_single(kt, 2, 2, 2, 2, evecs)
            Wyyzz = Wrotate_element_single(kt, 1, 1, 2, 2, evecs)

            rk_val = (G1m_single(L1, L2, L3) * Wyyyy +
                      G1m_single(L1, L3, L2) * Wzzzz +
                      G2m_single(L1, L2, L3) * Wyyzz)

            if rk_val < min_kurtosis:
                rk_val = min_kurtosis
            if rk_val > max_kurtosis:
                rk_val = max_kurtosis

            RK[v] = rk_val

    return np.asarray(RK)



def kurtosis_fractional_anisotropy_c(double[:, :] dki_params_flat):
    """Compute kurtosis fractional anisotropy (KFA).

    Parameters
    ----------
    dki_params_flat : ndarray (n, 27)
        Flattened DKI parameters array.

    Returns
    -------
    KFA : ndarray (n,)
        Kurtosis fractional anisotropy values.
    """
    cdef:
        cnp.npy_intp n_voxels = dki_params_flat.shape[0]
        cnp.npy_intp v
        double[:] KFA = np.zeros(n_voxels, dtype=np.float64)
        double Wxxxx, Wyyyy, Wzzzz
        double Wxxxy, Wxxxz, Wxyyy, Wyyyz, Wxzzz, Wyzzz
        double Wxxyy, Wxxzz, Wyyzz
        double Wxxyz, Wxyyz, Wxyzz
        double W, A, B

    with nogil:
        for v in range(n_voxels):
            Wxxxx = dki_params_flat[v, 12]
            Wyyyy = dki_params_flat[v, 13]
            Wzzzz = dki_params_flat[v, 14]
            Wxxxy = dki_params_flat[v, 15]
            Wxxxz = dki_params_flat[v, 16]
            Wxyyy = dki_params_flat[v, 17]
            Wyyyz = dki_params_flat[v, 18]
            Wxzzz = dki_params_flat[v, 19]
            Wyzzz = dki_params_flat[v, 20]
            Wxxyy = dki_params_flat[v, 21]
            Wxxzz = dki_params_flat[v, 22]
            Wyyzz = dki_params_flat[v, 23]
            Wxxyz = dki_params_flat[v, 24]
            Wxyyz = dki_params_flat[v, 25]
            Wxyzz = dki_params_flat[v, 26]

            W = (1.0 / 5.0) * (Wxxxx + Wyyyy + Wzzzz +
                                2.0 * Wxxyy + 2.0 * Wxxzz + 2.0 * Wyyzz)

            A = ((Wxxxx - W) * (Wxxxx - W) +
                 (Wyyyy - W) * (Wyyyy - W) +
                 (Wzzzz - W) * (Wzzzz - W) +
                 4.0 * (Wxxxy * Wxxxy + Wxxxz * Wxxxz +
                        Wxyyy * Wxyyy + Wyyyz * Wyyyz +
                        Wxzzz * Wxzzz + Wyzzz * Wyzzz) +
                 6.0 * ((Wxxyy - W / 3.0) * (Wxxyy - W / 3.0) +
                        (Wxxzz - W / 3.0) * (Wxxzz - W / 3.0) +
                        (Wyyzz - W / 3.0) * (Wyyzz - W / 3.0)) +
                 12.0 * (Wxxyz * Wxxyz + Wxyyz * Wxyyz + Wxyzz * Wxyzz))

            B = (Wxxxx * Wxxxx + Wyyyy * Wyyyy + Wzzzz * Wzzzz +
                 4.0 * (Wxxxy * Wxxxy + Wxxxz * Wxxxz +
                        Wxyyy * Wxyyy + Wyyyz * Wyyyz +
                        Wxzzz * Wxzzz + Wyzzz * Wyzzz) +
                 6.0 * (Wxxyy * Wxxyy + Wxxzz * Wxxzz + Wyyzz * Wyyzz) +
                 12.0 * (Wxxyz * Wxxyz + Wxyyz * Wxyyz + Wxyzz * Wxyzz))

            if B > 0.0 and W > 1e-8:
                KFA[v] = sqrt(A / B)
            else:
                KFA[v] = 0.0

    return np.asarray(KFA)
