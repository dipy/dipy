#!/usr/bin/python
""" Classes and functions for fitting the diffusion kurtosis model """

import numpy as np
import warnings
import scipy.optimize as opt
import dipy.core.sphere as dps
from dipy.reconst.multi_voxel import multi_voxel_fit
from dipy.reconst.dti import (TensorFit, mean_diffusivity,
                              from_lower_triangular,
                              lower_triangular, decompose_tensor,
                              MIN_POSITIVE_SIGNAL, nlls_fit_tensor,
                              restore_fit_tensor)
from dipy.reconst.utils import dki_design_matrix as design_matrix
from dipy.reconst.recspeed import local_maxima
from dipy.reconst.base import ReconstModel
from dipy.core.ndindex import ndindex
from dipy.core.geometry import (sphere2cart, cart2sphere,
                                perpendicular_directions)
from dipy.core.optimize import PositiveDefiniteLeastSquares
from dipy.data import get_sphere, get_fnames, load_sdp_constraints
from dipy.reconst.vec_val_sum import vec_val_vect
from dipy.core.gradients import check_multi_b


def _positive_evals(L1, L2, L3, er=2e-7):
    """ Helper function that identifies which voxels in a array have all
    eigenvalues significantly larger than zero

    Parameters
    ----------
    L1 : ndarray
        First independent variable of the integral.
    L2 : ndarray
        Second independent variable of the integral.
    L3 : ndarray
        Third independent variable of the integral.
    er : float, optional
        A eigenvalues is classified as larger than zero if it is larger than er

    Returns
    -------
    ind : boolean (n,)
        Array that marks the voxels that have all eigenvalues are larger than
        zero.

    """
    ind = np.logical_and(L1 > er, np.logical_and(L2 > er, L3 > er))

    return ind


def carlson_rf(x, y, z, errtol=3e-4):
    r""" Compute the Carlson's incomplete elliptic integral of the first kind

    Carlson's incomplete elliptic integral of the first kind is defined as:

    .. math::

        R_F = \frac{1}{2} \int_{0}^{\infty} \left [(t+x)(t+y)(t+z)  \right ]
        ^{-\frac{1}{2}}dt

    Parameters
    ----------
    x : ndarray
        First independent variable of the integral.
    y : ndarray
        Second independent variable of the integral.
    z : ndarray
        Third independent variable of the integral.
    errtol : float
        Error tolerance. Integral is computed with relative error less in
        magnitude than the defined value

    Returns
    -------
    RF : ndarray
        Value of the incomplete first order elliptic integral

    Notes
    -----
    x, y, and z have to be nonnegative and at most one of them is zero.

    References
    ----------
    .. [1] Carlson, B.C., 1994. Numerical computation of real or complex
           elliptic integrals. arXiv:math/9409227 [math.CA]

    """
    xn = x.copy()
    yn = y.copy()
    zn = z.copy()
    An = (xn + yn + zn) / 3.0
    Q = (3. * errtol) ** (-1 / 6.) * \
        np.max(np.abs([An - xn, An - yn, An - zn]), axis=0)
    # Convergence has to be done voxel by voxel
    index = ndindex(x.shape)
    for v in index:
        n = 0
        # Convergence condition
        while 4.**(-n) * Q[v] > abs(An[v]):
            xnroot = np.sqrt(xn[v])
            ynroot = np.sqrt(yn[v])
            znroot = np.sqrt(zn[v])
            lamda = xnroot * (ynroot + znroot) + ynroot * znroot
            n = n + 1
            xn[v] = (xn[v] + lamda) * 0.250
            yn[v] = (yn[v] + lamda) * 0.250
            zn[v] = (zn[v] + lamda) * 0.250
            An[v] = (An[v] + lamda) * 0.250

    # post convergence calculation
    X = 1. - xn / An
    Y = 1. - yn / An
    Z = - X - Y
    E2 = X * Y - Z * Z
    E3 = X * Y * Z
    RF = An**(-1 / 2.) * \
        (1 - E2 / 10. + E3 / 14. + (E2**2) / 24. - 3 / 44. * E2 * E3)

    return RF


def carlson_rd(x, y, z, errtol=1e-4):
    r""" Compute the Carlson's incomplete elliptic integral of the second kind

    Carlson's incomplete elliptic integral of the second kind is defined as:

    .. math::

        R_D = \frac{3}{2} \int_{0}^{\infty} (t+x)^{-\frac{1}{2}}
        (t+y)^{-\frac{1}{2}}(t+z)  ^{-\frac{3}{2}}

    Parameters
    ----------
    x : ndarray
        First independent variable of the integral.
    y : ndarray
        Second independent variable of the integral.
    z : ndarray
        Third independent variable of the integral.
    errtol : float
        Error tolerance. Integral is computed with relative error less in
        magnitude than the defined value

    Returns
    -------
    RD : ndarray
        Value of the incomplete second order elliptic integral

    Notes
    -----
    x, y, and z have to be nonnegative and at most x or y is zero.

    """
    xn = x.copy()
    yn = y.copy()
    zn = z.copy()
    A0 = (xn + yn + 3. * zn) / 5.0
    An = A0.copy()
    Q = (errtol / 4.) ** (-1 / 6.) * \
        np.max(np.abs([An - xn, An - yn, An - zn]), axis=0)
    sum_term = np.zeros(x.shape, dtype=x.dtype)
    n = np.zeros(x.shape)

    # Convergence has to be done voxel by voxel
    index = ndindex(x.shape)
    for v in index:
        # Convergence condition
        while 4.**(-n[v]) * Q[v] > abs(An[v]):
            xnroot = np.sqrt(xn[v])
            ynroot = np.sqrt(yn[v])
            znroot = np.sqrt(zn[v])
            lamda = xnroot * (ynroot + znroot) + ynroot * znroot
            sum_term[v] = sum_term[v] + \
                4.**(-n[v]) / (znroot * (zn[v] + lamda))
            n[v] = n[v] + 1
            xn[v] = (xn[v] + lamda) * 0.250
            yn[v] = (yn[v] + lamda) * 0.250
            zn[v] = (zn[v] + lamda) * 0.250
            An[v] = (An[v] + lamda) * 0.250

    # post convergence calculation
    X = (A0 - x) / (4.**n * An)
    Y = (A0 - y) / (4.**n * An)
    Z = - (X + Y) / 3.
    E2 = X * Y - 6. * Z * Z
    E3 = (3. * X * Y - 8. * Z * Z) * Z
    E4 = 3. * (X * Y - Z * Z) * Z**2.
    E5 = X * Y * Z**3.
    RD = \
        4**(-n) * An**(-3 / 2.) * \
        (1 - 3 / 14. * E2 + 1 / 6. * E3 +
         9 / 88. * (E2**2) - 3 / 22. * E4 - 9 / 52. * E2 * E3 +
         3 / 26. * E5) + 3 * sum_term

    return RD


def _F1m(a, b, c):
    r""" Helper function that computes function $F_1$ which is required to
    compute the analytical solution of the Mean kurtosis

    Parameters
    ----------
    a : ndarray
        Array containing the values of parameter $\lambda_1$ of function $F_1$
    b : ndarray
        Array containing the values of parameter $\lambda_2$ of function $F_1$
    c : ndarray
        Array containing the values of parameter $\lambda_3$ of function $F_1$

    Returns
    -------
    F1 : ndarray
       Value of the function $F_1$ for all elements of the arrays a, b, and c

    Notes
    -----
    Function $F_1$ is defined as [1]_:

    .. math::

        F_1(\lambda_1,\lambda_2,\lambda_3)=
        \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
        {18(\lambda_1-\lambda_2)(\lambda_1-\lambda_3)}
        [\frac{\sqrt{\lambda_2\lambda_3}}{\lambda_1}
        R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
        \frac{3\lambda_1^2-\lambda_1\lambda_2-\lambda_2\lambda_3-
        \lambda_1\lambda_3}
        {3\lambda_1 \sqrt{\lambda_2 \lambda_3}}
        R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-1 ]

    References
    ----------
    .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836

    """
    # Eigenvalues are considered equal if they are not 2.5% different to each
    # other. This value is adjusted according to the analysis reported in:
    # https://gsoc2015dipydki.blogspot.com/2015/08/rnh-post-13-start-wrapping-up-test.html
    er = 2.5e-2

    # Initialize F1
    F1 = np.zeros(a.shape)

    # Only computes F1 in voxels that have all eigenvalues larger than zero
    cond0 = _positive_evals(a, b, c)

    # Apply formula for non problematic plausible cases, i.e. a!=b and a!=c
    cond1 = np.logical_and(cond0, np.logical_and(abs(a - b) >= a * er,
                                                 abs(a - c) >= a * er))
    if np.sum(cond1) != 0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        RFm = carlson_rf(L1 / L2, L1 / L3, np.ones(len(L1)))
        RDm = carlson_rd(L1 / L2, L1 / L3, np.ones(len(L1)))
        F1[cond1] = ((L1 + L2 + L3) ** 2) / (18 * (L1 - L2) * (L1 - L3)) * \
                    (np.sqrt(L2 * L3) / L1 * RFm +
                     (3 * L1**2 - L1 * L2 - L1 * L3 - L2 * L3) /
                     (3 * L1 * np.sqrt(L2 * L3)) * RDm - 1)

    # Resolve possible singularity a==b
    cond2 = np.logical_and(cond0, np.logical_and(abs(a - b) < a * er,
                                                 abs(a - c) > a * er))
    if np.sum(cond2) != 0:
        L1 = (a[cond2] + b[cond2]) / 2.
        L3 = c[cond2]
        F1[cond2] = _F2m(L3, L1, L1) / 2.

    # Resolve possible singularity a==c
    cond3 = np.logical_and(cond0, np.logical_and(abs(a - c) < a * er,
                                                 abs(a - b) > a * er))
    if np.sum(cond3) != 0:
        L1 = (a[cond3] + c[cond3]) / 2.
        L2 = b[cond3]
        F1[cond3] = _F2m(L2, L1, L1) / 2

    # Resolve possible singularity a==b and a==c
    cond4 = np.logical_and(cond0, np.logical_and(abs(a - c) < a * er,
                                                 abs(a - b) < a * er))
    if np.sum(cond4) != 0:
        F1[cond4] = 1 / 5.

    return F1


def _F2m(a, b, c):
    r""" Helper function that computes function $F_2$ which is required to
    compute the analytical solution of the Mean kurtosis

    Parameters
    ----------
    a : ndarray
        Array containing the values of parameter $\lambda_1$ of function $F_2$
    b : ndarray
        Array containing the values of parameter $\lambda_2$ of function $F_2$
    c : ndarray
        Array containing the values of parameter $\lambda_3$ of function $F_2$

    Returns
    -------
    F2 : ndarray
       Value of the function $F_2$ for all elements of the arrays a, b, and c

    Notes
    -----
    Function $F_2$ is defined as [1]_:

    .. math::

        F_2(\lambda_1,\lambda_2,\lambda_3)=
        \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
        {3(\lambda_2-\lambda_3)^2}
        [\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}
        R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
        \frac{2\lambda_1-\lambda_2-\lambda_3}{3\sqrt{\lambda_2 \lambda_3}}
        R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-2]

    References
    ----------
    .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836

    """
    # Eigenvalues are considered equal if they are not 2.5% different to each
    # other. This value is adjusted according to the analysis reported in:
    # https://gsoc2015dipydki.blogspot.com/2015/08/rnh-post-13-start-wrapping-up-test.html
    er = 2.5e-2

    # Initialize F2
    F2 = np.zeros(a.shape)

    # Only computes F2 in voxels that have all eigenvalues larger than zero
    cond0 = _positive_evals(a, b, c)

    # Apply formula for non problematic plausible cases, i.e. b!=c
    cond1 = np.logical_and(cond0, (abs(b - c) > b * er))
    if np.sum(cond1) != 0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        RF = carlson_rf(L1 / L2, L1 / L3, np.ones(len(L1)))
        RD = carlson_rd(L1 / L2, L1 / L3, np.ones(len(L1)))
        F2[cond1] = (((L1 + L2 + L3) ** 2) / (3. * (L2 - L3) ** 2)) * \
                    (((L2 + L3) / (np.sqrt(L2 * L3))) * RF +
                     ((2. * L1 - L2 - L3) / (3. * np.sqrt(L2 * L3))) * RD - 2.)

    # Resolve possible singularity b==c
    cond2 = np.logical_and(cond0, np.logical_and(abs(b - c) < b * er,
                                                 abs(a - b) > b * er))
    if np.sum(cond2) != 0:
        L1 = a[cond2]
        L3 = (c[cond2] + b[cond2]) / 2.

        # Compute alfa [1]_
        x = 1. - (L1 / L3)
        alpha = np.zeros(len(L1))
        for i in range(len(x)):
            if x[i] > 0:
                alpha[i] = 1. / np.sqrt(x[i]) * np.arctanh(np.sqrt(x[i]))
            else:
                alpha[i] = 1. / np.sqrt(-x[i]) * np.arctan(np.sqrt(-x[i]))

        F2[cond2] = \
            6. * ((L1 + 2. * L3)**2) / (144. * L3**2 * (L1 - L3)**2) * \
            (L3 * (L1 + 2. * L3) + L1 * (L1 - 4. * L3) * alpha)

    # Resolve possible singularity a==b and a==c
    cond3 = np.logical_and(cond0, np.logical_and(abs(b - c) < b * er,
                                                 abs(a - b) < b * er))
    if np.sum(cond3) != 0:
        F2[cond3] = 6 / 15.

    return F2


def directional_diffusion(dt, V, min_diffusivity=0):
    r""" Calculate the apparent diffusion coefficient (adc) in each direction
    of a sphere for a single voxel [1]_

    Parameters
    ----------
    dt : array (6,)
        elements of the diffusion tensor of the voxel.
    V : array (g, 3)
        g directions of a Sphere in Cartesian coordinates
    min_diffusivity : float (optional)
        Because negative eigenvalues are not physical and small eigenvalues
        cause quite a lot of noise in diffusion-based metrics, diffusivity
        values smaller than `min_diffusivity` are replaced with
        `min_diffusivity`. Default = 0

    Returns
    -------
    adc : ndarray (g,)
        Apparent diffusion coefficient (adc) in all g directions of a sphere
        for a single voxel.

    References
    ----------
    .. [1] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
           Exploring the 3D geometry of the diffusion kurtosis tensor -
           Impact on the development of robust tractography procedures and
           novel biomarkers, NeuroImage 111: 85-99
    """
    adc = \
        V[:, 0] * V[:, 0] * dt[0] + \
        2 * V[:, 0] * V[:, 1] * dt[1] + \
        V[:, 1] * V[:, 1] * dt[2] + \
        2 * V[:, 0] * V[:, 2] * dt[3] + \
        2 * V[:, 1] * V[:, 2] * dt[4] + \
        V[:, 2] * V[:, 2] * dt[5]

    if min_diffusivity is not None:
        adc = adc.clip(min=min_diffusivity)
    return adc


def directional_diffusion_variance(kt, V, min_kurtosis=-3/7):
    r""" Calculate the apparent diffusion variance (adv) in each direction of a
    sphere for a single voxel [1]_

    Parameters
    ----------
    dt : array (6,)
        elements of the diffusion tensor of the voxel.
    kt : array (15,)
        elements of the kurtosis tensor of the voxel.
    V : array (g, 3)
        g directions of a Sphere in Cartesian coordinates
    min_kurtosis : float (optional)
        Because high-amplitude negative values of kurtosis are not physically
        and biologicaly pluasible, and these cause artefacts in
        kurtosis-based measures, directional kurtosis values smaller than
        `min_kurtosis` are replaced with `min_kurtosis`. Default = -3./7
        (theoretical kurtosis limit for regions that consist of water confined
        to spherical pores [2]_)
    adc : ndarray(g,) (optional)
        Apparent diffusion coefficient (adc) in all g directions of a sphere
        for a single voxel.
    adv : ndarray(g,) (optional)
        Apparent diffusion variance coefficient (advc) in all g directions of
        a sphere for a single voxel.

    Returns
    -------
    adv : ndarray (g,)
        Apparent diffusion variance (adv) in all g directions of a sphere for
        a single voxel.

    References
    ----------
    .. [1] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
           Exploring the 3D geometry of the diffusion kurtosis tensor -
           Impact on the development of robust tractography procedures and
           novel biomarkers, NeuroImage 111: 85-99

    """
    adv = \
        V[:, 0] * V[:, 0] * V[:, 0] * V[:, 0] * kt[0] + \
        V[:, 1] * V[:, 1] * V[:, 1] * V[:, 1] * kt[1] + \
        V[:, 2] * V[:, 2] * V[:, 2] * V[:, 2] * kt[2] + \
        4 * V[:, 0] * V[:, 0] * V[:, 0] * V[:, 1] * kt[3] + \
        4 * V[:, 0] * V[:, 0] * V[:, 0] * V[:, 2] * kt[4] + \
        4 * V[:, 0] * V[:, 1] * V[:, 1] * V[:, 1] * kt[5] + \
        4 * V[:, 1] * V[:, 1] * V[:, 1] * V[:, 2] * kt[6] + \
        4 * V[:, 0] * V[:, 2] * V[:, 2] * V[:, 2] * kt[7] + \
        4 * V[:, 1] * V[:, 2] * V[:, 2] * V[:, 2] * kt[8] + \
        6 * V[:, 0] * V[:, 0] * V[:, 1] * V[:, 1] * kt[9] + \
        6 * V[:, 0] * V[:, 0] * V[:, 2] * V[:, 2] * kt[10] + \
        6 * V[:, 1] * V[:, 1] * V[:, 2] * V[:, 2] * kt[11] + \
        12 * V[:, 0] * V[:, 0] * V[:, 1] * V[:, 2] * kt[12] + \
        12 * V[:, 0] * V[:, 1] * V[:, 1] * V[:, 2] * kt[13] + \
        12 * V[:, 0] * V[:, 1] * V[:, 2] * V[:, 2] * kt[14]

    return adv


def directional_kurtosis(dt, md, kt, V, min_diffusivity=0, min_kurtosis=-3/7,
                         adc=None, adv=None):
    r""" Calculate the apparent kurtosis coefficient (akc) in each direction of
    a sphere for a single voxel [1]_

    Parameters
    ----------
    dt : array (6,)
        elements of the diffusion tensor of the voxel.
    md : float
        mean diffusivity of the voxel
    kt : array (15,)
        elements of the kurtosis tensor of the voxel.
    V : array (g, 3)
        g directions of a Sphere in Cartesian coordinates
    min_diffusivity : float (optional)
        Because negative eigenvalues are not physical and small eigenvalues
        cause quite a lot of noise in diffusion-based metrics, diffusivity
        values smaller than `min_diffusivity` are replaced with
        `min_diffusivity`. Default = 0
    min_kurtosis : float (optional)
        Because high-amplitude negative values of kurtosis are not physically
        and biologicaly pluasible, and these cause artefacts in
        kurtosis-based measures, directional kurtosis values smaller than
        `min_kurtosis` are replaced with `min_kurtosis`. Default = -3./7
        (theoretical kurtosis limit for regions that consist of water confined
        to spherical pores [2]_)
    adc : ndarray(g,) (optional)
        Apparent diffusion coefficient (adc) in all g directions of a sphere
        for a single voxel.
    adv : ndarray(g,) (optional)
        Apparent diffusion variance (advc) in all g directions of a sphere for
        a single voxel.

    Returns
    -------
    akc : ndarray (g,)
        Apparent kurtosis coefficient (AKC) in all g directions of a sphere for
        a single voxel.

    References
    ----------
    .. [1] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
           Exploring the 3D geometry of the diffusion kurtosis tensor -
           Impact on the development of robust tractography procedures and
           novel biomarkers, NeuroImage 111: 85-99
    .. [2] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
           Robust estimation from DW-MRI using homogeneous polynomials.
           Proceedings of the 8th {IEEE} International Symposium on
           Biomedical Imaging: From Nano to Macro, ISBI 2011, 262-265.
           doi: 10.1109/ISBI.2011.5872402

    """
    if adc is None:
        adc = directional_diffusion(dt, V, min_diffusivity=min_diffusivity)
    if adv is None:
        adv = directional_diffusion_variance(kt, V)

    akc = adv * (md / adc) ** 2

    if min_kurtosis is not None:
        akc = akc.clip(min=min_kurtosis)

    return akc


def apparent_kurtosis_coef(dki_params, sphere, min_diffusivity=0,
                           min_kurtosis=-3./7):
    r""" Calculate the apparent kurtosis coefficient (AKC) in each direction
    of a sphere [1]_

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
    All parameters estimated from the diffusion kurtosis model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the first,
            second and third coordinates of the eigenvectors respectively
        3) Fifteen elements of the kurtosis tensor
    sphere : a Sphere class instance
        The AKC will be calculated for each of the vertices in the sphere
    min_diffusivity : float (optional)
        Because negative eigenvalues are not physical and small eigenvalues
        cause quite a lot of noise in diffusion-based metrics, diffusivity
        values smaller than `min_diffusivity` are replaced with
        `min_diffusivity`. Default = 0
    min_kurtosis : float (optional)
        Because high-amplitude negative values of kurtosis are not physically
        and biologicaly pluasible, and these cause artefacts in
        kurtosis-based measures, directional kurtosis values smaller than
        `min_kurtosis` are replaced with `min_kurtosis`. Default = -3./7
        (theoretical kurtosis limit for regions that consist of water confined
        to spherical pores [2]_)

    Returns
    -------
    akc : ndarray (x, y, z, g) or (n, g)
        Apparent kurtosis coefficient (AKC) for all g directions of a sphere.

    Notes
    -----
    For each sphere direction with coordinates $(n_{1}, n_{2}, n_{3})$, the
    calculation of AKC is done using formula [1]_:

    .. math ::

        AKC(n)=\frac{MD^{2}}{ADC(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}
        \sum_{k=1}^{3}\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

    where $W_{ijkl}$ are the elements of the kurtosis tensor, MD the mean
    diffusivity and ADC the apparent diffusion coefficient computed as:

    .. math ::

        ADC(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

    where $D_{ij}$ are the elements of the diffusion tensor.

    References
    ----------
    .. [1] Neto Henriques R, Correia MM, Nunes RG, Ferreira HA (2015).
           Exploring the 3D geometry of the diffusion kurtosis tensor -
           Impact on the development of robust tractography procedures and
           novel biomarkers, NeuroImage 111: 85-99
    .. [2] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
           Robust estimation from DW-MRI using homogeneous polynomials.
           Proceedings of the 8th {IEEE} International Symposium on
           Biomedical Imaging: From Nano to Macro, ISBI 2011, 262-265.
           doi: 10.1109/ISBI.2011.5872402

    """
    # Flat parameters
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split data
    evals, evecs, kt = split_dki_param(dki_params)

    # Initialize AKC matrix
    V = sphere.vertices
    akc = np.zeros((len(kt), len(V)))

    # select relevant voxels to process
    rel_i = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
    kt = kt[rel_i]
    evecs = evecs[rel_i]
    evals = evals[rel_i]
    akci = akc[rel_i]

    # Compute MD and DT
    md = mean_diffusivity(evals)
    dt = lower_triangular(vec_val_vect(evecs, evals))

    # loop over all relevant voxels
    for vox in range(len(kt)):
        akci[vox] = directional_kurtosis(dt[vox], md[vox], kt[vox], V,
                                         min_diffusivity=min_diffusivity,
                                         min_kurtosis=min_kurtosis)

    # reshape data according to input data
    akc[rel_i] = akci

    return akc.reshape((outshape + (len(V),)))


def mean_kurtosis(dki_params, min_kurtosis=-3./7, max_kurtosis=3,
                  analytical=True):
    r""" Compute mean kurtosis (MK) from the kurtosis tensor

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
    All parameters estimated from the diffusion kurtosis model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the first,
            second and third coordinates of the eigenvector
        3) Fifteen elements of the kurtosis tensor
    min_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, mean
        kurtosis values that are smaller than `min_kurtosis` are replaced with
        `min_kurtosis`. Default = -3./7 (theoretical kurtosis limit for regions
        that consist of water confined to spherical pores [4]_)
    max_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, mean
        kurtosis values that are larger than `max_kurtosis` are replaced with
        `max_kurtosis`. Default = 10
    analytical : bool (optional)
        If True, MK is calculated using its analytical solution, otherwise an
        exact numerical estimator is used (see Notes). Default is set to True

    Returns
    -------
    mk : array
        Calculated MK.

    Notes
    -----
    The MK is defined as the average of directional kurtosis coefficients
    across all spatial directions, which can be formulated by the following
    surface integral[1]_:

    .. math::

         MK \equiv \frac{1}{4\pi} \int d\Omega_\mathbf{n} K(\mathbf{n})

    This integral can be numerically solved by averaging directional
    kurtosis values sampled for directions of a spherical t-design [2]_.

    Alternatively, MK can be solved from the analytical solution derived by
    Tabesh et al. [3]_. This solution is given by:

    .. math::

        MK=F_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{1111}+
           F_1(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{2222}+
           F_1(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{3333}+ \\
           F_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}+
           F_2(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{1133}+
           F_2(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{1122}

    where $\hat{W}_{ijkl}$ are the components of the $W$ tensor in the
    coordinates system defined by the eigenvectors of the diffusion tensor
    $\mathbf{D}$ and

    .. math::

        F_1(\lambda_1,\lambda_2,\lambda_3)=
        \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
        {18(\lambda_1-\lambda_2)(\lambda_1-\lambda_3)}
        [\frac{\sqrt{\lambda_2\lambda_3}}{\lambda_1}
        R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
        \frac{3\lambda_1^2-\lambda_1\lambda_2-\lambda_2\lambda_3-
        \lambda_1\lambda_3}
        {3\lambda_1 \sqrt{\lambda_2 \lambda_3}}
        R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-1 ]

        F_2(\lambda_1,\lambda_2,\lambda_3)=
        \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
        {3(\lambda_2-\lambda_3)^2}
        [\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}
        R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
        \frac{2\lambda_1-\lambda_2-\lambda_3}{3\sqrt{\lambda_2 \lambda_3}}
        R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-2]

    where $R_f$ and $R_d$ are the Carlson's elliptic integrals.

    References
    ----------
    .. [1] Jensen, J.H., Helpern, J.A., 2010. MRI quantification of
           non-Gaussian water diffusion by kurtosis analysis. NMR in
           Biomedicine 23(7): 698-710
    .. [2] Hardin, R.H., Sloane, N.J.A., 1996. McLaren's Improved Snub Cube and
           Other New Spherical Designs in Three Dimensions. Discrete and
           Computational Geometry 15, 429-441.
    .. [3] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    .. [4] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
           Robust estimation from DW-MRI using homogeneous polynomials.
           Proceedings of the 8th {IEEE} International Symposium on
           Biomedical Imaging: From Nano to Macro, ISBI 2011, 262-265.
           doi: 10.1109/ISBI.2011.5872402

    """
    # Flat parameters. For numpy versions more recent than 1.6.0, this step
    # isn't required
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    if analytical:
        # Split the model parameters to three variable containing the evals,
        # evecs, and kurtosis elements
        evals, evecs, kt = split_dki_param(dki_params)

        # Rotate the kurtosis tensor from the standard Cartesian coordinate
        # system to another coordinate system in which the 3 orthonormal
        # eigenvectors of DT are the base coordinate
        Wxxxx = Wrotate_element(kt, 0, 0, 0, 0, evecs)
        Wyyyy = Wrotate_element(kt, 1, 1, 1, 1, evecs)
        Wzzzz = Wrotate_element(kt, 2, 2, 2, 2, evecs)
        Wxxyy = Wrotate_element(kt, 0, 0, 1, 1, evecs)
        Wxxzz = Wrotate_element(kt, 0, 0, 2, 2, evecs)
        Wyyzz = Wrotate_element(kt, 1, 1, 2, 2, evecs)

        # Compute MK
        MK = \
            _F1m(evals[..., 0], evals[..., 1], evals[..., 2]) * Wxxxx + \
            _F1m(evals[..., 1], evals[..., 0], evals[..., 2]) * Wyyyy + \
            _F1m(evals[..., 2], evals[..., 1], evals[..., 0]) * Wzzzz + \
            _F2m(evals[..., 0], evals[..., 1], evals[..., 2]) * Wyyzz + \
            _F2m(evals[..., 1], evals[..., 0], evals[..., 2]) * Wxxzz + \
            _F2m(evals[..., 2], evals[..., 1], evals[..., 0]) * Wxxyy

    else:
        # Numerical Solution using t-design of 45 directions
        V = np.loadtxt(get_fnames("t-design"))
        sph = dps.Sphere(xyz=V)
        KV = apparent_kurtosis_coef(dki_params, sph, min_kurtosis=min_kurtosis)
        MK = np.mean(KV, axis=-1)

    if min_kurtosis is not None:
        MK = MK.clip(min=min_kurtosis)

    if max_kurtosis is not None:
        MK = MK.clip(max=max_kurtosis)

    return MK.reshape(outshape)


def _G1m(a, b, c):
    r""" Helper function that computes function $G_1$ which is required to
    compute the analytical solution of the Radial kurtosis

    Parameters
    ----------
    a : ndarray
        Array containing the values of parameter $\lambda_1$ of function $G_1$
    b : ndarray
        Array containing the values of parameter $\lambda_2$ of function $G_1$
    c : ndarray
        Array containing the values of parameter $\lambda_3$ of function $G_1$

    Returns
    -------
    G1 : ndarray
       Value of the function $G_1$ for all elements of the arrays a, b, and c

    Notes
    -----
    Function $G_1$ is defined as [1]_:

    .. math::

        G_1(\lambda_1,\lambda_2,\lambda_3)=
        \frac{(\lambda_1+\lambda_2+\lambda_3)^2}{18\lambda_2(\lambda_2-
        \lambda_3)} \left (2\lambda_2 +
        \frac{\lambda_3^2-3\lambda_2\lambda_3}{\sqrt{\lambda_2\lambda_3}}
        \right)

    References
    ----------
    .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836

    """
    # Float error used to compare two floats, abs(l1 - l2) < er for l1 = l2
    # Error is defined as five orders of magnitude larger than system's epslon
    er = np.finfo(a.ravel()[0]).eps * 1e5

    # Initialize G1
    G1 = np.zeros(a.shape)

    # Only computes G1 in voxels that have all eigenvalues larger than zero
    cond0 = _positive_evals(a, b, c)

    # Apply formula for non problematic plausible cases, i.e. b!=c
    cond1 = np.logical_and(cond0, (abs(b - c) > er))
    if np.sum(cond1) != 0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        G1[cond1] = \
            (L1 + L2 + L3)**2 / (18 * L2 * (L2 - L3)**2) * \
            (2. * L2 + (L3**2 - 3 * L2 * L3) / np.sqrt(L2 * L3))

    # Resolve possible singularity b==c
    cond2 = np.logical_and(cond0, abs(b - c) < er)
    if np.sum(cond2) != 0:
        L1 = a[cond2]
        L2 = b[cond2]
        G1[cond2] = (L1 + 2. * L2)**2 / (24. * L2**2)

    return G1


def _G2m(a, b, c):
    r""" Helper function that computes function $G_2$ which is required to
    compute the analytical solution of the Radial kurtosis

    Parameters
    ----------
    a : ndarray
        Array containing the values of parameter $\lambda_1$ of function $G_2$
    b : ndarray
        Array containing the values of parameter $\lambda_2$ of function $G_2$
    c : ndarray (n,)
        Array containing the values of parameter $\lambda_3$ of function $G_2$

    Returns
    -------
    G2 : ndarray
       Value of the function $G_2$ for all elements of the arrays a, b, and c

    Notes
    -----
    Function $G_2$ is defined as [1]_:

    .. math::

        G_2(\lambda_1,\lambda_2,\lambda_3)=
        \frac{(\lambda_1+\lambda_2+\lambda_3)^2}{(\lambda_2-\lambda_3)^2}
        \left ( \frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}-2\right )

    References
    ----------
    .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836

    """
    # Float error used to compare two floats, abs(l1 - l2) < er for l1 = l2
    # Error is defined as five order of magnitude larger than system's epsilon
    er = np.finfo(a.ravel()[0]).eps * 1e5

    # Initialize G2
    G2 = np.zeros(a.shape)

    # Only computes G2 in voxels that have all eigenvalues larger than zero
    cond0 = _positive_evals(a, b, c)

    # Apply formula for non problematic plausible cases, i.e. b!=c
    cond1 = np.logical_and(cond0, (abs(b - c) > er))
    if np.sum(cond1) != 0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        G2[cond1] = \
            (L1 + L2 + L3)**2 / (3 * (L2 - L3)**2) * \
            ((L2 + L3) / np.sqrt(L2 * L3) - 2)

    # Resolve possible singularity b==c
    cond2 = np.logical_and(cond0, abs(b - c) < er)
    if np.sum(cond2) != 0:
        L1 = a[cond2]
        L2 = b[cond2]
        G2[cond2] = (L1 + 2. * L2)**2 / (12. * L2**2)

    return G2


def radial_kurtosis(dki_params, min_kurtosis=-3./7, max_kurtosis=10,
                    analytical=True):
    r""" Compute radial kurtosis (RK) of a diffusion kurtosis tensor [1]_, [2]_

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
    All parameters estimated from the diffusion kurtosis model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the first,
            second and third coordinates of the eigenvector
        3) Fifteen elements of the kurtosis tensor
    min_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, radial
        kurtosis values that are smaller than `min_kurtosis` are replaced with
        `min_kurtosis`. Default = -3./7 (theoretical kurtosis limit for regions
        that consist of water confined to spherical pores [3]_)
    max_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, radial
        kurtosis values that are larger than `max_kurtosis` are replaced with
        `max_kurtosis`. Default = 10
    analytical : bool (optional)
        If True, RK is calculated using its analytical solution, otherwise an
        exact numerical estimator is used (see Notes). Default is set to True.

    Returns
    -------
    rk : array
        Calculated RK.

    Notes
    -----
    RK is defined as the average of the directional kurtosis perpendicular
    to the fiber's main direction e1 [1]_, [2]_:

    .. math::

    RK \equiv \frac{1}{2\pi} \int d\Omega _\mathbf{\theta} K(\mathbf{\theta})
              \delta (\mathbf{\theta}\cdot \mathbf{e}_1)

    This equation can be numerically computed by averaging apparent
    directional kurtosis samples for directions perpendicular to e1.

    Otherwise, RK can be calculated from its analytical solution [2]_:

    .. math::

        K_{\bot} = G_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2222} +
                   G_1(\lambda_1,\lambda_3,\lambda_2)\hat{W}_{3333} +
                   G_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}

    where:

    .. math::

        G_1(\lambda_1,\lambda_2,\lambda_3)=
        \frac{(\lambda_1+\lambda_2+\lambda_3)^2}{18\lambda_2(\lambda_2-
        \lambda_3)} \left (2\lambda_2 +
        \frac{\lambda_3^2-3\lambda_2\lambda_3}{\sqrt{\lambda_2\lambda_3}}
        \right)

    and

    .. math::

        G_2(\lambda_1,\lambda_2,\lambda_3)=
        \frac{(\lambda_1+\lambda_2+\lambda_3)^2}{(\lambda_2-\lambda_3)^2}
        \left ( \frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}-2\right )

    References
    ----------
    .. [1] Jensen, J.H., Helpern, J.A., 2010. MRI quantification of
           non-Gaussian water diffusion by kurtosis analysis. NMR in
           Biomedicine 23(7): 698-710
    .. [2] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    .. [3] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
           Robust estimation from DW-MRI using homogeneous polynomials.
           Proceedings of the 8th {IEEE} International Symposium on Biomedical
           Imaging: From Nano to Macro, ISBI 2011, 262-265.
           doi: 10.1109/ISBI.2011.5872402

    """
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split the model parameters to three variable containing the evals,
    # evecs, and kurtosis elements
    evals, evecs, kt = split_dki_param(dki_params)

    if analytical:
        # Rotate the kurtosis tensor from the standard Cartesian coordinate
        # system to another coordinate system in which the 3 orthonormal
        # eigenvectors of DT are the base coordinate
        Wyyyy = Wrotate_element(kt, 1, 1, 1, 1, evecs)
        Wzzzz = Wrotate_element(kt, 2, 2, 2, 2, evecs)
        Wyyzz = Wrotate_element(kt, 1, 1, 2, 2, evecs)

        # Compute RK
        RK = \
            _G1m(evals[..., 0], evals[..., 1], evals[..., 2]) * Wyyyy + \
            _G1m(evals[..., 0], evals[..., 2], evals[..., 1]) * Wzzzz + \
            _G2m(evals[..., 0], evals[..., 1], evals[..., 2]) * Wyyzz

    else:
        # Numerical Solution using 10 perpendicular directions samples
        npa = 10

        # Initialize RK
        RK = np.zeros(kt.shape[:-1])

        # select relevant voxels to process
        rel_i = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
        dki_params = dki_params[rel_i]
        evecs = evecs[rel_i]
        RKi = RK[rel_i]

        # loop over all voxels
        KV = np.zeros((dki_params.shape[0], npa))
        for vox in range(len(dki_params)):
            V = perpendicular_directions(np.array(evecs[vox, :, 0]), num=npa,
                                         half=True)
            sph = dps.Sphere(xyz=V)
            KV[vox, :] = apparent_kurtosis_coef(dki_params[vox], sph,
                                                min_kurtosis=min_kurtosis)
        RKi = np.mean(KV, axis=-1)

        RK[rel_i] = RKi

    if min_kurtosis is not None:
        RK = RK.clip(min=min_kurtosis)

    if max_kurtosis is not None:
        RK = RK.clip(max=max_kurtosis)

    return RK.reshape(outshape)


def axial_kurtosis(dki_params, min_kurtosis=-3./7, max_kurtosis=10,
                   analytical=True):
    r"""  Compute axial kurtosis (AK) from the kurtosis tensor [1]_, [2]_

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
    All parameters estimated from the diffusion kurtosis model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the first,
            second and third coordinates of the eigenvector
        3) Fifteen elements of the kurtosis tensor
    min_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, axial
        kurtosis values that are smaller than `min_kurtosis` are replaced with
        `min_kurtosis`. Default = -3./7 (theoretical kurtosis limit for regions
        that consist of water confined to spherical pores [3]_)
    max_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, axial
        kurtosis values that are larger than `max_kurtosis` are replaced with
        `max_kurtosis`. Default = 10
    analytical : bool (optional)
        If True, AK is calculated from rotated diffusion kurtosis tensor,
        otherwise it will be computed from the apparent diffusion kurtosis
        values along the principal axis of the diffusion tensor (see notes).
        Default is set to True.

    Returns
    -------
    ak : array
        Calculated AK.

    Notes
    -----
    AK is defined as the directional kurtosis parallel to the fiber's main
    direction e1 [1]_, [2]_. You can compute AK using to approaches:

    1) AK is calculated from rotated diffusion kurtosis tensor [2]_, i.e.:

    .. math::
        AK = \hat{W}_{1111}
            \frac{(\lambda_{1}+\lambda_{2}+\lambda_{3})^2}{(9 \lambda_{1}^2)}

    2) AK can be sampled from the principal axis of the diffusion tensor:

    .. math::
        AK = K(\mathbf{\mathbf{e}_1)

    Although both approaches leads to an exact calculation of AK, the first
    approach will be referred to as the analytical method while the second
    approach will be referred to as the numerical method based on their analogy
    to the estimation strategies for MK and RK.

    References
    ----------
    .. [1] Jensen, J.H., Helpern, J.A., 2010. MRI quantification of
           non-Gaussian water diffusion by kurtosis analysis. NMR in
           Biomedicine 23(7): 698-710
    .. [2] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    .. [3] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
           Robust estimation from DW-MRI using homogeneous polynomials.
           Proceedings of the 8th {IEEE} International Symposium on
           Biomedical Imaging: From Nano to Macro, ISBI 2011, 262-265.
           doi: 10.1109/ISBI.2011.5872402

    """
    # Flat parameters
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split data
    evals, evecs, kt = split_dki_param(dki_params)

    # Initialize AK
    AK = np.zeros(kt.shape[:-1])

    # select relevant voxels to process
    rel_i = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
    kt = kt[rel_i]
    evecs = evecs[rel_i]
    evals = evals[rel_i]
    AKi = AK[rel_i]

    # Compute mean diffusivity
    md = mean_diffusivity(evals)

    if analytical:
        # Rotate the kurtosis tensor from the standard Cartesian coordinate
        # system to another coordinate system in which the 3 orthonormal
        # eigenvectors of DT are the base coordinate
        Wxxxx = Wrotate_element(kt, 0, 0, 0, 0, evecs)
        AKi = Wxxxx * (md ** 2) / (evals[..., 0] ** 2)

    else:
        # Compute apparent directional kurtosis along evecs[0]
        dt = lower_triangular(vec_val_vect(evecs, evals))
        for vox in range(len(kt)):
            AKi[vox] = directional_kurtosis(dt[vox], md[vox], kt[vox],
                                            np.array([evecs[vox, :, 0]]))

    # reshape data according to input data
    AK[rel_i] = AKi

    if min_kurtosis is not None:
        AK = AK.clip(min=min_kurtosis)

    if max_kurtosis is not None:
        AK = AK.clip(max=max_kurtosis)

    return AK.reshape(outshape)


def _kt_maximum_converge(ang, dt, md, kt):
    """ Helper function that computes the inverse of the directional kurtosis
    of a voxel along a given direction in polar coordinates

    Parameters
    ----------
    ang : array (2,)
        array containing the two polar angles
    dt : array (6,)
        elements of the diffusion tensor of the voxel.
    md : float
        mean diffusivity of the voxel
    kt : array (15,)
        elements of the kurtosis tensor of the voxel.

    Returns
    -------
    neg_kt : float
        The inverse value of the apparent kurtosis for the given direction.

    Notes
    -----
    This function is used to refine the kurtosis maximum estimate

    See Also
    --------
    dipy.reconst.dki.kurtosis_maximum

    """
    n = np.array([sphere2cart(1, ang[0], ang[1])])
    return -1. * directional_kurtosis(dt, md, kt, n)


def _voxel_kurtosis_maximum(dt, md, kt, sphere, gtol=1e-2):
    """ Compute the maximum value of a single voxel kurtosis tensor

    Parameters
    ----------
    dt : array (6,)
        elements of the diffusion tensor of the voxel.
    md : float
        mean diffusivity of the voxel
    kt : array (15,)
        elements of the kurtosis tensor of the voxel.
    sphere : Sphere class instance, optional
        The sphere providing sample directions for the initial search of the
        maximum value of kurtosis.
    gtol : float, optional
        This input is to refine kurtosis maximum under the precision of the
        directions sampled on the sphere class instance. The gradient of the
        convergence procedure must be less than gtol before successful
        termination. If gtol is None, fiber direction is directly taken from
        the initial sampled directions of the given sphere object

    Returns
    -------
    max_value : float
        kurtosis tensor maximum value
    max_dir : array (3,)
        Cartesian coordinates of the direction of the maximal kurtosis value

    """
    # Estimation of maximum kurtosis candidates
    akc = directional_kurtosis(dt, md, kt, sphere.vertices)
    max_val, ind = local_maxima(akc, sphere.edges)
    n = len(max_val)

    # case that none maximum was find (spherical or null kurtosis tensors)
    if n == 0:
        return np.mean(akc), np.zeros(3)

    max_dir = sphere.vertices[ind]

    # Select the maximum from the candidates
    max_value = max(max_val)
    max_direction = max_dir[np.argmax(max_val.argmax)]

    # refine maximum direction
    if gtol is not None:
        for p in range(n):
            r, theta, phi = cart2sphere(max_dir[p, 0], max_dir[p, 1],
                                        max_dir[p, 2])
            ang = np.array([theta, phi])
            ang[:] = opt.fmin_bfgs(_kt_maximum_converge, ang,
                                   args=(dt, md, kt), disp=False,
                                   retall=False, gtol=gtol)
            k_dir = np.array([sphere2cart(1., ang[0], ang[1])])
            k_val = directional_kurtosis(dt, md, kt, k_dir)
            if k_val > max_value:
                max_value = k_val
                max_direction = k_dir

    return max_value.item(), max_direction


def kurtosis_maximum(dki_params, sphere='repulsion100', gtol=1e-2,
                     mask=None):
    """ Compute kurtosis maximum value

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
    All parameters estimated from the diffusion kurtosis model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eingenvalues
        2) Three lines of the eigenvector matrix each containing the first,
            second and third coordinates of the eigenvector
        3) Fifteen elements of the kurtosis tensor
    sphere : Sphere class instance, optional
        The sphere providing sample directions for the initial search of the
        maximal value of kurtosis.
    gtol : float, optional
        This input is to refine kurtosis maximum under the precision of the
        directions sampled on the sphere class instance. The gradient of the
        convergence procedure must be less than gtol before successful
        termination. If gtol is None, fiber direction is directly taken from
        the initial sampled directions of the given sphere object
    mask : ndarray
        A boolean array used to mark the coordinates in the data that should be
        analyzed that has the shape dki_params.shape[:-1]

    Returns
    -------
    max_value : float
        kurtosis tensor maximum value
    max_dir : array (3,)
        Cartesian coordinates of the direction of the maximal kurtosis value

    """
    shape = dki_params.shape[:-1]

    # load gradient directions
    if not isinstance(sphere, dps.Sphere):
        sphere = get_sphere('repulsion100')

    # select voxels where to find fiber directions
    if mask is None:
        mask = np.ones(shape, dtype='bool')
    else:
        if mask.shape != shape:
            raise ValueError("Mask is not the same shape as dki_params.")

    evals, evecs, kt = split_dki_param(dki_params)

    # select non-zero voxels
    pos_evals = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
    mask = np.logical_and(mask, pos_evals)

    kt_max = np.zeros(mask.shape)
    dt = lower_triangular(vec_val_vect(evecs, evals))
    md = mean_diffusivity(evals)

    for idx in ndindex(shape):
        if not mask[idx]:
            continue
        kt_max[idx], da = _voxel_kurtosis_maximum(dt[idx], md[idx], kt[idx],
                                                  sphere, gtol=gtol)

    return kt_max


def mean_kurtosis_tensor(dki_params, min_kurtosis=-3./7, max_kurtosis=10):
    r""" Compute mean of the kurtosis tensor (MKT) [1]_

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
    All parameters estimated from the diffusion kurtosis model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the first,
            second and third coordinates of the eigenvector
        3) Fifteen elements of the kurtosis tensor
    min_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, mean
        kurtosis values that are smaller than `min_kurtosis` are replaced with
        `min_kurtosis`. Default = -3./7 (theoretical kurtosis limit for regions
        that consist of water confined to spherical pores [2]_)
    max_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, mean
        kurtosis values that are larger than `max_kurtosis` are replaced with
        `max_kurtosis`. Default = 10
    Returns
    -------
    mkt : array
        Calculated mean kurtosis tensor.

    Notes
    -----
    The MKT is defined as [1]_:

    .. math::

         MKT \equiv \frac{1}{4\pi} \int d
         \Omega_{\mathnbf{n}} n_i n_j n_k n_l W_{ijkl}


    which can be directly computed from the trace of the kurtosis tensor:

    .. math::

    MKT = \frac{1}{5} Tr(\mathbf{W}) = \frac{1}{5}
    (W_{1111} + W_{2222} + W_{3333} + 2W_{1122} + 2W_{1133} + 2W_{2233})

    References
    ----------
    .. [1] Hansen, B., Lund, T. E., Sangill, R., and Jespersen, S. N. (2013).
           Experimentally and computationally fast method for estimation of
           a mean kurtosis.Magnetic Resonance in Medicine69,  1754–1760.388
           doi:10.1002/mrm.24743
    .. [2] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
           Robust estimation from DW-MRI using homogeneous polynomials.
           Proceedings of the 8th {IEEE} International Symposium on
           Biomedical Imaging: From Nano to Macro, ISBI 2011, 262-265.
           doi: 10.1109/ISBI.2011.5872402

    """
    MKT = 1/5 * (dki_params[..., 12] + dki_params[..., 13] +
                 dki_params[..., 14] + 2 * dki_params[..., 21] +
                 2 * dki_params[..., 22] + 2 * dki_params[..., 23])

    if min_kurtosis is not None:
        MKT = MKT.clip(min=min_kurtosis)

    if max_kurtosis is not None:
        MKT = MKT.clip(max=max_kurtosis)

    return MKT


def kurtosis_fractional_anisotropy(dki_params):
    r""" Compute the anisotropy of the kurtosis tensor (KFA) [1]_

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
    All parameters estimated from the diffusion kurtosis model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the first,
            second and third coordinates of the eigenvector
        3) Fifteen elements of the kurtosis tensor
    Returns
    -------
    kfa : array
        Calculated mean kurtosis tensor.

    Notes
    -----
    The KFA is defined as [1]_:

    .. math::

         KFA \equiv
         \frac{||\mathbf{W} - MKT \mathbf{I}^{(4)}||_F}{||\mathbf{W}||_F}

    where $W$ is the kurtosis tensor, MKT the kurtosis tensor mean, $I^(4)$ is
    the fully symmetric rank 2 isotropic tensor and $||...||_F$ is the tensor's
    Frobenius norm [1]_.

    References
    ----------
    .. [1] Glenn, G. R., Helpern, J. A., Tabesh, A., and Jensen, J. H. (2015).
           Quantitative assessment of diffusional kurtosis anisotropy.
           NMR in Biomedicine 28, 448–459. doi:10.1002/nbm.3271

    """
    Wxxxx = dki_params[..., 12]
    Wyyyy = dki_params[..., 13]
    Wzzzz = dki_params[..., 14]
    Wxxxy = dki_params[..., 15]
    Wxxxz = dki_params[..., 16]
    Wxyyy = dki_params[..., 17]
    Wyyyz = dki_params[..., 18]
    Wxzzz = dki_params[..., 19]
    Wyzzz = dki_params[..., 20]
    Wxxyy = dki_params[..., 21]
    Wxxzz = dki_params[..., 22]
    Wyyzz = dki_params[..., 23]
    Wxxyz = dki_params[..., 24]
    Wxyyz = dki_params[..., 25]
    Wxyzz = dki_params[..., 26]

    W = 1.0/5.0 * (Wxxxx + Wyyyy + Wzzzz + 2*Wxxyy + 2*Wxxzz + 2*Wyyzz)

    # Compute's equation numerator
    A = (
          (Wxxxx - W) ** 2 +
          (Wyyyy - W) ** 2 +
          (Wzzzz - W) ** 2 +
          4 * (Wxxxy ** 2 +
               Wxxxz ** 2 +
               Wxyyy ** 2 +
               Wyyyz ** 2 +
               Wxzzz ** 2 +
               Wyzzz ** 2) +
          6 * ((Wxxyy - W/3) ** 2 +
               (Wxxzz - W/3) ** 2 +
               (Wyyzz - W/3) ** 2) +
          12 * (Wxxyz ** 2 +
                Wxyyz ** 2 +
                Wxyzz ** 2)
      )

    # Compute's equation denominator
    B = (
          Wxxxx ** 2 +
          Wyyyy ** 2 +
          Wzzzz ** 2 +
          4 * (Wxxxy ** 2 +
               Wxxxz ** 2 +
               Wxyyy ** 2 +
               Wyyyz ** 2 +
               Wxzzz ** 2 +
               Wyzzz ** 2) +
          6 * (Wxxyy ** 2 +
               Wxxzz ** 2 +
               Wyyzz ** 2) +
          12 * (Wxxyz ** 2 +
                Wxyyz ** 2 +
                Wxyzz ** 2)
      )

    # Compute KFA
    KFA = np.zeros(A.shape)
    cond1 = B > 0  # Avoiding Singularity (if B = 0, KFA = 0)
    # Avoiding overestimating KFA for small MKT values (KFA=0, MKT < tol)
    cond2 = W > 1e-8
    cond = np.logical_and(cond1, cond2)
    KFA[cond] = np.sqrt(A[cond]/B[cond])

    return KFA


def dki_prediction(dki_params, gtab, S0=1.):
    """ Predict a signal given diffusion kurtosis imaging parameters

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
    All parameters estimated from the diffusion kurtosis model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the first,
            second and third coordinates of the eigenvector
        3) Fifteen elements of the kurtosis tensor
    gtab : a GradientTable class instance
        The gradient table for this prediction
    S0 : float or ndarray (optional)
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 1

    Returns
    -------
    S : (..., N) ndarray
        Simulated signal based on the DKI model:

    .. math::

        S=S_{0}e^{-bD+\frac{1}{6}b^{2}D^{2}K}

    """
    evals, evecs, kt = split_dki_param(dki_params)

    # Define DKI design matrix according to given gtab
    A = design_matrix(gtab)

    # Flat parameters and initialize pred_sig
    fevals = evals.reshape((-1, evals.shape[-1]))
    fevecs = evecs.reshape((-1,) + evecs.shape[-2:])
    fkt = kt.reshape((-1, kt.shape[-1]))
    pred_sig = np.zeros((len(fevals), len(gtab.bvals)))
    if isinstance(S0, np.ndarray):
        S0_vol = np.reshape(S0, (len(fevals)))
    else:
        S0_vol = S0
    # looping for all voxels
    for v in range(len(pred_sig)):
        DT = np.dot(np.dot(fevecs[v], np.diag(fevals[v])), fevecs[v].T)
        dt = lower_triangular(DT)
        MD = (dt[0] + dt[2] + dt[5]) / 3
        if isinstance(S0_vol, np.ndarray):
            this_S0 = S0_vol[v]
        else:
            this_S0 = S0_vol
        X = np.concatenate((dt, fkt[v] * MD * MD,
                            np.array([-np.log(this_S0)])),
                           axis=0)
        pred_sig[v] = np.exp(np.dot(A, X))

    # Reshape data according to the shape of dki_params
    pred_sig = pred_sig.reshape(dki_params.shape[:-1] + (pred_sig.shape[-1],))

    return pred_sig


class DiffusionKurtosisModel(ReconstModel):
    """ Class for the Diffusion Kurtosis Model
    """

    def __init__(self, gtab, fit_method="WLS", return_S0_hat=False,
                 *args, **kwargs):
        """ Diffusion Kurtosis Tensor Model [1]

        Parameters
        ----------
        gtab : GradientTable instance
            The gradient table for the data set.
        fit_method : str or callable, optional
            str be one of the following:
                'OLS' or 'ULLS' for ordinary least squares.
                'WLS', 'WLLS' or 'UWLLS' for weighted ordinary least squares.
                    See dki.ls_fit_dki.
                'CLS' for LMI constrained ordinary least squares [2].
                'CWLS' for LMI constrained weighted least squares [2].
                    See dki.cls_fit_dki.
            callable has to have the signature:
                fit_method(design_matrix, data, *args, **kwargs).
            Default: "WLS"
        return_S0_hat : bool
            Boolean to return (True) or not (False) the S0 values for the fit.
        args, kwargs :
            arguments and key-word arguments passed to the fit_method.

        References
        ----------
        .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
               Estimation of tensors and tensor-derived measures in diffusional
               kurtosis imaging. Magn Reson Med. 65(3), 823-836
        .. [2] Dela Haije et al. "Enforcing necessary non-negativity
               constraints for common diffusion MRI models using sum of
               squares programming". NeuroImage 209, 2020, 116405.

        """
        ReconstModel.__init__(self, gtab)

        # Check if at least three b-values are given
        enough_b = check_multi_b(self.gtab, 3, non_zero=False)
        if not enough_b:
            msg = "DKI requires at least 3 b-values (which can include b=0)."
            raise ValueError(msg)

        self.common_fit_method = not callable(fit_method)
        if self.common_fit_method:
            try:
                self.fit_method = common_fit_methods[fit_method]
            except KeyError:
                msg = '"' + str(fit_method) + '" is not a known fit method. '
                msg += ' The fit method should either be a function or one of '
                msg += ' thecommon fit methods.'
                raise ValueError(msg)

        self.return_S0_hat = return_S0_hat
        self.args = args
        self.kwargs = kwargs

        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is None:
            self.min_signal = MIN_POSITIVE_SIGNAL
        elif self.min_signal <= 0:
            msg = "The `min_signal` key-word argument needs to be strictly"
            msg += " positive."
            raise ValueError(msg)

        self.design_matrix = design_matrix(self.gtab)
        self.inverse_design_matrix = np.linalg.pinv(self.design_matrix)

        tol = 1e-6
        self.min_diffusivity = tol / -self.design_matrix.min()

        self.convexity_constraint = fit_method in {'CLS', 'CWLS'}
        if self.convexity_constraint:
            self.cvxpy_solver = self.kwargs.pop('cvxpy_solver', None)
            self.convexity_level = self.kwargs.pop('convexity_level', 'full')
            msg = "convexity_level must be a positive, even number, or 'full'."
            if isinstance(self.convexity_level, str):
                if self.convexity_level == 'full':
                    self.sdp_constraints = load_sdp_constraints('dki')
                else:
                    raise ValueError(msg)
            elif self.convexity_level < 0 or self.convexity_level % 2:
                raise ValueError(msg)
            else:
                if self.convexity_level > 4:
                    msg = "Maximum convexity_level supported is 4."
                    warnings.warn(msg)
                    self.convexity_level = 4
                self.sdp_constraints = load_sdp_constraints(
                    'dki', self.convexity_level)
            self.sdp = PositiveDefiniteLeastSquares(22, A=self.sdp_constraints)

        self.weights = fit_method in {'WLS', 'WLLS', 'UWLLS', 'CWLS'}
        self.is_multi_method = fit_method in ['WLS', 'OLS', 'UWLLS', 'ULLS',
                                              'WLLS', 'OLLS', 'CLS', 'CWLS']

    def fit(self, data, mask=None):
        """ Fit method of the DKI model.

        Parameters
        ----------
        data : array
            The measured signal from one voxel.
        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]

        """
        data_thres = np.maximum(data, self.min_signal)
        if self.is_multi_method:
            return self.multi_fit(data_thres, mask=mask)

        S0_params = None
        if data.ndim == 1:
            params, extra = self.fit_method(self.design_matrix, data_thres,
                                            return_S0_hat=self.return_S0_hat,
                                            *self.args, **self.kwargs)
            if self.return_S0_hat:
                params, S0_params = params
            return DiffusionKurtosisFit(self, params, model_S0=S0_params)

        if mask is not None:
            # Check for valid shape of the mask
            if data.ndim == 4 and mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)

        data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))
        data_in_mask = np.maximum(data_in_mask, self.min_signal)

        params, extra = self.fit_method(self.design_matrix, data_in_mask,
                                        return_S0_hat=self.return_S0_hat,
                                        *self.args, **self.kwargs)

        if self.return_S0_hat:
            params, S0_params = params

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            dki_params = params.reshape(out_shape)
            if self.return_S0_hat:
                S0_params = S0_params.reshape(out_shape).squeeze()
        else:
            dki_params = np.zeros(data.shape[:-1] + (27,))
            dki_params[mask, :] = params
            if self.return_S0_hat:
                S0_params_in_mask = np.zeros(data.shape[:-1])
                S0_params_in_mask[mask] = S0_params.squeeze()
                S0_params = S0_params_in_mask

        return DiffusionKurtosisFit(self, dki_params, model_S0=S0_params)

    @multi_voxel_fit
    def multi_fit(self, data_thres, mask=None):
        extra_args = {} if not self.convexity_constraint else {
            'cvxpy_solver': self.cvxpy_solver,
            'sdp': self.sdp,
            }
        params, extra = self.fit_method(self.design_matrix, data_thres,
                                        self.inverse_design_matrix,
                                        return_S0_hat=self.return_S0_hat,
                                        weights=self.weights,
                                        min_diffusivity=self.min_diffusivity,
                                        **extra_args)

        S0_params = None
        if self.return_S0_hat:
            params, S0_params = params

        return DiffusionKurtosisFit(self, params, model_S0=S0_params)

    def predict(self, dki_params, S0=1.):
        """ Predict a signal for this DKI model class instance given parameters

        Parameters
        ----------
        dki_params : ndarray (x, y, z, 27) or (n, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follows::

        1. Three diffusion tensor's eigenvalues
        2. Three lines of the eigenvector matrix each containing the
        first, second and third coordinates of the eigenvector
        3. Fifteen elements of the kurtosis tensor


        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        """
        return dki_prediction(dki_params, self.gtab, S0)


class DiffusionKurtosisFit(TensorFit):
    """ Class for fitting the Diffusion Kurtosis Model"""

    def __init__(self, model, model_params, model_S0=None):
        """ Initialize a DiffusionKurtosisFit class instance

        Since DKI is an extension of DTI, class instance is defined as subclass
        of the TensorFit from dti.py

        Parameters
        ----------
        model : DiffusionKurtosisModel Class instance
            Class instance containing the Diffusion Kurtosis Model for the fit
        model_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the diffusion kurtosis model,
            not including S0.
            Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the
                first, second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
        model_S0 : ndarray (x, y, z, 1) or (n, 1), optional
            S0 estimated from the diffusion kurtosis model.

        """

        TensorFit.__init__(self, model, model_params, model_S0=model_S0)

    @property
    def kt(self):
        """
        Return the 15 independent elements of the kurtosis tensor as an array
        """
        return self.model_params[..., 12:27]

    def akc(self, sphere):
        r""" Calculate the apparent kurtosis coefficient (AKC) in each
        direction on the sphere for each voxel in the data

        Parameters
        ----------
        sphere : Sphere class instance

        Returns
        -------
        akc : ndarray
           The estimates of the apparent kurtosis coefficient in every
           direction on the input sphere

        Notes
        -----
        For each sphere direction with coordinates $(n_{1}, n_{2}, n_{3})$, the
        calculation of AKC is done using formula:

        .. math ::

            AKC(n)=\frac{MD^{2}}{ADC(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}
            \sum_{k=1}^{3}\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

        where $W_{ijkl}$ are the elements of the kurtosis tensor, MD the mean
        diffusivity and ADC the apparent diffusion coefficient computed as:

        .. math ::

            ADC(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

        where $D_{ij}$ are the elements of the diffusion tensor.

        """
        return apparent_kurtosis_coef(self.model_params, sphere)

    def mk(self, min_kurtosis=-3./7, max_kurtosis=10, analytical=True):
        r""" Compute mean kurtosis (MK) from the kurtosis tensor

        Parameters
        ----------
        min_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, mean
            kurtosis values that are smaller than `min_kurtosis` are replaced
            with `min_kurtosis`. Default = -3./7 (theoretical kurtosis limit
            for regions that consist of water confined to spherical pores [4]_)
        max_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, mean
            kurtosis values that are larger than `max_kurtosis` are replaced
            with `max_kurtosis`. Default = 10
        analytical : bool (optional)
            If True, MK is calculated using its analytical solution, otherwise
            an exact numerical estimator is used (see Notes). Default is set to
            True.

        Returns
        -------
        mk : array
            Calculated MK.

        Notes
        -----
        The MK is defined as the average of directional kurtosis coefficients
        across all spatial directions, which can be formulated by the following
        surface integral[1]_:

        .. math::

             MK \equiv \frac{1}{4\pi} \int d\Omega_\mathbf{n} K(\mathbf{n})

        This integral can be numerically solved by averaging directional
        kurtosis values sampled for directions of a spherical t-design [2]_.

        Alternatively, MK can be solved from the analytical solution derived by
        Tabesh et al. [3]_. This solution is given by:

        .. math::

            MK=F_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{1111}+
               F_1(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{2222}+
               F_1(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{3333}+ \\
               F_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}+
               F_2(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{1133}+
               F_2(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{1122}

        where $\hat{W}_{ijkl}$ are the components of the $W$ tensor in the
        coordinates system defined by the eigenvectors of the diffusion tensor
        $\mathbf{D}$ and

        .. math::

            F_1(\lambda_1,\lambda_2,\lambda_3)=
            \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
            {18(\lambda_1-\lambda_2)(\lambda_1-\lambda_3)}
            [\frac{\sqrt{\lambda_2\lambda_3}}{\lambda_1}
            R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
            \frac{3\lambda_1^2-\lambda_1\lambda_2-\lambda_2\lambda_3-
            \lambda_1\lambda_3}
            {3\lambda_1 \sqrt{\lambda_2 \lambda_3}}
            R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-1 ]

            F_2(\lambda_1,\lambda_2,\lambda_3)=
            \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
            {3(\lambda_2-\lambda_3)^2}
            [\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}
            R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
            \frac{2\lambda_1-\lambda_2-\lambda_3}{3\sqrt{\lambda_2 \lambda_3}}
            R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-2]

        where $R_f$ and $R_d$ are the Carlson's elliptic integrals.

        References
        ----------
        .. [1] Jensen, J.H., Helpern, J.A., 2010. MRI quantification of
               non-Gaussian water diffusion by kurtosis analysis. NMR in
               Biomedicine 23(7): 698-710
        .. [2] Hardin, R.H., Sloane, N.J.A., 1996. McLaren's Improved Snub Cube
               and Other New Spherical Designs in Three Dimensions. Discrete
               and Computational Geometry 15, 429-441.
        .. [3] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
               Estimation of tensors and tensor-derived measures in diffusional
               kurtosis imaging. Magn Reson Med. 65(3), 823-836
        .. [4] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
               Robust estimation from DW-MRI using homogeneous polynomials.
               Proceedings of the 8th {IEEE} International Symposium on
               Biomedical Imaging: From Nano to Macro, ISBI 2011, 262-265.
               doi: 10.1109/ISBI.2011.5872402

        """
        return mean_kurtosis(self.model_params, min_kurtosis, max_kurtosis,
                             analytical)

    def ak(self, min_kurtosis=-3./7, max_kurtosis=10, analytical=True):
        r"""
        Compute axial kurtosis (AK) of a diffusion kurtosis tensor [1]_

        Parameters
        ----------
        min_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, axial
            kurtosis values that are smaller than `min_kurtosis` are replaced
            with -3./7 (theoretical kurtosis limit
            for regions that consist of water confined to spherical pores [2]_)
        max_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, axial
            kurtosis values that are larger than `max_kurtosis` are replaced
            with `max_kurtosis`. Default = 10
        analytical : bool (optional)
            If True, AK is calculated from rotated diffusion kurtosis tensor,
            otherwise it will be computed from the apparent diffusion kurtosis
            values along the principal axis of the diffusion tensor
            (see notes). Default is set to True.

        Returns
        -------
        ak : array
            Calculated AK.

        Notes
        -----
        AK is defined as the directional kurtosis parallel to the fiber's main
        direction e1 [1]_, [2]_. You can compute AK using to approaches:

        1) AK is calculated from rotated diffusion kurtosis tensor [2]_, i.e.:

        .. math::
            AK = \hat{W}_{1111}
            \frac{(\lambda_{1}+\lambda_{2}+\lambda_{3})^2}{(9 \lambda_{1}^2)}

        2) AK can be sampled from the principal axis of the diffusion tensor:

        .. math::
            AK = K(\mathbf{\mathbf{e}_1)

        Although both approaches leads to an exact calculation of AK, the
        first approach will be referred to as the analytical method while the
        second approach will be referred to as the numerical method based on
        their analogy to the estimation strategies for MK and RK.

        References
        ----------
        .. [1] Jensen, J.H., Helpern, J.A., 2010. MRI quantification of
               non-Gaussian water diffusion by kurtosis analysis. NMR in
               Biomedicine 23(7): 698-710
        .. [2] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
               Estimation of tensors and tensor-derived measures in diffusional
               kurtosis imaging. Magn Reson Med. 65(3), 823-836
        .. [3] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
               Robust estimation from DW-MRI using homogeneous polynomials.
               Proceedings of the 8th {IEEE} International Symposium on
               Biomedical Imaging: From Nano to Macro, ISBI 2011, 262-265.
               doi: 10.1109/ISBI.2011.5872402

        """
        return axial_kurtosis(self.model_params, min_kurtosis, max_kurtosis,
                              analytical)

    def rk(self, min_kurtosis=-3./7, max_kurtosis=10, analytical=True):
        r""" Compute radial kurtosis (RK) of a diffusion kurtosis tensor [1]_

        Parameters
        ----------
        min_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range,
            radial kurtosis values that are smaller than `min_kurtosis` are
            replaced with `min_kurtosis`. Default = -3./7 (theoretical kurtosis
            limit for regions that consist of water confined to spherical pores
            [3]_)
        max_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range,
            radial kurtosis values that are larger than `max_kurtosis` are
            replaced with `max_kurtosis`. Default = 10
        analytical : bool (optional)
            If True, RK is calculated using its analytical solution, otherwise
            an exact numerical estimator is used (see Notes). Default is set to
            True

        Returns
        -------
        rk : array
            Calculated RK.

        Notes
        -----
        RK is defined as the average of the directional kurtosis perpendicular
        to the fiber's main direction e1 [1]_, [2]_:

        .. math::

        RK \equiv \frac{1}{2\pi} \int d\Omega _\mathbf{\theta}
            K(\mathbf{\theta}) \delta (\mathbf{\theta}\cdot \mathbf{e}_1)

        This equation can be numerically computed by averaging apparent
        directional kurtosis samples for directions perpendicular to e1.

        Otherwise, RK can be calculated from its analytical solution [2]_:

        .. math::

            K_{\bot} = G_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2222} +
                       G_1(\lambda_1,\lambda_3,\lambda_2)\hat{W}_{3333} +
                       G_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}

        where:

        .. math::

            G_1(\lambda_1,\lambda_2,\lambda_3)=
            \frac{(\lambda_1+\lambda_2+\lambda_3)^2}{18\lambda_2(\lambda_2-
            \lambda_3)} \left (2\lambda_2 +
            \frac{\lambda_3^2-3\lambda_2\lambda_3}{\sqrt{\lambda_2\lambda_3}}
            \right)

        and

        .. math::

            G_2(\lambda_1,\lambda_2,\lambda_3)=
           \frac{(\lambda_1+\lambda_2+\lambda_3)^2}{(\lambda_2-\lambda_3)^2}
           \left ( \frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}-
           2\right )

        References
        ----------
        .. [1] Jensen, J.H., Helpern, J.A., 2010. MRI quantification of
               non-Gaussian water diffusion by kurtosis analysis. NMR in
               Biomedicine 23(7): 698-710
        .. [2] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
               Estimation of tensors and tensor-derived measures in diffusional
               kurtosis imaging. Magn Reson Med. 65(3), 823-836
        .. [3] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
               Robust estimation from DW-MRI using homogeneous polynomials.
               Proceedings of the 8th {IEEE} International Symposium on
               Biomedical Imaging: From Nano to Macro, ISBI 2011, 262-265.
               doi: 10.1109/ISBI.2011.5872402

        """
        return radial_kurtosis(self.model_params, min_kurtosis, max_kurtosis,
                               analytical)

    def kmax(self, sphere='repulsion100', gtol=1e-5, mask=None):
        r""" Compute the maximum value of a single voxel kurtosis tensor

        Parameters
        ----------
        sphere : Sphere class instance, optional
            The sphere providing sample directions for the initial search of
            the maximum value of kurtosis.
        gtol : float, optional
            This input is to refine kurtosis maximum under the precision of the
            directions sampled on the sphere class instance. The gradient of
            the convergence procedure must be less than gtol before successful
            termination. If gtol is None, fiber direction is directly taken
            from the initial sampled directions of the given sphere object

        Returns
        -------
        max_value : float
            kurtosis tensor maximum value

        """
        return kurtosis_maximum(self.model_params, sphere, gtol, mask)

    def mkt(self, min_kurtosis=-3./7, max_kurtosis=10):
        r""" Compute mean of the kurtosis tensor (MKT) [1]_

        Parameters
        ----------
        min_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, mean
            kurtosis values that are smaller than `min_kurtosis` are replaced
            with `min_kurtosis`. Default = -3./7 (theoretical kurtosis limit
            for regions that consist of water confined to spherical pores [2]_)
        max_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, mean
            kurtosis values that are larger than `max_kurtosis` are replaced
            with `max_kurtosis`. Default = 10

        Returns
        -------
        mkt : array
            Calculated mean kurtosis tensor.

        Notes
        -----
        The MKT is defined as [1]_:

        .. math::

             MKT \equiv \frac{1}{4\pi} \int d
             \Omega_{\mathnbf{n}} n_i n_j n_k n_l W_{ijkl}


        which can be directly computed from the trace of the kurtosis tensor:

        .. math::

        MKT = \frac{1}{5} Tr(\mathbf{W}) = \frac{1}{5}
        (W_{1111} + W_{2222} + W_{3333} + 2W_{1122} + 2W_{1133} + 2W_{2233})

        References
        ----------
        .. [1] Hansen, B., Lund, T. E., Sangill, R., and Jespersen, S. N. 2013.
               Experimentally and computationally fast method for estimation
               of a mean kurtosis. Magnetic Resonance in Medicine69, 1754–1760.
               388. doi:10.1002/mrm.24743
        .. [2] Barmpoutis, A., & Zhuo, J., 2011. Diffusion kurtosis imaging:
               Robust estimation from DW-MRI using homogeneous polynomials.
               Proceedings of the 8th {IEEE} International Symposium on
               Biomedical Imaging: From Nano to Macro, ISBI 2011, 262-265.
               doi: 10.1109/ISBI.2011.5872402
        """
        return mean_kurtosis_tensor(self.model_params, min_kurtosis,
                                    max_kurtosis)

    @property
    def kfa(self):
        r""" Return the kurtosis tensor (KFA) [1]_

        Notes
        -----
        The KFA is defined as [1]_:

        .. math::

             KFA \equiv
             \frac{||\mathbf{W} - MKT \mathbf{I}^{(4)}||_F}{||\mathbf{W}||_F}

        where $W$ is the kurtosis tensor, MKT the kurtosis tensor mean, $I^(4)$
        is the fully symmetric rank 2 isotropic tensor and $||...||_F$ is the
        tensor's Frobenius norm [1]_.

        References
        ----------
        .. [1] Glenn, G. R., Helpern, J. A., Tabesh, A., and Jensen, J. H.
               (2015). Quantitative assessment of diffusional kurtosis
               anisotropy. NMR in Biomedicine 28, 448–459. doi:10.1002/nbm.3271

        """
        return kurtosis_fractional_anisotropy(self.model_params)

    def predict(self, gtab, S0=1.):
        r""" Given a DKI model fit, predict the signal on the vertices of a
        gradient table

        Parameters
        ----------
        gtab : a GradientTable class instance
            The gradient table for this prediction

        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        Notes
        -----
        The predicted signal is given by:

        .. math::

            S(n,b)=S_{0}e^{-bD(n)+\frac{1}{6}b^{2}D(n)^{2}K(n)}

        $\mathbf{D(n)}$ and $\mathbf{K(n)}$ can be computed from the DT and KT
        using the following equations:

        .. math::

            D(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

        and

        .. math::

            K(n)=\frac{MD^{2}}{D(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}
            \sum_{k=1}^{3}\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

        where $D_{ij}$ and $W_{ijkl}$ are the elements of the second-order DT
        and the fourth-order KT tensors, respectively, and $MD$ is the mean
        diffusivity.

        """
        return dki_prediction(self.model_params, gtab, S0)


def params_to_dki_params(result, min_diffusivity=0):
    # Extracting the diffusion tensor parameters from solution
    DT_elements = result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = evals.mean(0)**2
    KT_elements = result[6:21] / MD_square if MD_square else 0.*result[6:21]
    S0 = np.exp(-result[[-1]])

    # Write output
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 KT_elements, S0), axis=0)

    return dki_params


def ls_fit_dki(design_matrix, data, inverse_design_matrix,
               return_S0_hat=False, weights=True,
               min_diffusivity=0):
    r""" Compute the diffusion and kurtosis tensors using an ordinary or
    weighted linear least squares approach [1]_

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array (g)
        Data or response variables holding the data.
    inverse_design_matrix : array (22, g)
        Inverse of the design matrix.
    return_S0_hat : bool, optional
        Boolean to return (True) or not (False) the S0 values for the fit.
    weights : bool, optional
        Parameter indicating whether weights are used. Default: True.
    min_diffusivity : float, optional
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    dki_params : array (27)
    All parameters estimated from the diffusion kurtosis model for all N
    voxels. Parameters are ordered as follows:
        1) Three diffusion tensor eigenvalues.
        2) Three blocks of three elements, containing the first second and
            third coordinates of the diffusion tensor eigenvectors.
        3) Fifteen elements of the kurtosis tensor.

    References
    ----------
    [1] Veraart, J., Sijbers, J., Sunaert, S., Leemans, A., Jeurissen, B.,
        2013. Weighted linear least squares estimation of diffusion MRI
        parameters: Strengths, limitations, and pitfalls. Magn Reson Med 81,
        335-346.

    """
    # Set up least squares problem
    A = design_matrix
    y = np.log(data)

    # DKI ordinary linear least square solution
    result = np.dot(inverse_design_matrix, y)

    # Define weights as diag(yn**2)
    if weights:
        W = np.diag(np.exp(2 * np.dot(A, result)))
        AT_W = np.dot(A.T, W)
        inv_AT_W_A = np.linalg.pinv(np.dot(AT_W, A))
        AT_W_LS = np.dot(AT_W, y)
        result = np.dot(inv_AT_W_A, AT_W_LS)

    # Write output
    dki_params = params_to_dki_params(result, min_diffusivity=min_diffusivity)

    if return_S0_hat:
        return (dki_params[..., 0:-1], dki_params[..., -1]), None
    else:
        return dki_params[..., 0:-1], None


def cls_fit_dki(design_matrix, data, inverse_design_matrix, sdp,
                return_S0_hat=False, weights=True,
                min_diffusivity=0, cvxpy_solver=None):
    r""" Compute the diffusion and kurtosis tensors using a constrained
    ordinary or weighted linear least squares approach [1]_

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array (g)
        Data or response variables holding the data.
    inverse_design_matrix : array (22, g)
        Inverse of the design matrix.
    sdp : PositiveDefiniteLeastSquares instance
        A CVXPY representation of a regularized least squares optimization
        problem.
    return_S0_hat : bool, optional
        Boolean to return (True) or not (False) the S0 values for the fit.
    weights : bool, optional
        Parameter indicating whether weights are used. Default: True.
    min_diffusivity : float, optional
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.
    cvxpy_solver : str, optional
        cvxpy solver name. Optionally optimize the positivity constraint with a
        particular cvxpy solver. See https://www.cvxpy.org/ for details.
        Default: None (cvxpy chooses its own solver).

    Returns
    -------
    dki_params : array (27)
    All parameters estimated from the diffusion kurtosis model for all N
    voxels. Parameters are ordered as follows:
        1) Three diffusion tensor eigenvalues.
        2) Three blocks of three elements, containing the first second and
            third coordinates of the diffusion tensor eigenvectors.
        3) Fifteen elements of the kurtosis tensor.

    References
    ----------
    .. [1] Dela Haije et al. "Enforcing necessary non-negativity constraints
           for common diffusion MRI models using sum of squares programming".
           NeuroImage 209, 2020, 116405.
    """
    # Set up least squares problem
    A = design_matrix
    y = np.log(data)

    # Define sqrt weights as diag(yn)
    if weights:
        result = np.dot(inverse_design_matrix, y)
        W = np.diag(np.exp(np.dot(A, result)))
        A = np.dot(W, A)
        y = np.dot(W, y)

    # Solve sdp
    result = sdp.solve(A, y, check=True, solver=cvxpy_solver)

    # Write output
    dki_params = params_to_dki_params(result, min_diffusivity=min_diffusivity)

    if return_S0_hat:
        return (dki_params[..., 0:-1], dki_params[..., -1]), None
    else:
        return dki_params[..., 0:-1], None


def Wrotate(kt, Basis):
    r""" Rotate a kurtosis tensor from the standard Cartesian coordinate system
    to another coordinate system basis

    Parameters
    ----------
    kt : (15,)
        Vector with the 15 independent elements of the kurtosis tensor
    Basis : array (3, 3)
        Vectors of the basis column-wise oriented
    inds : array(m, 4) (optional)
        Array of vectors containing the four indexes of m specific elements of
        the rotated kurtosis tensor. If not specified all 15 elements of the
        rotated kurtosis tensor are computed.

    Returns
    -------
    Wrot : array (m,) or (15,)
        Vector with the m independent elements of the rotated kurtosis tensor.
        If 'indices' is not specified all 15 elements of the rotated kurtosis
        tensor are computed.

    Notes
    -----
    KT elements are assumed to be ordered as follows:

    .. math::

    \begin{matrix} ( & W_{xxxx} & W_{yyyy} & W_{zzzz} & W_{xxxy} & W_{xxxz}
                     & ... \\
                     & W_{xyyy} & W_{yyyz} & W_{xzzz} & W_{yzzz} & W_{xxyy}
                     & ... \\
                     & W_{xxzz} & W_{yyzz} & W_{xxyz} & W_{xyyz} & W_{xyzz}
                     & & )\end{matrix}

    References
    ----------
    [1] Hui ES, Cheung MM, Qi L, Wu EX, 2008. Towards better MR
    characterization of neural tissues using directional diffusion kurtosis
    analysis. Neuroimage 42(1): 122-34

    """
    inds = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2],
                     [0, 0, 0, 1], [0, 0, 0, 2], [0, 1, 1, 1],
                     [1, 1, 1, 2], [0, 2, 2, 2], [1, 2, 2, 2],
                     [0, 0, 1, 1], [0, 0, 2, 2], [1, 1, 2, 2],
                     [0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 2, 2]])

    Wrot = np.zeros(kt.shape)

    for e in range(len(inds)):
        Wrot[..., e] = Wrotate_element(kt, inds[e][0], inds[e][1], inds[e][2],
                                       inds[e][3], Basis)

    return Wrot


# Defining keys to select a kurtosis tensor element with indexes (i, j, k, l)
# on a kt vector that contains only the 15 independent elements of the kurtosis
# tensor: Considering y defined by (i+1) * (j+1) * (k+1) * (l+1). Two elements
# of the full 4D kurtosis tensor are equal if y obtain from the indexes of
# these two element are equal. Therefore, the possible values of y (1, 16, 81,
# 2, 3, 8, 24 27, 54, 4, 9, 36, 6, 12, 18) are used to point each element of
# the kurtosis tensor on the format of a vector containing the 15 independent
# elements.
ind_ele = {1: 0, 16: 1, 81: 2, 2: 3, 3: 4, 8: 5, 24: 6, 27: 7, 54: 8, 4: 9,
           9: 10, 36: 11, 6: 12, 12: 13, 18: 14}


def Wrotate_element(kt, indi, indj, indk, indl, B):
    r""" Compute the the specified index element of a kurtosis tensor rotated
    to the coordinate system basis B

    Parameters
    ----------
    kt : ndarray (x, y, z, 15) or (n, 15)
        Array containing the 15 independent elements of the kurtosis tensor
    indi : int
        Rotated kurtosis tensor element index i (0 for x, 1 for y, 2 for z)
    indj : int
        Rotated kurtosis tensor element index j (0 for x, 1 for y, 2 for z)
    indk : int
        Rotated kurtosis tensor element index k (0 for x, 1 for y, 2 for z)
    indl: int
        Rotated kurtosis tensor element index l (0 for x, 1 for y, 2 for z)
    B: array (x, y, z, 3, 3) or (n, 15)
        Vectors of the basis column-wise oriented

    Returns
    -------
    Wre : float
          rotated kurtosis tensor element of index ind_i, ind_j, ind_k, ind_l

    Notes
    -----
    It is assumed that initial kurtosis tensor elementes are defined on the
    Cartesian coordinate system.

    References
    ----------
    [1] Hui ES, Cheung MM, Qi L, Wu EX, 2008. Towards better MR
    characterization of neural tissues using directional diffusion kurtosis
    analysis. Neuroimage 42(1): 122-34

    """
    Wre = 0

    xyz = [0, 1, 2]
    for il in xyz:
        for jl in xyz:
            for kl in xyz:
                for ll in xyz:
                    key = (il + 1) * (jl + 1) * (kl + 1) * (ll + 1)
                    multiplyB = \
                        B[..., il, indi] * B[..., jl, indj] * \
                        B[..., kl, indk] * B[..., ll, indl]
                    Wre = Wre + multiplyB * kt[..., ind_ele[key]]
    return Wre


def Wcons(k_elements):
    r""" Construct the full 4D kurtosis tensors from its 15 independent
    elements

    Parameters
    ----------
    k_elements : (15,)
        elements of the kurtosis tensor in the following order:

    .. math::

    \begin{matrix} ( & W_{xxxx} & W_{yyyy} & W_{zzzz} & W_{xxxy} & W_{xxxz}
                     & ... \\
                     & W_{xyyy} & W_{yyyz} & W_{xzzz} & W_{yzzz} & W_{xxyy}
                     & ... \\
                     & W_{xxzz} & W_{yyzz} & W_{xxyz} & W_{xyyz} & W_{xyzz}
                     & & )\end{matrix}

    Returns
    -------
    W : array(3, 3, 3, 3)
        Full 4D kurtosis tensor

    """
    W = np.zeros((3, 3, 3, 3))

    xyz = [0, 1, 2]
    for ind_i in xyz:
        for ind_j in xyz:
            for ind_k in xyz:
                for ind_l in xyz:
                    key = (ind_i + 1) * (ind_j + 1) * (ind_k + 1) * (ind_l + 1)
                    W[ind_i][ind_j][ind_k][ind_l] = k_elements[ind_ele[key]]

    return W


def split_dki_param(dki_params):
    r""" Extract the diffusion tensor eigenvalues, the diffusion tensor
    eigenvector matrix, and the 15 independent elements of the kurtosis tensor
    from the model parameters estimated from the DKI model

    Parameters
    ----------
    dki_params : ndarray (x, y, z, 27) or (n, 27)
    All parameters estimated from the diffusion kurtosis model.
    Parameters are ordered as follows:
        1) Three diffusion tensor's eigenvalues
        2) Three lines of the eigenvector matrix each containing the first,
            second and third coordinates of the eigenvector
        3) Fifteen elements of the kurtosis tensor

    Returns
    -------
    eigvals : array (x, y, z, 3) or (n, 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (x, y, z, 3, 3) or (n, 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])
    kt : array (x, y, z, 15) or (n, 15)
        Fifteen elements of the kurtosis tensor

    """
    evals = dki_params[..., :3]
    evecs = dki_params[..., 3:12].reshape(dki_params.shape[:-1] + (3, 3))
    kt = dki_params[..., 12:27]

    return evals, evecs, kt


common_fit_methods = {'WLS': ls_fit_dki,
                      'OLS': ls_fit_dki,
                      'NLS': nlls_fit_tensor,
                      'UWLLS': ls_fit_dki,
                      'ULLS': ls_fit_dki,
                      'WLLS': ls_fit_dki,
                      'OLLS': ls_fit_dki,
                      'NLLS': nlls_fit_tensor,
                      'RT': restore_fit_tensor,
                      'restore': restore_fit_tensor,
                      'RESTORE': restore_fit_tensor,
                      'CLS': cls_fit_dki,
                      'CWLS': cls_fit_dki
                      }
