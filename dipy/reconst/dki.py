#!/usr/bin/python
""" Classes and functions for fitting the diffusion kurtosis model """
from __future__ import division, print_function, absolute_import

import numpy as np
from dipy.reconst.dti import (TensorFit, mean_diffusivity, axial_diffusivity,
                              radial_diffusivity, from_lower_triangular,
                              lower_triangular, decompose_tensor,
                              _min_positive_signal)

from dipy.reconst.utils import dki_design_matrix as design_matrix
from dipy.utils.six.moves import range
from ..core.onetime import auto_attr
from .base import ReconstModel
from dipy.core.ndindex import ndindex


def _positive_evals(L1, L2, L3, er=2e-7):
    """ Helper function that indentifies which voxels in a array have all
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
    r""" Computes the Carlson's incomplete elliptic integral of the first kind
    defined as:

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

    Note
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
    Q = (3.*errtol) ** (-1/6.) * np.max(np.abs([An - xn, An - yn, An - zn]),
                                        axis=0)
    # Convergence has to be done voxel by voxel
    index = ndindex(x.shape)
    for v in index:
        n = 0
        # Convergence condition
        while 4.**(-n) * Q[v] > abs(An[v]):
            xnroot = np.sqrt(xn[v])
            ynroot = np.sqrt(yn[v])
            znroot = np.sqrt(zn[v])
            lamda = xnroot*(ynroot + znroot) + ynroot*znroot
            n = n + 1
            xn[v] = (xn[v]+lamda) * 0.250
            yn[v] = (yn[v]+lamda) * 0.250
            zn[v] = (zn[v]+lamda) * 0.250
            An[v] = (An[v]+lamda) * 0.250

    # post convergence calculation
    X = 1. - xn/An
    Y = 1. - yn/An
    Z = - X - Y
    E2 = X*Y - Z*Z
    E3 = X * Y * Z
    RF = An**(-1/2.) * (1 - E2/10. + E3/14. + (E2**2)/24. - 3/44.*E2*E3)

    return RF


def carlson_rd(x, y, z, errtol=1e-4):
    r""" Computes the Carlson's incomplete elliptic integral of the second kind
    defined as:

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

    Note
    -----
    x, y, and z have to be nonnegative and at most x or y is zero.
    """
    xn = x.copy()
    yn = y.copy()
    zn = z.copy()
    A0 = (xn + yn + 3.*zn) / 5.0
    An = A0.copy()
    Q = (errtol/4.) ** (-1/6.) * np.max(np.abs([An - xn, An - yn, An - zn]),
                                        axis=0)
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
            lamda = xnroot*(ynroot + znroot) + ynroot*znroot
            sum_term[v] = sum_term[v] + 4.**(-n[v]) / (znroot * (zn[v]+lamda))
            n[v] = n[v] + 1
            xn[v] = (xn[v]+lamda) * 0.250
            yn[v] = (yn[v]+lamda) * 0.250
            zn[v] = (zn[v]+lamda) * 0.250
            An[v] = (An[v]+lamda) * 0.250

    # post convergence calculation
    X = (A0 - x) / (4.**(n) * An)
    Y = (A0 - y) / (4.**(n) * An)
    Z = - (X+Y) / 3.
    E2 = X*Y - 6.*Z*Z
    E3 = (3.*X*Y - 8.*Z*Z) * Z
    E4 = 3. * (X*Y - Z*Z) * Z**2.
    E5 = X * Y * Z**3.
    RD = \
        4**(-n) * An**(-3/2.) * \
        (1 - 3/14.*E2 + 1/6.*E3 + 9/88.*(E2**2) - 3/22.*E4 - 9/52.*E2*E3 +
         3/26.*E5) + 3*sum_term

    return RD


def _F1m(a, b, c):
    """ Helper function that computes function $F_1$ which is required to
    compute the analytical solution of the Mean kurtosis.

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
    --------
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
    # http://gsoc2015dipydki.blogspot.co.uk/2015/08/rnh-post-13-start-wrapping-up-test.html
    er = 2.5e-2

    # Initialize F1
    F1 = np.zeros(a.shape)

    # Only computes F1 in voxels that have all eigenvalues larger than zero
    cond0 = _positive_evals(a, b, c)

    # Apply formula for non problematic plaussible cases, i.e. a!=b and a!=c
    cond1 = np.logical_and(cond0, np.logical_and(abs(a - b) >= a * er,
                                                 abs(a - c) >= a * er))
    if np.sum(cond1) != 0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        RFm = carlson_rf(L1/L2, L1/L3, np.ones(len(L1)))
        RDm = carlson_rd(L1/L2, L1/L3, np.ones(len(L1)))
        F1[cond1] = ((L1+L2+L3) ** 2) / (18 * (L1-L2) * (L1-L3)) * \
                    (np.sqrt(L2*L3) / L1 * RFm +
                     (3 * L1**2 - L1*L2 - L1*L3 - L2*L3) /
                     (3 * L1 * np.sqrt(L2*L3)) * RDm - 1)

    # Resolve possible sigularity a==b
    cond2 = np.logical_and(cond0, np.logical_and(abs(a - b) < a * er,
                                                 abs(a - c) > a * er))
    if np.sum(cond2) != 0:
        L1 = (a[cond2]+b[cond2]) / 2.
        L3 = c[cond2]
        F1[cond2] = _F2m(L3, L1, L1) / 2.

    # Resolve possible sigularity a==c
    cond3 = np.logical_and(cond0, np.logical_and(abs(a - c) < a * er,
                                                 abs(a - b) > a * er))
    if np.sum(cond3) != 0:
        L1 = (a[cond3]+c[cond3]) / 2.
        L2 = b[cond3]
        F1[cond3] = _F2m(L2, L1, L1) / 2

    # Resolve possible sigularity a==b and a==c
    cond4 = np.logical_and(cond0, np.logical_and(abs(a - c) < a * er,
                                                 abs(a - b) < a * er))
    if np.sum(cond4) != 0:
        F1[cond4] = 1/5.

    return F1


def _F2m(a, b, c):
    """ Helper function that computes function $F_2$ which is required to
    compute the analytical solution of the Mean kurtosis.

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
    --------
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
    # http://gsoc2015dipydki.blogspot.co.uk/2015/08/rnh-post-13-start-wrapping-up-test.html
    er = 2.5e-2

    # Initialize F2
    F2 = np.zeros(a.shape)

    # Only computes F2 in voxels that have all eigenvalues larger than zero
    cond0 = _positive_evals(a, b, c)

    # Apply formula for non problematic plaussible cases, i.e. b!=c
    cond1 = np.logical_and(cond0, (abs(b - c) > b * er))
    if np.sum(cond1) != 0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        RF = carlson_rf(L1/L2, L1/L3, np.ones(len(L1)))
        RD = carlson_rd(L1/L2, L1/L3, np.ones(len(L1)))
        F2[cond1] = (((L1+L2+L3) ** 2) / (3. * (L2-L3) ** 2)) * \
                    (((L2+L3) / (np.sqrt(L2*L3))) * RF +
                     ((2.*L1-L2-L3) / (3.*np.sqrt(L2*L3))) * RD - 2.)

    # Resolve possible sigularity b==c
    cond2 = np.logical_and(cond0, np.logical_and(abs(b - c) < b * er,
                                                 abs(a - b) > b * er))
    if np.sum(cond2) != 0:
        L1 = a[cond2]
        L3 = (c[cond2]+b[cond2]) / 2.

        # Cumpute alfa [1]_
        x = 1. - (L1/L3)
        alpha = np.zeros(len(L1))
        for i in range(len(x)):
            if x[i] > 0:
                alpha[i] = 1./np.sqrt(x[i]) * np.arctanh(np.sqrt(x[i]))
            else:
                alpha[i] = 1./np.sqrt(-x[i]) * np.arctan(np.sqrt(-x[i]))

        F2[cond2] = \
            6. * ((L1 + 2.*L3)**2) / (144. * L3**2 * (L1-L3)**2) * \
            (L3 * (L1 + 2.*L3) + L1 * (L1 - 4.*L3) * alpha)

    # Resolve possible sigularity a==b and a==c
    cond3 = np.logical_and(cond0, np.logical_and(abs(b - c) < b * er,
                                                 abs(a - b) < b * er))
    if np.sum(cond3) != 0:
        F2[cond3] = 6/15.

    return F2


def apparent_kurtosis_coef(dki_params, sphere, min_diffusivity=0,
                           min_kurtosis=-1):
    r""" Calculate the apparent kurtosis coefficient (AKC) in each direction
    of a sphere.

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
        cause quite a lot of noise in diffusion based metrics, diffusivity
        values smaller than `min_diffusivity` are replaced with
        `min_diffusivity`. defaut = 0
    min_kurtosis : float (optional)
        Because high amplitude negative values of kurtosis are not physicaly
        and biologicaly pluasible, and these causes huge artefacts in kurtosis
        based measures, directional kurtosis values than `min_kurtosis` are
        replaced with `min_kurtosis`. defaut = -1

    Returns
    --------
    AKC : ndarray (x, y, z, g) or (n, g)
        Apparent kurtosis coefficient (AKC) for all g directions of a sphere.

    Notes
    -----
    For each sphere direction with coordinates $(n_{1}, n_{2}, n_{3})$, the
    calculation of AKC is done using formula:

    .. math ::
        AKC(n)=\frac{MD^{2}}{ADC(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}
        \sum_{k=1}^{3}\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

    where $W_{ijkl}$ are the elements of the kurtosis tensor, MD the mean
    diffusivity and ADC the apparent diffusion coefficent computed as:

    .. math ::

        ADC(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

    where $D_{ij}$ are the elements of the diffusion tensor.
    """
    # Flat parameters
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split data
    evals, evecs, kt = split_dki_param(dki_params)

    # Initialize AKC matrix
    V = sphere.vertices
    AKC = np.zeros((len(kt), len(V)))

    # select relevant voxels to process
    rel_i = _positive_evals(evals[..., 0], evals[..., 1], evals[..., 2])
    kt = kt[rel_i]
    evecs = evecs[rel_i]
    evals = evals[rel_i]
    AKCi = AKC[rel_i]

    # Compute MD
    MD = mean_diffusivity(evals)

    # loop over all relevant voxels
    for vox in range(len(kt)):
        R = evecs[vox]
        dt = lower_triangular(np.dot(np.dot(R, np.diag(evals[vox])), R.T))
        AKCi[vox] = _directional_kurtosis(dt, MD[vox], kt[vox], V,
                                          min_diffusivity=min_diffusivity,
                                          min_kurtosis=min_kurtosis)

    # reshape data according to input data
    AKC[rel_i] = AKCi

    return AKC.reshape((outshape + (len(V),)))


def _directional_kurtosis(dt, MD, kt, V, min_diffusivity=0, min_kurtosis=-1):
    r""" Helper function that calculate the apparent kurtosis coefficient (AKC)
    in each direction of a sphere for a single voxel.

    Parameters
    ----------
    dt : array (6,)
        elements of the diffusion tensor of the voxel.
    MD : float
        mean diffusivity of the voxel
    kt : array (15,)
        elements of the kurtosis tensor of the voxel.
    V : array (g, 3)
        g directions of a Sphere in Cartesian coordinates
    min_diffusivity : float (optional)
        Because negative eigenvalues are not physical and small eigenvalues
        cause quite a lot of noise in diffusion based metrics, diffusivity
        values smaller than `min_diffusivity` are replaced with
        `min_diffusivity`. defaut = 0
    min_kurtosis : float (optional)
        Because high amplitude negative values of kurtosis are not physicaly
        and biologicaly pluasible, and these causes huge artefacts in kurtosis
        based measures, directional kurtosis values than `min_kurtosis` are
        replaced with `min_kurtosis`. defaut = -1

    Returns
    --------
    AKC : ndarray (g,)
        Apparent kurtosis coefficient (AKC) in all g directions of a sphere for
        a single voxel.

    See Also
    --------
    apparent_kurtosis_coef
    """
    ADC = \
        V[:, 0] * V[:, 0] * dt[0] + \
        2 * V[:, 0] * V[:, 1] * dt[1] + \
        V[:, 1] * V[:, 1] * dt[2] + \
        2 * V[:, 0] * V[:, 2] * dt[3] + \
        2 * V[:, 1] * V[:, 2] * dt[4] + \
        V[:, 2] * V[:, 2] * dt[5]

    if min_diffusivity is not None:
        ADC = ADC.clip(min=min_diffusivity)

    AKC = \
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

    if min_kurtosis is not None:
        AKC = AKC.clip(min=min_kurtosis)

    return (MD/ADC) ** 2 * AKC


def mean_kurtosis(dki_params, min_kurtosis=0, max_kurtosis=3):
    r""" Computes mean Kurtosis (MK) from the kurtosis tensor.

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
        `min_kurtosis`. defaut = 0
    max_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, mean
        kurtosis values that are larger than `max_kurtosis` are replaced with
        `max_kurtosis`. defaut = 3

    Returns
    -------
    mk : array
        Calculated MK.

    Notes
    --------
    The MK analytical solution is calculated using the following equation [1]_:
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
    .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """
    # Flat parameters. For numpy versions more recent than 1.6.0, this step
    # isn't required
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split the model parameters to three variable containing the evals, evecs,
    # and kurtosis elements
    evals, evecs, kt = split_dki_param(dki_params)

    # Rotate the kurtosis tensor from the standard Cartesian coordinate system
    # to another coordinate system in which the 3 orthonormal eigenvectors of
    # DT are the base coordinate
    Wxxxx = Wrotate_element(kt, 0, 0, 0, 0, evecs)
    Wyyyy = Wrotate_element(kt, 1, 1, 1, 1, evecs)
    Wzzzz = Wrotate_element(kt, 2, 2, 2, 2, evecs)
    Wxxyy = Wrotate_element(kt, 0, 0, 1, 1, evecs)
    Wxxzz = Wrotate_element(kt, 0, 0, 2, 2, evecs)
    Wyyzz = Wrotate_element(kt, 1, 1, 2, 2, evecs)

    # Compute MK
    MK = \
        _F1m(evals[..., 0], evals[..., 1], evals[..., 2])*Wxxxx + \
        _F1m(evals[..., 1], evals[..., 0], evals[..., 2])*Wyyyy + \
        _F1m(evals[..., 2], evals[..., 1], evals[..., 0])*Wzzzz + \
        _F2m(evals[..., 0], evals[..., 1], evals[..., 2])*Wyyzz + \
        _F2m(evals[..., 1], evals[..., 0], evals[..., 2])*Wxxzz + \
        _F2m(evals[..., 2], evals[..., 1], evals[..., 0])*Wxxyy

    if min_kurtosis is not None:
        MK = MK.clip(min=min_kurtosis)

    if max_kurtosis is not None:
        MK = MK.clip(max=max_kurtosis)

    return MK.reshape(outshape)


def _G1m(a, b, c):
    """ Helper function that computes function $G_1$ which is required to
    compute the analytical solution of the Radial kurtosis.

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
    --------
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
    # Error is defined as three order of magnitude larger than system's epslon
    er = np.finfo(a.ravel()[0]).eps * 1e3

    # Initialize G1
    G1 = np.zeros(a.shape)

    # Only computes G1 in voxels that have all eigenvalues larger than zero
    cond0 = _positive_evals(a, b, c)

    # Apply formula for non problematic plaussible cases, i.e. b!=c
    cond1 = np.logical_and(cond0, (abs(b - c) > er))
    if np.sum(cond1) != 0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        G1[cond1] = \
            (L1+L2+L3)**2 / (18 * L2 * (L2-L3)**2) * \
            (2.*L2 + (L3**2 - 3*L2*L3) / np.sqrt(L2*L3))

    # Resolve possible sigularity b==c
    cond2 = np.logical_and(cond0, abs(b - c) < er)
    if np.sum(cond2) != 0:
        L1 = a[cond2]
        L2 = b[cond2]
        G1[cond2] = (L1 + 2.*L2)**2 / (24.*L2**2)

    return G1


def _G2m(a, b, c):
    """ Helper function that computes function $G_2$ which is required to
    compute the analytical solution of the Radial kurtosis.

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
    --------
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
    # Error is defined as three order of magnitude larger than system's epslon
    er = np.finfo(a.ravel()[0]).eps * 1e3

    # Initialize G2
    G2 = np.zeros(a.shape)

    # Only computes G2 in voxels that have all eigenvalues larger than zero
    cond0 = _positive_evals(a, b, c)

    # Apply formula for non problematic plaussible cases, i.e. b!=c
    cond1 = np.logical_and(cond0, (abs(b - c) > er))
    if np.sum(cond1) != 0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        G2[cond1] = \
            (L1+L2+L3)**2 / (3 * (L2-L3)**2) * ((L2+L3) / np.sqrt(L2*L3) - 2)

    # Resolve possible sigularity b==c
    cond2 = np.logical_and(cond0, abs(b - c) < er)
    if np.sum(cond2) != 0:
        L1 = a[cond2]
        L2 = b[cond2]
        G2[cond2] = (L1 + 2.*L2)**2 / (12.*L2**2)

    return G2


def radial_kurtosis(dki_params, min_kurtosis=0, max_kurtosis=3):
    r""" Radial Kurtosis (RK) of a diffusion kurtosis tensor.

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
        `min_kurtosis`. defaut = 0
    max_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, radial
        kurtosis values that are larger than `max_kurtosis` are replaced with
        `max_kurtosis`. defaut = 3

    Returns
    -------
    rk : array
        Calculated RK.

    Notes
    --------
    RK is calculated with the following equation:
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
    """
    # Flat parameters. For numpy versions more recent than 1.6.0, this step
    # isn't required
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split the model parameters to three variable containing the evals, evecs,
    # and kurtosis elements
    evals, evecs, kt = split_dki_param(dki_params)

    # Rotate the kurtosis tensor from the standard Cartesian coordinate system
    # to another coordinate system in which the 3 orthonormal eigenvectors of
    # DT are the base coordinate
    Wyyyy = Wrotate_element(kt, 1, 1, 1, 1, evecs)
    Wzzzz = Wrotate_element(kt, 2, 2, 2, 2, evecs)
    Wyyzz = Wrotate_element(kt, 1, 1, 2, 2, evecs)

    # Compute RK
    RK = \
        _G1m(evals[..., 0], evals[..., 1], evals[..., 2]) * Wyyyy + \
        _G1m(evals[..., 0], evals[..., 2], evals[..., 1]) * Wzzzz + \
        _G2m(evals[..., 0], evals[..., 1], evals[..., 2]) * Wyyzz

    if min_kurtosis is not None:
        RK = RK.clip(min=min_kurtosis)

    if max_kurtosis is not None:
        RK = RK.clip(max=max_kurtosis)

    return RK.reshape(outshape)


def axial_kurtosis(dki_params, min_kurtosis=0, max_kurtosis=3):
    r"""  Computes axial Kurtosis (AK) from the kurtosis tensor.

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
        `min_kurtosis`. defaut = 0
    max_kurtosis : float (optional)
        To keep kurtosis values within a plausible biophysical range, axial
        kurtosis values that are larger than `max_kurtosis` are replaced with
        `max_kurtosis`. defaut = 3

    Returns
    -------
    ak : array
        Calculated AK.
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

    # Compute MD
    MD = mean_diffusivity(evals)

    # loop over all voxels
    for vox in range(len(kt)):
        R = evecs[vox]
        dt = lower_triangular(np.dot(np.dot(R, np.diag(evals[vox])), R.T))
        AKi[vox] = _directional_kurtosis(dt, MD[vox], kt[vox],
                                         np.array([R[:, 0]]))

    # reshape data according to input data
    AK[rel_i] = AKi

    if min_kurtosis is not None:
        AK = AK.clip(min=min_kurtosis)

    if max_kurtosis is not None:
        AK = AK.clip(max=max_kurtosis)

    return AK.reshape(outshape)


def dki_prediction(dki_params, gtab, S0=150):
    """ Predict a signal given diffusion kurtosis imaging parameters.

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
        voxels. Default: 150

    Returns
    --------
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

    # lopping for all voxels
    for v in range(len(pred_sig)):
        DT = np.dot(np.dot(fevecs[v], np.diag(fevals[v])), fevecs[v].T)
        dt = lower_triangular(DT)
        MD = (dt[0] + dt[2] + dt[5]) / 3
        X = np.concatenate((dt, fkt[v]*MD*MD, np.array([np.log(S0)])), axis=0)
        pred_sig[v] = np.exp(np.dot(A, X))

    # Reshape data according to the shape of dki_params
    pred_sig = pred_sig.reshape(dki_params.shape[:-1] + (pred_sig.shape[-1],))

    return pred_sig


class DiffusionKurtosisModel(ReconstModel):
    """ Class for the Diffusion Kurtosis Model
    """
    def __init__(self, gtab, fit_method="OLS", *args, **kwargs):
        """ Diffusion Kurtosis Tensor Model [1]

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable
            str can be one of the following:
            'OLS' or 'ULLS' for ordinary least squares
                dki.ols_fit_dki
            'WLS' or 'UWLLS' for weighted ordinary least squares
                dki.wls_fit_dki

            callable has to have the signature:
                fit_method(design_matrix, data, *args, **kwargs)

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dki.ols_fit_dki, dki.wls_fit_dki for details

        References
        ----------
           [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
        """
        ReconstModel.__init__(self, gtab)

        if not callable(fit_method):
            try:
                self.fit_method = common_fit_methods[fit_method]
            except KeyError:
                raise ValueError('"' + str(fit_method) + '" is not a known '
                                 'fit method, the fit method should either be '
                                 'a function or one of the common fit methods')

        self.design_matrix = design_matrix(self.gtab)
        self.args = args
        self.kwargs = kwargs
        self.min_signal = self.kwargs.pop('min_signal', None)
        if self.min_signal is not None and self.min_signal <= 0:
            e_s = "The `min_signal` key-word argument needs to be strictly"
            e_s += " positive."
            raise ValueError(e_s)

    def fit(self, data, mask=None):
        """ Fit method of the DKI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]
        """
        if mask is None:
            # Flatten it to 2D either way:
            data_in_mask = np.reshape(data, (-1, data.shape[-1]))
        else:
            # Check for valid shape of the mask
            if mask.shape != data.shape[:-1]:
                raise ValueError("Mask is not the same shape as data.")
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = np.reshape(data[mask], (-1, data.shape[-1]))

        if self.min_signal is None:
            min_signal = _min_positive_signal(data)
        else:
            min_signal = self.min_signal

        data_in_mask = np.maximum(data_in_mask, min_signal)
        params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
                                         *self.args, **self.kwargs)

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            dki_params = params_in_mask.reshape(out_shape)
        else:
            dki_params = np.zeros(data.shape[:-1] + (27,))
            dki_params[mask, :] = params_in_mask

        return DiffusionKurtosisFit(self, dki_params)

    def predict(self, dki_params, S0=1):
        """ Predict a signal for this DKI model class instance given
        parameters.

        Parameters
        ----------
        dki_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor
        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return dki_prediction(dki_params, self.gtab, S0)


class DiffusionKurtosisFit(TensorFit):
    """ Class for fitting the Diffusion Kurtosis Model"""
    def __init__(self, model, model_params):
        """ Initialize a DiffusionKurtosisFit class instance.

        Since DKI is an extension of DTI, class instance is defined as subclass
        of the TensorFit from dti.py

        Parameters
        ----------
        model : DiffusionKurtosisModel Class instance
            Class instance containing the Diffusion Kurtosis Model for the fit
        model_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor
        """
        TensorFit.__init__(self, model, model_params)

    @property
    def kt(self):
        """
        Returns the 15 independent elements of the kurtosis tensor as an array
        """
        return self.model_params[..., 12:]

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
        diffusivity and ADC the apparent diffusion coefficent computed as:

        .. math ::
            ADC(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

        where $D_{ij}$ are the elements of the diffusion tensor.
        """
        return apparent_kurtosis_coef(self.model_params, sphere)

    def mk(self, min_kurtosis=0, max_kurtosis=3):
        r""" Computes mean Kurtosis (MK) from the kurtosis tensor.

        Parameters
        ----------
        min_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, mean
            kurtosis values that are smaller than `min_kurtosis` are replaced
            with `min_kurtosis`. defaut = 0
        max_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, mean
            kurtosis values that are larger than `max_kurtosis` are replaced
            with `max_kurtosis`. defaut = 3

        Returns
        -------
        mk : array
            Calculated MK.

        Notes
        --------
        The MK analytical solution is calculated using the following equation
        [1]_:

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

        and

        .. math::

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
        .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
               Estimation of tensors and tensor-derived measures in diffusional
               kurtosis imaging. Magn Reson Med. 65(3), 823-836
        """
        return mean_kurtosis(self.model_params, min_kurtosis, max_kurtosis)

    def ak(self, min_kurtosis=0, max_kurtosis=3):
        r"""
        Axial Kurtosis (AK) of a diffusion kurtosis tensor.

        Parameters
        ----------
        min_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, axial
            kurtosis values that are smaller than `min_kurtosis` are replaced
            with `min_kurtosis`. defaut = 0
        max_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, axial
            kurtosis values that are larger than `max_kurtosis` are replaced
            with `max_kurtosis`. defaut = 3

        Returns
        -------
        ak : array
            Calculated AK.
        """
        return axial_kurtosis(self.model_params, min_kurtosis, max_kurtosis)

    def rk(self, min_kurtosis=0, max_kurtosis=3):
        r""" Radial Kurtosis (RK) of a diffusion kurtosis tensor.

        Parameters
        ----------
        min_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, axial
            kurtosis values that are smaller than `min_kurtosis` are replaced
            with `min_kurtosis`. defaut = 0
        max_kurtosis : float (optional)
            To keep kurtosis values within a plausible biophysical range, axial
            kurtosis values that are larger than `max_kurtosis` are replaced
            with `max_kurtosis`. defaut = 3

        Returns
        -------
        rk : array
            Calculated RK.

        Notes
        ------
        RK is calculated with the following equation:

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
            \left ( \frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}-2
            \right )
        """
        return radial_kurtosis(self.model_params, min_kurtosis, max_kurtosis)

    def predict(self, gtab, S0=1):
        r""" Given a DKI model fit, predict the signal on the vertices of a
        gradient table

        Parameters
        ----------
        dki_params : ndarray (x, y, z, 27) or (n, 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follows:
                1) Three diffusion tensor's eigenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor

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


def ols_fit_dki(design_matrix, data):
    r""" Computes ordinary least squares (OLS) fit to calculate the diffusion
    tensor and kurtosis tensor using a linear regression diffusion kurtosis
    model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array (N, g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.

    Returns
    -------
    dki_params : array (N, 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    See Also
    --------
    wls_fit_dki

    References
    ----------
       [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """
    tol = 1e-6

    # preparing data and initializing parameters
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dki_params = np.empty((len(data_flat), 27))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    # lopping OLS solution on all data voxels
    for vox in range(len(data_flat)):
        dki_params[vox] = _ols_iter(inv_design, data_flat[vox],
                                    min_diffusivity)

    # Reshape data according to the input data shape
    dki_params = dki_params.reshape((data.shape[:-1]) + (27,))

    return dki_params


def _ols_iter(inv_design, sig, min_diffusivity):
    """ Helper function used by ols_fit_dki - Applies OLS fit of the diffusion
    kurtosis model to single voxel signals.

    Parameters
    ----------
    inv_design : array (g, 22)
        Inverse of the design matrix holding the covariants used to solve for
        the regression coefficients.
    sig : array (g,)
        Diffusion-weighted signal for a single voxel data.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    dki_params : array (27,)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    """
    # DKI ordinary linear least square solution
    log_s = np.log(sig)
    result = np.dot(inv_design, log_s)

    # Extracting the diffusion tensor parameters from solution
    DT_elements = result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = (evals.mean(0))**2
    KT_elements = result[6:21] / MD_square

    # Write output
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 KT_elements), axis=0)

    return dki_params


def wls_fit_dki(design_matrix, data):
    r""" Computes weighted linear least squares (WLS) fit to calculate
    the diffusion tensor and kurtosis tensor using a weighted linear
    regression diffusion kurtosis model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array (N, g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avoid taking log(0) durring the tensor fitting.

    Returns
    -------
    dki_params : array (N, 27)
        All parameters estimated from the diffusion kurtosis model for all N
        voxels.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the first
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    References
    ----------
       [1] Veraart, J., Sijbers, J., Sunaert, S., Leemans, A., Jeurissen, B.,
           2013. Weighted linear least squares estimation of diffusion MRI
           parameters: Strengths, limitations, and pitfalls. Magn Reson Med 81,
           335-346.
    """

    tol = 1e-6

    # preparing data and initializing parametres
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dki_params = np.empty((len(data_flat), 27))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    # lopping WLS solution on all data voxels
    for vox in range(len(data_flat)):
        dki_params[vox] = _wls_iter(design_matrix, inv_design, data_flat[vox],
                                    min_diffusivity)

    # Reshape data according to the input data shape
    dki_params = dki_params.reshape((data.shape[:-1]) + (27,))

    return dki_params


def _wls_iter(design_matrix, inv_design, sig, min_diffusivity):
    """ Helper function used by wls_fit_dki - Applies WLS fit of the diffusion
    kurtosis model to single voxel signals.

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients
    inv_design : array (g, 22)
        Inverse of the design matrix.
    sig : array (g, )
        Diffusion-weighted signal for a single voxel data.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    dki_params : array (27, )
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follows:
            1) Three diffusion tensor's eigenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    """
    A = design_matrix

    # DKI ordinary linear least square solution
    log_s = np.log(sig)
    ols_result = np.dot(inv_design, log_s)

    # Define weights as diag(yn**2)
    W = np.diag(np.exp(2 * np.dot(A, ols_result)))

    # DKI weighted linear least square solution
    inv_AT_W_A = np.linalg.pinv(np.dot(np.dot(A.T, W), A))
    AT_W_LS = np.dot(np.dot(A.T, W), log_s)
    wls_result = np.dot(inv_AT_W_A, AT_W_LS)

    # Extracting the diffusion tensor parameters from solution
    DT_elements = wls_result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = (evals.mean(0))**2
    KT_elements = wls_result[6:21] / MD_square

    # Write output
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2],
                                 KT_elements), axis=0)

    return dki_params


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
    --------
    Wrot : array (m,) or (15,)
        Vector with the m independent elements of the rotated kurtosis tensor.
        If 'indices' is not specified all 15 elements of the rotated kurtosis
        tensor are computed.
    Note
    ------
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
    r""" Computes the the specified index element of a kurtosis tensor rotated
    to the coordinate system basis B.

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

    Note
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
                    key = (il+1) * (jl+1) * (kl+1) * (ll+1)
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
                    key = (ind_i+1) * (ind_j+1) * (ind_k+1) * (ind_l+1)
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
    --------
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
    kt = dki_params[..., 12:]

    return evals, evecs, kt

common_fit_methods = {'WLS': wls_fit_dki,
                      'OLS': ols_fit_dki,
                      'UWLLS': wls_fit_dki,
                      'ULLS': ols_fit_dki,
                      }
