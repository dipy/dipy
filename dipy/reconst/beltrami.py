""" Functions that implement a beltrami framework algorithm to
fit the Free Water elimination model for single shell datasets """
from __future__ import division
import numpy as np


def x_manifold(D, mask):
    """
    Converts the diffusion tensor components
    into Iwasawa coordinates, according to [1], [2]

    Parameters
    ----------
    D : (..., 6) ndarray
        The lower triangular components of the diffusion tensor,
        in the following order: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz
    mask : boolean array
        boolean mask that marks indices of the data that
        should be converted, with shape X.shape[:-1]

    Returns
    -------
    X : (..., 6) ndarray
        The six independent Iwasawa coordinates
    
    References
    ----------
    .. [1] Pasternak, O., Sochen, N., Gur, Y., Intrator, N., & Assaf, Y. (2009).
        Free water elimination and mapping from diffusion MRI.
        Magnetic Resonance in Medicine: An Official Journal of 
        the International Society for Magnetic Resonance in Medicine, 62(3), 717-730.

    .. [2] Gur, Y., & Sochen, N. (2007, October).
        Fast invariant Riemannian DT-MRI regularization.
        In 2007 IEEE 11th International Conference on Computer Vision (pp. 1-7). IEEE.

    """

    Dxx = D[mask, 0]
    Dxy = D[mask, 1]
    Dyy = D[mask, 2]
    Dxz = D[mask, 3]
    Dyz = D[mask, 4]
    Dzz = D[mask, 5]

    X = np.zeros(D.shape)
    X[mask, 0] = Dxx  # X1
    X[mask, 3] = Dxy / Dxx  # X4
    X[mask, 4] = Dxz / Dxx  # X5
    X[mask, 1] = Dyy - X[mask, 0] * X[mask, 3]**2  # X2
    X[mask, 5] = (Dyz - X[mask, 0] * X[mask, 3] * X[mask, 4]) / X[mask, 1]  # X6
    X[mask, 2] = Dzz - X[mask, 0] * X[mask, 4]**2 - X[mask, 1] * X[mask, 5]**2  # X3

    return X


def d_manifold(X, mask):
    """
    Converts Iwasawa coordinates back to
    tensor components
    Parameters
    ----------
    X : (..., 6) ndarray
        The Iwasawa coordinetes, in the following order:
        X1, X2, X3, X4, X5, X6
    mask : boolean array
        boolean mask that marks indices of the data that
        should be converted, with shape X.shape[:-1]

    Returns
    -------
    D : (..., 6) ndarray
        The lower triangular components of the diffusion tensor,
        in the following order: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz

    References
    ----------
    See referneces section of the fucntion "x_manifold"

    """

    X1 = X[mask, 0]
    X2 = X[mask, 1]
    X3 = X[mask, 2]
    X4 = X[mask, 3]
    X5 = X[mask, 4]
    X6 = X[mask, 5]

    D = np.zeros(X.shape)
    D[mask, 0] = X1  # Dxx
    D[mask, 1] = X1 * X4  # Dxy
    D[mask, 2] = X2 + X1 * X4**2  # Dyy
    D[mask, 3] = X1 * X5  # Dxz
    D[mask, 4] = X1 * X4 * X5 + X2 * X6  # Dyz
    D[mask, 5] = X3 + X1 * X5**2 + X2 * X6**2  # Dzz

    return D


def forward_diff(Xi, dim, vox_size):
    """
    Forward finite differences

    Parameters
    ----------
    Xi : (X, Y, Z) ndarray
        A single Iwasawa coordinate (i.e. X1) for all voxels.
    dim : int
        The dimension along which to perform the finite difference:
        0 - derivative along x axis,
        1 - derivative along y axis,
        2 - derivative along z axis.
    vox_size : float
        The normalized voxel size along the chosen dimension.

    Returns
    -------
    dfX : (X, Y, Z) ndarray
        The forward difference of Xi along the chosen axis,
        normalized by voxel size.

    Notes
    -----
    The forward difference of a sequence $a$ at postition $n$ is defined as:

        $\delta a_{n} = a_{n+1} - a_{n}$

    """

    n = Xi.shape[dim]
    shift = np.append(np.arange(1, n), n-1)

    if dim == 0:
        dfX = (Xi[shift, ...] - Xi) / vox_size
    elif: dim == 1:
        dfX = (Xi[:, shift, :] - Xi) / vox_size
    elif: dim == 2:
        dfX = (Xi[..., shift] - Xi) / vox_size

    return dfX


def backward_diff(Xi, dim, vox_size):
    """
    Backward finite differences

    Parameters
    ----------
    Xi : (X, Y, Z) ndarray
        A single Iwasawa coordinate (i.e. X1) for all voxels.
    dim : int
        The dimension along which to perform the finite difference:
        0 - derivative along x axis,
        1 - derivative along y axis,
        2 - derivative along z axis.
    vox_size : float
        The normalized voxel size along the chosen dimension.

    Returns
    -------
    dbX : (X, Y, Z) ndarray
        The backward difference of Xi along the chosen axis,
        normalized by voxel size.

    Notes
    -----
    The backward difference of a sequence $a$ at postition $n$ is defined as:

        $\delta a_{n} = a_{n} - a_{n-1}$

    """

    n = Xi.shape[dim]
    shift = np.append(0, np.arange(0, n-1))

    if dim == 0:
        dbX = (Xi - Xi[shift, ...]) / vox_size

    elif: dim == 1:
        dbX = (Xi - Xi[:, shift, :]) / vox_size

    elif: dim == 2:
        dbX = Xi - (Xi[..., shift]) / vox_size

    return dbX


def beltrami_euclidean(D, mask, zooms, beta):
    """
    Computes the Beltrami increments to update the diffusion tensor
    components, using the euclidean metric [1].

    Parameters
    ----------
    D : (..., 6) ndarray
        The lower triangular components of the diffusion tensor,
        in the following order: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz.
    mask : boolean array
        Boolean mask that marks indices of the data that should be updated.
    zooms : (3,) tuple of floats
        Tuple containing the normalized voxel sizes (dx, dy, dz)-
    beta : float
        The ratio between the the spatial feature manifold and
        the image domain metrics, controls how isottropic is the smoothing of the
        D manifold, see [2] for more details.

    Returns
    -------
    bad_g : boolean array
        Boolean mask that marks the indices of the data where the induced metric
        tensor "g" is unstable (i.e. the determinant of g is negative or increases to
        extreme values), and where the update should be null.
    gsqr : ndarray
        The square root of the determinant of the metric tensor g.
    dB : (..., 6) ndarray
        The increments to be applied to each component of D, computed by applying
        the Beltrami operator.

    Notes
    -----
    In the Beltrami framework for DTI, a 3 by 3 symmetric tensor $g$ is used
    to measure distances on the 3D image domain, defined (in Einstein summation) by:

        $g_{ij} = \partial_{i} x^{m} \partial_{j} x^{n} h_{mn}$,

    where the $$x^{m}$ denotes the "mth" component of a given manifold
    (e.g. Dxx of D), the $\partial_{i}$ denotes the derivative along the
    "ith" axis and $h_{mn}$ is the spatial feature metric tensor.
    However, if this metric is chosen to be euclidean, then there is no need
    for the Iwasawa conversion and $h$ becomes a 9 by 9 diagonal matrix, with
    the first 3 diagonal elements equal to 1 and the rest equal to the constant
    beta [1].

    The Beltrami operator for the "nth" component of a given manifold $x^{n}$
    is defined in terms of the inverse of $g$:

        $\Delta B^{n} = 
        /frac{1}{\sqrt{|g|}} \partial_{i} \sqrt{|g|} g^{ij} \partial_{j} x^{n}$,
    
    where $g|$ denotes the determinant of $g$ and  $g^{ij}$  its inverse.
    
    Applying this operator refularizes (smooths) the manifold D, small "beta"
    ratios result in more isotropic smoothing. See references for more details.

    References
    ----------
    .. [1] Pasternak, O., Maier-Hein, K., Baumgartner,
        C., Shenton, M. E., Rathi, Y., & Westin, C. F. (2014).
        The estimation of free-water corrected diffusion tensors.
        In Visualization and Processing of Tensors and Higher Order
        Descriptors for Multi-Valued Data (pp. 249-270). Springer,
        Berlin, Heidelberg.

    .. [2] Gur, Y., & Sochen, N. (2007, October).
        Fast invariant Riemannian DT-MRI regularization.
        In 2007 IEEE 11th International Conference on Computer Vision (pp. 1-7). IEEE.

    """

    nx, ny, nz = D.shape[:-1]
    dx, dy, dz = zooms

    Dxx = D[..., 0]
    Dxy = D[..., 1]
    Dyy = D[..., 2]
    Dxz = D[..., 3]
    Dyz = D[..., 4]
    Dzz = D[..., 5]

    # computing all spatial derivatives of D,
    # Xi denotes the "ith" components of D (e.g. X1x = d(Dxx) / dx)
    X1x = forward_diff(Dxx, 0, dx) * mask  # not sure if it is efficient to constantly multiply by this mask,
    X1y = forward_diff(Dxx, 1, dy) * mask  # this is done to avoid unstable derivatives at the the brain border,
    X1z = forward_diff(Dxx, 2, dz) * mask  # maybe theres a better way to do this

    X2x = forward_diff(Dxy, 0, dx) * mask
    X2y = forward_diff(Dxy, 1, dy) * mask
    X2z = forward_diff(Dxy, 2, dz) * mask

    X3x = forward_diff(Dyy, 0, dx) * mask
    X3y = forward_diff(Dyy, 1, dy) * mask
    X3z = forward_diff(Dyy, 2, dz) * mask

    X4x = forward_diff(Dxz, 0, dx) * mask
    X4y = forward_diff(Dxz, 1, dy) * mask
    X4z = forward_diff(Dxz, 2, dz) * mask

    X5x = forward_diff(Dyz, 0, dx) * mask
    X5y = forward_diff(Dyz, 1, dy) * mask
    X5z = forward_diff(Dyz, 2, dz) * mask

    X6x = forward_diff(Dzz, 0, dx) * mask
    X6y = forward_diff(Dzz, 1, dy) * mask
    X6z = forward_diff(Dzz, 2, dz) * mask

    # computing the components of the g metric
    g_11 = (X1x**2 + X2x**2 + X3x**2 +
            X4x**2 + X5x**2 + X6x**2) * beta + 1

    g_22 = (X1y**2 + X2y**2 + X3y**2 +
            X4y**2 + X5y**2 + X6y**2) * beta + 1

    g_33 = (X1z**2 + X2z**2 + X3z**2 +
            X4z**2 + X5z**2 + X6z**2) * beta + 1

    g_12 = (X1x * X1y + X2x * X2y + X3x * X3y +
            X4x * X4y + X5x * X5y + X6x * X6y) * beta

    g_13 = (X1x * X1z + X2x * X2z + X3x * X3z +
            X4x * X4z + X5x * X5z + X6x * X6z) * beta

    g_23 = (X1y * X1z + X2y * X2z + X3y * X3z +
            X4y * X4z + X5y * X5z + X6y * X6z) * beta

    # computing g inverse
    gdet = (g_11 * g_22 * g_33 +
            g_12 * g_13 * g_23 * 2 -
            g_22 * g_13**2 -
            g_33 * g_12**2 -
            g_11 * g_23**2)

    # where g is unstable, replace by thed identity matrix
    bad_g = np.logical_or(gdet <= 0, gdet > 1000)
    bad_g = np.logical_and(bad_g, mask)
    gdet[bad_g] = 1
    g_11[bad_g] = 1
    g_22[bad_g] = 1
    g_33[bad_g] = 1
    g_12[bad_g] = 0
    g_13[bad_g] = 0
    g_23[bad_g] = 0

    g11 = (g_22 * g_33 - g_23**2) / gdet
    g22 = (g_11 * g_33 - g_13**2) / gdet
    g33 = (g_11 * g_22 - g_12**2) / gdet
    g12 = (g_13 * g_23 - g_12 * g_33) / gdet
    g13 = (g_12 * g_23 - g_13 * g_22) / gdet
    g23 = (g_12 * g_13 - g_11 * g_23) / gdet

    # computing an auxuliary matrix matrix A,
    # that stores the product g^{ij} \partial_{j} x^{n}
    A1x = g11 * X1x + g12 * X1y + g13 * X1z
    A1y = g12 * X1x + g22 * X1y + g23 * X1z
    A1z = g13 * X1x + g23 * X1y + g33 * X1z

    A2x = g11 * X2x + g12 * X2y + g13 * X2z
    A2y = g12 * X2x + g22 * X2y + g23 * X2z
    A2z = g13 * X2x + g23 * X2y + g33 * X2z

    A3x = g11 * X3x + g12 * X3y + g13 * X3z
    A3y = g12 * X3x + g22 * X3y + g23 * X3z
    A3z = g13 * X3x + g23 * X3y + g33 * X3z

    A4x = g11 * X4x + g12 * X4y + g13 * X4z
    A4y = g12 * X4x + g22 * X4y + g23 * X4z
    A4z = g13 * X4x + g23 * X4y + g33 * X4z

    A5x = g11 * X5x + g12 * X5y + g13 * X5z
    A5y = g12 * X5x + g22 * X5y + g23 * X5z
    A5z = g13 * X5x + g23 * X5y + g33 * X5z

    A6x = g11 * X6x + g12 * X6y + g13 * X6z
    A6y = g12 * X6x + g22 * X6y + g23 * X6z
    A6z = g13 * X6x + g23 * X6y + g33 * X6z

    # computing beltrami increment
    g = np.sqrt(gdet)
    dB = np.zeros(X.shape)
    dB[..., 0] = 1 / g * (backward_diff(g * A1x, 0, dx) * mask +
                          backward_diff(g * A1y, 1, dy) * mask +
                          backward_diff(g * A1z, 2, dz) * mask)

    dB[..., 1] = 1 / g * (backward_diff(g * A2x, 0, dx) * mask +
                          backward_diff(g * A2y, 1, dy) * mask +
                          backward_diff(g * A2z, 2, dz) * mask)

    dB[..., 2] = 1 / g * (backward_diff(g * A3x, 0, dx) * mask +
                          backward_diff(g * A3y, 1, dy) * mask +
                          backward_diff(g * A3z, 2, dz) * mask)

    dB[..., 3] = 1 / g * (backward_diff(g * A4x, 0, dx) * mask +
                          backward_diff(g * A4y, 1, dy) * mask +
                          backward_diff(g * A4z, 2, dz) * mask)

    dB[..., 4] = 1 / g * (backward_diff(g * A5x, 0, dx) * mask +
                          backward_diff(g * A5y, 1, dy) * mask +
                          backward_diff(g * A5z, 2, dz) * mask)

    dB[..., 5] = 1 / g * (backward_diff(g * A6x, 0, dx) * mask +
                          backward_diff(g * A6y, 1, dy) * mask +
                          backward_diff(g * A6z, 2, dz) * mask)

    return (bad_g, g, dB)
