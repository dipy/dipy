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


def civ1(X1, X2, X3, X6, X1x, X1y, X1z,
         X4x, X4y, X4z, X5x, X5y, X5z,
         A1x, A1y, A1z, A4x, A4y, A4z, A5x, A5y, A5z, mask):
    """
    Computes the Levi-Civita term to update the Iwasawa coodinate X1,
    when the affine metric is used instead of the euclidean.
    """

    x, y, z = mask.shape

    # computing Christoffel symbols
    G11, G44, G45, G55 = np.zeros((4, x, y, z)) 
    G11[mask] = -1 / X1[mask]
    G44[mask] = - X1[mask]**2 * (X3[mask] + X2[mask] * X6[mask]**2) / (X2[mask] * X3[mask])
    G45[mask] = X1[mask]**2 * X6[mask] / X3[mask]
    G55[mask] = -X1[mask]**2 / X3[mask]
    # computing auxiliary B matrix
    B1x = G11 * X1x
    B1y = G11 * X1y
    B1z = G11 * X1z

    B4x = G44 * X4x + G45 * X5x
    B4y = G44 * X4y + G45 * X5y
    B4z = G44 * X4z + G45 * X5z

    B5x = G45 * X4x + G55 * X5x
    B5y = G45 * X4y + G55 * X5y
    B5z = G45 * X4z + G55 * X5z

    # computing the civita term
    C1 = (A1x * B1x + A1y * B1y + A1z * B1z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z)

    return C1


def civ2(X1, X2, X3, X2x, X2y, X2z,
         X4x, X4y, X4z, X6x, X6y, X6z,
         A2x, A2y, A2z, A4x, A4y, A4z, A6x, A6y, A6z, mask):
    """
    Computes the Levi-Civita term to update the Iwasawa coodinate X2,
    when the affine metric is used instead of the euclidean.
    """

    x, y, z = mask.shape
    # computing Christoffel symbols
    G22, G44, G66 = np.zeros((3, x, y, z))
    G22[mask] = -1 / X2[mask]
    G44[mask] = X1[mask]
    G66[mask] = -X2[mask]**2 / X3[mask]
    # computing auxiliary B matrix
    B2x = G22 * X2x
    B2y = G22 * X2y
    B2z = G22 * X2z

    B4x = G44 * X4x
    B4y = G44 * X4y
    B4z = G44 * X4z

    B6x = G66 * X6x
    B6y = G66 * X6y
    B6z = G66 * X6z

    # computing the civita term
    C2 = (A2x * B2x + A2y * B2y + A2z * B2z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C2


def civ3(X1, X2, X3, X6, X3x, X3y, X3z, X4x, X4y, X4z,
         X5x, X5y, X5z, X6x, X6y, X6z, A3x, A3y, A3z,
         A4x, A4y, A4z, A5x, A5y, A5z, A6x, A6y, A6z, mask):
    """
    Computes the Levi-Civita term to update the Iwasawa coodinate X3,
    when the affine metric is used instead of the euclidean.
    """

    x, y, z = mask.shape

    # computing Christoffel symbols
    G33, G44, G45, G55, G66 = np.zeros((5, x, y, z)) 
    G33[mask] = -1 / X3[mask]
    G44[mask] = X1[mask] * X6[mask]**2
    G45[mask] = -X1[mask] * X6[mask]
    G55[mask] = X1[mask]
    G66[mask] = X2[mask]

    # computing auxiliary B matrix
    B3x = G33 * X3x
    B3y = G33 * X3y
    B3z = G33 * X3z

    B4x = G44 * X4x + G45 * X5x
    B4y = G44 * X4y + G45 * X5y
    B4z = G44 * X4z + G45 * X5z

    B5x = G45 * X4x + G55 * X5x
    B5y = G45 * X4y + G55 * X5y
    B5z = G45 * X4z + G55 * X5z

    B6x = G66 * X6x
    B6y = G66 * X6y
    B6z = G66 * X6z

    # computing the civita term
    C3 = (A3x * B3x + A3y * B3y + A3z * B3z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C3


def civ4(X1, X2, X3, X6, X1x, X1y, X1z, X2x, X2y, X2z, X4x, X4y, X4z,
         X5x, X5y, X5z, X6x, X6y, X6z, A1x, A1y, A1z, A2x, A2y, A2z,
         A4x, A4y, A4z, A5x, A5y, A5z, A6x, A6y, A6z, mask):
    """
    Computes the Levi-Civita term to update the Iwasawa coodinate X4,
    when the affine metric is used instead of the euclidean.
    """

    x, y, z = mask.shape

    # computing Christoffel symbols
    G14, G24, G46, G56 = np.zeros((4, x, y, z))
    G14[mask] = 1 / (2 * X1[mask])
    G24[mask] = -1 / (2 * X2[mask])
    G46[mask] = X2[mask] * X6[mask] / (2 * X3[mask])
    G56[mask] = -X2[mask] / (2 * X3[mask])
    # computing auxiliary B matrix
    B1x = G14 * X4x
    B1y = G14 * X4y
    B1z = G14 * X4z

    B2x = G24 * X4x
    B2y = G24 * X4y
    B2z = G24 * X4z

    B4x = G14 * X1x + G24 * X2x + G46 * X6x
    B4y = G14 * X1y + G24 * X2y + G46 * X6y
    B4z = G14 * X1z + G24 * X2z + G46 * X6z

    B5x = G56 * X6x
    B5y = G56 * X6y
    B5z = G56 * X6z

    B6x = G46 * X4x + G56 * X5x
    B6y = G46 * X4y + G56 * X5y
    B6z = G46 * X4z + G56 * X5z

    # computing the civita term
    C4 = (A1x * B1x + A1y * B1y + A1z * B1z +
          A2x * B2x + A2y * B2y + A2z * B2z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C4


def civ5(X1, X2, X3, X6, X1x, X1y, X1z, X2x, X2y, X2z, X3x, X3y, X3z,
         X4x, X4y, X4z, X5x, X5y, X5z, X6x, X6y, X6z, A1x, A1y, A1z,
         A2x, A2y, A2z, A3x, A3y, A3z, A4x, A4y, A4z, A5x, A5y, A5z,
         A6x, A6y, A6z, mask):
    """
    Computes the Levi-Civita term to update the Iwasawa coodinate X5,
    when the affine metric is used instead of the euclidean.
    """

    x, y, z = mask.shape

    # computing Christoffel symbols
    G24, G34, G15, G35, G46, G56 = np.zeros((6, x, y, z))
    G24[mask] = -X6[mask] / (2 * X2[mask])
    G34[mask] = X6[mask] / (2 * X3[mask])
    G15[mask] = 1 / (2 * X1[mask])
    G35[mask] = -1 / (2 * X3[mask])
    G46[mask] = (-1 + X2[mask] * X6[mask]**2 / X3[mask]) / 2
    G56[mask] = -X2[mask] * X6[mask] / (2 * X3[mask])

    # computing auxiliary B matrix
    B1x = G15 * X5x
    B1y = G15 * X5y
    B1z = G15 * X5z

    B2x = G24 * X4x
    B2y = G24 * X4y
    B2z = G24 * X4z

    B3x = G34 * X4x + G35 * X5x
    B3y = G34 * X4y + G35 * X5y
    B3z = G34 * X4z + G35 * X5z

    B4x = G24 * X2x + G34 * X3x + G46 * X6x
    B4y = G24 * X2y + G34 * X3y + G46 * X6y
    B4z = G24 * X2z + G34 * X3z + G46 * X6z

    B5x = G15 * X1x + G35 * X3x + G56 * X6x
    B5y = G15 * X1y + G35 * X3y + G56 * X6y
    B5z = G15 * X1z + G35 * X3z + G56 * X6z

    B6x = G46 * X4x + G56 * X5x
    B6y = G46 * X4y + G56 * X5y
    B6z = G46 * X4z + G56 * X5z

    # computing the civita term
    C5 = (A1x * B1x + A1y * B1y + A1z * B1z +
          A2x * B2x + A2y * B2y + A2z * B2z +
          A3x * B3x + A3y * B3y + A3z * B3z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C5


def civ6(X1, X2, X3, X6, X2x, X2y, X2z, X3x, X3y, X3z,
         X4x, X4y, X4z, X5x, X5y, X5z, X6x, X6y, X6z,
         A2x, A2y, A2z, A3x, A3y, A3z, A4x, A4y, A4z,
         A5x, A5y, A5z, A6x, A6y, A6z, mask):
    """
    Computes the Levi-Civita term to update the Iwasawa coodinate X6,
    when the affine metric is used instead of the euclidean.
    """

    x, y, z = mask.shape

    # computing Christoffel symbols
    G44, G45, G26, G36 = np.zeros((4, x, y, z))
    G44[mask] = -X1[mask] * X6[mask] / X2[mask]
    G45[mask] = X1[mask] / (2 * X2[mask])
    G26[mask] = 1 / (2 * X2[mask])
    G36[mask] = -1 / (2 * X3[mask])

    # computing auxiliary B matrix
    B2x = G26 * X6x
    B2y = G26 * X6y
    B2z = G26 * X6z

    B3x = G36 * X6x + G45 * X5x
    B3y = G36 * X6y + G45 * X5y
    B3z = G36 * X6z + G45 * X5z

    B4x = G44 * X4x
    B4y = G44 * X4y
    B4z = G44 * X4z

    B5x = G45 * X4x
    B5y = G45 * X4y
    B5z = G45 * X4z

    B6x = G26 * X2x + G36 * X3x
    B6y = G26 * X2y + G36 * X3y
    B6z = G26 * X2z + G36 * X3z

    # computing the civita term
    C6 = (A2x * B2x + A2y * B2y + A2z * B2z +
          A3x * B3x + A3y * B3y + A3z * B3z +
          A4x * B4x + A4y * B4y + A4z * B4z +
          A5x * B5x + A5y * B5y + A5z * B5z +
          A6x * B6x + A6y * B6y + A6z * B6z)

    return C6


def beltrami_affine(X, mask, zooms, beta):
    """ 
    Computes Beltrami and Levi-Civita increments to uppdate the Iwasawa coordinates
    when the affine metric is chosen instead of the euclidean metric.
    """

    x, y, z = X.shape[:-1]
    dx, dy, dz = zooms
    X1, X2, X3, X4, X5, X6 = np.rollaxis(X, axis=-1)

    # computing all spatial derivatives of X,
    # Xi denotes the "ith" components of X (e.g. X1x = d(X1) / dx)
    X1x = forward_diff(Dxx, 0, dx) * mask
    X1y = forward_diff(Dxx, 1, dy) * mask
    X1z = forward_diff(Dxx, 2, dz) * mask

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

    h44, h55, h66, h77, h78, h88, h99 = np.zeros((7, x, y, z))

    h44[mask] = 1 / (X1[mask]**2)
    h55[mask] = 1 / (X2[mask]**2)
    h66[mask] = 1 / (X3[mask]**2)
    h77[mask] = (2 * X1[mask] * (X3[mask] + X2[mask] * X6[mask]**2)) / (X2[mask] * X3[mask])
    h78[mask] = -(2 * X1[mask] * X6[mask]) / X3[mask]
    h88[mask] = 2 * X1[mask] / X3[mask]
    h99[mask] = 2 * X2[mask] / X3[mask]

    g_11 = 1 + (X1x**2 * h44 + X2x**2 * h55 + X3x**2 * h66 +
                X4x**2 * h77 + 2 * X4x * X5x * h78 +
                X5x**2 * h88 + X6x**2 * h99) * beta

    g_22 = 1 + (X1y**2 * h44 + X2y**2 * h55 + X3y**2 * h66 +
                X4y**2 * h77 + 2 * X4y * X5y * h78 +
                X5y**2 * h88 + X6y**2 * h99) * beta

    g_33 = 1 + (X1z**2 * h44 + X2z**2 * h55 + X3z**2 * h66 +
                X4z**2 * h77 + 2 * X4z * X5z * h78 +
                X5z**2 * h88 + X6z**2 * h99) * beta

    g_12 = (X1x * X1y * h44 + X2x * X2y * h55 +
            X3x * X3y * h66 + X4x * X4y * h77 +
            X5x * X4y * h78 + X4x * X5y * h78 +
            X5x * X5y * h88 + X6x * X6y * h99) * beta

    g_13 = (X1x * X1z * h44 + X2x * X2z * h55 +
            X3x * X3z * h66 + X4x * X4z * h77 +
            X5x * X4z * h78 + X4x * X5z * h78 +
            X5x * X5z * h88 + X6x * X6z * h99) * beta

    g_23 = (X1y * X1z * h44 + X2y * X2z * h55 +
            X3y * X3z * h66 + X4y * X4z * h77 +
            X5y * X4z * h78 + X4y * X5z * h78 +
            X5y * X5z * h88 + X6y * X6z * h99) * beta
    # computing g inverse
    gdet = (g_11 * g_22 * g_33 + g_12 * g_13 * g_23 * 2 -
            g_22 * g_13**2 - g_33 * g_12**2 - g_11 * g_23**2)

    # gdet[gdet <= 0] = 0.0001
    # bad_g = np.logical_and(mask, gdet <= 0)
    bad_g = np.logical_or(gdet <= 0, gdet > 1000)
    bad_g = np.logical_and(bad_g, mask)
    gdet[bad_g] = 1
    g_11[bad_g] = 1
    g_22[bad_g] = 1
    g_33[bad_g] = 1
    g_12[bad_g] = 0
    g_13[bad_g] = 0
    g_23[bad_g] = 0

    g11, g22, g33, g12, g13, g23 = np.zeros((6, x, y, z))

    g11 = (g_22 * g_33 - g_23**2) / gdet
    g22 = (g_11 * g_33 - g_13**2) / gdet
    g33 = (g_11 * g_22 - g_12**2) / gdet
    g12 = (g_13 * g_23 - g_12 * g_33) / gdet
    g13 = (g_12 * g_23 - g_13 * g_22) / gdet
    g23 = (g_12 * g_13 - g_11 * g_23) / gdet

    # computing A matrix
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

    # computing civita increments
    dC = np.zeros(X.shape)
    dC[..., 0] = civ1(X1, X2, X3, X6, X1x, X1y, X1z, X4x, X4y, X4z,
                      X5x, X5y, X5z, A1x, A1y, A1z, A4x, A4y, A4z,
                      A5x, A5y, A5z, mask)

    dC[..., 1] = civ2(X1, X2, X3, X2x, X2y, X2z, X4x, X4y, X4z, X6x, X6y, X6z,
                      A2x, A2y, A2z, A4x, A4y, A4z, A6x, A6y, A6z, mask)

    dC[..., 2] = civ3(X1, X2, X3, X6, X3x, X3y, X3z, X4x, X4y, X4z,
                      X5x, X5y, X5z, X6x, X6y, X6z, A3x, A3y, A3z,
                      A4x, A4y, A4z, A5x, A5y, A5z, A6x, A6y, A6z, mask)

    dC[..., 3] = civ4(X1, X2, X3, X6, X1x, X1y, X1z, X2x, X2y, X2z,
                      X4x, X4y, X4z, X5x, X5y, X5z, X6x, X6y, X6z,
                      A1x, A1y, A1z, A2x, A2y, A2z, A4x, A4y, A4z,
                      A5x, A5y, A5z, A6x, A6y, A6z, mask)

    dC[..., 4] = civ5(X1, X2, X3, X6, X1x, X1y, X1z, X2x, X2y, X2z,
                      X3x, X3y, X3z, X4x, X4y, X4z, X5x, X5y, X5z,
                      X6x, X6y, X6z, A1x, A1y, A1z, A2x, A2y, A2z,
                      A3x, A3y, A3z, A4x, A4y, A4z, A5x, A5y, A5z,
                      A6x, A6y, A6z, mask)
    dC[..., 5] = civ6(X1, X2, X3, X6, X2x, X2y, X2z, X3x, X3y, X3z,
                      X4x, X4y, X4z, X5x, X5y, X5z, X6x, X6y, X6z,
                      A2x, A2y, A2z, A3x, A3y, A3z, A4x, A4y, A4z,
                      A5x, A5y, A5z, A6x, A6y, A6z, mask)

    # dB[bad_g, :] = 0
    # dC[bad_g, :] = 0

    return (bad_g, g, dB, dC)


def fidelity_euclidean(Ak, D, f, g, bvals, H, mask, Diso):
    """
    Computes the Fidelity increments to update the diffusion tensor
    components, when the euclidean metric is chosen.

    Parameters
    ----------
    Ak : (...) ndarray
        Observed signal attenuatiions.
    D : (..., 6) ndarray
        The lower triangular components of the diffusion tensor,
        in the following order: Dxx, Dxy, Dyy, Dxz, Dyz, Dzz.
    f : (...) ndarray
        Tissue volume fraction (f = 1 - fw)
    g : (...) ndarray
        Square root of the determinant of the euclidean metric tensor g.
    bvals : (k) ndarray
        Vector containing the bvals for all 'k' gradient directions.
    H : (k, 6) ndarray
        Transposed design matrix.
    mask : boolean array
        Boolean mask that marks indices of the data that should be updated.
    Diso : float
        The diffusion constant of isotropic Free Water.

    Returns
    -------
    df : (...) ndarray
        Increments to update the tissue volume fracion.
    dF : (..., 6) ndarray
        The Fidelity increments to be applied to each component of D.

    Notes
    -----
    The H matrix is a transposed version of the design matrix implemented in
    dipy, mutiplied by -1, also, the 'dummy' vector is cropped.

    e.g. let A denote the design matrix implemented in dipy, then:

    H = A[1:, :-1]
    H = -1 * H.T
    """

    D_inmask = D[mask, :]
    f_inmask = f[mask]
    f_inmask = f_inmask[..., np.newaxis]
    Ak_inmask = Ak[mask, :]
    g_inmask = g[mask]

    # computing the part of fidelity common to all X1,..X6
    Aw = np.exp(-bvals * Diso)
    Cw = (1 - f_inmask) * Aw
    qDq = np.dot(D_inmask, H)
    np.clip(qDq, a_min=10**-7, a_max=None, out=qDq[...])
    At = np.exp(-qDq)
    Ct = f_inmask * At
    Abi = Ct + Cw
    aux = (Abi - Ak_inmask) * At

    # computing total fidelity terms
    dF = np.zeros(D.shape)

    dF[mask, 0] = np.sum(aux * H[0, :], axis=-1)
    dF[mask, 0] /= -g_inmask

    dF[mask, 1] = np.sum(aux * H[1, :], axis=-1)
    dF[mask, 1] /= -g_inmask

    dF[mask, 2] = np.sum(aux * H[2, :], axis=-1)
    dF[mask, 2] /= -g_inmask

    dF[mask, 3] = np.sum(aux * H[3, :], axis=-1)
    dF[mask, 3] /= -g_inmask

    dF[mask, 4] = np.sum(aux * H[4, :], axis=-1)
    dF[mask, 4] /= -g_inmask

    dF[mask, 5] = np.sum(aux * H[5, :], axis=-1)
    dF[mask, 5] /= -g_inmask

    # computing tissue fraction increment
    df = np.zeros(f.shape)
    df[mask] = -np.sum(bvals * (Abi - Ak_inmask) * (At - Aw), axis=-1)
    
    return (df, dF)
