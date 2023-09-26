""" Utility functions for algebra etc """

import itertools
import math
import numpy as np
import numpy.linalg as npl

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def sphere2cart(r, theta, phi):
    """ Spherical to Cartesian coordinates

    This is the standard physics convention where `theta` is the
    inclination (polar) angle, and `phi` is the azimuth angle.

    Imagine a sphere with center (0,0,0).  Orient it with the z axis
    running south-north, the y axis running west-east and the x axis
    from posterior to anterior.  `theta` (the inclination angle) is the
    angle to rotate from the z-axis (the zenith) around the y-axis,
    towards the x axis.  Thus the rotation is counter-clockwise from the
    point of view of positive y.  `phi` (azimuth) gives the angle of
    rotation around the z-axis towards the y axis.  The rotation is
    counter-clockwise from the point of view of positive z.

    Equivalently, given a point P on the sphere, with coordinates x, y,
    z, `theta` is the angle between P and the z-axis, and `phi` is
    the angle between the projection of P onto the XY plane, and the X
    axis.

    Geographical nomenclature designates theta as 'co-latitude', and phi
    as 'longitude'

    Parameters
    ----------
    r : array_like
       radius
    theta : array_like
       inclination or polar angle
    phi : array_like
       azimuth angle

    Returns
    -------
    x : array
       x coordinate(s) in Cartesian space
    y : array
       y coordinate(s) in Cartesian space
    z : array
       z coordinate

    Notes
    -----
    See these pages:

    * https://en.wikipedia.org/wiki/Spherical_coordinate_system
    * https://mathworld.wolfram.com/SphericalCoordinates.html

    for excellent discussion of the many different conventions
    possible.  Here we use the physics conventions, used in the
    wikipedia page.

    Derivations of the formulae are simple. Consider a vector x, y, z of
    length r (norm of x, y, z).  The inclination angle (theta) can be
    found from: cos(theta) == z / r -> z == r * cos(theta).  This gives
    the hypotenuse of the projection onto the XY plane, which we will
    call Q. Q == r*sin(theta). Now x / Q == cos(phi) -> x == r *
    sin(theta) * cos(phi) and so on.

    We have deliberately named this function ``sphere2cart`` rather than
    ``sph2cart`` to distinguish it from the Matlab function of that
    name, because the Matlab function uses an unusual convention for the
    angles that we did not want to replicate.  The Matlab function is
    trivial to implement with the formulae given in the Matlab help.

    """
    sin_theta = np.sin(theta)
    x = r * np.cos(phi) * sin_theta
    y = r * np.sin(phi) * sin_theta
    z = r * np.cos(theta)
    x, y, z = np.broadcast_arrays(x, y, z)
    return x, y, z


def cart2sphere(x, y, z):
    r""" Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    See doc for ``sphere2cart`` for angle conventions and derivation
    of the formulae.

    $0\le\theta\mathrm{(theta)}\le\pi$ and $-\pi\le\phi\mathrm{(phi)}\le\pi$

    Parameters
    ----------
    x : array_like
       x coordinate in Cartesian space
    y : array_like
       y coordinate in Cartesian space
    z : array_like
       z coordinate

    Returns
    -------
    r : array
       radius
    theta : array
       inclination (polar) angle
    phi : array
       azimuth angle

    """
    r = np.sqrt(x * x + y * y + z * z)
    theta = np.arccos(np.divide(z, r, where=r > 0))
    theta = np.where(r > 0, theta, 0.)
    phi = np.arctan2(y, x)
    r, theta, phi = np.broadcast_arrays(r, theta, phi)
    return r, theta, phi


def sph2latlon(theta, phi):
    """Convert spherical coordinates to latitude and longitude.

    Returns
    -------
    lat, lon : ndarray
        Latitude and longitude.

    """
    return np.rad2deg(theta - np.pi / 2), np.rad2deg(phi - np.pi)


def normalized_vector(vec, axis=-1):
    """ Return vector divided by its Euclidean (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    ----------
    vec : array_like shape (3,)

    Returns
    -------
    nvec : array shape (3,)
       vector divided by L2 norm

    Examples
    --------
    >>> vec = [1, 2, 3]
    >>> l2n = np.sqrt(np.dot(vec, vec))
    >>> nvec = normalized_vector(vec)
    >>> np.allclose(np.array(vec) / l2n, nvec)
    True
    >>> vec = np.array([[1, 2, 3]])
    >>> vec.shape == (1, 3)
    True
    >>> normalized_vector(vec).shape == (1, 3)
    True

    """
    return vec / vector_norm(vec, axis, keepdims=True)


def vector_norm(vec, axis=-1, keepdims=False):
    """ Return vector Euclidean (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    ----------
    vec : array_like
        Vectors to norm.
    axis : int
        Axis over which to norm. By default norm over last axis. If `axis` is
        None, `vec` is flattened then normed.
    keepdims : bool
        If True, the output will have the same number of dimensions as `vec`,
        with shape 1 on `axis`.

    Returns
    -------
    norm : array
        Euclidean norms of vectors.

    Examples
    --------
    >>> import numpy as np
    >>> vec = [[8, 15, 0], [0, 36, 77]]
    >>> vector_norm(vec)
    array([ 17.,  85.])
    >>> vector_norm(vec, keepdims=True)
    array([[ 17.],
           [ 85.]])
    >>> vector_norm(vec, axis=0)
    array([  8.,  39.,  77.])

    """
    vec = np.asarray(vec)
    vec_norm = np.sqrt((vec * vec).sum(axis))
    if keepdims:
        if axis is None:
            shape = [1] * vec.ndim
        else:
            shape = list(vec.shape)
            shape[axis] = 1
        vec_norm = vec_norm.reshape(shape)
    return vec_norm


def rodrigues_axis_rotation(r, theta):
    """ Rodrigues formula

    Rotation matrix for rotation around axis r for angle theta.

    The rotation matrix is given by the Rodrigues formula:

    R = Id + sin(theta)*Sn + (1-cos(theta))*Sn^2

    with::

             0  -nz  ny
      Sn =   nz   0 -nx
            -ny  nx   0

    where n = r / ||r||

    In case the angle ||r|| is very small, the above formula may lead
    to numerical instabilities. We instead use a Taylor expansion
    around theta=0:

    R = I + sin(theta)/tetha Sr + (1-cos(theta))/teta2 Sr^2

    leading to:

    R = I + (1-theta2/6)*Sr + (1/2-theta2/24)*Sr^2

    Parameters
    ----------
    r :  array_like shape (3,), axis
    theta : float, angle in degrees

    Returns
    -------
    R : array, shape (3,3), rotation matrix

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.core.geometry import rodrigues_axis_rotation
    >>> v=np.array([0,0,1])
    >>> u=np.array([1,0,0])
    >>> R=rodrigues_axis_rotation(v,40)
    >>> ur=np.dot(R,u)
    >>> np.round(np.rad2deg(np.arccos(np.dot(ur,u))))
    40.0

    """
    theta = np.deg2rad(theta)
    if theta > 1e-30:
        n = r / np.linalg.norm(r)
        Sn = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        R = np.eye(3) + np.sin(theta) * Sn + \
            (1 - np.cos(theta)) * np.dot(Sn, Sn)
    else:
        Sr = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
        theta2 = theta * theta
        R = np.eye(3) + (1 - theta2 / 6.) * \
            Sr + (.5 - theta2 / 24.) * np.dot(Sr, Sr)
    return R


def nearest_pos_semi_def(B):
    """ Least squares positive semi-definite tensor estimation

    Parameters
    ----------
    B : (3,3) array_like
       B matrix - symmetric. We do not check the symmetry.

    Returns
    -------
    npds : (3,3) array
       Estimated nearest positive semi-definite array to matrix `B`.

    Examples
    --------
    >>> B = np.diag([1, 1, -1])
    >>> nearest_pos_semi_def(B)
    array([[ 0.75,  0.  ,  0.  ],
           [ 0.  ,  0.75,  0.  ],
           [ 0.  ,  0.  ,  0.  ]])

    References
    ----------
    .. [1] Niethammer M, San Jose Estepar R, Bouix S, Shenton M, Westin CF.
           On diffusion tensor estimation. Conf Proc IEEE Eng Med Biol Soc.
           2006;1:2622-5. PubMed PMID: 17946125; PubMed Central PMCID:
           PMC2791793.

    """
    B = np.asarray(B)
    vals, vecs = npl.eigh(B)
    # indices of eigenvalues in descending order
    inds = np.argsort(vals)[::-1]
    vals = vals[inds]
    cardneg = np.sum(vals < 0)
    if cardneg == 0:
        return B
    if cardneg == 3:
        return np.zeros((3, 3))
    lam1a, lam2a, lam3a = vals
    scalers = np.zeros((3,))
    if cardneg == 2:
        b112 = np.max([0, lam1a + (lam2a + lam3a) / 3.])
        scalers[0] = b112
    elif cardneg == 1:
        lam1b = lam1a + 0.25 * lam3a
        lam2b = lam2a + 0.25 * lam3a
        if lam1b >= 0 and lam2b >= 0:
            scalers[:2] = lam1b, lam2b
        else:  # one of the lam1b, lam2b is < 0
            if lam2b < 0:
                b111 = np.max([0, lam1a + (lam2a + lam3a) / 3.])
                scalers[0] = b111
            if lam1b < 0:
                b221 = np.max([0, lam2a + (lam1a + lam3a) / 3.])
                scalers[1] = b221
    # resort the scalers to match the original vecs
    scalers = scalers[np.argsort(inds)]
    return np.dot(vecs, np.dot(np.diag(scalers), vecs.T))


def sphere_distance(pts1, pts2, radius=None, check_radius=True):
    """ Distance across sphere surface between `pts1` and `pts2`

    Parameters
    ----------
    pts1 : (N,R) or (R,) array_like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D)
    pts2 : (N,R) or (R,) array_like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D).  It should be
       possible to broadcast `pts1` against `pts2`
    radius : None or float, optional
       Radius of sphere.  Default is to work out radius from mean of the
       length of each point vector
    check_radius : bool, optional
       If True, check if the points are on the sphere surface - i.e
       check if the vector lengths in `pts1` and `pts2` are close to
       `radius`.  Default is True.

    Returns
    -------
    d : (N,) or (0,) array
       Distances between corresponding points in `pts1` and `pts2`
       across the spherical surface, i.e. the great circle distance

    See Also
    --------
    cart_distance : cartesian distance between points
    vector_cosine : cosine of angle between vectors

    Examples
    --------
    >>> print('%.4f' % sphere_distance([0,1],[1,0]))
    1.5708
    >>> print('%.4f' % sphere_distance([0,3],[3,0]))
    4.7124
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    lens1 = np.sqrt(np.sum(pts1 ** 2, axis=-1))
    lens2 = np.sqrt(np.sum(pts2 ** 2, axis=-1))
    if radius is None:
        radius = (np.mean(lens1) + np.mean(lens2)) / 2.0
    if check_radius:
        if not (np.allclose(radius, lens1) and
                np.allclose(radius, lens2)):
            raise ValueError('Radii do not match sphere surface')
    # Get angle with vector cosine
    dots = np.inner(pts1, pts2)
    lens = lens1 * lens2
    angle_cos = np.arccos(dots / lens)
    return angle_cos * radius


def cart_distance(pts1, pts2):
    """ Cartesian distance between `pts1` and `pts2`

    If either of `pts1` or `pts2` is 2D, then we take the first
    dimension to index points, and the second indexes coordinate.  More
    generally, we take the last dimension to be the coordinate
    dimension.

    Parameters
    ----------
    pts1 : (N,R) or (R,) array_like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D)
    pts2 : (N,R) or (R,) array_like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D).  It should be
       possible to broadcast `pts1` against `pts2`

    Returns
    -------
    d : (N,) or (0,) array
       Cartesian distances between corresponding points in `pts1` and
       `pts2`

    See Also
    --------
    sphere_distance : distance between points on sphere surface

    Examples
    --------
    >>> cart_distance([0,0,0], [0,0,3])
    3.0
    """
    sqs = np.subtract(pts1, pts2) ** 2
    return np.sqrt(np.sum(sqs, axis=-1))


def vector_cosine(vecs1, vecs2):
    """ Cosine of angle between two (sets of) vectors

    The cosine of the angle between two vectors ``v1`` and ``v2`` is
    given by the inner product of ``v1`` and ``v2`` divided by the
    product of the vector lengths::

       v_cos = np.inner(v1, v2) / (np.sqrt(np.sum(v1**2)) *
                                   np.sqrt(np.sum(v2**2)))

    Parameters
    ----------
    vecs1 : (N, R) or (R,) array_like
       N vectors (as rows) or single vector.  Vectors have R elements.
    vecs1 : (N, R) or (R,) array_like
       N vectors (as rows) or single vector.  Vectors have R elements.
       It should be possible to broadcast `vecs1` against `vecs2`

    Returns
    -------
    vcos : (N,) or (0,) array
       Vector cosines.  To get the angles you will need ``np.arccos``

    Notes
    -----
    The vector cosine will be the same as the correlation only if all
    the input vectors have zero mean.

    """
    vecs1 = np.asarray(vecs1)
    vecs2 = np.asarray(vecs2)
    lens1 = np.sqrt(np.sum(vecs1 ** 2, axis=-1))
    lens2 = np.sqrt(np.sum(vecs2 ** 2, axis=-1))
    dots = np.inner(vecs1, vecs2)
    lens = lens1 * lens2
    return dots / lens


def lambert_equal_area_projection_polar(theta, phi):
    r""" Lambert Equal Area Projection from polar sphere to plane

    Return positions in (y1,y2) plane corresponding to the points
    with polar coordinates (theta, phi) on the unit sphere, under the
    Lambert Equal Area Projection mapping (see Mardia and Jupp (2000),
    Directional Statistics, p. 161).

    See doc for ``sphere2cart`` for angle conventions

    - $0 \le \theta \le \pi$ and $0 \le \phi \le 2 \pi$
    - $|(y_1,y_2)| \le 2$

    The Lambert EAP maps the upper hemisphere to the planar disc of radius 1
    and the lower hemisphere to the planar annulus between radii 1 and 2,
    and *vice versa*.

    Parameters
    ----------
    theta : array_like
       theta spherical coordinates
    phi : array_like
       phi spherical coordinates

    Returns
    -------
    y : (N,2) array
       planar coordinates of points following mapping by Lambert's EAP.
    """

    return 2 * np.repeat(np.sin(theta / 2), 2).reshape((theta.shape[0], 2)) * \
        np.column_stack((np.cos(phi), np.sin(phi)))


def lambert_equal_area_projection_cart(x, y, z):
    r""" Lambert Equal Area Projection from cartesian vector to plane

    Return positions in $(y_1,y_2)$ plane corresponding to the
    directions of the vectors with cartesian coordinates xyz under the
    Lambert Equal Area Projection mapping (see Mardia and Jupp (2000),
    Directional Statistics, p. 161).

    The Lambert EAP maps the upper hemisphere to the planar disc of radius 1
    and the lower hemisphere to the planar annulus between radii 1 and 2,
    The Lambert EAP maps the upper hemisphere to the planar disc of radius 1
    and the lower hemisphere to the planar annulus between radii 1 and 2.
    and *vice versa*.

    See doc for ``sphere2cart`` for angle conventions

    Parameters
    ----------
    x : array_like
       x coordinate in Cartesian space
    y : array_like
       y coordinate in Cartesian space
    z : array_like
       z coordinate

    Returns
    -------
    y : (N,2) array
       planar coordinates of points following mapping by Lambert's EAP.

    """
    (r, theta, phi) = cart2sphere(x, y, z)
    return lambert_equal_area_projection_polar(theta, phi)


def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    Code modified from the work of Christoph Gohlke, link provided here
    https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py

    Parameters
    ----------
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    Returns
    -------
    matrix : ndarray (4, 4)

    Code modified from the work of Christoph Gohlke, link provided here
    https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py

    Examples
    --------
    >>> import numpy
    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    _ = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    _ = euler_matrix(ai, aj, ak, axes)

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj * si
        M[i, k] = sj * ci
        M[j, i] = sj * sk
        M[j, j] = -cj * ss + cc
        M[j, k] = -cj * cs - sc
        M[k, i] = -sj * ck
        M[k, j] = cj * sc + cs
        M[k, k] = cj * cc - ss
    else:
        M[i, i] = cj * ck
        M[i, j] = sj * sc - cs
        M[i, k] = sj * cc + ss
        M[j, i] = cj * sk
        M[j, j] = sj * ss + cc
        M[j, k] = sj * cs - sc
        M[k, i] = -sj
        M[k, j] = cj * si
        M[k, k] = cj * ci
    return M


def compose_matrix(scale=None, shear=None, angles=None, translate=None,
                   perspective=None):
    """Return 4x4 transformation matrix from sequence of
    transformations.

    Code modified from the work of Christoph Gohlke, link provided here
    https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py

    This is the inverse of the ``decompose_matrix`` function.

    Parameters
    ----------
    scale : (3,) array_like
        Scaling factors.
    shear : array_like
        Shear factors for x-y, x-z, y-z axes.
    angles : array_like
        Euler angles about static x, y, z axes.
    translate : array_like
        Translation vector along x, y, z axes.
    perspective : array_like
        Perspective partition of matrix.

    Returns
    -------
    matrix : 4x4 array


    Examples
    --------
    >>> import math
    >>> import numpy as np
    >>> import dipy.core.geometry as gm
    >>> scale = np.random.random(3) - 0.5
    >>> shear = np.random.random(3) - 0.5
    >>> angles = (np.random.random(3) - 0.5) * (2*math.pi)
    >>> trans = np.random.random(3) - 0.5
    >>> persp = np.random.random(4) - 0.5
    >>> M0 = gm.compose_matrix(scale, shear, angles, trans, persp)

    """
    M = np.identity(4)
    if perspective is not None:
        P = np.identity(4)
        P[3, :] = perspective[:4]
        M = np.dot(M, P)
    if translate is not None:
        T = np.identity(4)
        T[:3, 3] = translate[:3]
        M = np.dot(M, T)
    if angles is not None:
        R = euler_matrix(angles[0], angles[1], angles[2], 'sxyz')
        M = np.dot(M, R)
    if shear is not None:
        Z = np.identity(4)
        Z[1, 2] = shear[2]
        Z[0, 2] = shear[1]
        Z[0, 1] = shear[0]
        M = np.dot(M, Z)
    if scale is not None:
        S = np.identity(4)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        M = np.dot(M, S)
    M /= M[3, 3]
    return M


def decompose_matrix(matrix):
    """Return sequence of transformations from transformation matrix.

    Code modified from the excellent work of Christoph Gohlke, link
    provided here:
    https://github.com/cgohlke/transformations/blob/master/transformations/transformations.py

    Parameters
    ----------
    matrix : array_like
        Non-degenerate homogeneous transformation matrix

    Returns
    -------
    scale : (3,) ndarray
        Three scaling factors.
    shear : (3,) ndarray
        Shear factors for x-y, x-z, y-z axes.
    angles : (3,) ndarray
        Euler angles about static x, y, z axes.
    translate : (3,) ndarray
        Translation vector along x, y, z axes.
    perspective : ndarray
        Perspective partition of matrix.

    Raises
    ------
    ValueError
        If matrix is of wrong type or degenerate.

    Examples
    --------
    >>> import numpy as np
    >>> T0=np.diag([2,1,1,1])
    >>> scale, shear, angles, trans, persp = decompose_matrix(T0)

    """
    M = np.array(matrix, dtype=np.float64, copy=True).T
    if abs(M[3, 3]) < _EPS:
        raise ValueError("M[3, 3] is zero")
    M /= M[3, 3]
    P = M.copy()
    P[:, 3] = 0, 0, 0, 1
    if not np.linalg.det(P):
        raise ValueError("matrix is singular")

    scale = np.zeros((3, ), dtype=np.float64)
    shear = [0, 0, 0]
    angles = [0, 0, 0]

    if any(abs(M[:3, 3]) > _EPS):
        perspective = np.dot(M[:, 3], np.linalg.inv(P.T))
        M[:, 3] = 0, 0, 0, 1
    else:
        perspective = np.array((0, 0, 0, 1), dtype=np.float64)

    translate = M[3, :3].copy()
    M[3, :3] = 0

    row = M[:3, :3].copy()
    scale[0] = vector_norm(row[0])
    row[0] /= scale[0]
    shear[0] = np.dot(row[0], row[1])
    row[1] -= row[0] * shear[0]
    scale[1] = vector_norm(row[1])
    row[1] /= scale[1]
    shear[0] /= scale[1]
    shear[1] = np.dot(row[0], row[2])
    row[2] -= row[0] * shear[1]
    shear[2] = np.dot(row[1], row[2])
    row[2] -= row[1] * shear[2]
    scale[2] = vector_norm(row[2])
    row[2] /= scale[2]
    shear[1:] /= scale[2]

    if np.dot(row[0], np.cross(row[1], row[2])) < 0:
        scale *= -1
        row *= -1

    angles[1] = math.asin(-row[0, 2])
    if math.cos(angles[1]):
        angles[0] = math.atan2(row[1, 2], row[2, 2])
        angles[2] = math.atan2(row[0, 1], row[0, 0])
    else:
        # angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective


def circumradius(a, b, c):
    """ a, b and c are 3-dimensional vectors which are the vertices of a
    triangle. The function returns the circumradius of the triangle, i.e
    the radius of the smallest circle that can contain the triangle. In
    the degenerate case when the 3 points are collinear it returns
    half the distance between the furthest apart points.

    Parameters
    ----------
    a, b, c : (3,) array_like
       the three vertices of the triangle

    Returns
    -------
    circumradius : float
        the desired circumradius
    """
    x = a - c
    xx = np.linalg.norm(x) ** 2
    y = b - c
    yy = np.linalg.norm(y) ** 2
    z = np.cross(x, y)
    # test for collinearity
    if np.linalg.norm(z) == 0:
        return np.sqrt(np.max(np.dot(x, x), np.dot(y, y),
                              np.dot(a - b, a - b))) / 2.
    else:
        m = np.vstack((x, y, z))
        w = np.dot(np.linalg.inv(m.T), np.array([xx / 2., yy / 2., 0]))
        return np.linalg.norm(w) / 2.


def vec2vec_rotmat(u, v):
    r""" rotation matrix from 2 unit vectors

    u, v being unit 3d vectors return a 3x3 rotation matrix R than aligns u to
    v.

    In general there are many rotations that will map u to v. If S is any
    rotation using v as an axis then R.S will also map u to v since (S.R)u =
    S(Ru) = Sv = v.  The rotation R returned by vec2vec_rotmat leaves fixed the
    perpendicular to the plane spanned by u and v.

    The transpose of R will align v to u.

    Parameters
    ----------
    u : array, shape(3,)
    v : array, shape(3,)

    Returns
    -------
    R : array, shape(3,3)

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.core.geometry import vec2vec_rotmat
    >>> u=np.array([1,0,0])
    >>> v=np.array([0,1,0])
    >>> R=vec2vec_rotmat(u,v)
    >>> np.dot(R,u)
    array([ 0.,  1.,  0.])
    >>> np.dot(R.T,v)
    array([ 1.,  0.,  0.])

    """
    # Cross product is the first step to find R
    # Rely on numpy instead of manual checking for failing
    # cases
    w = np.cross(u, v)
    wn = np.linalg.norm(w)

    # Check that cross product is OK and vectors
    # u, v are not collinear (norm(w)>0.0)
    if np.isnan(wn) or wn < np.finfo(float).eps:
        norm_u_v = np.linalg.norm(u - v)
        # This is the case of two antipodal vectors:
        # ** former checking assumed norm(u) == norm(v)
        if norm_u_v > np.linalg.norm(u):
            return -np.eye(3)
        return np.eye(3)

    # if everything ok, normalize w
    w = w / wn

    # vp is in plane of u,v,  perpendicular to u
    vp = (v - (np.dot(u, v) * u))
    vp = vp / np.linalg.norm(vp)

    # (u vp w) is an orthonormal basis
    P = np.array([u, vp, w])
    Pt = P.T
    cosa = np.clip(np.dot(u, v), -1, 1)
    sina = np.sqrt(1 - cosa ** 2)
    R = np.array([[cosa, -sina, 0], [sina, cosa, 0], [0, 0, 1]])
    Rp = np.dot(Pt, np.dot(R, P))

    # make sure that you don't return any Nans
    # check using the appropriate tool in numpy
    if np.any(np.isnan(Rp)):
        return np.eye(3)

    return Rp


def compose_transformations(*mats):
    """ Compose multiple 4x4 affine transformations in one 4x4 matrix

    Parameters
    ----------

    mat1 : array, (4, 4)
    mat2 : array, (4, 4)
    ...
    matN : array, (4, 4)

    Returns
    -------
    matN x ... x mat2 x mat1 : array, (4, 4)

    """
    prev = mats[0]
    if len(mats) < 2:
        raise ValueError('At least two or more matrices are needed')

    for mat in mats[1:]:

        prev = np.dot(mat, prev)

    return prev


def perpendicular_directions(v, num=30, half=False):
    r""" Computes n evenly spaced perpendicular directions relative to a given
    vector v

    Parameters
    ----------
    v : array (3,)
        Array containing the three cartesian coordinates of vector v
    num : int, optional
        Number of perpendicular directions to generate
    half : bool, optional
        If half is True, perpendicular directions are sampled on half of the
        unit circumference perpendicular to v, otherwive perpendicular
        directions are sampled on the full circumference. Default of half is
        False

    Returns
    -------
    psamples : array (n, 3)
        array of vectors perpendicular to v

    Notes
    -----
    Perpendicular directions are estimated using the following two step
    procedure:

        1) the perpendicular directions are first sampled in a unit
        circumference parallel to the plane normal to the x-axis.

        2) Samples are then rotated and aligned to the plane normal to vector
        v. The rotational matrix for this rotation is constructed as reference
        frame basis which axis are the following:
            - The first axis is vector v
            - The second axis is defined as the normalized vector given by the
            cross product between vector v and the unit vector aligned to the
            x-axis
            - The third axis is defined as the cross product between the
            previous computed vector and vector v.

    Following this two steps, coordinates of the final perpendicular directions
    are given as:

    .. math::

        \left [ -\sin(a_{i}) \sqrt{{v_{y}}^{2}+{v_{z}}^{2}}
        \; , \;
        \frac{v_{x}v_{y}\sin(a_{i})-v_{z}\cos(a_{i})}
        {\sqrt{{v_{y}}^{2}+{v_{z}}^{2}}}
        \; , \;
        \frac{v_{x}v_{z}\sin(a_{i})-v_{y}\cos(a_{i})}
        {\sqrt{{v_{y}}^{2}+{v_{z}}^{2}}} \right  ]

    This procedure has a singularity when vector v is aligned to the x-axis. To
    solve this singularity, perpendicular directions in procedure's step 1 are
    defined in the plane normal to y-axis and the second axis of the rotated
    frame of reference is computed as the normalized vector given by the cross
    product between vector v and the unit vector aligned to the y-axis.
    Following this, the coordinates of the perpendicular directions are given
    as:

        \left [ -\frac{\left (v_{x}v_{y}\sin(a_{i})+v_{z}\cos(a_{i}) \right )}
        {\sqrt{{v_{x}}^{2}+{v_{z}}^{2}}}
        \; , \;
        \sin(a_{i}) \sqrt{{v_{x}}^{2}+{v_{z}}^{2}}
        \; , \;
        \frac{v_{y}v_{z}\sin(a_{i})+v_{x}\cos(a_{i})}
        {\sqrt{{v_{x}}^{2}+{v_{z}}^{2}}} \right  ]

    For more details on this calculation, see
    `here <https://gsoc2015dipydki.blogspot.com/2015/07/rnh-post-8-computing-perpendicular.html>`_.

    """  # noqa: E501
    v = np.array(v, dtype=float)

    # Float error used for floats comparison
    er = np.finfo(v[0]).eps * 1e3

    # Define circumference or semi-circumference
    if half is True:
        a = np.linspace(0., math.pi, num=num, endpoint=False)
    else:
        a = np.linspace(0., 2 * math.pi, num=num, endpoint=False)

    cosa = np.cos(a)
    sina = np.sin(a)

    # Check if vector is not aligned to the x axis
    if abs(v[0] - 1.) > er:
        sq = np.sqrt(v[1]**2 + v[2]**2)
        psamples = np.array([- sq*sina, (v[0]*v[1]*sina - v[2]*cosa) / sq,
                             (v[0]*v[2]*sina + v[1]*cosa) / sq])
    else:
        sq = np.sqrt(v[0]**2 + v[2]**2)
        psamples = np.array([- (v[2]*cosa + v[0]*v[1]*sina) / sq, sina*sq,
                             (v[0]*cosa - v[2]*v[1]*sina) / sq])

    return psamples.T


def dist_to_corner(affine):
    """Calculate the maximal distance from the center to a corner of a voxel,
    given an affine

    Parameters
    ----------
    affine : 4 by 4 array.
        The spatial transformation from the measurement to the scanner space.

    Returns
    -------
    dist: float
        The maximal distance to the corner of a voxel, given voxel size encoded
        in the affine.

    """
    R = affine[0:3, 0:3]
    vox_dim = np.diag(np.linalg.cholesky(R.T.dot(R)))
    return np.sqrt(np.sum((vox_dim / 2) ** 2))


def is_hemispherical(vecs):
    """Test whether all points on a unit sphere lie in the same hemisphere.

    Parameters
    ----------
    vecs : numpy.ndarray
        2D numpy array with shape (N, 3) where N is the number of points.
        All points must lie on the unit sphere.

    Returns
    -------
    is_hemi : bool
        If True, one can find a hemisphere that contains all the points.
        If False, then the points do not lie in any hemisphere

    pole : numpy.ndarray
        If `is_hemi == True`, then pole is the "central" pole of the
        input vectors. Otherwise, pole is the zero vector.

    References
    ----------
    https://rstudio-pubs-static.s3.amazonaws.com/27121_a22e51b47c544980bad594d5e0bb2d04.html  # noqa

    """
    if vecs.shape[1] != 3:
        raise ValueError("Input vectors must be 3D vectors")
    if not np.allclose(1, np.linalg.norm(vecs, axis=1)):
        raise ValueError("Input vectors must be unit vectors")

    # Generate all pairwise cross products
    v0, v1 = zip(*[p for p in itertools.permutations(vecs, 2)])
    cross_prods = np.cross(v0, v1)

    # Normalize them
    cross_prods /= np.linalg.norm(cross_prods, axis=1)[:, np.newaxis]

    # `cross_prods` now contains all candidate vertex points for "the polygon"
    # in the reference. "The polygon" is a subset. Find which points belong to
    # the polygon using a dot product test with each of the original vectors
    angles = np.arccos(np.dot(cross_prods, vecs.transpose()))

    # And test whether it is orthogonal or less
    dot_prod_test = angles <= np.pi / 2.0

    # If there is at least one point that is orthogonal or less to each
    # input vector, then the points lie on some hemisphere
    is_hemi = len(vecs) in np.sum(dot_prod_test.astype(int), axis=1)

    if is_hemi:
        vertices = cross_prods[
            np.sum(dot_prod_test.astype(int), axis=1) == len(vecs)
        ]

        pole = np.mean(vertices, axis=0)
        pole /= np.linalg.norm(pole)
    else:
        pole = np.array([0.0, 0.0, 0.0])
    return is_hemi, pole
