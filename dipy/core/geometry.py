''' Utility functions for algebra etc '''

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
    ''' Spherical to Cartesian coordinates

    This is the standard physics convention where `theta` is the
    inclination (polar) angle, and `phi` is the azimuth angle.

    Imagine a sphere with center (0,0,0).  Orient it with the z axis
    running south->north, the y axis running west-east and the x axis
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
    ------------
    r : array-like
       radius
    theta : array-like
       inclination or polar angle
    phi : array-like
       azimuth angle

    Returns
    ---------
    x : array
       x coordinate(s) in Cartesion space
    y : array
       y coordinate(s) in Cartesian space
    z : array
       z coordinate

    Notes
    --------
    See these pages:

    * http://en.wikipedia.org/wiki/Spherical_coordinate_system
    * http://mathworld.wolfram.com/SphericalCoordinates.html

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
    '''
    sin_theta = np.sin(theta)
    x = r * np.cos(phi) * sin_theta
    y = r * np.sin(phi) * sin_theta
    z = r * np.cos(theta)
    return x, y, z


def cart2sphere(x, y, z):
    r''' Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    See doc for ``sphere2cart`` for angle conventions and derivation
    of the formulae.

    $0 \le \theta \mathrm{(theta)} \le \pi$ and $0 \le \phi \mathrm{(phi)} \le 2 \pi$

    Parameters
    ------------
    x : array-like
       x coordinate in Cartesion space
    y : array-like
       y coordinate in Cartesian space
    z : array-like
       z coordinate

    Returns
    ---------
    r : array
       radius
    theta : array
       inclination (polar) angle
    phi : array
       azimuth angle
    '''
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def normalized_vector(vec):
    ''' Return vector divided by its Euclidean (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    ------------
    vec : array-like shape (3,)

    Returns
    ----------
    nvec : array shape (3,)
       vector divided by L2 norm

    Examples
    -----------
    >>> vec = [1, 2, 3]
    >>> l2n = np.sqrt(np.dot(vec, vec))
    >>> nvec = normalized_vector(vec)
    >>> np.allclose(np.array(vec) / l2n, nvec)
    True
    >>> vec = np.array([[1, 2, 3]])
    >>> vec.shape
    (1, 3)
    >>> normalized_vector(vec).shape
    (3,)
    '''
    vec = np.asarray(vec).squeeze()
    return vec / math.sqrt((vec**2).sum())


def vector_norm(vec):
    ''' Return vector Euclidaan (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    -------------
    vec : array-like shape (3,)

    Returns
    ---------
    norm : scalar

    Examples
    --------
    >>> import numpy as np
    >>> vec = [1, 2, 3]
    >>> l2n = np.sqrt(np.dot(vec, vec))
    >>> nvec = vector_norm(vec)
    >>> np.allclose(nvec, np.sqrt(np.dot(vec, vec)))
    True
    '''
    vec = np.asarray(vec)
    return math.sqrt((vec**2).sum())
    

def nearest_pos_semi_def(B):
    ''' Least squares positive semi-definite tensor estimation
    
    Reference: Niethammer M, San Jose Estepar R, Bouix S, Shenton M,
    Westin CF.  On diffusion tensor estimation. Conf Proc IEEE Eng Med
    Biol Soc.  2006;1:2622-5. PubMed PMID: 17946125; PubMed Central
    PMCID: PMC2791793.
 
    Parameters
    ------------
    B : (3,3) array-like
       B matrix - symmetric. We do not check the symmetry.
       
    Returns
    ---------
    npds : (3,3) array
       Estimated nearest positive semi-definite array to matrix `B`.

    Examples
    ----------
    >>> B = np.diag([1, 1, -1])
    >>> nearest_pos_semi_def(B)
    array([[ 0.75,  0.  ,  0.  ],
           [ 0.  ,  0.75,  0.  ],
           [ 0.  ,  0.  ,  0.  ]])
    '''
    B = np.asarray(B)
    vals, vecs = npl.eigh(B)
    # indices of eigenvalues in descending order
    inds = np.argsort(vals)[::-1]
    vals = vals[inds]
    cardneg = np.sum(vals < 0)
    if cardneg == 0:
        return B
    if cardneg == 3:
        return np.zeros((3,3))
    lam1a, lam2a, lam3a = vals
    scalers = np.zeros((3,))
    if cardneg == 2:
        b112 = np.max([0,lam1a+(lam2a+lam3a)/3.])
        scalers[0] = b112
    elif cardneg == 1:
        lam1b=lam1a+0.25*lam3a
        lam2b=lam2a+0.25*lam3a
        if lam1b >= 0 and lam2b >= 0:
            scalers[:2] = lam1b, lam2b
        else: # one of the lam1b, lam2b is < 0
            if lam2b < 0:
                b111=np.max([0,lam1a+(lam2a+lam3a)/3.])
                scalers[0] = b111
            if lam1b < 0:
                b221=np.max([0,lam2a+(lam1a+lam3a)/3.])
                scalers[1] = b221
    # resort the scalers to match the original vecs
    scalers = scalers[np.argsort(inds)]
    return np.dot(vecs, np.dot(np.diag(scalers), vecs.T))


def sphere_distance(pts1, pts2, radius=None, check_radius=True):
    """ Distance across sphere surface between `pts1` and `pts2`

    Parameters
    ------------
    pts1 : (N,R) or (R,) array-like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D)
    pts2 : (N,R) or (R,) array-like
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
    ---------
    d : (N,) or (0,) array
       Distances between corresponding points in `pts1` and `pts2`
       across the spherical surface, i.e. the great circle distance

    See also
    ----------
    cart_distance : cartesian distance between points
    vector_cosine : cosine of angle between vectors
    
    Examples
    ----------
    >>> print '%.4f' % sphere_distance([0,1],[1,0])
    1.5708
    >>> print '%.4f' % sphere_distance([0,3],[3,0])
    4.7124
    """
    pts1 = np.asarray(pts1)
    pts2 = np.asarray(pts2)
    lens1 = np.sqrt(np.sum(pts1**2, axis=-1))
    lens2 = np.sqrt(np.sum(pts2**2, axis=-1))
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
    ''' Cartesian distance between `pts1` and `pts2`

    If either of `pts1` or `pts2` is 2D, then we take the first
    dimension to index points, and the second indexes coordinate.  More
    generally, we take the last dimension to be the coordinate
    dimension. 
    
    Parameters
    ------------
    pts1 : (N,R) or (R,) array-like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D)
    pts2 : (N,R) or (R,) array-like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D).  It should be
       possible to broadcast `pts1` against `pts2`

    Returns
    ---------
    d : (N,) or (0,) array
       Cartesian distances between corresponding points in `pts1` and
       `pts2`

    See also
    ----------
    sphere_distance : distance between points on sphere surface

    Examples
    ----------
    >>> cart_distance([0,0,0], [0,0,3])
    3.0
    '''
    sqs = np.subtract(pts1, pts2)**2
    return np.sqrt(np.sum(sqs, axis=-1))


def vector_cosine(vecs1, vecs2):
    """ Cosine of angle between two (sets of) vectors

    The cosine of the angle between two vectors ``v1`` and ``v2`` is
    given by the inner product of ``v1`` and ``v2`` divided by the
    product of the vector lengths::

       v_cos = np.inner(v1, v2) / (np.sqrt(np.sum(v1**2)) *
                                   np.sqrt(np.sum(v2**2)))

    Parameters
    -------------
    vecs1 : (N, R) or (R,) array-like
       N vectors (as rows) or single vector.  Vectors have R elements.
    vecs1 : (N, R) or (R,) array-like
       N vectors (as rows) or single vector.  Vectors have R elements.
       It should be possible to broadcast `vecs1` against `vecs2`
    
    Returns
    ----------
    vcos : (N,) or (0,) array
       Vector cosines.  To get the angles you will need ``np.arccos``

    Notes
    --------
    The vector cosine will be the same as the correlation only if all
    the input vectors have zero mean.
    """
    vecs1 = np.asarray(vecs1)
    vecs2 = np.asarray(vecs2)
    lens1 = np.sqrt(np.sum(vecs1**2, axis=-1))
    lens2 = np.sqrt(np.sum(vecs2**2, axis=-1))
    dots = np.inner(vecs1, vecs2)
    lens = lens1 * lens2
    return dots / lens

def lambert_equal_area_projection_polar(theta, phi):
    r''' Lambert Equal Area Projection from polar sphere to plane

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
    theta : array-like
       theta spherical coordinates
    phi : array-like
       phi spherical coordinates

    Returns
    ---------
    y : (N,2) array
       planar coordinates of points following mapping by Lambert's EAP.
    '''

    return 2 * np.repeat(np.sin(theta/2),2).reshape((theta.shape[0],2)) * np.column_stack((np.cos(phi), np.sin(phi)))


def lambert_equal_area_projection_cart(x,y,z):
    r''' Lambert Equal Area Projection from cartesian vector to plane

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
    ------------
    x : array-like
       x coordinate in Cartesion space
    y : array-like
       y coordinate in Cartesian space
    z : array-like
       z coordinate

    Returns
    ----------
    y : (N,2) array
       planar coordinates of points following mapping by Lambert's EAP.
    '''

    (r, theta, phi) = cart2sphere(x,y,z)
    return lambert_equal_area_projection_polar(theta, phi)




def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.

    Code modified from the work of Christoph Gohlke link provided here
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

    Parameters
    ------------    
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple

    Returns
    ---------
    matrix: 4x4 numpy array

    Code modified from the work of Christoph Gohlke link provided here
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

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
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)

    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M


def compose_matrix(scale=None, shear=None, angles=None, translate=None,perspective=None):
    """Return 4x4 transformation matrix from sequence of
    transformations.

    Code modified from the work of Christoph Gohlke link provided here
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html

    This is the inverse of the decompose_matrix function.

    Parameters
    -------------    
    scale : vector of 3 scaling factors
    shear : list of shear factors for x-y, x-z, y-z axes
    angles : list of Euler angles about static x, y, z axes
    translate : translation vector along x, y, z axes
    perspective : perspective partition of matrix

    Returns
    ---------
    matrix : 4x4 array
    

    Examples
    ----------
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

    Code modified from the excellent work of Christoph Gohlke link provided here
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    
    Parameters
    ------------
    matrix : array_like
        Non-degenerative homogeneous transformation matrix

    Returns
    ---------
    tuple of:    
        scale : vector of 3 scaling factors
        shear : list of shear factors for x-y, x-z, y-z axes
        angles : list of Euler angles about static x, y, z axes
        translate : translation vector along x, y, z axes
        perspective : perspective partition of matrix

    
    Raise ValueError if matrix is of wrong type or degenerative.

    Examples
    -----------
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
        #angles[0] = math.atan2(row[1, 0], row[1, 1])
        angles[0] = math.atan2(-row[2, 1], row[1, 1])
        angles[2] = 0.0

    return scale, shear, angles, translate, perspective

def vector_norm(data, axis=None, out=None):
    """Return length, i.e. euclidean norm, of ndarray along axis.

    Examples
    ----------
    
    >>> import numpy
    >>> v = numpy.random.random(3)
    >>> n = vector_norm(v)
    >>> numpy.allclose(n, numpy.linalg.norm(v))
    True
    >>> v = numpy.random.rand(6, 5, 3)
    >>> n = vector_norm(v, axis=-1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=2)))
    True
    >>> n = vector_norm(v, axis=1)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> v = numpy.random.rand(5, 4, 3)
    >>> n = numpy.empty((5, 3), dtype=numpy.float64)
    >>> vector_norm(v, axis=1, out=n)
    >>> numpy.allclose(n, numpy.sqrt(numpy.sum(v*v, axis=1)))
    True
    >>> vector_norm([])
    0.0
    >>> vector_norm([1.0])
    1.0

    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


        
def circumradius(a, b, c):
    ''' a, b and c are 3-dimensional vectors which are the vertices of a
    triangle. The function returns the circumradius of the triangle, i.e
    the radius of the smallest circle that can contain the triangle. In
    the degenerate case when the 3 points are collinear it returns
    half the distance between the furthest apart points.

    Parameters
    ----------
    a, b, c : (3,) arraylike
       the three vertices of the triangle
    
    Returns
    -------
    circumradius : float
        the desired circumradius
    '''
    x = a-c
    xx = np.linalg.norm(x)**2
    y = b-c
    yy = np.linalg.norm(y)**2
    z = np.cross(x,y)
    # test for collinearity
    if np.linalg.norm(z) == 0:
        return np.sqrt(np.max(np.dot(x,x), np.dot(y,y), np.dot(a-b,a-b)))/2.
    else:
        m = np.vstack((x,y,z))
        w = np.dot(np.linalg.inv(m.T),np.array([xx/2.,yy/2.,0]))
        return np.linalg.norm(w)/2.
