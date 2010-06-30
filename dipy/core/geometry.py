''' Utility functions for algebra etc '''

import math

import numpy as np
import numpy.linalg as npl


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

    Parameters
    ----------
    r : array-like
       radius
    theta : array-like
       inclination or polar angle
    phi : array-like
       azimuth angle

    Returns
    -------
    x : array
       x coordinate(s) in Cartesion space
    y : array
       y coordinate(s) in Cartesian space
    z : array
       z coordinate

    Notes
    -----
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
    ''' Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    See doc for ``sphere2cart`` for angle conventions and derivation
    of the formulae.

    0 >= `theta` >= pi and 0 >= `phi` >= 2*pi

    Parameters
    ----------
    x : array-like
       x coordinate in Cartesion space
    y : array-like
       y coordinate in Cartesian space
    z : array-like
       z coordinate

    Returns
    -------
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
    ''' Return vector divided by Euclidean (L2) norm

    See :term:`unit vector` and :term:`Euclidean norm`

    Parameters
    ----------
    vec : array-like shape (3,)

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
    ----------
    vec : array-like shape (3,)

    Returns
    -------
    norm : scalar

    Examples
    --------
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
    ----------
    B : (3,3) array-like
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
    ----------
    pts1 : (N,R) or (R,) array-like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D)
    pts2 : (N,R) or (R,) array-like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D).  It should be
       possible to broadcast `pts1` against `pts2`
    radius : None or float, optional
       Radius of sphere.  Default is to work out radius from points
    check_radius : bool, optional
       If True, check if the points are on the sphere surface - i.e
       check if the vector lengths in `pts1` and `pts2` are close to
       `radius`.  Default is True.
       
    Returns
    -------
    d : (N,) or (0,) array
       Distances between corresponding points in `pts1` and `pts2`
       across the spherical surface

    See also
    --------
    cart_distance : cartesian distance between points
    vector_cosine : cosine of angle between vectors
    
    Examples
    --------
    >>> print '%.4f' % sphere_distance([0,1],[1,0])
    1.5708
    >>> print '%.4f' % sphere_distance([0,3],[3,0])
    4.7124
    """
    # Get angle with vector cosine
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
    dots = np.inner(pts1, pts2)
    lens = lens1 * lens2
    angle_cos = np.arccos(dots / lens)
    return angle_cos * radius


def cart_distance(pts1, pts2):
    ''' Cartesian distance between `pts1` and `pts2`

    If either of `pts1` or 'pts2` is 2D, then we take the first
    dimension to index points, and the second indexes coordinate.  More
    generally, we take the last dimension to be the coordinate
    dimension. 
    
    Parameters
    ----------
    pts1 : (N,R) or (R,) array-like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D)
    pts2 : (N,R) or (R,) array-like
       where N is the number of points and R is the number of
       coordinates defining a point (``R==3`` for 3D).  It should be
       possible to broadcast `pts1` against `pts2`

    Returns
    -------
    d : (N,) or (0,) array
       Cartesian distances between corresponding points in `pts1` and
       `pts2`

    See also
    --------
    sphere_distance : distance between points on sphere surface

    Examples
    --------
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
    ----------
    vecs1 : (N, R) or (R,) array-like
       N vectors (as rows) or single vector.  Vectors have R elements.
    vecs1 : (N, R) or (R,) array-like
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
    lens1 = np.sqrt(np.sum(vecs1**2, axis=-1))
    lens2 = np.sqrt(np.sum(vecs2**2, axis=-1))
    dots = np.inner(vecs1, vecs2)
    lens = lens1 * lens2
    return dots / lens
