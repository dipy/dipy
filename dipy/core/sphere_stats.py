""" Statistics on spheres
"""

import numpy as np
import dipy.core.geometry as geometry
from itertools import permutations


def random_uniform_on_sphere(n=1, coords='xyz'):
    r"""Random unit vectors from a uniform distribution on the sphere.

    Parameters
    ----------
    n : int
        Number of random vectors
    coords : {'xyz', 'radians', 'degrees'}
        'xyz' for cartesian form
        'radians' for spherical form in rads
        'degrees' for spherical form in degrees

    Notes
    -----
    The uniform distribution on the sphere, parameterized by spherical
    coordinates $(\theta, \phi)$, should verify $\phi\sim U[0,2\pi]$, while
    $z=\cos(\theta)\sim U[-1,1]$.

    References
    ----------
    .. [1] https://mathworld.wolfram.com/SpherePointPicking.html.

    Returns
    -------
    X : array, shape (n,3) if coords='xyz' or shape (n,2) otherwise
        Uniformly distributed vectors on the unit sphere.

    Examples
    --------
    >>> from dipy.core.sphere_stats import random_uniform_on_sphere
    >>> X = random_uniform_on_sphere(4, 'radians')
    >>> X.shape == (4, 2)
    True
    >>> X = random_uniform_on_sphere(4, 'xyz')
    >>> X.shape == (4, 3)
    True
    """
    rng = np.random.default_rng()
    z = rng.uniform(-1, 1, n)
    theta = np.arccos(z)
    phi = rng.uniform(0, 2*np.pi, n)
    if coords == 'xyz':
        r = np.ones(n)
        return np.vstack(geometry.sphere2cart(r, theta, phi)).T
    angles = np.vstack((theta, phi)).T
    if coords == 'radians':
        return angles
    if coords == 'degrees':
        return np.rad2deg(angles)


def eigenstats(points, alpha=0.05):
    r"""Principal direction and confidence ellipse

    Implements equations in section 6.3.1(ii) of Fisher, Lewis and
    Embleton, supplemented by equations in section 3.2.5.

    Parameters
    ----------
    points : array_like (N,3)
        array of points on the sphere of radius 1 in $\mathbb{R}^3$
    alpha : real or None
        1 minus the coverage for the confidence ellipsoid, e.g. 0.05 for 95%
        coverage.

    Returns
    -------
    centre : vector (3,)
        centre of ellipsoid
    b1 : vector (2,)
        lengths of semi-axes of ellipsoid
    """
    n = points.shape[0]
    # the number of points

    rad2deg = 180/np.pi
    # scale angles from radians to degrees

    # there is a problem with averaging and axis data.
    """
    centroid = np.sum(points, axis=0)/n
    normed_centroid = geometry.normalized_vector(centroid)
    x,y,z = normed_centroid
    #coordinates of normed centroid
    polar_centroid = np.array(geometry.cart2sphere(x,y,z))*rad2deg
    """

    cross = np.dot(points.T, points)/n
    # cross-covariance of points

    evals, evecs = np.linalg.eigh(cross)
    # eigen decomposition assuming that cross is symmetric

    order = np.argsort(evals)
    # eigenvalues don't necessarily come in an particular order?

    tau = evals[order]
    # the ordered eigenvalues

    h = evecs[:, order]
    # the eigenvectors in corresponding order

    h[:, 2] = h[:, 2]*np.sign(h[2, 2])
    # map the first principal direction into upper hemisphere

    centre = np.array(geometry.cart2sphere(*h[:, 2]))[1:]*rad2deg
    # the spherical coordinates of the first principal direction

    e = np.zeros((2, 2))

    p0 = np.dot(points, h[:, 0])
    p1 = np.dot(points, h[:, 1])
    p2 = np.dot(points, h[:, 2])
    # the principal coordinates of the points

    e[0, 0] = np.sum((p0**2)*(p2**2))/(n*(tau[0]-tau[2])**2)
    e[1, 1] = np.sum((p1**2)*(p2**2))/(n*(tau[1]-tau[2])**2)
    e[0, 1] = np.sum((p0*p1*(p2**2))/(n*(tau[0]-tau[2])*(tau[1]-tau[2])))
    e[1, 0] = e[0, 1]
    # e is a 2x2 helper matrix

    d = -2*np.log(alpha)/n
    s, w = np.linalg.eig(e)
    g = np.sqrt(d*s)
    b1 = np.arcsin(g)*rad2deg
    # b1 are the estimated 100*(1-alpha)% confidence ellipsoid semi-axes
    # in degrees

    return centre, b1


    # # b2 is equivalent to b1 above
    #
    # # try to invert e and calculate vector b the standard errors of
    # # centre - these are forced to a mixture of NaN and/or 0 in singular cases
    # b2 = np.array([np.NaN,np.NaN])
    # if np.abs(np.linalg.det(e)) < 10**-20:
    #     b2 = np.array([0,np.NaN])
    # else:
    #     try:
    #         f = np.linalg.inv(e)
    #     except np.linalg.LigAlgError:
    #         b2 = np.array([np.NaN, np.NaN])
    #     else:
    #         t, y = np.linalg.eig(f)
    #         d = -2*np.log(alpha)/n
    #         g = np.sqrt(d/t)
    #         b2= np.arcsin(g)*rad2deg



def compare_orientation_sets(S, T):
    r"""Computes the mean cosine distance of the best match between
    points of two sets of vectors S and T (angular similarity)

    Parameters
    ----------
    S : array, shape (m,d)
        First set of vectors.
    T : array, shape (n,d)
        Second set of vectors.

    Returns
    -------
    max_mean_cosine : float
        Maximum mean cosine distance.

    Examples
    --------
    >>> from dipy.core.sphere_stats import compare_orientation_sets
    >>> S=np.array([[1,0,0],[0,1,0],[0,0,1]])
    >>> T=np.array([[1,0,0],[0,0,1]])
    >>> compare_orientation_sets(S,T)
    1.0
    >>> T=np.array([[0,1,0],[1,0,0],[0,0,1]])
    >>> S=np.array([[1,0,0],[0,0,1]])
    >>> compare_orientation_sets(S,T)
    1.0
    >>> from dipy.core.sphere_stats import compare_orientation_sets
    >>> S=np.array([[-1,0,0],[0,1,0],[0,0,1]])
    >>> T=np.array([[1,0,0],[0,0,-1]])
    >>> compare_orientation_sets(S,T)
    1.0

    """

    m = len(S)
    n = len(T)
    if m < n:
        A = S.copy()
        a = m
        S = T
        T = A
        n = a

    v = [np.sum([np.abs(np.dot(p[i], T[i])) for i in range(n)])
         for p in permutations(S, n)]
    return np.max(v)/float(n)
    # return np.max(v)*float(n)/float(m)


def angular_similarity(S, T):
    r"""Computes the cosine distance of the best match between
    points of two sets of vectors S and T

    Parameters
    ----------
    S : array, shape (m,d)
    T : array, shape (n,d)

    Returns
    -------
    max_cosine_distance:float

    Examples
    --------
    >>> import numpy as np
    >>> from dipy.core.sphere_stats import angular_similarity
    >>> S=np.array([[1,0,0],[0,1,0],[0,0,1]])
    >>> T=np.array([[1,0,0],[0,0,1]])
    >>> angular_similarity(S,T)
    2.0
    >>> T=np.array([[0,1,0],[1,0,0],[0,0,1]])
    >>> S=np.array([[1,0,0],[0,0,1]])
    >>> angular_similarity(S,T)
    2.0
    >>> S=np.array([[-1,0,0],[0,1,0],[0,0,1]])
    >>> T=np.array([[1,0,0],[0,0,-1]])
    >>> angular_similarity(S,T)
    2.0
    >>> T=np.array([[0,1,0],[1,0,0],[0,0,1]])
    >>> S=np.array([[1,0,0],[0,1,0],[0,0,1]])
    >>> angular_similarity(S,T)
    3.0
    >>> S=np.array([[0,1,0],[1,0,0],[0,0,1]])
    >>> T=np.array([[1,0,0],[0,np.sqrt(2)/2.,np.sqrt(2)/2.],[0,0,1]])
    >>> angular_similarity(S,T)
    2.7071067811865475
    >>> S=np.array([[0,1,0],[1,0,0],[0,0,1]])
    >>> T=np.array([[1,0,0]])
    >>> angular_similarity(S,T)
    1.0
    >>> S=np.array([[0,1,0],[1,0,0]])
    >>> T=np.array([[0,0,1]])
    >>> angular_similarity(S,T)
    0.0
    >>> S=np.array([[0,1,0],[1,0,0]])
    >>> T=np.array([[0,np.sqrt(2)/2.,np.sqrt(2)/2.]])

    Now we use ``print`` to reduce the precision of of the printed output
    (so the doctests don't detect unimportant differences)

    >>> print('%.12f' % angular_similarity(S,T))
    0.707106781187
    >>> S=np.array([[0,1,0]])
    >>> T=np.array([[0,np.sqrt(2)/2.,np.sqrt(2)/2.]])
    >>> print('%.12f' % angular_similarity(S,T))
    0.707106781187
    >>> S=np.array([[0,1,0],[0,0,1]])
    >>> T=np.array([[0,np.sqrt(2)/2.,np.sqrt(2)/2.]])
    >>> print('%.12f' % angular_similarity(S,T))
    0.707106781187
    """
    m = len(S)
    n = len(T)
    if m < n:
        A = S.copy()
        a = m
        S = T
        T = A
        n = a

    """
    v=[]
    for p in permutations(S,n):
        angles=[]
        for i in range(n):
            angles.append(np.abs(np.dot(p[i],T[i])))
        v.append(np.sum(angles))
    print(v)
    """
    v = [np.sum([np.abs(np.dot(p[i], T[i])) for i in range(n)])
         for p in permutations(S, n)]

    return float(np.max(v))  # *float(n)/float(m)
