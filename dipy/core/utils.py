''' Utility functions for algebra etc '''

import numpy as np


def sphere2cart(theta, phi, r=1.0):
    ''' Spherical to Cartesian coordinates

    This is the standard physics convention where `theta` is the
    inclination (polar) angle, and `phi` is the azimuth angle.

    Imagine a sphere with center (0,0,0).  Orient it with the z axis
    running south->north, the y axis running east-west and the x axis
    from anterior to posterior.  `theta` (the inclination angle) is the
    angle to rotate from the z-axis around the x-axis.

    We have deliberately named this function ``sphere2cart`` rather than
    ``sph2cart`` to distinguish it from the Matlab function of that
    name, because the Matlab function uses an odd convention for the
    angles that we did not want to replicate.
    
    Parameters
    ----------
    theta : array-like
       inclination or polar angle
    phi : array-like
       azimuth angle
    r : array-like
       radius

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
    the hypotenuse of the projection onto the XY plane - say P, where P
    == r*sin(theta). Now x / P == cos(phi) -> x == r * sin(theta) *
    cos(phi) and so on.
    '''
    sin_theta = np.sin(theta)
    x = r * np.cos(phi) * sin_theta
    y = r * np.sin(phi) * sin_theta
    z = np.cos(theta)
    return x, y, z


def cart2sphere(x, y, z):
    ''' Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    See doc for ``sphere2cart`` for angle conventions and derivation
    of the formulae.

    Parameters
    ----------
    x : scalar or (N,) array-like
       x coordinate in Cartesion space
    y : scalar or (N,) array-like
       y coordinate in Cartesian space
    z : scalar or (N,) array-like
       z coordinate

    Returns
    -------
    theta : (N,) array
       inclination (polar) angle
    phi : (N,) array
       azimuth angle
    r : (N,) array
       radius
    '''
    r = np.sqrt(x*x + y*y + z*z)
    theta = np.arccos(z/r)
    phi = np.arctan2(y, x)
    return theta, phi, r


def matlab_sph2cart(azimuth, zenith, r=1.0):
    ''' Cartesian 3D coordinates for angles `azimuth` and `zenith`

    Using rather unusual Matlab convention of the zenith angle taken as
    rotation from the y axis towards the z axis.

    As usual, we define the sphere as having a center at Cartesian
    coordinates 0,0,0.  Imagining the sphere as a globe viewed from the
    front, x, y and z axes are oriented posterior->anterior, east->west
    and south->north.  The `azimuth` angle is counter-clockwise rotation
    around the z axis (viewed from anterior, positive x), towards
    positive y.  Imagine we rotate the y axis to Y' with the azimuth
    rotation.  Then the zenith rotation (in this convention) is
    clock-wise (from positive Y') rotation around Y' towards positive z.

    The azimuth angle is therefore the angle between the x axis and the
    projection of the vector onto the x-y plane.
    
    Parameters
    ----------
    azimuth : (N,) array-like
       azimuth angle
    zenith : (N,) array-like
       zenith angle
    r : float or (N, array-like), optional
       radius.  Default is 1.0

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
    There are different conventions for the order in which the azimuth
    angles and the zenith angles are specified, and the greek letters
    corresponding to `azimuth` and `zenith`; see:
    
    See: http://mathworld.wolfram.com/SphericalCoordinates.html

    It's unpleasant, but the zenith angle can also be from the z axis,
    or from the Y' axis.  The latter appears to be rare, but it's the
    convention used in Matlab.
    
    Here we follow the conventions of Matlab.

    Derivations of the formulae are simple. Consider a vector x, y, z of
    length r (norm of x, y, z).  The zenith angle (in this convention)
    can be found from sin(zenith) = z / r -> z = r * sin(zenith).  This
    gives the hypotenuse of the projection onto the XY plane - say P =
    r*cos(zenith). Now x / P = cos(azimuth) -> x = r * cos(zenith) *
    cos(azimuth).
    '''
    cos_zen = np.cos(zenith)
    x = r * np.cos(azimuth) * cos_zen
    y = r * np.sin(azimuth) * cos_zen
    z = r * np.sin(zenith)
    return x, y, z


def matlab_cart2sph(x, y, z):
    ''' Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    See doc for ``matlab_sph2cart`` for angle conventions and derivation
    of the formulae.

    Parameters
    ----------
    x : scalar or (N,) array-like
       x coordinate in Cartesion space
    y : scalar or (N,) array-like
       y coordinate in Cartesian space
    z : scalar or (N,) array-like
       z coordinate

    Returns
    -------
    azimuth : (N,) array
       azimuth angle
    zenith : (N,) array
       zenith angle
    r : (N,) array
       radius

    Notes
    -----
    in this convention), when `zenith` is pi/2 or 3*pi/4,
    then we are on the z axis, and `azimuth` is undefined; we
    arbitrarily set it to 0.
    '''
    r = np.sqrt(x*x + y*y + z*z)
    zenith = np.arcsin(z/r)
    azimuth = np.arctan2(y, x)
    return azimuth, zenith, r
