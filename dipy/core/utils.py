''' Utility functions for algebra etc '''

import numpy as np


def matlab_sph2cart(azimuth, zenith, r=1.0):
    ''' Cartesian 3D coordinates for angles `azimuth` and `zenith`

    Using rather unusual Matlab convention of the zenith angle taken as
    rotation from the y axis towards the z axis.

    As usual, we define the sphere as having a center at Cartesian
    coordinates 0,0,0.  Imagining the sphere as a globe viewed from the
    front, x, y and z axes are oriented south->north, east->west and
    posterior -> anterior.  The `azimuth` angle is counter-clockwise
    rotation around the z axis (viewed from anterior, positive z),
    towards positive y.  Imagine we rotate the y axis to Y' with the
    azimuth rotation.  Then the zenith rotation (in this convention) is
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
