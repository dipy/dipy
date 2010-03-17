''' Utility functions for algebra etc '''

import numpy as np


def sph2cart(azimuth, zenith, r=1.0):
    ''' Return Cartesian 3D coordinates for angles `theta` and `phi`

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
    See: http://mathworld.wolfram.com/SphericalCoordinates.html
    '''
    sin_zen = np.sin(zenith)
    x = r * np.cos(azimuth) * sin_zen
    y = r * np.sin(azimuth) * sin_zen
    z = r * np.cos(zenith)
    return x, y, z


def cart2sph(x, y, z):
    ''' Return angles for Cartesian 3D coordinates `x`, `y`, and `z`

    Parameters
    ----------
    x : (N,) array-like
       x coordinate in Cartesion space
    y : (N,) array-like
       y coordinate in Cartesian space
    z : (N,) array-like
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
    See: http://mathworld.wolfram.com/SphericalCoordinates.html
    '''
    r = np.sqrt(x*x + y*y + z*z)
    azimuth = np.arctan2(y, x)
    zenith = np.arccos(z/r)
    return azimuth, zenith, r
