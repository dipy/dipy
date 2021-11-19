""" Create example sphere points """

import numpy as np


def _make_pts():
    """ Make points around sphere quadrants """
    thetas = np.arange(1, 4) * np.pi/4
    phis = np.arange(8) * np.pi/4
    north_pole = (0, 0, 1)
    south_pole = (0, 0, -1)
    points = [north_pole, south_pole]
    for theta in thetas:
        for phi in phis:
            x = np.sin(theta) * np.cos(phi)
            y = np.sin(theta) * np.sin(phi)
            z = np.cos(theta)
            points.append((x, y, z))
    return np.array(points)


sphere_points = _make_pts()
