"""
FORCE: Fast Orientation Reconstruction and Compartment Estimation

This module provides signal simulation for dictionary-based diffusion MRI
reconstruction using multi-compartment tissue models.
"""

import numpy as np

from dipy.sims.voxel import all_tensor_evecs


def bingham_to_sf(f0, k1, k2, major_axis, minor_axis, vertices):
    """
    Evaluate Bingham distribution on a sphere.

    The Bingham distribution models fiber orientation dispersion
    in diffusion MRI.

    Parameters
    ----------
    f0 : float
        Maximum amplitude of the distribution.
    k1 : float
        Concentration parameter along major axis.
    k2 : float
        Concentration parameter along minor axis.
    major_axis : ndarray (3,)
        Major axis of the distribution.
    minor_axis : ndarray (3,)
        Minor axis of the distribution.
    vertices : ndarray (N, 3)
        Unit sphere directions for evaluation.

    Returns
    -------
    sf : ndarray (N,)
        Spherical function values at each vertex.
    """
    sf = f0 * np.exp(
        -k1 * vertices.dot(major_axis) ** 2
        - k2 * vertices.dot(minor_axis) ** 2
    )
    return sf.T


def bingham_dictionary(target_sphere, odi_list):
    """
    Generate dictionary of Bingham spherical functions.

    Pre-computes Bingham distributions for all sphere directions
    and ODI values for efficient signal simulation.

    Parameters
    ----------
    target_sphere : ndarray (N, 3)
        Unit sphere vertices.
    odi_list : ndarray
        List of orientation dispersion index values.

    Returns
    -------
    bingham_sf : dict
        Nested dictionary mapping (vertex_index, odi) to
        spherical function values.
    """
    bingham_sf = {}
    for i in range(len(target_sphere)):
        vertex_key = tuple(target_sphere[i])
        bingham_sf[i] = {}

        for odi in odi_list:
            k = 1 / np.tan(np.pi / 2 * odi)
            evecs = all_tensor_evecs(vertex_key)
            major_axis, minor_axis = evecs[:, 1], evecs[:, 2]

            bingham_sf[i][odi] = bingham_to_sf(
                1, k, k, major_axis, minor_axis, target_sphere
            )

    return bingham_sf
