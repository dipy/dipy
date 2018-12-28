""" Utilities to generate affine transformation matrices. """
# -*- coding: utf-8 -*-

import numpy as np


def generate_unit_determinant_matrix():
    """Generate a unit determinant matrix.

    Parameters
    ----------
    None.

    Returns
    -------
    affine : ndarray (4, 4)
        A unit determinant matrix.

    """

    # Generate a random matrix
    mat = np.random.random((4, 4))

    # Check whether the determinant is zero
    det = np.linalg.det(mat)
    while det == 0:
        mat = np.random.random((4, 4))
        det = np.linalg.det(mat)

    # Divide every element of the matrix by |det A|^{1/n}
    n = np.linalg.matrix_rank(mat)
    divisor = np.abs(det) / n
    mat = np.divide(mat, divisor)

    # If the determinant is negative, flip the sign of the first row
    det = np.linalg.det(mat)
    if det < 0:
        mat = np.concatenate((np.negative(mat[0, :])[np.newaxis, :],
                              mat[1:, :]), axis=0)

    return mat


def generate_random_affine():
    """Generate a random affine transformation matrix.

    Parameters
    ----------
    None.

    Returns
    -------
    affine : ndarray (4, 4)
        A random affine transformation matrix.

    """

    translation_matrix = rand_translation_matrix()
    rotation_matrix = rand_rotation_matrix()
    shear_matrix = rand_shear_matrix()
    affine = np.dot(translation_matrix, rotation_matrix)
    affine = np.dot(affine, shear_matrix)

    return affine


def rand_translation_matrix():
    """Create a random translation matrix.

    """

    tx, ty, tz = np.random.random(size=(3,))

    translation_matrix = np.identity(4)
    translation_matrix[:3, 3] = (tx, ty, tz)

    return translation_matrix


def rand_rotation_matrix():
    """Create a random rotation matrix.

    """

    theta_x, theta_y, theta_z = np.random.uniform(0, 2.0*np.pi, size=(3,))

    rot_x = np.array(((1, 0, 0, 0),
                      (0, np.cos(theta_x), -np.sin(theta_x), 0),
                      (0, np.sin(theta_x), np.cos(theta_x), 0),
                      (0, 0, 0, 1)))
    rot_y = np.array(((np.cos(theta_y), 0, np.sin(theta_y), 0),
                      (0, 1, 0, 0),
                      (-np.sin(theta_y), 0, np.cos(theta_y), 0),
                      (0, 0, 0, 1)))
    rot_z = np.array(((np.cos(theta_z), -np.sin(theta_z), 0, 0),
                      (np.sin(theta_z), np.cos(theta_z), 0, 0),
                      (0, 0, 1, 0),
                      (0, 0, 0, 1)))

    rotation_matrix = np.dot(rot_x, np.dot(rot_y, rot_z))

    return rotation_matrix


def rand_shear_matrix():
    """Create a random shear matrix.

    """

    shear_xy, shear_xz, shear_yx, shear_yz, shear_zx, shear_zy = \
        np.random.random(size=(6,))

    shear_matrix = np.array(((1, shear_yx, shear_zx, 0),
                             (shear_xy, 1, shear_zy, 0),
                             (shear_xz, shear_yz, 0, 1),
                             (0, 0, 0, 1)))

    return shear_matrix
