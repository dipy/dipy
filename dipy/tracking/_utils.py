"""This is a helper module for dipy.tracking.utils."""

import numpy as np


def _mapping_to_voxel(affine):
    """Invert affine and return a mapping to voxel coordinates.

    This function is an implementation detail and only meant to be
    used with ``_to_voxel_coordinates``.

    Parameters
    ----------
    affine : array-like, shape (4, 4)
        The mapping from voxel indices, [i, j, k], to real world
        coordinates. The inverse of this mapping is used unless
        `affine` is None.

    Returns
    -------
    lin_T : ndarray, shape (3, 3)
        Transpose of the linear part of the mapping to voxel space,
        i.e. ``inv(affine)[:3, :3].T``.
    offset : ndarray, shape (3,)
        Offset part of the mapping, i.e. ``inv(affine)[:3, 3]``
        plus ``.5``. The half voxel shift ensures that truncating
        the result gives the correct integer voxel coordinate.

    Raises
    ------
    ValueError
        If both affine and voxel_size are None.
    """
    if affine is None:
        raise ValueError("no affine specified")

    affine = np.array(affine, dtype=float)
    inv_affine = np.linalg.inv(affine)
    lin_T = inv_affine[:3, :3].T.copy()
    offset = inv_affine[:3, 3] + 0.5

    return lin_T, offset


def _to_voxel_coordinates(streamline, lin_T, offset):
    """Apply a mapping from streamline coordinates to voxel coordinates.

    Raises an error if any voxel indices are negative.

    Parameters
    ----------
    streamline : ndarray, shape (N, 3)
        A single streamline represented as an array of N
        points in 3D space.
    lin_T : ndarray, shape (3, 3)
        Linear transformation matrix (transposed) used to
        map streamline coordinates to voxel space.
    offset : ndarray, shape (3,)
        Offset vector applied after the linear transformation
        to shift coordinates into voxel space.

    Returns
    -------
    inds : ndarray, shape (N, 3)
        Voxel coordinates corresponding to each point of the
        streamline, as integer indices.

    Raises
    ------
    IndexError
        If any of the mapped voxel indices are negative.
    """
    inds = np.dot(streamline, lin_T)
    inds += offset
    if inds.min().round(decimals=6) < 0:
        raise IndexError("streamline has points that map to negative voxel indices")
    return inds.astype(np.intp)
