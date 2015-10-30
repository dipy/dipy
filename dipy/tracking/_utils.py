from __future__ import division, print_function, absolute_import
"""This is a helper module for dipy.tracking.utils"""

from warnings import warn
import numpy as np


def _voxel_size_deprecated():
    m = DeprecationWarning('the voxel_size argument to this function is '
                           'deprecated, use the affine argument instead')
    warn(m)


def _mapping_to_voxel(affine, voxel_size):
    """Inverts affine and returns a mapping so voxel coordinates. This
    function is an implementation detail and only meant to be used with
    ``_to_voxel_coordinates``.

    Parameters
    ----------
    affine : array_like (4, 4)
        The mapping from voxel indices, [i, j, k], to real world coordinates.
        The inverse of this mapping is used unless `affine` is None.
    voxel_size : array_like (3,)
        Used to support deprecated trackvis space.

    Return
    ------
    lin_T : array (3, 3)
        Transpose of the linear part of the mapping to voxel space, (ie
        ``inv(affine)[:3, :3].T``)
    offset : array or scaler
        Offset part of the mapping (ie, ``inv(affine)[:3, 3]``) + ``.5``. The
        half voxel shift is so that truncating the result of this mapping
        will give the correct integer voxel coordinate.

    Raises
    ------
    ValueError
        If both affine and voxel_size are None.

    """
    if affine is not None:
        affine = np.array(affine, dtype=float)
        inv_affine = np.linalg.inv(affine)
        lin_T = inv_affine[:3, :3].T.copy()
        offset = inv_affine[:3, 3] + .5
    elif voxel_size is not None:
        _voxel_size_deprecated()
        voxel_size = np.asarray(voxel_size, dtype=float)
        lin_T = np.diag(1. / voxel_size)
        offset = 0.
    else:
        raise ValueError("no affine specified")
    return lin_T, offset


def _to_voxel_coordinates(streamline, lin_T, offset):
    """Applies a mapping from streamline coordinates to voxel_coordinates,
    raises an error for negative voxel values."""
    inds = np.dot(streamline, lin_T)
    inds += offset
    if inds.min().round(decimals=6) < 0:
        raise IndexError('streamline has points that map to negative voxel'
                         ' indices')
    return inds.astype(int)


