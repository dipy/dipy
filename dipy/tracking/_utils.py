from __future__ import division, print_function, absolute_import
"""This is a helper module for dipy.tracking.utils"""

from functools import wraps
from warnings import warn
import numpy as np

from ..utils.six.moves import xrange


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
    if inds.min() < 0:
        raise IndexError('streamline has points that map to negative voxel'
                         ' indices')
    return inds.astype(int)


def transform_sl(sl, affine=None):
    """
    Helper function that moves and generates the streamline. Thin wrapper
    around move_streamlines

    Parameters
    ----------
    sl : list
        A list of streamline coordinates

    affine : 4 by 4 array
        Affine mapping from fibers to data
    """
    if affine is None:
        affine = np.eye(4)
    # Generate these immediately:
    return [s for s in move_streamlines(sl, affine)]


def unique_rows(in_array, dtype='f4'):
    """
    This (quickly) finds the unique rows in an array

    Parameters
    ----------
    in_array: ndarray
        The array for which the unique rows should be found

    dtype: str, optional
        This determines the intermediate representation used for the
        values. Should at least preserve the values of the input array.

    Returns
    -------
    u_return: ndarray
       Array with the unique rows of the original array.

    """
    x = np.array([tuple(in_array.T[:,i]) for i in
                  xrange(in_array.shape[0])],
        dtype=(''.join(['%s,'%dtype]* in_array.shape[-1])[:-1]))

    u,i = np.unique(x, return_index=True)
    u_i = x[np.sort(i)]
    u_return = np.empty((in_array.shape[-1],len(u_i)))
    for j in xrange(len(u_i)):
        u_return[:,j] = np.array([x for x in u_i[j]])

    # Return back the same dtype as you originally had:
    return u_return.T.astype(in_array.dtype)


def _with_initialize(generator):
    """Allows one to write a generator with initialization code.

    All code up to the first yield is run as soon as the generator function is
    called and the first yield value is ignored.
    """
    @wraps(generator)
    def helper(*args, **kwargs):
        gen = generator(*args, **kwargs)
        next(gen)
        return gen

    return helper


@_with_initialize
def move_streamlines(streamlines, output_space, input_space=None):
    """Applies a linear transformation, given by affine, to streamlines.

    Parameters
    ----------
    streamlines : sequence
        A set of streamlines to be transformed.
    output_space : array (4, 4)
        An affine matrix describing the target space to which the streamlines
        will be transformed.
    input_space : array (4, 4), optional
        An affine matrix describing the current space of the streamlines, if no
        ``input_space`` is specified, it's assumed the streamlines are in the
        reference space. The reference space is the same as the space
        associated with the affine matrix ``np.eye(4)``.

    Returns
    -------
    streamlines : generator
        A sequence of transformed streamlines.

    """
    if input_space is None:
        affine = output_space
    else:
        inv = np.linalg.inv(input_space)
        affine = np.dot(output_space, inv)

    lin_T = affine[:3, :3].T.copy()
    offset = affine[:3, 3].copy()
    yield
    # End of initialization

    for sl in streamlines:
        yield np.dot(sl, lin_T) + offset
