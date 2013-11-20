from __future__ import division, print_function, absolute_import
"""Various tools related to creating and working with streamlines

Important Note:
---------------
At this time all the tools in this module are using the "trackvis coordinate
system." This coordinate system places the origin at the corner of the first
voxel of an image. The diagonal corner of the image is [x*i, j*y, k*z] where
[x, y, z] is the voxel size and [i, j, k] is the dimention of the image. A
2d example is shown below where the dimention is [3, 3] and the voxel size is
[1, 3]:

Trackvis:
A------------
|   |   |   |
| C |   |   |
|   |   |   |
----B--------
|   |   |   |
|   |   |   |
|   |   |   |
-------------
|   |   |   |
|   |   |   |
|   |   |   |
------------D

A = [0, 0]
B = [1, 3]
C = [.5, 1.5]
D = [3, 9]

Any streamlines created using a differnt coordinate system should be moved to
this coordinate system before any of the functions from this module are used.
"""
"""
This module uses the trackvis coordinate system, for more information about
this coordinate system please see dipy.tracking.utils
The following modules also use this coordinate system:
dipy.tracking.utils
dipy.tracking.integration
dipy.reconst.interpolate
"""

from collections import defaultdict
from ..utils.six.moves import xrange

import numpy as np
from numpy import (asarray, array, atleast_3d, ceil, concatenate, empty,
                   eye, mgrid, sqrt, zeros, linalg, diag, dot)
from dipy.io.bvectxt import ornt_mapping


def _rmi(index, dims):
    """An alternate implementation of numpy.ravel_multi_index for older
    versions of numpy.

    """
    index = np.asarray(index)
    dims = np.asarray(dims)
    if index.ndim > 2:
        raise ValueError("Index should be 1 or 2-D")
    elif index.ndim == 2:
        index = index.T
    if (index >= dims).any():
        raise ValueError("Index exceeds dimensions")
    strides = np.r_[dims[:0:-1].cumprod()[::-1], 1]
    return (strides * index).sum(-1)


try:
    from numpy import ravel_multi_index
except ImportError:
    ravel_multi_index = _rmi


def _mapping_to_voxel(affine, voxel_size):
    """ Returns the best nearest neighbor mapping from real world coordinates
    to voxel indices based on available information.

    Parameters
    ----------
    affine : array_like (4, 4)
        The mapping from voxel indices, [i, j, k], to real world coordinates.
        The inverse of this mapping is used unless `affine` is None.
    voxel_size : array_like (3,)
        If `affine` is not None, `voxel_size` is ignored, otherwise the
        the mapping is assumed to be in trk space. IE the corners of the image
        are ``[0, 0, 0]`` and ``voxel_size * dim``.

    If both are None, then coordinates are taken to be nifti-style voxel
    indices. IE [0, 0, 0] is the center of the first voxel.

    Return
    ------
    lin : array (3, 3)
        Transpose of the linear part of the mapping. (ie ``affine[:3, :3].T``)
    offset : array or scaler
        Offset part of the mapping. (ie, ``affine[:3, 3]``)

    Note
    ----
    This is meant to be used along with `_to_voxel_coordinates`.

    """
    if affine is not None:
        affine = np.array(affine, dtype=float)
        inv_affine = np.linalg.inv(affine)
        lin = inv_affine[:-1, :-1].T
        offset = inv_affine[:-1, -1] + .5
    elif voxel_size is not None:
        voxel_size = np.asarray(voxel_size, dtype=float)
        lin = np.diag(1. / voxel_size)
        offset = 0.
    else:
        lin = np.eye(3)
        offset = .5
    return lin, offset


def _to_voxel_coordinates(streamline, lin, offset):
    """Applies a mapping from streamline coordinates to voxel_coordinates"""
    inds = np.dot(streamline, lin)
    inds += offset
    if inds.min() < 0:
        raise IndexError('streamline has points that map to negative voxel'
                         ' indices')
    return inds.astype(int)


def density_map(streamlines, vol_dims, voxel_size=None, affine=None):
    """Counts the number of unique streamlines that pass though each voxel

    Counts the number of points in each streamline that lie inside each voxel.

    Parameters
    ----------
    streamlines : iterable
        A sequence of arrays, each streamline should a list of points in
        3-space, where (0,0,0) is one corner of the first voxel in image
        volume, voxel_size the diagonal corner of the same voxel and
        voxel_size*vol_dims is the diagonal corner of the image.
    vol_dims : 3 ints
        The shape of the volume to be returned containing the streamlines
        counts
    voxel_size : array_like (3,), optional
        The size of the voxels in the image volume. This is ignored if affine
        is set.
    affine : array_like (4, 4), optional
        The mapping from voxel coordinates to streamline coordinates. If
        neither `affine` or `voxel_size` is set, the streamline values are
        assumed to be in voxel coordinates. IE ``[0, 0, 0]`` is the center of
        the first voxel and the voxel size is ``[1, 1, 1]``.

    Returns
    -------
    image_volume : ndarray, shape=vol_dims
        The number of streamline points in each voxel of volume

    Raises
    ------
    IndexError
        When the points of the streamlines lie outside of the return volume

    Notes
    -----
    A streamline can pass though a voxel even if one of the points of the
    streamline does not lie in the voxel. For example a step from [0,0,0] to
    [0,0,2] passes though [0,0,1]. Consider subsegmenting the streamlines when
    the edges of the voxels are smaller than the steps of the streamlines.

    """
    lin, offset = _mapping_to_voxel(affine, voxel_size)
    counts = zeros(vol_dims, 'int')
    for sl in streamlines:
        inds = _to_voxel_coordinates(sl, lin, offset)
        i, j, k = inds.T
        #this takes advantage of the fact that numpy's += operator only acts
        #once even if there are repeats in inds
        counts[i, j, k] += 1
    return counts

def connectivity_matrix(streamlines, label_volume, voxel_size=None,
                        affine=None, symmetric=False, return_mapping=False,
                        mapping_as_streamlines=False):
    """Counts the streamlines that start and end at each label pair

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    label_volume : ndarray
        An image volume with an integer data type, where the intensities in the
        volume map to anatomical structures.
    voxel_size : array_like (3,), optional
        The size of the voxels in the image volume. This is ignored if affine
        is set.
    affine : array_like (4, 4), optional
        The mapping from voxel coordinates to streamline coordinates. If
        neither `affine` or `voxel_size` is set, the streamline values are
        assumed to be in voxel coordinates. IE ``[0, 0, 0]`` is the center of
        the first voxel and the voxel size is ``[1, 1, 1]``.
    symmetric : bool, False by default
        Symmetric means we don't distinguish between start and end points. If
        symmetric is True, ``matrix[i, j] == matrix[j, i]``.
    return_mapping : bool, False by default
        If True, a mapping is returned which maps matrix indices to
        streamlines.
    mapping_as_streamlines : bool, False by default
        If True voxel indices map to lists of streamline objects. Otherwise
        voxel indices map to lists of integers.

    Returns
    -------
    matrix : ndarray
        The number of connection between each pair of regions in
        `label_volume`.
    mapping : defaultdict(list)
        ``mapping[i, j]`` returns all the streamlines that connect region `i`
        to region `j`. If `symmetric` is True mapping will only have one key
        for each start end pair such that if ``i < j`` mapping will have key
        ``(i, j)`` but not key ``(j, i)``.

    """
    # Error checking on label_volume
    kind = label_volume.dtype.kind
    labels_possitive = ((kind == 'u') or
                        ((kind == 'i') and (label_volume.min() >= 0))
                       )
    valid_label_volume = (labels_possitive and label_volume.ndim == 3)
    if not valid_label_volume:
        raise ValueError("label_volume must be a 3d integer array with"
                         "non-negative label values")

    # If streamlines is an iterators
    if return_mapping and mapping_as_streamlines:
        streamlines = list(streamlines)
    #take the first and last point of each streamline
    endpoints = [sl[0::len(sl)-1] for sl in streamlines]

    # Map the streamlines coordinates to voxel coordinates
    lin, offset = _mapping_to_voxel(affine, voxel_size)
    endpoints = _to_voxel_coordinates(endpoints, lin, offset)

    #get labels for label_volume
    i, j, k = endpoints.T
    endlabels = label_volume[i, j, k]
    if symmetric:
        endlabels.sort(0)
    mx = endlabels.max() + 1
    matrix = ndbincount(endlabels, shape=(mx, mx))
    if symmetric:
        matrix = np.maximum(matrix, matrix.T)

    if return_mapping:
        mapping = defaultdict(list)
        for i, (a, b) in enumerate(endlabels.T):
            mapping[a, b].append(i)

        # Replace each list of indices with the streamlines they index
        if mapping_as_streamlines:
            for key in mapping:
                mapping[key] = [streamlines[i] for i in mapping[key]]

        # Return the mapping matrix and the mapping
        return matrix, mapping
    else:
        return matrix


def ndbincount(x, weights=None, shape=None):
    """Like bincount, but for nd-indicies

    Parameters
    ----------
    x : array_like (N, M)
        M indices to a an Nd-array
    weights : array_like (M,), optional
        Weights associated with indices
    shape : optional
        the shape of the output
    """
    x = np.asarray(x)
    if shape is None:
        shape = x.max(1) + 1

    x = ravel_multi_index(x, shape)
    # out = np.bincount(x, weights, minlength=np.prod(shape))
    # out.shape = shape
    # Use resize to be compatible with numpy < 1.6, minlength new in 1.6
    out = np.bincount(x, weights)
    out.resize(shape)

    return out


def reduce_labels(label_volume):
    """Reduces an array of labels to the integers from 0 to n with smallest
    possible n

    Examples
    --------
    >>> labels = np.array([[1, 3, 9],
    ...                    [1, 3, 8],
    ...                    [1, 3, 7]])
    >>> new_labels, lookup = reduce_labels(labels)
    >>> lookup
    array([1, 3, 7, 8, 9])
    >>> new_labels #doctest: +ELLIPSIS
    array([[0, 1, 4],
           [0, 1, 3],
           [0, 1, 2]]...)
    >>> (lookup[new_labels] == labels).all()
    True
    """
    lookup_table = np.unique(label_volume)
    label_volume = lookup_table.searchsorted(label_volume)
    return label_volume, lookup_table

def length(streamlines):
    """Calculates the lenth of each streamline in a sequence of streamlines

    Sums the lenths of each segment in a streamline to get the length of the
    streamline. Returns a generator.

    Example:
    >>> streamlines = [np.array([[0., 0., 0.],
    ...                          [0., 0., 1.],
    ...                          [3., 4., 1.]]),
    ...                np.array([[0., 0., 0.]])]
    >>> list(length(streamlines))
    [6.0, 0.0]
    """
    for sl in streamlines:
        if len(sl) == 1:
            yield 0.
        else:
            diff = sl[1:] - sl[:-1]
            seglen = sqrt((diff * diff).sum(-1))
            yield seglen.sum()


def streamline_mapping(streamlines, voxel_size=None, affine=None,
                       mapping_as_streamlines=False):
    """Creates a mapping from voxel indices to streamlines

    Returns a dictionary where each key is a 3d voxel index and the associated
    value is a list of the streamlines that pass through that voxel.

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    voxel_size : array_like (3,), optional
        The size of the voxels in the image volume. This is ignored if affine
        is set.
    affine : array_like (4, 4), optional
        The mapping from voxel coordinates to streamline coordinates. If
        neither `affine` or `voxel_size` is set, the streamline values are
        assumed to be in voxel coordinates. IE ``[0, 0, 0]`` is the center of
        the first voxel and the voxel size is ``[1, 1, 1]``.
    mapping_as_streamlines : bool, optional, False by default
        If True voxel indices map to lists of streamline objects. Otherwise
        voxel indices map to lists of integers.

    Returns
    -------
    mapping : defaultdict(list)
        A mapping from voxel indices to the streamlines that pass though that
        voxel.

    Examples
    --------
    >>> streamlines = [np.array([[0., 0., 0.],
    ...                          [1., 1., 1.],
    ...                          [2., 3., 4.]]),
    ...                np.array([[0., 0., 0.],
    ...                          [1., 2., 3.]])]
    >>> mapping = streamline_mapping(streamlines, (1, 1, 1))
    >>> mapping[0, 0, 0]
    [0, 1]
    >>> mapping[1, 1, 1]
    [0]
    >>> mapping[1, 2, 3]
    [1]
    >>> mapping.get((3, 2, 1), 'no streamlines')
    'no streamlines'
    >>> mapping = streamline_mapping(streamlines, (1, 1, 1),
    ...                              mapping_as_streamlines=True)
    >>> mapping[1, 2, 3][0] is streamlines[1]
    True

    """
    lin, offset = _mapping_to_voxel(affine, voxel_size)
    mapping = defaultdict(list)
    if mapping_as_streamlines:
        streamlines = list(streamlines)
    for i, sl in enumerate(streamlines):
        voxel_indices = _to_voxel_coordinates(sl, lin, offset)
        uniq_points = set(tuple(point) for point in voxel_indices)
        for point in uniq_points:
            mapping[point].append(i)
    if mapping_as_streamlines:
        for key in mapping:
            mapping[key] = [streamlines[i] for i in mapping[key]]
    return mapping


def subsegment(streamlines, max_segment_length):
    """Splits the segments of the streamlines into small segments

    Replaces each segment of each of the streamlines with the smallest possible
    number ofequally sized smaller segments such that no segmentment is longer
    than max_segment_length. Among other things, this can useful for getting
    streamline counts on a grid that is smaller than the length of the
    streamline segments.

    Parameters
    ----------
    streamlines : sequence of ndarrays
        The streamlines to be subsegmented.
    max_segment_length : float
        The longest allowable segment length.

    Returns
    -------
    output_streamlines : generator
        A set of streamlines.

    Notes
    -----
    Segments of 0 length are removed. If unchanged

    Examples
    --------
    >>> streamlines = [array([[0,0,0],[2,0,0],[5,0,0]])]
    >>> list(subsegment(streamlines, 3.))
    [array([[ 0.,  0.,  0.],
           [ 2.,  0.,  0.],
           [ 5.,  0.,  0.]])]
    >>> list(subsegment(streamlines, 1))
    [array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 2.,  0.,  0.],
           [ 3.,  0.,  0.],
           [ 4.,  0.,  0.],
           [ 5.,  0.,  0.]])]
    >>> list(subsegment(streamlines, 1.6))
    [array([[ 0. ,  0. ,  0. ],
           [ 1. ,  0. ,  0. ],
           [ 2. ,  0. ,  0. ],
           [ 3.5,  0. ,  0. ],
           [ 5. ,  0. ,  0. ]])]
    """
    for sl in streamlines:
        diff = (sl[1:] - sl[:-1])
        length = sqrt((diff*diff).sum(-1))
        num_segments = ceil(length/max_segment_length).astype('int')

        output_sl = empty((num_segments.sum()+1, 3), 'float')
        output_sl[0] = sl[0]

        count = 1
        for ii in xrange(len(num_segments)):
            ns = num_segments[ii]
            if ns == 1:
                output_sl[count] = sl[ii+1]
                count += 1
            elif ns > 1:
                small_d = diff[ii]/ns
                point = sl[ii]
                for jj in xrange(ns):
                    point = point + small_d
                    output_sl[count] = point
                    count += 1
            elif ns == 0:
                pass
                #repeated point
            else:
                #this should never happen because ns should be a positive int
                assert(ns >= 0)
        yield output_sl

def seeds_from_mask(mask, density=[1, 1, 1], voxel_size=None, affine=None):
    """Takes a binary mask and returns seeds in voxels != 0

    places evanly spaced points in nonzero voxels of mask, spaces the points
    based on density. For example if density is [1, 2, 3], there will be 6
    points in each voxel, at x=.5, y=[.25, .75] and z=[.166, .5, .833].
    density=a is the same as density = [a, a, a]

    Examples
    --------
    >>> mask = zeros((3,3,3), 'bool')
    >>> mask[0,0,0] = 1
    >>> seeds_from_mask(mask, [1,1,1], [1,1,1])
    array([[ 0.5,  0.5,  0.5]])

    >>> seeds_from_mask(mask, [1,2,3], [1,1,1])
    array([[ 0.5       ,  0.25      ,  0.16666667],
           [ 0.5       ,  0.75      ,  0.16666667],
           [ 0.5       ,  0.25      ,  0.5       ],
           [ 0.5       ,  0.75      ,  0.5       ],
           [ 0.5       ,  0.25      ,  0.83333333],
           [ 0.5       ,  0.75      ,  0.83333333]])
    >>> mask[0,1,2] = 1
    >>> seeds_from_mask(mask, [1,1,2], [1.1,1.1,2.5])
    array([[ 0.55 ,  0.55 ,  0.625],
           [ 0.55 ,  0.55 ,  1.875],
           [ 0.55 ,  1.65 ,  5.625],
           [ 0.55 ,  1.65 ,  6.875]])

    """
    mask = np.array(mask, dtype=bool, copy=False, ndmin=3)
    if mask.ndim != 3:
        raise ValueError('mask cannot be more than 3d')
    density = asarray(density, int)
    if density.shape == (1,):
        density = density + [0, 0, 0]
    elif density.shape != (3,):
        raise ValueError("density should be in integer array of shape (3,)")

    # Grid of points between -.5 and .5, centered at 0, with given density
    grid = mgrid[0:density[0], 0:density[1], 0:density[2]]
    grid = grid.T.reshape((-1, 3))
    grid = grid / density
    grid += (.5 / density - .5)

    # Add the grid of points to each voxel in mask
    where = np.argwhere(mask)
    seeds = where[:, np.newaxis, :] + grid[np.newaxis, :, :]
    seeds.shape = (-1, 3)

    # Apply the spacial transform
    if affine is not None:
        # Use affine to move seeds int real world coordinates
        seeds = np.dot(seeds, affine[:3, :3].T)
        seeds += affine[:3, 3]
    elif voxel_size is not None:
        # Use voxel_size to move seeds into trackvis space
        seeds += .5
        seeds *= voxel_size

    return seeds


def target(streamlines, target_mask, voxel_size=None, affine=None):
    """Retain tracks that pass though target_mask

    This function loops over the streamlines and returns streamlines that pass
    though target_mask.

    Parameters
    ----------
    streamlines : iterable
        A squence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    target_mask : array-like
        A mask used as a target
    voxel_size
        Size of the voxels in the target_mask

    Returns
    -------
    streamlines : generator
        A sequence of streamlines that pass though target_mask

    Raises
    ------
    IndexError
        When the points of the streamlines lie outside of the target_mask

    See Also
    --------
    density_map

    """
    ones = np.ones(3.)
    lin, offset = _mapping_to_voxel(affine, voxel_size)
    for sl in streamlines:
        ind = _to_voxel_coordinates(sl, lin, offset)
        i, j, k = ind.T
        try:
            state = target_mask[i, j, k]
        except IndexError:
            volume_size = tuple(voxel_size * target_mask.shape)
            raise IndexError('streamline has values greater than the size of '
                             'the target mask, ' + str(volume_size))
        if state.any():
            yield sl


def merge_streamlines(backward, forward):
    """Merges two sets of streamlines seeded at the same points

    Because the first point of each streamline pair should be the same, only
    one is kept

    Parameters
    ----------
    backward : iterable
        a sequence of streamlines, will be returned in reversed order in the
        result
    forward : iterable
        a sequence of streamlines, will be returned in same order in the result

    Returns
    -------
    streamlines : generator
        generator of merged streamlines

    Examples
    --------
    >>> A = [array([[0,0,0],[1,1,1],[2,2,2]])]
    >>> B = [array([[0,0,0],[-1,-1,-1],[-2,-2,-2]])]
    >>> list(merge_streamlines(A,B))
    [array([[ 2,  2,  2],
           [ 1,  1,  1],
           [ 0,  0,  0],
           [-1, -1, -1],
           [-2, -2, -2]])]
    >>> list(merge_streamlines(B,A))
    [array([[-2, -2, -2],
           [-1, -1, -1],
           [ 0,  0,  0],
           [ 1,  1,  1],
           [ 2,  2,  2]])]
    """
    B = iter(backward)
    F = iter(forward)
    while True:
        yield concatenate((next(B)[:0:-1], next(F)))

def move_streamlines(streamlines, affine):
    """Applies a linear transformation, given by affine, to streamlines

    Parameters
    ----------
    streamlines : sequence
        A set of streamlines to be transformed.
    affine : array (4, 4)
        A linear tranformation to be applied to the streamlines. The last row
        of affine should be [0, 0, 0, 1].

    Returns
    -------
    streamlines : generator
        A sequence of transformed streamlines
    """
    for sl in streamlines:
        yield dot(sl, affine[:3,:3].T) + affine[:3,3]

def reorder_voxels_affine(input_ornt, output_ornt, shape, voxel_size):
    """Calculates a linear tranformation equivelent to chaning voxel order

    Calculates a linear tranformation A such that [a, b, c, 1] = A[x, y, z, 1].
    where [x, y, z] is a point in the coordinate system defined by input_ornt
    and [a, b, c] is the same point in the coordinate system defined by
    output_ornt.

    Parameters
    ----------
    input_ornt : array (n, 2)
        A description of the orientation of a point in n-space. See
        ``nibabel.orientation`` or ``dipy.io.bvectxt`` for more information.
    output_ornt : array (n, 2)
        A description of the orientation of a point in n-space.
    shape : tuple of int
        Shape of the image in the input orientation.
        ``map = ornt_mapping(input_ornt, output_ornt)``
    voxel_size : int
        Voxel size of the image in the input orientation.

    Returns
    -------
    A : array (n+1, n+1)
        Affine matrix of the transformation between input_ornt and output_ornt.

    See Also
    --------
    nibabel.orientation
    dipy.io.bvectxt.orientation_to_string
    dipy.io.bvectxt.orientation_from_string
    """
    map = ornt_mapping(input_ornt, output_ornt)
    if input_ornt.shape != output_ornt.shape:
        raise ValueError("input_ornt and output_ornt must have the same shape")
    affine = eye(len(input_ornt)+1)
    affine[:3] = affine[map[:, 0]]
    corner = asarray(voxel_size) * shape
    affine[:3, 3] = (map[:, 1] < 0) * corner[map[:, 0]]
    #multiply the rows of affine to get right sign
    affine[:3, :3] *= map[:, 1:]
    return affine

def affine_from_fsl_mat_file(mat_affine, input_voxsz, output_voxsz):
    """It takes the affine matrix from flirt (FSLdot) and the voxel size of the
    input and output images and it returns the adjusted affine matrix for
    trackvis.
    """
    input_voxsz = asarray(input_voxsz)
    output_voxsz = asarray(output_voxsz)
    shift = eye(4)
    shift[:3,3] = -input_voxsz/2

    affine = dot(mat_affine, shift)
    affine[:3,3] += output_voxsz/2

    return affine
