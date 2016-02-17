"""Various tools related to creating and working with streamlines

This module provides tools for targeting streamlines using ROIs, for making
connectivity matrices from whole brain fiber tracking and some other tools that
allow streamlines to interact with image data.

Important Note:
---------------
Dipy uses affine matrices to represent the relationship between streamline
points, which are defined as points in a continuous 3d space, and image voxels,
which are typically arranged in a discrete 3d grid. Dipy uses a convention
similar to nifti files to interpret these affine matrices. This convention is
that the point at the center of voxel ``[i, j, k]`` is represented by the point
``[x, y, z]`` where ``[x, y, z, 1] = affine * [i, j, k, 1]``.  Also when the
phrase "voxel coordinates" is used, it is understood to be the same as ``affine
= eye(4)``.

As an example, lets take a 2d image where the affine is::

    [[1., 0., 0.],
     [0., 2., 0.],
     [0., 0., 1.]]

The pixels of an image with this affine would look something like:

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

And the letters A-D represent the following points in
"real world coordinates"::

    A = [-.5, -1.]
    B = [ .5,  1.]
    C = [ 0.,  0.]
    D = [ 2.5,  5.]

"""
from __future__ import division, print_function, absolute_import

from functools import wraps
from warnings import warn

from nibabel.affines import apply_affine
from scipy.spatial.distance import cdist

from dipy.core.geometry import dist_to_corner

from collections import defaultdict
from ..utils.six.moves import xrange, map

import numpy as np
from numpy import (asarray, ceil, dot, empty, eye, sqrt)
from dipy.io.bvectxt import ornt_mapping
from dipy.tracking import metrics

# Import helper functions shared with vox2track
from ._utils import (_mapping_to_voxel, _to_voxel_coordinates)


def _rmi(index, dims):
    """An alternate implementation of numpy.ravel_multi_index for older
    versions of numpy.

    Assumes array layout is C contiguous
    """
    # Upcast to integer type capable of holding largest array index
    index = np.asarray(index, dtype=np.intp)
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


def density_map(streamlines, vol_dims, voxel_size=None, affine=None):
    """Counts the number of unique streamlines that pass through each voxel.

    Parameters
    ----------
    streamlines : iterable
        A sequence of streamlines.

    vol_dims : 3 ints
        The shape of the volume to be returned containing the streamlines
        counts
    voxel_size :
        This argument is deprecated.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.

    Returns
    -------
    image_volume : ndarray, shape=vol_dims
        The number of streamline points in each voxel of volume.

    Raises
    ------
    IndexError
        When the points of the streamlines lie outside of the return volume.

    Notes
    -----
    A streamline can pass through a voxel even if one of the points of the
    streamline does not lie in the voxel. For example a step from [0,0,0] to
    [0,0,2] passes through [0,0,1]. Consider subsegmenting the streamlines when
    the edges of the voxels are smaller than the steps of the streamlines.

    """
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    counts = np.zeros(vol_dims, 'int')
    for sl in streamlines:
        inds = _to_voxel_coordinates(sl, lin_T, offset)
        i, j, k = inds.T
        # this takes advantage of the fact that numpy's += operator only
        # acts once even if there are repeats in inds
        counts[i, j, k] += 1
    return counts


def connectivity_matrix(streamlines, label_volume, voxel_size=None,
                        affine=None, symmetric=True, return_mapping=False,
                        mapping_as_streamlines=False):
    """Counts the streamlines that start and end at each label pair.

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    label_volume : ndarray
        An image volume with an integer data type, where the intensities in the
        volume map to anatomical structures.
    voxel_size :
        This argument is deprecated.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline coordinates.
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
    labels_positive = ((kind == 'u') or
                       ((kind == 'i') and (label_volume.min() >= 0)))
    valid_label_volume = (labels_positive and label_volume.ndim == 3)
    if not valid_label_volume:
        raise ValueError("label_volume must be a 3d integer array with"
                         "non-negative label values")

    # If streamlines is an iterators
    if return_mapping and mapping_as_streamlines:
        streamlines = list(streamlines)
    # take the first and last point of each streamline
    endpoints = [sl[0::len(sl)-1] for sl in streamlines]

    # Map the streamlines coordinates to voxel coordinates
    lin_T, offset = _mapping_to_voxel(affine, voxel_size)
    endpoints = _to_voxel_coordinates(endpoints, lin_T, offset)

    # get labels for label_volume
    i, j, k = endpoints.T
    endlabels = label_volume[i, j, k]
    if symmetric:
        endlabels.sort(0)
    mx = label_volume.max() + 1
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
    """Like bincount, but for nd-indicies.

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
    possible n.

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


def subsegment(streamlines, max_segment_length):
    """Splits the segments of the streamlines into small segments.

    Replaces each segment of each of the streamlines with the smallest possible
    number of equally sized smaller segments such that no segment is longer
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
    >>> streamlines = [np.array([[0,0,0],[2,0,0],[5,0,0]])]
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
                # repeated point
            else:
                # this should never happen because ns should be a positive
                # int
                assert(ns >= 0)
        yield output_sl


def seeds_from_mask(mask, density=[1, 1, 1], voxel_size=None, affine=None):
    """Creates seeds for fiber tracking from a binary mask.

    Seeds points are placed evenly distributed in all voxels of ``mask`` which
    are ``True``.

    Parameters
    ----------
    mask : binary 3d array_like
        A binary array specifying where to place the seeds for fiber tracking.
    density : int or array_like (3,)
        Specifies the number of seeds to place along each dimension. A
        ``density`` of `2` is the same as ``[2, 2, 2]`` and will result in a
        total of 8 seeds per voxel.
    voxel_size :
        This argument is deprecated.
    affine : array, (4, 4)
        The mapping between voxel indices and the point space for seeds. A
        seed point at the center the voxel ``[i, j, k]`` will be represented as
        ``[x, y, z]`` where ``[x, y, z, 1] == np.dot(affine, [i, j, k , 1])``.

    See Also
    --------
    random_seeds_from_mask

    Raises
    ------
    ValueError
        When ``mask`` is not a three-dimensional array

    Examples
    --------
    >>> mask = np.zeros((3,3,3), 'bool')
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
    if density.size == 1:
        d = density
        density = np.empty(3, dtype=int)
        density.fill(d)
    elif density.shape != (3,):
        raise ValueError("density should be in integer array of shape (3,)")

    # Grid of points between -.5 and .5, centered at 0, with given density
    grid = np.mgrid[0:density[0], 0:density[1], 0:density[2]]
    grid = grid.T.reshape((-1, 3))
    grid = grid / density
    grid += (.5 / density - .5)

    where = np.argwhere(mask)

    # Add the grid of points to each voxel in mask
    seeds = where[:, np.newaxis, :] + grid[np.newaxis, :, :]
    seeds = seeds.reshape((-1, 3))

    # Apply the spatial transform
    if affine is not None:
        # Use affine to move seeds into real world coordinates
        seeds = np.dot(seeds, affine[:3, :3].T)
        seeds += affine[:3, 3]
    elif voxel_size is not None:
        # Use voxel_size to move seeds into trackvis space
        seeds += .5
        seeds *= voxel_size

    return seeds


def random_seeds_from_mask(mask, seeds_count=1, seed_count_per_voxel=True, affine=None):
    """Creates randomly placed seeds for fiber tracking from a binary mask.

    Seeds points are placed randomly distributed in voxels of ``mask``
    which are ``True``.
    If ``seed_count_per_voxel`` is ``True``, this function is
    similar to ``seeds_from_mask()``, with the difference that instead of evenly
    distributing the seeds, it randomly places the seeds within the voxels
    specified by the ``mask``. The initial random conditions can be set using
    ``numpy.random.seed(...)``, prior to calling this function.

    Parameters
    ----------
    mask : binary 3d array_like
        A binary array specifying where to place the seeds for fiber tracking.
    seeds_count : int
        The number of seeds to generate. If ``seed_count_per_voxel`` is True,
        specifies the number of seeds to place in each voxel. Otherwise,
        specifies the total number of seeds to place in the mask.
    seed_count_per_voxel: bool
        If True, seeds_count is per voxel, else seeds_count is the total number
        of seeds.
    affine : array, (4, 4)
        The mapping between voxel indices and the point space for seeds. A
        seed point at the center the voxel ``[i, j, k]`` will be represented as
        ``[x, y, z]`` where ``[x, y, z, 1] == np.dot(affine, [i, j, k , 1])``.

    See Also
    --------
    seeds_from_mask

    Raises
    ------
    ValueError
        When ``mask`` is not a three-dimensional array

    Examples
    --------
    >>> mask = np.zeros((3,3,3), 'bool')
    >>> mask[0,0,0] = 1

    >>> np.random.seed(1)
    >>> random_seeds_from_mask(mask, seeds_count=1, seed_count_per_voxel=True)
    array([[-0.082978  ,  0.22032449, -0.49988563]])

    >>> random_seeds_from_mask(mask, seeds_count=6, seed_count_per_voxel=True)
    array([[-0.19766743, -0.35324411, -0.40766141],
           [-0.31373979, -0.15443927, -0.10323253],
           [ 0.03881673, -0.08080549,  0.1852195 ],
           [-0.29554775,  0.37811744, -0.47261241],
           [ 0.17046751, -0.0826952 ,  0.05868983],
           [-0.35961306, -0.30189851,  0.30074457]])
    >>> mask[0,1,2] = 1
    >>> random_seeds_from_mask(mask, seeds_count=2, seed_count_per_voxel=True)
    array([[ 0.46826158, -0.18657582,  0.19232262],
           [ 0.37638915,  0.39460666, -0.41495579],
           [-0.46094522,  0.66983042,  2.3781425 ],
           [-0.40165317,  0.92110763,  2.45788953]])

    """
    mask = np.array(mask, dtype=bool, copy=False, ndmin=3)
    if mask.ndim != 3:
        raise ValueError('mask cannot be more than 3d')

    where = np.argwhere(mask)
    num_voxels = len(where)

    if not seed_count_per_voxel:
        # Generate enough seeds per voxel
        seeds_per_voxel = seeds_count // num_voxels + 1
    else:
        seeds_per_voxel = seeds_count

    # Generate as many random triplets as the number of seeds needed
    grid = np.random.random([seeds_per_voxel * num_voxels, 3])
    # Repeat elements of 'where' so that it can be added to grid
    where = np.repeat(where, seeds_per_voxel, axis=0)
    seeds = where + grid - .5
    seeds = asarray(seeds)

    if not seed_count_per_voxel:
        # Randomize the seeds and select the requested amount
        np.random.shuffle(seeds)
        seeds = seeds[:seeds_count]

    # Apply the spatial transform
    if affine is not None:
        # Use affine to move seeds into real world coordinates
        seeds = np.dot(seeds, affine[:3, :3].T)
        seeds += affine[:3, 3]

    return seeds


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
def target(streamlines, target_mask, affine, include=True):
    """Filters streamlines based on whether or not they pass through an ROI.

    Parameters
    ----------
    streamlines : iterable
        A sequence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    target_mask : array-like
        A mask used as a target. Non-zero values are considered to be within
        the target region.
    affine : array (4, 4)
        The affine transform from voxel indices to streamline points.
    include : bool, default True
        If True, streamlines passing through `target_mask` are kept. If False,
        the streamlines not passing through `target_mask` are kept.

    Returns
    -------
    streamlines : generator
        A sequence of streamlines that pass through `target_mask`.

    Raises
    ------
    IndexError
        When the points of the streamlines lie outside of the `target_mask`.

    See Also
    --------
    density_map

    """
    target_mask = np.array(target_mask, dtype=bool, copy=True)
    lin_T, offset = _mapping_to_voxel(affine, voxel_size=None)
    yield
    # End of initialization

    for sl in streamlines:
        try:
            ind = _to_voxel_coordinates(sl, lin_T, offset)
            i, j, k = ind.T
            state = target_mask[i, j, k]
        except IndexError:
            raise ValueError("streamlines points are outside of target_mask")
        if state.any() == include:
            yield sl


def streamline_near_roi(streamline, roi_coords, tol, mode='any'):
    """Is a streamline near an ROI.

    Implements the inner loops of the :func:`near_roi` function.

    Parameters
    ----------
    streamline : array, shape (N, 3)
        A single streamline
    roi_coords : array, shape (M, 3)
        ROI coordinates transformed to the streamline coordinate frame.
    tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, this function returns True.
    mode : string
        One of {"any", "all", "either_end", "both_end"}, where return True
        if:

        "any" : any point is within tol from ROI.

        "all" : all points are within tol from ROI.

        "either_end" : either of the end-points is within tol from ROI

        "both_end" : both end points are within tol from ROI.

    Returns
    -------
    out : boolean
    """
    if len(roi_coords) == 0:
        return False
    if mode == "any" or mode == "all":
        s = streamline
    elif mode == "either_end" or mode == "both_end":
        # 'end' modes, use a streamline with 2 nodes:
        s = np.vstack([streamline[0], streamline[-1]])
    else:
        e_s = "For determining relationship to an array, you can use "
        e_s += "one of the following modes: 'any', 'all', 'both_end',"
        e_s += "'either_end', but you entered: %s." % mode
        raise ValueError(e_s)

    dist = cdist(s, roi_coords, 'euclidean')

    if mode == "any" or mode == "either_end":
        return np.min(dist) <= tol
    else:
        return np.all(np.min(dist, -1) <= tol)


def near_roi(streamlines, region_of_interest, affine=None, tol=None,
             mode="any"):
    """Provide filtering criteria for a set of streamlines based on whether
    they fall within a tolerance distance from an ROI

    Parameters
    ----------
    streamlines : list or generator
        A sequence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    region_of_interest : ndarray
        A mask used as a target. Non-zero values are considered to be within
        the target region.
    affine : ndarray
        Affine transformation from voxels to streamlines. Default: identity.
    tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.
    mode : string, optional
        One of {"any", "all", "either_end", "both_end"}, where return True
        if:

        "any" : any point is within tol from ROI. Default.

        "all" : all points are within tol from ROI.

        "either_end" : either of the end-points is within tol from ROI

        "both_end" : both end points are within tol from ROI.

    Returns
    -------
    1D array of boolean dtype, shape (len(streamlines), )

    This contains `True` for indices corresponding to each streamline
    that passes within a tolerance distance from the target ROI, `False`
    otherwise.
    """
    if affine is None:
        affine = np.eye(4)
    dtc = dist_to_corner(affine)
    if tol is None:
        tol = dtc
    elif tol < dtc:
        w_s = "Tolerance input provided would create gaps in your"
        w_s += " inclusion ROI. Setting to: %s" % dtc
        warn(w_s)
        tol = dtc

    roi_coords = np.array(np.where(region_of_interest)).T
    x_roi_coords = apply_affine(affine, roi_coords)

    # If it's already a list, we can save time by preallocating the output
    if isinstance(streamlines, list):
        out = np.zeros(len(streamlines), dtype=bool)
        for ii, sl in enumerate(streamlines):
            out[ii] = streamline_near_roi(sl, x_roi_coords, tol=tol,
                                          mode=mode)
        return out
    # If it's a generator, we'll need to generate the output into a list
    else:
        out = []
        for sl in streamlines:
            out.append(streamline_near_roi(sl, x_roi_coords, tol=tol,
                                           mode=mode))

        return(np.array(out, dtype=bool))


def reorder_voxels_affine(input_ornt, output_ornt, shape, voxel_size):
    """Calculates a linear transformation equivalent to changing voxel order.

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
    # multiply the rows of affine to get right sign
    affine[:3, :3] *= map[:, 1:]
    return affine


def affine_from_fsl_mat_file(mat_affine, input_voxsz, output_voxsz):
    """
    Converts an affine matrix from flirt (FSLdot) and a given voxel size for
    input and output images and returns an adjusted affine matrix for trackvis.

    Parameters
    ----------
    mat_affine : array of shape (4, 4)
       An FSL flirt affine.
    input_voxsz : array of shape (3,)
       The input image voxel dimensions.
    output_voxsz : array of shape (3,)

    Returns
    -------
    affine : array of shape (4, 4)
      A trackvis-compatible affine.

    """
    # TODO the affine returned by this function uses a different reference than
    # the nifti-style index coordinates dipy has adopted as a convention. We
    # should either fix this function in a backward compatible way or replace
    # and deprecate it.
    input_voxsz = asarray(input_voxsz)
    output_voxsz = asarray(output_voxsz)
    shift = eye(4)
    shift[:3, 3] = -input_voxsz / 2

    affine = dot(mat_affine, shift)
    affine[:3, 3] += output_voxsz / 2

    return affine


def affine_for_trackvis(voxel_size, voxel_order=None, dim=None,
                        ref_img_voxel_order=None):
    """Returns an affine which maps points for voxel indices to trackvis
    space.

    Parameters
    ----------
    voxel_size : array (3,)
        The sizes of the voxels in the reference image.

    Returns
    -------
    affine : array (4, 4)
        Mapping from the voxel indices of the reference image to trackvis
        space.
    """
    if (voxel_order is not None or dim is not None or
        ref_img_voxel_order is not None):
        raise NotImplemented

    # Create affine
    voxel_size = np.asarray(voxel_size)
    affine = np.eye(4)
    affine[[0, 1, 2], [0, 1, 2]] = voxel_size
    affine[:3, 3] = voxel_size / 2.
    return affine


def length(streamlines, affine=None):
    """
    Calculate the lengths of many streamlines in a bundle.

    Parameters
    ----------
    streamlines : list
        Each item in the list is an array with 3D coordinates of a streamline.
    affine : 4 x 4 array
        An affine transformation to move the fibers by, before computing their
        lengths.

    Returns
    -------
    Iterator object which then computes the length of each
    streamline in the bundle, upon iteration.
    """
    if affine is not None:
        streamlines = move_streamlines(streamlines, affine)
    return map(metrics.length, streamlines)


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
    # Sort input array
    order = np.lexsort(in_array.T)

    # Apply sort and compare neighbors
    x = in_array[order]
    diff_x = np.ones(len(x), dtype=bool)
    diff_x[1:] = (x[1:] != x[:-1]).any(-1)

    # Reverse sort and return unique rows
    un_order = order.argsort()
    diff_in_array = diff_x[un_order]
    return in_array[diff_in_array]


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


def reduce_rois(rois, include):
    """Reduce multiple ROIs to one inclusion and one exclusion ROI

    Parameters
    ----------
    rois : list or ndarray
        A list of 3D arrays, each with shape (x, y, z) corresponding to the
        shape of the brain volume, or a 4D array with shape (n_rois, x, y,
        z). Non-zeros in each volume are considered to be within the region.

    include : array or list
        A list or 1D array of boolean marking inclusion or exclusion
        criteria.

    Returns
    -------
    include_roi : boolean 3D array
        An array marking the inclusion mask.

    exclude_roi : boolean 3D array
        An array marking the exclusion mask

    Note
    ----
    The include_roi and exclude_roi can be used to perfom the operation: "(A
    or B or ...) and not (X or Y or ...)", where A, B are inclusion regions
    and X, Y are exclusion regions.
    """
    include_roi = np.zeros(rois[0].shape, dtype=bool)
    exclude_roi = np.zeros(rois[0].shape, dtype=bool)

    for i in range(len(rois)):
        if include[i]:
            include_roi |= rois[i]
        else:
            exclude_roi |= rois[i]

    return include_roi, exclude_roi
