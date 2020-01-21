"""Various tools related to creating and working with streamlines

This module provides tools for targeting streamlines using ROIs, for making
connectivity matrices from whole brain fiber tracking and some other tools that
allow streamlines to interact with image data.

Important Notes
-----------------
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

The pixels of an image with this affine would look something like::

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

from functools import wraps
from warnings import warn

from nibabel.affines import apply_affine
from scipy.spatial.distance import cdist
from numpy import ravel_multi_index

from dipy.core.geometry import dist_to_corner

from collections import defaultdict, OrderedDict
from itertools import combinations, groupby

import numpy as np
from numpy import (asarray, ceil, empty, sqrt)
from dipy.tracking import metrics
from dipy.tracking.vox2track import _streamlines_in_mask

# Import helper functions shared with vox2track
from dipy.tracking._utils import (_mapping_to_voxel, _to_voxel_coordinates)


def density_map(streamlines, affine, vol_dims):
    """Counts the number of unique streamlines that pass through each voxel.

    Parameters
    ----------
    streamlines : iterable
        A sequence of streamlines.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    vol_dims : 3 ints
        The shape of the volume to be returned containing the streamlines
        counts

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
    lin_T, offset = _mapping_to_voxel(affine)
    counts = np.zeros(vol_dims, 'int')
    for sl in streamlines:
        inds = _to_voxel_coordinates(sl, lin_T, offset)
        i, j, k = inds.T
        # this takes advantage of the fact that numpy's += operator only
        # acts once even if there are repeats in inds
        counts[i, j, k] += 1
    return counts


def connectivity_matrix(streamlines, affine, label_volume, inclusive=False,
                        symmetric=True, return_mapping=False,
                        mapping_as_streamlines=False):
    """Counts the streamlines that start and end at each label pair.

    Parameters
    ----------
    streamlines : sequence
        A sequence of streamlines.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline coordinates.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    label_volume : ndarray
        An image volume with an integer data type, where the intensities in the
        volume map to anatomical structures.
    inclusive: bool
        Whether to analyze the entire streamline, as opposed to just the
        endpoints. Allowing this will increase calculation time and mapping
        size, especially if mapping_as_streamlines is True. False by default.
    symmetric : bool, True by default
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

    # If streamlines is an iterator
    if return_mapping and mapping_as_streamlines:
        streamlines = list(streamlines)

    if inclusive:
        # Create ndarray to store streamline connections
        edges = np.ndarray(shape=(3, 0), dtype=int)
        lin_T, offset = _mapping_to_voxel(affine)
        for sl, _ in enumerate(streamlines):
            # Convert streamline to voxel coordinates
            entire = _to_voxel_coordinates(streamlines[sl], lin_T, offset)
            i, j, k = entire.T

            if symmetric:
                # Create list of all labels streamline passes through
                entirelabels = list(OrderedDict.fromkeys(label_volume[i, j, k]))
                # Append all connection combinations with streamline number
                for comb in combinations(entirelabels, 2):
                    edges = np.append(edges, [[comb[0]], [comb[1]], [sl]],
                                      axis=1)
            else:
                # Create list of all labels streamline passes through, keeping
                # order and whether a label was entered multiple times
                entirelabels = list(groupby(label_volume[i, j, k]))
                # Append connection combinations along with streamline number,
                # removing duplicates and connections from a label to itself
                combs = set(combinations([z[0] for z in entirelabels], 2))
                for comb in combs:
                    if comb[0] == comb[1]:
                        pass
                    else:
                        edges = np.append(edges, [[comb[0]], [comb[1]], [sl]],
                                          axis=1)
        if symmetric:
            edges[0:2].sort(0)
        mx = label_volume.max() + 1
        matrix = ndbincount(edges[0:2], shape=(mx, mx))

        if symmetric:
            matrix = np.maximum(matrix, matrix.T)
        if return_mapping:
            mapping = defaultdict(list)
            for i, (a, b, c) in enumerate(edges.T):
                mapping[a, b].append(c)
            # Replace each list of indices with the streamlines they index
            if mapping_as_streamlines:
                for key in mapping:
                    mapping[key] = [streamlines[i] for i in mapping[key]]

            return matrix, mapping

        return matrix
    else:
        # take the first and last point of each streamline
        endpoints = [sl[0::len(sl)-1] for sl in streamlines]

        # Map the streamlines coordinates to voxel coordinates
        lin_T, offset = _mapping_to_voxel(affine)
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

        return matrix


def ndbincount(x, weights=None, shape=None):
    """Like bincount, but for nd-indices.

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
    out = np.bincount(x, weights, minlength=np.prod(shape))
    out.shape = shape

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
        for ii in range(len(num_segments)):
            ns = num_segments[ii]
            if ns == 1:
                output_sl[count] = sl[ii+1]
                count += 1
            elif ns > 1:
                small_d = diff[ii]/ns
                point = sl[ii]
                for _ in range(ns):
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


def seeds_from_mask(mask, affine, density=[1, 1, 1]):
    """Create seeds for fiber tracking from a binary mask.

    Seeds points are placed evenly distributed in all voxels of ``mask`` which
    are ``True``.

    Parameters
    ----------
    mask : binary 3d array_like
        A binary array specifying where to place the seeds for fiber tracking.
    affine : array, (4, 4)
        The mapping between voxel indices and the point space for seeds.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
        A seed point at the center the voxel ``[i, j, k]``
        will be represented as ``[x, y, z]`` where
        ``[x, y, z, 1] == np.dot(affine, [i, j, k , 1])``.
    density : int or array_like (3,)
        Specifies the number of seeds to place along each dimension. A
        ``density`` of `2` is the same as ``[2, 2, 2]`` and will result in a
        total of 8 seeds per voxel.

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
    >>> seeds_from_mask(mask, np.eye(4), [1,1,1])
    array([[ 0.,  0.,  0.]])
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
    if seeds.any():
        # Use affine to move seeds into real world coordinates
        seeds = np.dot(seeds, affine[:3, :3].T)
        seeds += affine[:3, 3]

    return seeds


def random_seeds_from_mask(mask, affine, seeds_count=1,
                           seed_count_per_voxel=True, random_seed=None):
    """Create randomly placed seeds for fiber tracking from a binary mask.

    Seeds points are placed randomly distributed in voxels of ``mask``
    which are ``True``.
    If ``seed_count_per_voxel`` is ``True``, this function is
    similar to ``seeds_from_mask()``, with the difference that instead of
    evenly distributing the seeds, it randomly places the seeds within the
    voxels specified by the ``mask``.

    Parameters
    ----------
    mask : binary 3d array_like
        A binary array specifying where to place the seeds for fiber tracking.
    affine : array, (4, 4)
        The mapping between voxel indices and the point space for seeds.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
        A seed point at the center the voxel ``[i, j, k]``
        will be represented as ``[x, y, z]`` where
        ``[x, y, z, 1] == np.dot(affine, [i, j, k , 1])``.
    seeds_count : int
        The number of seeds to generate. If ``seed_count_per_voxel`` is True,
        specifies the number of seeds to place in each voxel. Otherwise,
        specifies the total number of seeds to place in the mask.
    seed_count_per_voxel: bool
        If True, seeds_count is per voxel, else seeds_count is the total number
        of seeds.
    random_seed : int
        The seed for the random seed generator (numpy.random.seed).

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
    >>> random_seeds_from_mask(mask, np.eye(4), seeds_count=1,
    ... seed_count_per_voxel=True, random_seed=1)
    array([[-0.0640051 , -0.47407377,  0.04966248]])
    >>> random_seeds_from_mask(mask, np.eye(4), seeds_count=6,
    ... seed_count_per_voxel=True, random_seed=1)
    array([[-0.0640051 , -0.47407377,  0.04966248],
           [ 0.0507979 ,  0.20814782, -0.20909526],
           [ 0.46702984,  0.04723225,  0.47268436],
           [-0.27800683,  0.37073231, -0.29328084],
           [ 0.39286015, -0.16802019,  0.32122912],
           [-0.42369171,  0.27991879, -0.06159077]])
    >>> mask[0,1,2] = 1
    >>> random_seeds_from_mask(mask, np.eye(4),
    ... seeds_count=2, seed_count_per_voxel=True, random_seed=1)
    array([[-0.0640051 , -0.47407377,  0.04966248],
           [-0.27800683,  1.37073231,  1.70671916],
           [ 0.0507979 ,  0.20814782, -0.20909526],
           [-0.48962585,  1.00187459,  1.99577329]])
    """
    mask = np.array(mask, dtype=bool, copy=False, ndmin=3)
    if mask.ndim != 3:
        raise ValueError('mask cannot be more than 3d')

    # Randomize the voxels
    np.random.seed(random_seed)
    shape = mask.shape
    mask = mask.flatten()
    indices = np.arange(len(mask))
    np.random.shuffle(indices)

    where = [np.unravel_index(i, shape) for i in indices if mask[i] == 1]
    num_voxels = len(where)

    if not seed_count_per_voxel:
        # Generate enough seeds per voxel
        seeds_per_voxel = seeds_count // num_voxels + 1
    else:
        seeds_per_voxel = seeds_count

    seeds = []
    for i in range(1, seeds_per_voxel + 1):
        for s in where:
            # Set the random seed with the current seed, the current value of
            # seeds per voxel and the global random seed.
            if random_seed is not None:
                s_random_seed = hash((np.sum(s) + 1) * i + random_seed) \
                    % (2**32 - 1)
                np.random.seed(s_random_seed)
            # Generate random triplet
            grid = np.random.random(3)
            seed = s + grid - .5
            seeds.append(seed)
    seeds = asarray(seeds)

    if not seed_count_per_voxel:
        # Select the requested amount
        seeds = seeds[:seeds_count]

    # Apply the spatial transform
    if seeds.any():
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
def target(streamlines, affine, target_mask, include=True):
    """Filters streamlines based on whether or not they pass through an ROI.

    Parameters
    ----------
    streamlines : iterable
        A sequence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    affine : array (4, 4)
        The mapping between voxel indices and the point space for seeds.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    target_mask : array-like
        A mask used as a target. Non-zero values are considered to be within
        the target region.
    include : bool, default True
        If True, streamlines passing through `target_mask` are kept. If False,
        the streamlines not passing through `target_mask` are kept.

    Returns
    -------
    streamlines : generator
        A sequence of streamlines that pass through `target_mask`.

    Raises
    ------
    ValueError
        When the points of the streamlines lie outside of the `target_mask`.

    See Also
    --------
    density_map
    """
    target_mask = np.array(target_mask, dtype=bool, copy=True)
    lin_T, offset = _mapping_to_voxel(affine)
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


@_with_initialize
def target_line_based(streamlines, affine, target_mask, include=True):
    """Filters streamlines based on whether or not they pass through a ROI,
    using a line-based algorithm. Mostly used as a replacement of `target`
    for compressed streamlines.

    This function never returns single-point streamlines, whatever the
    value of `include`.

    Parameters
    ----------
    streamlines : iterable
        A sequence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    affine : array (4, 4)
        The mapping between voxel indices and the point space for seeds.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    target_mask : array-like
        A mask used as a target. Non-zero values are considered to be within
        the target region.
    include : bool, default True
        If True, streamlines passing through `target_mask` are kept. If False,
        the streamlines not passing through `target_mask` are kept.

    Returns
    -------
    streamlines : generator
        A sequence of streamlines that pass through `target_mask`.

    References
    ----------
    [Bresenham5] Bresenham, Jack Elton. "Algorithm for computer control of a
                 digital plotter", IBM Systems Journal, vol 4, no. 1, 1965.
    [Houde15] Houde et al. How to avoid biased streamlines-based metrics for
              streamlines with variable step sizes, ISMRM 2015.

    See Also
    --------
    dipy.tracking.utils.density_map
    dipy.tracking.streamline.compress_streamlines
    """
    target_mask = np.array(target_mask, dtype=np.uint8, copy=True)
    lin_T, offset = _mapping_to_voxel(affine)
    streamline_index = _streamlines_in_mask(
        streamlines, target_mask, lin_T, offset)
    yield
    # End of initialization

    for idx in np.where(streamline_index == [0, 1][include])[0]:
        yield streamlines[idx]


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


def near_roi(streamlines, affine, region_of_interest, tol=None,
             mode="any"):
    """Provide filtering criteria for a set of streamlines based on whether
    they fall within a tolerance distance from an ROI

    Parameters
    ----------
    streamlines : list or generator
        A sequence of streamlines. Each streamline should be a (N, 3) array,
        where N is the length of the streamline.
    affine : array (4, 4)
        The mapping between voxel indices and the point space for seeds.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    region_of_interest : ndarray
        A mask used as a target. Non-zero values are considered to be within
        the target region.
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

    # If it's already a list, we can save time by pre-allocating the output
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


def length(streamlines):
    """
    Calculate the lengths of many streamlines in a bundle.

    Parameters
    ----------
    streamlines : list
        Each item in the list is an array with 3D coordinates of a streamline.

    Returns
    -------
    Iterator object which then computes the length of each
    streamline in the bundle, upon iteration.
    """

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
def transform_tracking_output(tracking_output, affine, save_seeds=False):
    """Applies a linear transformation, given by affine, to streamlines.
    Parameters
    ----------
    streamlines : Streamlines generator
        Either streamlines (list, ArraySequence) or a tuple with streamlines
        and seeds together
    affine : array (4, 4)
        The mapping between voxel indices and the point space for seeds.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    seeding_activated : bool, optional
        If set, seeds associated to streamlines will be also moved and returned
    Returns
    -------
    streamlines : generator
        A sequence of transformed streamlines.
        If return_seeds is True, also return seeds
    """
    if save_seeds:
        streamlines, seeds = zip(*tracking_output)
    else:
        streamlines = tracking_output
        seeds = None

    lin_T = affine[:3, :3].T.copy()
    offset = affine[:3, 3].copy()
    yield
    # End of initialization

    if seeds is not None:
        for sl, seed in zip(streamlines, seeds):
            yield np.dot(sl, lin_T) + offset, np.dot(seed, lin_T) + offset
    else:
        for sl in streamlines:
            yield np.dot(sl, lin_T) + offset


def reduce_rois(rois, include):
    """Reduce multiple ROIs to one inclusion and one exclusion ROI.

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

    Notes
    -----
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


def _min_at(a, index, value):
    index = np.asarray(index)
    sort_keys = [value] + list(index)
    order = np.lexsort(sort_keys)
    index = index[:, order]
    value = value[order]
    uniq = np.ones(index.shape[1], dtype=bool)
    uniq[1:] = (index[:, 1:] != index[:, :-1]).any(axis=0)

    index = index[:, uniq]
    value = value[uniq]

    a[tuple(index)] = np.minimum(a[tuple(index)], value)


try:
    minimum_at = np.minimum.at
except AttributeError:
    minimum_at = _min_at


def path_length(streamlines, affine, aoi, fill_value=-1):
    """ Computes the shortest path, along any streamline, between aoi and
    each voxel.

    Parameters
    ----------
    streamlines : seq of (N, 3) arrays
        A sequence of streamlines, path length is given in mm along the curve
        of the streamline.
    aoi : array, 3d
        A mask (binary array) of voxels from which to start computing distance.
    affine : array (4, 4)
        The mapping between voxel indices and the point space for seeds.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    fill_value : float
        The value of voxel in the path length map that are not connected to the
        aoi.

    Returns
    -------
    plm : array
        Same shape as aoi. The minimum distance between every point and aoi
        along the path of a streamline.
    """
    aoi = np.asarray(aoi, dtype=bool)

    # path length map
    plm = np.empty(aoi.shape, dtype=float)
    plm[:] = np.inf
    lin_T, offset = _mapping_to_voxel(affine)
    for sl in streamlines:
        seg_ind = _to_voxel_coordinates(sl, lin_T, offset)
        i, j, k = seg_ind.T
        # Get where streamlines passes through aoi
        breaks = aoi[i, j, k]
        # Where streamline passes aoi, dist is zero
        i, j, k = seg_ind[breaks].T
        plm[i, j, k] = 0

        # If a streamline crosses aoi >1, re-start counting distance for each
        for seg in _as_segments(sl, breaks):
            i, j, k = _to_voxel_coordinates(seg[1:], lin_T, offset).T
            # Get the distance, in mm, between streamline points
            segment_length = np.sqrt(((seg[1:] - seg[:-1]) ** 2).sum(1))
            dist = segment_length.cumsum()
            # Updates path length map with shorter distances
            minimum_at(plm, (i, j, k), dist)
    if fill_value != np.inf:
        plm = np.where(plm == np.inf, fill_value, plm)
    return plm


def _part_segments(streamline, break_points):
    segments = np.split(streamline, break_points.nonzero()[0])
    # Skip first segment, all points before first break
    # first segment is empty when break_points[0] == 0
    segments = segments[1:]
    for each in segments:
        if len(each) > 1:
            yield each


def _as_segments(streamline, break_points):
    for seg in _part_segments(streamline, break_points):
        yield seg
    for seg in _part_segments(streamline[::-1], break_points[::-1]):
        yield seg
