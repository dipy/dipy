from warnings import warn

import numpy as np
from nibabel.affines import apply_affine
from dipy.tracking.streamlinespeed import set_number_of_points
from dipy.tracking.streamlinespeed import length
from dipy.tracking.streamlinespeed import compress_streamlines
import dipy.tracking.utils as ut
from dipy.tracking.utils import streamline_near_roi
from dipy.core.geometry import dist_to_corner
from scipy.spatial.distance import cdist
from copy import deepcopy

def unlist_streamlines(streamlines):
    """ Return the streamlines not as a list but as an array and an offset

    Parameters
    ----------
    streamlines: sequence

    Returns
    -------
    points : array
    offsets : array

    """

    points = np.concatenate(streamlines, axis=0)
    offsets = np.zeros(len(streamlines), dtype='i8')

    curr_pos = 0
    prev_pos = 0
    for (i, s) in enumerate(streamlines):

            prev_pos = curr_pos
            curr_pos += s.shape[0]
            points[prev_pos:curr_pos] = s
            offsets[i] = curr_pos

    return points, offsets


def relist_streamlines(points, offsets):
    """ Given a representation of a set of streamlines as a large array and
    an offsets array return the streamlines as a list of shorter arrays.

    Parameters
    -----------
    points : array
    offsets : array

    Returns
    -------
    streamlines: sequence
    """

    streamlines = []

    streamlines.append(points[0: offsets[0]])

    for i in range(len(offsets) - 1):
        streamlines.append(points[offsets[i]: offsets[i + 1]])

    return streamlines


def center_streamlines(streamlines):
    """ Move streamlines to the origin

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    Returns
    -------
    new_streamlines : list
        List of 2D ndarrays of shape[-1]==3
    inv_shift : ndarray
        Translation in x,y,z to go back in the initial position

    """
    center = np.mean(np.concatenate(streamlines, axis=0), axis=0)
    return [s - center for s in streamlines], center


def transform_streamlines(streamlines, mat):
    """ Apply affine transformation to streamlines

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3
    mat : array, (4, 4)
        transformation matrix

    Returns
    -------
    new_streamlines : list
        List of the transformed 2D ndarrays of shape[-1]==3
    """
    return [apply_affine(mat, s) for s in streamlines]


def select_random_set_of_streamlines(streamlines, select):
    """ Select a random set of streamlines

    Parameters
    ----------
    streamlines : list
        List of 2D ndarrays of shape[-1]==3

    select : int
        Number of streamlines to select. If there are less streamlines
        than ``select`` then ``select=len(streamlines)``.

    Returns
    -------
    selected_streamlines : list

    Notes
    -----
    The same streamline will not be selected twice.
    """
    len_s = len(streamlines)
    index = np.random.choice(len_s, min(select, len_s), replace=False)
    return [streamlines[i] for i in index]


def select_by_rois(streamlines, rois, include, mode=None, affine=None,
                   tol=None):
    """Select streamlines based on logical relations with several regions of
    interest (ROIs). For example, select streamlines that pass near ROI1,
    but only if they do not pass near ROI2.

    Parameters
    ----------
    streamlines : list
        A list of candidate streamlines for selection
    rois : list or ndarray
        A list of 3D arrays, each with shape (x, y, z) corresponding to the
        shape of the brain volume, or a 4D array with shape (n_rois, x, y,
        z). Non-zeros in each volume are considered to be within the region
    include : array or list
        A list or 1D array of boolean values marking inclusion or exclusion
        criteria. If a streamline is near any of the inclusion ROIs, it
        should evaluate to True, unless it is also near any of the exclusion
        ROIs.
    mode : string, optional
        One of {"any", "all", "either_end", "both_end"}, where a streamline is
        associated with an ROI if:

        "any" : any point is within tol from ROI. Default.

        "all" : all points are within tol from ROI.

        "either_end" : either of the end-points is within tol from ROI

        "both_end" : both end points are within tol from ROI.

    affine : ndarray
        Affine transformation from voxels to streamlines. Default: identity.
    tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.

    Notes
    -----
    The only operation currently possible is "(A or B or ...) and not (X or Y
    or ...)", where A, B are inclusion regions and X, Y are exclusion regions.

    Returns
    -------
    generator
       Generates the streamlines to be included based on these criteria.

    See also
    --------
    :func:`dipy.tracking.utils.near_roi`
    :func:`dipy.tracking.utils.reduce_rois`

    Examples
    --------
    >>> streamlines = [np.array([[0, 0., 0.9],
    ...                          [1.9, 0., 0.]]),
    ...                np.array([[0., 0., 0],
    ...                          [0, 1., 1.],
    ...                          [0, 2., 2.]]),
    ...                np.array([[2, 2, 2],
    ...                          [3, 3, 3]])]
    >>> mask1 = np.zeros((4, 4, 4), dtype=bool)
    >>> mask2 = np.zeros_like(mask1)
    >>> mask1[0, 0, 0] = True
    >>> mask2[1, 0, 0] = True
    >>> selection = select_by_rois(streamlines, [mask1, mask2],
    ...                            [True, True],
    ...                            tol=1)
    >>> list(selection) # The result is a generator
    [array([[ 0. ,  0. ,  0.9],
           [ 1.9,  0. ,  0. ]]), array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  2.,  2.]])]
    >>> selection = select_by_rois(streamlines, [mask1, mask2],
    ...                            [True, False],
    ...                            tol=0.87)
    >>> list(selection)
    [array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  2.,  2.]])]
    >>> selection = select_by_rois(streamlines, [mask1, mask2],
    ...                            [True, True],
    ...                            mode="both_end",
    ...                            tol=1.0)
    >>> list(selection)
    [array([[ 0. ,  0. ,  0.9],
           [ 1.9,  0. ,  0. ]])]
    >>> mask2[0, 2, 2] = True
    >>> selection = select_by_rois(streamlines, [mask1, mask2],
    ...                            [True, True],
    ...                            mode="both_end",
    ...                            tol=1.0)
    >>> list(selection)
    [array([[ 0. ,  0. ,  0.9],
           [ 1.9,  0. ,  0. ]]), array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  2.,  2.]])]
    """
    if affine is None:
        affine = np.eye(4)
    # This calculates the maximal distance to a corner of the voxel:
    dtc = dist_to_corner(affine)
    if tol is None:
        tol = dtc
    elif tol < dtc:
        w_s = "Tolerance input provided would create gaps in your"
        w_s += " inclusion ROI. Setting to: %s" % dist_to_corner
        warn(w_s)
        tol = dtc
    include_roi, exclude_roi = ut.reduce_rois(rois, include)
    include_roi_coords = np.array(np.where(include_roi)).T
    x_include_roi_coords = apply_affine(affine, include_roi_coords)
    exclude_roi_coords = np.array(np.where(exclude_roi)).T
    x_exclude_roi_coords = apply_affine(affine, exclude_roi_coords)

    if mode is None:
        mode = "any"
    for sl in streamlines:
        include = streamline_near_roi(sl, x_include_roi_coords, tol=tol,
                                      mode=mode)
        exclude = streamline_near_roi(sl, x_exclude_roi_coords, tol=tol,
                                      mode=mode)
        if include & ~exclude:
            yield sl


def orient_by_rois(streamlines, roi1, roi2, affine=None, copy=True):
    """Orient a set of streamlines according to a pair of ROIs

    Parameters
    ----------
    streamlines : list
        List of 3d arrays. Each array contains the xyz coordinates of a single
        streamline.
    roi1, roi2 : ndarray
        Binary masks designating the location of the regions of interest, or
        coordinate arrays (n-by-3 array with ROI coordinate in each row).
    affine : ndarray
        Affine transformation from voxels to streamlines. Default: identity.
    copy : bool
        Whether to make a copy of the input, or mutate the input inplace.

    Returns
    -------
    streamlines : list
        The same 3D arrays, but reoriented with respect to the ROIs

    Examples
    --------
    >>> streamlines = [np.array([[0, 0., 0],
    ...                          [1, 0., 0.],
    ...                          [2, 0., 0.]]),
    ...                np.array([[2, 0., 0.],
    ...                          [1, 0., 0],
    ...                          [0, 0,  0.]])]
    >>> roi1 = np.zeros((4, 4, 4), dtype=bool)
    >>> roi2 = np.zeros_like(roi1)
    >>> roi1[0, 0, 0] = True
    >>> roi2[1, 0, 0] = True
    >>> orient_by_rois(streamlines, roi1, roi2)
    [array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 2.,  0.,  0.]]), array([[ 0.,  0.,  0.],
           [ 1.,  0.,  0.],
           [ 2.,  0.,  0.]])]

    """

    # If we don't already have coordinates on our hands:
    if len(roi1.shape) == 3:
        roi1 = np.asarray(np.where(roi1.astype(bool))).T
    if len(roi2.shape) == 3:
        roi2 = np.asarray(np.where(roi2.astype(bool))).T

    if affine is not None:
        roi1 = apply_affine(affine, roi1)
        roi2 = apply_affine(affine, roi2)

    # Make a copy, so you don't change the output in place:
    if copy:
        new_sl = deepcopy(streamlines)
    else:
        new_sl = streamlines

    for idx, sl in enumerate(streamlines):
        dist1 = cdist(sl, roi1, 'euclidean')
        dist2 = cdist(sl, roi2, 'euclidean')
        min1 = np.argmin(dist1, 0)
        min2 = np.argmin(dist2, 0)
        if min1[0] > min2[0]:
            new_sl[idx] = sl[::-1]

    return new_sl
