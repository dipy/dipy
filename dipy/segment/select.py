"""
Select streamlines
"""


from warnings import warn
import numpy as np
from nibabel.affines import apply_affine
from dipy.tracking.utils import _near_roi


def reduce_rois(rois, include):
    """
    Reduce multiple ROIs to one inclusion and one exclusion ROI

    Parameters
    ----------
    rois : list or ndarray
        A list of 3D arrays, each with shape (x, y, z) corresponding to the
        shape of the brain volume, or a 4D array with shape (n_rois, x, y,
        z). Non-zeros in each volume are considered to be within the region.

    include: array or list
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


def select_by_roi(streamlines, rois, include, mode=None, affine=None,
                  tol=None):
    """
    Select streamlines based on logical relations with several regions of
    interest (ROIs). For example, select streamlines that pass near ROI1,
    but only if they do not pass near ROI2.

    Parameters
    ----------
    streamlines: list
        A list of candidate streamlines for selection
    rois: list or ndarray
        A list of 3D arrays, each with shape (x, y, z) corresponding to the
        shape of the brain volume, or a 4D array with shape (n_rois, x, y,
        z). Non-zeros in each volume are considered to be within the region
    include: array or list
        A list or 1D array of boolean values marking inclusion or exclusion
        criteria. If a streamline is near any of the inclusion ROIs, it
        should evaluate to True, unless it is also near any of the exclusion
        ROIs.
    mode : string, optional
		One of {"any", "all", "either_end", "both_end"}, where a
            streamline is associated with an ROI if:
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
    :func:`reduce_rois`

    Examples
    --------
    >>> streamlines = [np.array([[0, 0., 0.9],
    ...                                  [1.9, 0., 0.]]),
    ...                        np.array([[0., 0., 0],
    ...                                  [0, 1., 1.],
    ...                                  [0, 2., 2.]]),
    ...                        np.array([[2, 2, 2],
    ...                                  [3, 3, 3]])]
    >>> mask1 = np.zeros((4, 4, 4), dtype=bool)
    >>> mask2 = np.zeros_like(mask1)
    >>> mask1[0, 0, 0] = True
    >>> mask2[1, 0, 0] = True
    >>> selection = select_by_roi(streamlines, [mask1, mask2], [True, True],
    ...                           tol=1)
    >>> list(selection) # The result is a generator
    [array([[ 0. ,  0. ,  0.9],
           [ 1.9,  0. ,  0. ]]), array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  2.,  2.]])]
    >>> selection = select_by_roi(streamlines, [mask1, mask2], [True, False],
    ...                           tol=0.87)
    >>> list(selection)
    [array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  2.,  2.]])]
    >>> selection = select_by_roi(streamlines, [mask1, mask2],
    ...                      [True, True],
    ...                      mode="both_end",
    ...                      tol=1.0)
    >>> list(selection)
    [array([[ 0. ,  0. ,  0.9],
           [ 1.9,  0. ,  0. ]])]
    >>> mask2[0, 2, 2] = True
    >>> selection = select_by_roi(streamlines, [mask1, mask2],
    ...                      [True, True],
    ...                      mode="both_end",
    ...                      tol=1.0)
    >>> list(selection)
    [array([[ 0. ,  0. ,  0.9],
           [ 1.9,  0. ,  0. ]]), array([[ 0.,  0.,  0.],
           [ 0.,  1.,  1.],
           [ 0.,  2.,  2.]])]
    """
    if affine is None:
        affine = np.eye(4)
    # This calculates the maximal distance to a corner of the voxel:
    dist_to_corner = np.sqrt(np.sum((np.diag(affine)[:-1] / 2) ** 2))
    if tol is None:
        tol = dist_to_corner
    elif tol < dist_to_corner:
        w_s = "Tolerance input provided would create gaps in your"
        w_s += " inclusion ROI. Setting to: %s"%dist_to_corner
        warn(w_s)
        tol = dist_to_corner
    include_roi, exclude_roi = reduce_rois(rois, include)
    include_roi_coords = np.array(np.where(include_roi)).T
    x_include_roi_coords = apply_affine(affine, include_roi_coords)
    exclude_roi_coords = np.array(np.where(exclude_roi)).T
    x_exclude_roi_coords = apply_affine(affine, exclude_roi_coords)

    if mode is None:
        mode = "any"
    for sl in streamlines:
        include = _near_roi(sl, x_include_roi_coords, tol=tol,
                                mode=mode)
        exclude = _near_roi(sl, x_exclude_roi_coords, tol=tol,
                                mode=mode)
        if include & ~exclude:
            yield sl
