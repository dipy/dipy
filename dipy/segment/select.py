import numpy as np
from dipy.tracking.utils import near_roi


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


def select_by_roi(streamlines, rois, include, affine=None, tol=None,
                  endpoints=False):
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
        A list or 1D array of boolean marking inclusion or exclusion
        criteria. If a streamline is near any of the inclusion ROIs, it
        should evaluate to True, unless it is also near any of the exclusion
        ROIs.
    affine : ndarray
        Affine transformation from voxels to streamlines. Default: identity.
    tol : float
        Distance (in the units of the streamlines, usually mm). If any
        coordinate in the streamline is within this distance from the center
        of any voxel in the ROI, the filtering criterion is set to True for
        this streamline, otherwise False. Defaults to the distance between
        the center of each voxel and the corner of the voxel.
    endpoints : bool, optional
        Use only the streamline endpoints as criteria. Default: False

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
    """
    include_roi, exclude_roi = reduce_rois(rois, include)
    for idx, sl in enumerate(streamlines):
        include = near_roi([sl], include_roi, affine, tol, endpoints)
        exclude = near_roi([sl], exclude_roi, affine, tol, endpoints)
        if include & ~exclude:
            yield sl
