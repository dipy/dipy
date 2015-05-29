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


def select(streamlines, include_roi, exclude_roi, affine, tol):
    """
    Perform selection of streamlines based on an inclusion ROI and an
    exclusion ROI.

    Parameters
    ----------
    streamlines: list
        A list of candidate streamlines for selection.

    include_roi : boolean 3D array
        An array marking the inclusion mask.

    exclude_roi : boolean 3D array
        An array marking the exclusion mask

    Returns
    -------
    to_include : boolean array designating inclusion or exclusion of the
    streamlines given the ROIs and the inclusion/exclusion criteria.

    """
    include = near_roi(streamlines, include_roi, affine, tol)
    exclude = near_roi(streamlines, exclude_roi, affine, tol)
    return include & (~exclude)


def select_by_roi(streamlines, rois, include, affine=None, tol=0):
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

    Notes
    -----
    The only operation currently possible is "(A or B or ...) and not (X or Y
    or ...)", where A, B are inclusion regions and X, Y are exclusion regions.

    Returns
    -------
    to_include : boolean array designating inclusion or exclusion of the
    streamlines given the ROIs and the inclusion/exclusion criteria.

    See also
    --------
    :func:`dipy.tracking.utils.near_roi`
    :func:`select`
    :func:`reduce_rois`
    """

    include_roi, exclude_roi = reduce_rois(rois, include)
    return select(streamlines, include_roi, exclude_roi, affine, tol)