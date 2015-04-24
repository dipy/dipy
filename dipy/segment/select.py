import numpy as np
from dipy.tracking.utils import near_roi


def _multi_or(arr):
    """
    Helper function to compute the logical orness of
    """
    if len(arr.shape) == 1 or arr.shape[0]<2:
        return arr
    if arr.shape[0]==2:
        return np.logical_or(arr[0], arr[1])

    else:
        return np.logical_or(_multi_or(arr[1:]), arr[0])


def _get_near(streamlines, rois, include, affine=None, tol=0):
    """
    Helper function to reduce repeated operations

    """
    if not np.sum(include):
        return np.zeros(len(streamlines), dtype=bool)
    n_rois = np.sum(include)
    idx = np.where(include)
    is_near = np.zeros((n_rois, len(streamlines)), dtype=bool)
    for roi_idx, roi in enumerate(rois[idx]):
        is_near[roi_idx] = near_roi(streamlines, roi, affine=affine,
                                    tol=tol)
    return is_near


def select_by_roi(streamlines, rois, include, affine=None, tol=0):
    """
    Select streamlines based on logical relations with several regions of
    intererst (ROIs). For example, select streamlines that pass near ROI1, but
    only if they do not pass near ROI2.

    Parameters
    ----------
    streamlines: list
        A list of candidate streamlines for selection

    rois: list or ndarray
        A list of 3D arrays, each with shape (x, y, z) corresponding to the
        shape of the brain volume, or a 4D array with shape (n_rois, x, y,
        z). Non-zeros in each volume are considered to be within the regio

    include: array
        A 1D array of boolean marking inclusion or exclusion criteria. If a
        streamline is near any of the inclusion ROIs, it should evaluate to
        True, unless it is also near any of the exclusion ROIs.

    Notes
    -----
    The only operation currently possible is "(A or B or ...) and not (X or Y or
    ...)", where A, B are inclusion regions and X, Y are exclusion regions.

    Returns
    -------
    to_include: boolean array designating inclusion or exclusion of the
    streamlines given the ROIs and the inclusion/exclusion criteria.

    See also
    --------
    `func`:dipy.tracking.utils.near_roi:
    """
    include_a = np.asarray(include)
    rois_a = np.asarray(rois)
    include_near = _get_near(streamlines, rois_a, include_a,
                             affine=affine, tol=tol)
    exclude_near = _get_near(streamlines, rois_a, ~include_a,
                             affine=affine, tol=tol)
    to_include = _multi_or(include_near)
    to_exclude = _multi_or(exclude_near)
    return np.logical_and(to_include, ~to_exclude).squeeze()
