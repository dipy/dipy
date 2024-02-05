import os
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates
from scipy.spatial.distance import mahalanobis

from dipy.io.utils import save_buan_profiles_hdf5
from dipy.segment.clustering import QuickBundles
from dipy.segment.metricspeed import AveragePointwiseEuclideanMetric
from dipy.tracking.streamline import (set_number_of_points,
                                      values_from_volume,
                                      orient_by_streamline,
                                      Streamlines)


def peak_values(
        bundle, peaks, dt, pname, bname, subject, group_id, ind, dir_name):
    """ Peak_values function finds the generalized fractional anisotropy (gfa)
        and quantitative anisotropy (qa) values from peaks object (eg: csa) for
        every point on a streamline used while tracking and saves it in hd5
        file.

    Parameters
    ----------
    bundle : string
        Name of bundle being analyzed
    peaks : peaks
        contains peak directions and values
    dt : DataFrame
        DataFrame to be populated
    pname : string
        Name of the dti metric
    bname : string
        Name of bundle being analyzed.
    subject : string
        subject number as a string (e.g. 10001)
    group_id : integer
        which group subject belongs to 1 patient and 0 for control
    ind : integer list
        ind tells which disk number a point belong.
    dir_name : string
        path of output directory

    """

    gfa = peaks.gfa
    anatomical_measures(bundle, gfa, dt, pname+'_gfa', bname, subject, group_id,
                        ind, dir_name)

    qa = peaks.qa[...,0]
    anatomical_measures(bundle, qa, dt, pname+'_qa', bname, subject, group_id,
                        ind, dir_name)


def anatomical_measures(bundle, metric, dt, pname, bname, subject, group_id,
                        ind, dir_name):
    """ Calculates dti measure (eg: FA, MD) per point on streamlines and
        save it in hd5 file.

    Parameters
    ----------
    bundle : string
        Name of bundle being analyzed
    metric : matrix of float values
        dti metric e.g. FA, MD
    dt : DataFrame
        DataFrame to be populated
    pname : string
        Name of the dti metric
    bname : string
        Name of bundle being analyzed.
    subject : string
        subject number as a string (e.g. 10001)
    group_id : integer
        which group subject belongs to 1 for patient and 0 control
    ind : integer list
        ind tells which disk number a point belong.
    dir_name : string
        path of output directory

    """
    dt["streamline"] = []
    dt["disk"] = []
    dt["subject"] = []
    dt[pname] = []
    dt["group"] = []

    values = map_coordinates(metric, bundle._data.T,
                             order=1)

    dt["disk"].extend(ind[list(range(len(values)))]+1)
    dt["subject"].extend([subject]*len(values))
    dt["group"].extend([group_id]*len(values))
    dt[pname].extend(values)

    for st_i in range(len(bundle)):

        st = bundle[st_i]
        dt["streamline"].extend([st_i]*len(st))

    file_name = bname + "_" + pname

    save_buan_profiles_hdf5(os.path.join(dir_name, file_name), dt)


def assignment_map(target_bundle, model_bundle, no_disks):
    """
    Calculates assignment maps of the target bundle with reference to
    model bundle centroids.

    Parameters
    ----------
    target_bundle : streamlines
        target bundle extracted from subject data in common space
    model_bundle : streamlines
        atlas bundle used as reference
    no_disks : integer, optional
        Number of disks used for dividing bundle into disks.

    Returns
    -------
    indx : ndarray
        Assignment map of the target bundle streamline point indices to the
        model bundle centroid points.

    References
    ----------
    .. [Chandio2020] Chandio, B.Q., Risacher, S.L., Pestilli, F., Bullock, D.,
    Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and Garyfallidis, E.
    Bundle analytics, a computational framework for investigating the
    shapes and profiles of brain pathways across populations.
    Sci Rep 10, 17149 (2020)

    """

    mbundle_streamlines = set_number_of_points(model_bundle,
                                               nb_points=no_disks)

    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=85., metric=metric)
    clusters = qb.cluster(mbundle_streamlines)
    centroids = Streamlines(clusters.centroids)

    _, indx = cKDTree(centroids.get_data(), 1,
                      copy_data=True).query(target_bundle.get_data(), k=1)

    return indx


def gaussian_weights(bundle, n_points=100, return_mahalnobis=False,
                     stat=np.mean):
    """
    Calculate weights for each streamline/node in a bundle, based on a
    Mahalanobis distance from the core the bundle, at that node (mean, per
    default).

    Parameters
    ----------
    bundle : Streamlines
        The streamlines to weight.
    n_points : int, optional
        The number of points to resample to. *If the `bundle` is an array, this
        input is ignored*. Default: 100.
    return_mahalanobis : bool, optional
        Whether to return the Mahalanobis distance instead of the weights.
        Default: False.
    stat : callable, optional.
        The statistic used to calculate the central tendency of streamlines in
        each node. Can be one of {`np.mean`, `np.median`} or other functions
        that have similar API. Default: `np.mean`

    Returns
    -------
    w : array of shape (n_streamlines, n_points)
        Weights for each node in each streamline, calculated as its relative
        inverse of the Mahalanobis distance, relative to the distribution of
        coordinates at that node position across streamlines.

    """
    # Resample to same length for each streamline:
    bundle = set_number_of_points(bundle, n_points)

    # This is the output
    w = np.zeros((len(bundle), n_points))

    # If there's only one fiber here, it gets the entire weighting:
    if len(bundle) == 1:
        if return_mahalnobis:
            return np.array([np.nan])
        else:
            return np.array([1])

    for node in range(n_points):
        # This should come back as a 3D covariance matrix with the spatial
        # variance covariance of this node across the different streamlines
        # This is a 3-by-3 array:
        node_coords = bundle._data[node::n_points]
        c = np.cov(node_coords.T, ddof=0)
        # Reorganize as an upper diagonal matrix for expected Mahalanobis
        # input:
        c = np.array([[c[0, 0], c[0, 1], c[0, 2]],
                      [0, c[1, 1], c[1, 2]],
                      [0, 0, c[2, 2]]])
        # Calculate the mean or median of this node as well
        # delta = node_coords - np.mean(node_coords, 0)
        m = stat(node_coords, 0)
        # Weights are the inverse of the Mahalanobis distance
        for fn in range(len(bundle)):
            # In the special case where all the streamlines have the exact same
            # coordinate in this node, the covariance matrix is all zeros, so
            # we can't calculate the Mahalanobis distance, we will instead give
            # each streamline an identical weight, equal to the number of
            # streamlines:
            if np.allclose(c, 0):
                w[:, node] = len(bundle)
                break
            # Otherwise, go ahead and calculate Mahalanobis for node on
            # fiber[fn]:
            w[fn, node] = mahalanobis(node_coords[fn], m, np.linalg.inv(c))
    if return_mahalnobis:
        return w
    # weighting is inverse to the distance (the further you are, the less you
    # should be weighted)
    w = 1 / w
    # Normalize before returning, so that the weights in each node sum to 1:
    return w / np.sum(w, 0)


def afq_profile(data, bundle, affine, n_points=100, profile_stat=np.average,
                orient_by=None, weights=None, **weights_kwarg):
    """
    Calculates a summarized profile of data for a bundle or tract
    along its length.

    Follows the approach outlined in [Yeatman2012]_.

    Parameters
    ----------
    data : 3D volume
        The statistic to sample with the streamlines.

    bundle : StreamLines class instance
        The collection of streamlines (possibly already resampled into an array
         for each to have the same length) with which we are resampling. See
         Note below about orienting the streamlines.
    affine : array_like (4, 4)
        The mapping from voxel coordinates to streamline points.
        The voxel_to_rasmm matrix, typically from a NIFTI file.
    n_points: int, optional
        The number of points to sample along the bundle. Default: 100.
    orient_by: streamline, optional.
        A streamline to use as a standard to orient all of the streamlines in
        the bundle according to.
    weights : 1D array or 2D array or callable (optional)
        Weight each streamline (1D) or each node (2D) when calculating the
        tract-profiles. Must sum to 1 across streamlines (in each node if
        relevant). If callable, this is a function that calculates weights.
    profile_stat : callable
        The statistic used to average the profile across streamlines.
        If weights is not None, this must take weights as a keyword argument.
        The default, np.average, is the same as np.mean but takes weights
        as a keyword argument.
    weights_kwarg : key-word arguments
        Additional key-word arguments to pass to the weight-calculating
        function. Only to be used if weights is a callable.

    Returns
    -------
    ndarray : a 1D array with the profile of `data` along the length of
        `bundle`

    Notes
    -----
    Before providing a bundle as input to this function, you will need to make
    sure that the streamlines in the bundle are all oriented in the same
    orientation relative to the bundle (use :func:`orient_by_streamline`).

    References
    ----------
    .. [Yeatman2012] Yeatman, Jason D., Robert F. Dougherty,
       Nathaniel J. Myall, Brian A. Wandell, and Heidi M. Feldman. 2012.
       "Tract Profiles of White Matter Properties: Automating Fiber-Tract
       Quantification" PloS One 7 (11): e49790.

    """
    if orient_by is not None:
        bundle = orient_by_streamline(bundle, orient_by)
    if affine is None:
        affine = np.eye(4)
    if len(bundle) == 0:
        raise ValueError("The bundle contains no streamlines")

    # Resample each streamline to the same number of points:
    fgarray = set_number_of_points(bundle, n_points)

    # Extract the values
    values = np.array(values_from_volume(data, fgarray, affine))

    if weights is not None:
        if callable(weights):
            weights = weights(bundle, **weights_kwarg)
        else:
            # We check that weights *always sum to 1 across streamlines*:
            if not np.allclose(np.sum(weights, 0), np.ones(n_points)):
                raise ValueError(
                    "The sum of weights across streamlines",
                    " must be equal to 1")

        return profile_stat(values, weights=weights, axis=0)
    else:
        return profile_stat(values, axis=0)
