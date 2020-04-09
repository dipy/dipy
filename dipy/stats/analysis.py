
import os
import numpy as np
from time import time
from scipy.spatial import cKDTree
from scipy.ndimage.interpolation import map_coordinates
from scipy.spatial.distance import mahalanobis

from dipy.utils.optpkg import optional_package
from dipy.io.image import load_nifti
from dipy.io.streamline import load_tractogram
from dipy.io.utils import save_buan_profiles_hdf5
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import AveragePointwiseEuclideanMetric
from dipy.io.peaks import load_peaks
from dipy.tracking.streamline import (set_number_of_points,
                                      values_from_volume,
                                      orient_by_streamline,
                                      transform_streamlines,
                                      Streamlines)
from glob import glob
pd, have_pd, _ = optional_package("pandas")
_, have_tables, _ = optional_package("tables")

if have_pd:
    import pandas as pd


def peak_values(bundle, peaks, dt, pname, bname, subject, group, ind, dir):
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
        group : string
            which group subject belongs to (e.g. patient or control)
        ind : integer list
            ind tells which disk number a point belong.
        dir : string
            path of output directory

    """

    gfa = peaks.gfa
    anatomical_measures(bundle, gfa, dt, pname+'_gfa', bname, subject, group,
                        ind, dir)

    qa = peaks.qa[...,0]
    anatomical_measures(bundle, qa, dt, pname+'_qa', bname, subject, group,
                        ind, dir)


def anatomical_measures(bundle, metric, dt, pname, bname, subject, group,
                        ind, dir):
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
        group : string
            which group subject belongs to (e.g. patient or control)
        ind : integer list
            ind tells which disk number a point belong.
        dir : string
            path of output directory
    """

    if group == 'patient':
        group_id = 1  # 1 means patient
    else:
        group_id = 0  # 0 means control

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

    file_name = bname+"_"+pname

    save_buan_profiles_hdf5(os.path.join(dir, file_name), dt)


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
        Number of disks used for dividing bundle into disks. (Default 100)

    References
    ----------
    .. [Chandio19] Chandio, B.Q., S. Koudoro, D. Reagan, J. Harezlak,
    E. Garyfallidis, Bundle Analytics: a computational and statistical
    analyses framework for tractometric studies, Proceedings of:
    International Society of Magnetic Resonance in Medicine (ISMRM),
    Montreal, Canada, 2019.
    """

    mbundle_streamlines = set_number_of_points(model_bundle,
                                               nb_points=no_disks)

    metric = AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=85., metric=metric)
    clusters = qb.cluster(mbundle_streamlines)
    centroids = Streamlines(clusters.centroids)

    _, indx = cKDTree(centroids.data, 1,
                      copy_data=True).query(target_bundle.data, k=1)

    return indx


def buan_bundle_profiles(model_bundle_folder, bundle_folder,
                         orig_bundle_folder, metric_folder, group, subject,
                         no_disks=100, out_dir=''):
    """
    Applies statistical analysis on bundles and saves the results
    in a directory specified by ``out_dir``.

    Parameters
    ----------
    model_bundle_folder : string
        Path to the input model bundle files. This path may contain
        wildcards to process multiple inputs at once.
    bundle_folder : string
        Path to the input bundle files in common space. This path may
        contain wildcards to process multiple inputs at once.
    orig_folder : string
        Path to the input bundle files in native space. This path may
        contain wildcards to process multiple inputs at once.
    metric_folder : string
        Path to the input dti metric or/and peak files. It will be used as
        metric for statistical analysis of bundles.
    group : string
        what group subject belongs to either control or patient
    subject : string
        subject id e.g. 10001
    no_disks : integer, optional
        Number of disks used for dividing bundle into disks. (Default 100)
    out_dir : string, optional
        Output directory (default input file directory)

    References
    ----------
    .. [Chandio19] Chandio, B.Q., S. Koudoro, D. Reagan, J. Harezlak,
    E. Garyfallidis, Bundle Analytics: a computational and statistical
    analyses framework for tractometric studies, Proceedings of:
    International Society of Magnetic Resonance in Medicine (ISMRM),
    Montreal, Canada, 2019.

    """

    t = time()

    dt = dict()

    mb = glob(os.path.join(model_bundle_folder, "*.trk"))
    print(mb)

    mb.sort()

    bd = glob(os.path.join(bundle_folder, "*.trk"))

    bd.sort()
    print(bd)
    org_bd = glob(os.path.join(orig_bundle_folder, "*.trk"))
    org_bd.sort()
    print(org_bd)
    n = len(org_bd)
    n = len(mb)

    for io in range(n):

        mbundles = load_tractogram(mb[io], reference='same',
                                   bbox_valid_check=False).streamlines
        bundles = load_tractogram(bd[io], reference='same',
                                  bbox_valid_check=False).streamlines
        orig_bundles = load_tractogram(org_bd[io], reference='same',
                                       bbox_valid_check=False).streamlines

        if len(orig_bundles) > 5:

            indx = assignment_map(bundles, mbundles, no_disks)
            ind = np.array(indx)

            metric_files_names_dti = glob(os.path.join(metric_folder,
                                                       "*.nii.gz"))

            metric_files_names_csa = glob(os.path.join(metric_folder,
                                                       "*.pam5"))

            _, affine = load_nifti(metric_files_names_dti[0])

            affine_r = np.linalg.inv(affine)
            transformed_orig_bundles = transform_streamlines(orig_bundles,
                                                             affine_r)

            for mn in range(len(metric_files_names_dti)):

                ab = os.path.split(metric_files_names_dti[mn])
                metric_name = ab[1]

                fm = metric_name[:-7]
                bm = os.path.split(mb[io])[1][:-4]

                print("bm = ", bm)

                dt = dict()

                print("metric = ", metric_files_names_dti[mn])

                metric, _ = load_nifti(metric_files_names_dti[mn])

                anatomical_measures(transformed_orig_bundles, metric, dt, fm,
                                    bm, subject, group, ind, out_dir)

            for mn in range(len(metric_files_names_csa)):
                ab = os.path.split(metric_files_names_csa[mn])
                metric_name = ab[1]

                fm = metric_name[:-5]
                bm = os.path.split(mb[io])[1][:-4]

                print("bm = ", bm)
                print("metric = ", metric_files_names_csa[mn])
                dt = dict()
                metric = load_peaks(metric_files_names_csa[mn])

                peak_values(transformed_orig_bundles, metric, dt, fm, bm,
                            subject, group, ind, out_dir)

    print("total time taken in minutes = ", (-t + time())/60)


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


def afq_profile(data, bundle, affine, n_points=100,
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

    if weights is None:
        weights = np.ones(values.shape) / values.shape[0]
    elif callable(weights):
        weights = weights(bundle, **weights_kwarg)
    else:
        # We check that weights *always sum to 1 across streamlines*:
        if not np.allclose(np.sum(weights, 0), np.ones(n_points)):
            raise ValueError("The sum of weights across streamlines must ",
                             "be equal to 1")

    return np.sum(weights * values, 0)
