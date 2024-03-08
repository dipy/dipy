from time import time
from itertools import chain
import logging

import numpy as np
from dipy.tracking.streamline import (set_number_of_points, nbytes,
                                      select_random_set_of_streamlines)
from dipy.segment.clustering import qbx_and_merge
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from dipy.align.streamlinear import (StreamlineLinearRegistration,
                                     BundleMinDistanceMetric,
                                     BundleSumDistanceMatrixMetric,
                                     BundleMinDistanceAsymmetricMetric)

from dipy.tracking.streamline import Streamlines, length
from nibabel.affines import apply_affine


def check_range(streamline, gt, lt):
    length_s = length(streamline)
    if (length_s > gt) & (length_s < lt):
        return True
    else:
        return False


logger = logging.getLogger(__name__)


def bundle_adjacency(dtracks0, dtracks1, threshold):
    """ Find bundle adjacency between two given tracks/bundles

    Parameters
    ----------
    dtracks0 : Streamlines
        White matter tract from one subject
    dtracks1 : Streamlines
        White matter tract from another subject
    threshold : float
        Threshold controls
        how much strictness user wants while calculating bundle adjacency
        between two bundles. Smaller threshold means bundles should be strictly
        adjacent to get higher BA score.

    Returns
    -------
    res : Float
        Bundle adjacency score between two tracts

    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """
    d01 = bundles_distances_mdf(dtracks0, dtracks1)

    pair12 = []

    for i in range(len(dtracks0)):
        if np.min(d01[i, :]) < threshold:
            j = np.argmin(d01[i, :])
            pair12.append((i, j))

    pair12 = np.array(pair12)
    pair21 = []

    # solo2 = []
    for i in range(len(dtracks1)):
        if np.min(d01[:, i]) < threshold:
            j = np.argmin(d01[:, i])
            pair21.append((i, j))

    pair21 = np.array(pair21)
    A = len(pair12) / float(len(dtracks0))
    B = len(pair21) / float(len(dtracks1))
    res = 0.5 * (A + B)
    return res


def ba_analysis(recognized_bundle, expert_bundle, nb_pts=20, threshold=6.):
    """ Calculates bundle adjacency score between two given bundles

    Parameters
    ----------
    recognized_bundle : Streamlines
        Extracted bundle from the whole brain tractogram (eg: AF_L)
    expert_bundle : Streamlines
        Model bundle used as reference while extracting similar type bundle
        from input tractogram
    nb_pts : integer (default 20)
        Discretizing streamlines to have nb_pts number of points
    threshold : float (default 6)
        Threshold used for in computing bundle adjacency. Threshold controls
        how much strictness user wants while calculating bundle adjacency
        between two bundles. Smaller threshold means bundles should be strictly
        adjacent to get higher BA score.

    Returns
    -------
        Bundle adjacency score between two tracts

    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """

    recognized_bundle = set_number_of_points(recognized_bundle, nb_pts)

    expert_bundle = set_number_of_points(expert_bundle, nb_pts)

    return bundle_adjacency(recognized_bundle, expert_bundle, threshold)


def cluster_bundle(bundle, clust_thr, rng, nb_pts=20, select_randomly=500000):
    """ Clusters bundles

    Parameters
    ----------
    bundle : Streamlines
        White matter tract
    clust_thr : float
        clustering threshold used in quickbundlesX
    rng : np.random.Generator
        numpy's random generator for generating random values.
    nb_pts: integer (default 20)
        Discretizing streamlines to have nb_points number of points
    select_randomly: integer (default 500000)
        Randomly select streamlines from the input bundle

    Returns
    -------
    centroids : Streamlines
        clustered centroids of the input bundle

    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
   """

    model_cluster_map = qbx_and_merge(bundle, clust_thr,
                                      nb_pts=nb_pts,
                                      select_randomly=select_randomly,
                                      rng=rng)
    centroids = model_cluster_map.centroids

    return centroids


def bundle_shape_similarity(bundle1, bundle2, rng, clust_thr=(5, 3, 1.5),
                            threshold=6):
    """ Calculates bundle shape similarity between two given bundles using
    bundle adjacency (BA) metric

    Parameters
    ----------
    bundle1 : Streamlines
        White matter tract from one subject (eg: AF_L)
    bundle2 : Streamlines
        White matter tract from another subject (eg: AF_L)
    rng : np.random.Generator
    clust_thr : array-like, optional
        list of clustering thresholds used in quickbundlesX
    threshold : float, optional
        Threshold used for in computing bundle adjacency. Threshold controls
        how much strictness user wants while calculating shape similarity
        between two bundles. Smaller threshold means bundles should be strictly
        similar to get higher shape similarity score.

    Returns
    -------
    ba_value : Float
        Bundle similarity score between two tracts

    References
    ----------
    .. [Chandio2020] Chandio, B.Q., Risacher, S.L., Pestilli, F., Bullock, D.,
    Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and Garyfallidis, E.
    Bundle analytics, a computational framework for investigating the
    shapes and profiles of brain pathways across populations.
    Sci Rep 10, 17149 (2020)

    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """

    if len(bundle1) == 0 or len(bundle2) == 0:
        return 0

    bundle1_centroids = cluster_bundle(bundle1, clust_thr=clust_thr,
                                       rng=rng)
    bundle2_centroids = cluster_bundle(bundle2, clust_thr=clust_thr,
                                       rng=rng)
    bundle1_centroids = Streamlines(bundle1_centroids)
    bundle2_centroids = Streamlines(bundle2_centroids)

    ba_value = ba_analysis(recognized_bundle=bundle1_centroids,
                           expert_bundle=bundle2_centroids,
                           threshold=threshold)

    return ba_value


class RecoBundles:

    def __init__(self, streamlines,  greater_than=50, less_than=1000000,
                 cluster_map=None, clust_thr=15, nb_pts=20,
                 rng=None, verbose=False):
        """ Recognition of bundles

        Extract bundles from a participants' tractograms using model bundles
        segmented from a different subject or an atlas of bundles.
        See [Garyfallidis17]_ for the details.

        Parameters
        ----------
        streamlines : Streamlines
            The tractogram in which you want to recognize bundles.
        greater_than : int, optional
            Keep streamlines that have length greater than
            this value (default 50)
        less_than : int, optional
            Keep streamlines have length less than this value (default 1000000)
        cluster_map : QB map, optional.
            Provide existing clustering to start RB faster (default None).
        clust_thr : float, optional.
            Distance threshold in mm for clustering `streamlines`.
            Default: 15.
        nb_pts : int, optional.
            Number of points per streamline (default 20)
        rng : np.random.Generator
            If None define generator in initialization function.
            Default: None
        verbose: bool, optional.
            If True, log information.

        Notes
        -----
        Make sure that before creating this class that the streamlines and
        the model bundles are roughly in the same space.
        Also default thresholds are assumed in RAS 1mm^3 space. You may
        want to adjust those if your streamlines are not in world coordinates.

        References
        ----------
        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
            bundles using local and global streamline-based registration and
            clustering, Neuroimage, 2017.
        """
        map_ind = np.zeros(len(streamlines))
        for i in range(len(streamlines)):
            map_ind[i] = check_range(streamlines[i], greater_than, less_than)
        map_ind = map_ind.astype(bool)

        self.orig_indices = np.array(list(range(0, len(streamlines))))
        self.filtered_indices = np.array(self.orig_indices[map_ind])
        self.streamlines = Streamlines(streamlines[map_ind])
        self.nb_streamlines = len(self.streamlines)
        self.verbose = verbose
        if self.verbose:
            logger.info(f"target brain streamlines length = {len(streamlines)}")
            logger.info(f"After refining target brain streamlines"
                        f" length = {len(self.streamlines)}")

        self.start_thr = [40, 25, 20]
        if rng is None:
            self.rng = np.random.default_rng()
        else:
            self.rng = rng

        if cluster_map is None:
            self._cluster_streamlines(clust_thr=clust_thr, nb_pts=nb_pts)
        else:
            if self.verbose:
                t = time()

            self.cluster_map = cluster_map
            self.cluster_map.refdata = self.streamlines
            self.centroids = self.cluster_map.centroids
            self.nb_centroids = len(self.centroids)
            self.indices = [cluster.indices for cluster in self.cluster_map]

            if self.verbose:
                logger.info(f' Streamlines have {self.nb_centroids} centroids')
                logger.info(f' Total loading duration {time() - t:0.3f} s\n')

    def _cluster_streamlines(self, clust_thr, nb_pts):

        if self.verbose:
            t = time()
            logger.info('# Cluster streamlines using QBx')
            logger.info(f' Tractogram has {len(self.streamlines)} streamlines')
            logger.info(f' Size is {nbytes(self.streamlines):0.3f} MB')
            logger.info(f' Distance threshold {clust_thr:0.3f}')

        # TODO this needs to become a default parameter
        thresholds = self.start_thr + [clust_thr]

        merged_cluster_map = qbx_and_merge(self.streamlines, thresholds,
                                           nb_pts, None, self.rng,
                                           self.verbose)

        self.cluster_map = merged_cluster_map
        self.centroids = merged_cluster_map.centroids
        self.nb_centroids = len(self.centroids)
        self.indices = [cluster.indices for cluster in self.cluster_map]

        if self.verbose:
            logger.info(f' Streamlines have {self.nb_centroids} centroids')
            logger.info(f' Total duration {time() - t:0.3f} s\n')

    def recognize(self, model_bundle, model_clust_thr,
                  reduction_thr=10,
                  reduction_distance='mdf',
                  slr=True,
                  num_threads=None,
                  slr_metric=None,
                  slr_x0=None,
                  slr_bounds=None,
                  slr_select=(400, 600),
                  slr_method='L-BFGS-B',
                  pruning_thr=5,
                  pruning_distance='mdf'):
        """ Recognize the model_bundle in self.streamlines

        Parameters
        ----------
        model_bundle : Streamlines
            model bundle streamlines used as a reference to extract similar
            streamlines from input tractogram
        model_clust_thr : float
            MDF distance threshold for the model bundles
        reduction_thr : float, optional
            Reduce search space in the target tractogram by (mm) (default 10)
        reduction_distance : string, optional
            Reduction distance type can be mdf or mam (default mdf)
        slr : bool, optional
            Use Streamline-based Linear Registration (SLR) locally
            (default True)
        num_threads : int, optional
            Number of threads to be used for OpenMP parallelization. If None
            (default) the value of OMP_NUM_THREADS environment variable is used
            if it is set, otherwise all available threads are used. If < 0 the
            maximal number of threads minus |num_threads + 1| is used (enter -1
            to use as many threads as possible). 0 raises an error.
        slr_metric : BundleMinDistanceMetric
        slr_x0 : array or int or str, optional
            Transformation allowed. translation, rigid, similarity or scaling
            Initial parametrization for the optimization.

            If 1D array with:
                a) 6 elements then only rigid registration is performed with
                the 3 first elements for translation and 3 for rotation.
                b) 7 elements also isotropic scaling is performed (similarity).
                c) 12 elements then translation, rotation (in degrees),
                scaling and shearing are performed (affine).

                Here is an example of x0 with 12 elements:
                ``x0=np.array([0, 10, 0, 40, 0, 0, 2., 1.5, 1, 0.1, -0.5, 0])``

                This has translation (0, 10, 0), rotation (40, 0, 0) in
                degrees, scaling (2., 1.5, 1) and shearing (0.1, -0.5, 0).

            If int:
                a) 6
                    ``x0 = np.array([0, 0, 0, 0, 0, 0])``
                b) 7
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1.])``
                c) 12
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])``

            If str:
                a) "rigid"
                    ``x0 = np.array([0, 0, 0, 0, 0, 0])``
                b) "similarity"
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1.])``
                c) "affine"
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])
            (default None)
        slr_bounds : array, optional
            (default None)
        slr_select : tuple, optional
            Select the number of streamlines from model to neirborhood of
            model to perform the local SLR.
        slr_method : string, optional
            Optimization method 'L_BFGS_B' or 'Powell' optimizers can be used.
            (default 'L-BFGS-B')
        pruning_thr : float, optional
            Pruning after reducing the search space (default 5).
        pruning_distance : string, optional
            Pruning distance type can be mdf or mam (default mdf)

        Returns
        -------
        recognized_transf : Streamlines
            Recognized bundle in the space of the model tractogram
        recognized_labels : array
            Indices of recognized bundle in the original tractogram

        References
        ----------
        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
            bundles using local and global streamline-based registration and
            clustering, Neuroimage, 2017.
        """
        if self.verbose:
            t = time()
            logger.info('## Recognize given bundle ## \n')

        model_centroids = self._cluster_model_bundle(
            model_bundle,
            model_clust_thr=model_clust_thr)

        neighb_streamlines, neighb_indices = self._reduce_search_space(
            model_centroids,
            reduction_thr=reduction_thr,
            reduction_distance=reduction_distance)

        if len(neighb_streamlines) == 0:
            return Streamlines([]), []

        if slr:
            transf_streamlines, slr1_bmd = self._register_neighb_to_model(
                model_bundle,
                neighb_streamlines,
                metric=slr_metric,
                x0=slr_x0,
                bounds=slr_bounds,
                select_model=slr_select[0],
                select_target=slr_select[1],
                method=slr_method,
                num_threads=num_threads)
        else:
            transf_streamlines = neighb_streamlines

        pruned_streamlines, labels = self._prune_what_not_in_model(
            model_centroids,
            transf_streamlines,
            neighb_indices,
            pruning_thr=pruning_thr,
            pruning_distance=pruning_distance)
        if self.verbose:
            logger.info(f'Total duration of recognition time'
                        f' is {time()-t:0.3f} s\n')

        return pruned_streamlines, self.filtered_indices[labels]

    def refine(self, model_bundle, pruned_streamlines, model_clust_thr,
               reduction_thr=14,
               reduction_distance='mdf',
               slr=True,
               slr_metric=None,
               slr_x0=None,
               slr_bounds=None,
               slr_select=(400, 600),
               slr_method='L-BFGS-B',
               pruning_thr=6,
               pruning_distance='mdf'):
        """ Refine and recognize the model_bundle in self.streamlines
        This method expects once pruned streamlines as input. It refines the
        first output of recobundle by applying second local slr (optional),
        and second pruning. This method is useful when we are dealing with
        noisy data or when we want to extract small tracks from tractograms.
        This time, search space is created using pruned bundle and not model
        bundle.

        Parameters
        ----------
        model_bundle : Streamlines
            model bundle streamlines used as a reference to extract similar
            streamlines from input tractogram
        pruned_streamlines : Streamlines
            Recognized bundle from target tractogram by RecoBundles.
        model_clust_thr : float
            MDF distance threshold for the model bundles
        reduction_thr : float
            Reduce search space by (mm) (default 14)
        reduction_distance : string
            Reduction distance type can be mdf or mam (default mdf)
        slr : bool
            Use Streamline-based Linear Registration (SLR) locally
            (default True)
        slr_metric : BundleMinDistanceMetric
        slr_x0 : array or int or str
            Transformation allowed. translation, rigid, similarity or scaling
            Initial parametrization for the optimization.

            If 1D array with:
                a) 6 elements then only rigid registration is performed with
                the 3 first elements for translation and 3 for rotation.
                b) 7 elements also isotropic scaling is performed (similarity).
                c) 12 elements then translation, rotation (in degrees),
                scaling and shearing are performed (affine).

                Here is an example of x0 with 12 elements:
                ``x0=np.array([0, 10, 0, 40, 0, 0, 2., 1.5, 1, 0.1, -0.5, 0])``

                This has translation (0, 10, 0), rotation (40, 0, 0) in
                degrees, scaling (2., 1.5, 1) and shearing (0.1, -0.5, 0).

            If int:
                a) 6
                    ``x0 = np.array([0, 0, 0, 0, 0, 0])``
                b) 7
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1.])``
                c) 12
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])``

            If str:
                a) "rigid"
                    ``x0 = np.array([0, 0, 0, 0, 0, 0])``
                b) "similarity"
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1.])``
                c) "affine"
                    ``x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])
            (default None)
        slr_bounds : array
            (default None)
        slr_select : tuple
            Select the number of streamlines from model to neirborhood of
            model to perform the local SLR.
        slr_method : string
            Optimization method 'L_BFGS_B' or 'Powell' optimizers can be used.
            (default 'L-BFGS-B')
        pruning_thr : float
            Pruning after reducing the search space (default 6).
        pruning_distance : string
            Pruning distance type can be mdf or mam (default mdf)

        Returns
        -------
        recognized_transf : Streamlines
            Recognized bundle in the space of the model tractogram
        recognized_labels : array
            Indices of recognized bundle in the original tractogram

        References
        ----------
        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
            bundles using local and global streamline-based registration and
            clustering, Neuroimage, 2017.

        .. [Chandio2020] Chandio, B.Q., Risacher, S.L., Pestilli, F.,
        Bullock, D., Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and
        Garyfallidis, E. Bundle analytics, a computational framework for
        investigating the shapes and profiles of brain pathways across
        populations. Sci Rep 10, 17149 (2020)
        """
        if self.verbose:
            t = time()
            logger.info('## Refine recognize given bundle ## \n')

        model_centroids = self._cluster_model_bundle(
            model_bundle,
            model_clust_thr=model_clust_thr)

        pruned_model_centroids = self._cluster_model_bundle(
            pruned_streamlines,
            model_clust_thr=model_clust_thr)

        neighb_streamlines, neighb_indices = self._reduce_search_space(
            pruned_model_centroids,
            reduction_thr=reduction_thr,
            reduction_distance=reduction_distance)

        if len(neighb_streamlines) == 0:  # if no streamlines recognized
            return Streamlines([]), []

        if self.verbose:
            logger.info("2nd local Slr")

        if slr:
            transf_streamlines, slr2_bmd = self._register_neighb_to_model(
                model_bundle,
                neighb_streamlines,
                metric=slr_metric,
                x0=slr_x0,
                bounds=slr_bounds,
                select_model=slr_select[0],
                select_target=slr_select[1],
                method=slr_method)
        if self.verbose:
            logger.info("pruning after 2nd local Slr")

        pruned_streamlines, labels = self._prune_what_not_in_model(
            model_centroids,
            transf_streamlines,
            neighb_indices,
            pruning_thr=pruning_thr,
            pruning_distance=pruning_distance)

        if self.verbose:
            logger.info(f'Total duration of recognition time'
                        f' is {time()-t:0.3f} s\n')

        return pruned_streamlines, self.filtered_indices[labels]

    def evaluate_results(self, model_bundle, pruned_streamlines, slr_select):
        """ Compare the similarity between two given bundles, model bundle,
        and extracted bundle.

        Parameters
        ----------
        model_bundle : Streamlines
        pruned_streamlines : Streamlines
        slr_select : tuple
            Select the number of streamlines from model to neirborhood of
            model to perform the local SLR.

        Returns
        -------
        ba_value : float
            bundle adjacency value between model bundle and pruned bundle
        bmd_value : float
            bundle minimum distance value between model bundle and
            pruned bundle
        """

        spruned_streamlines = Streamlines(pruned_streamlines)
        recog_centroids = self._cluster_model_bundle(
            spruned_streamlines,
            model_clust_thr=1.25)
        mod_centroids = self._cluster_model_bundle(
            model_bundle,
            model_clust_thr=1.25)
        recog_centroids = Streamlines(recog_centroids)
        model_centroids = Streamlines(mod_centroids)
        ba_value = bundle_adjacency(set_number_of_points(recog_centroids, 20),
                                    set_number_of_points(model_centroids, 20),
                                    threshold=10)

        BMD = BundleMinDistanceMetric()
        static = select_random_set_of_streamlines(model_bundle,
                                                  slr_select[0])
        moving = select_random_set_of_streamlines(pruned_streamlines,
                                                  slr_select[1])
        nb_pts = 20
        static = set_number_of_points(static, nb_pts)
        moving = set_number_of_points(moving, nb_pts)

        BMD.setup(static, moving)
        x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1, 0, 0, 0])  # affine
        bmd_value = BMD.distance(x0.tolist())

        return ba_value, bmd_value

    def _cluster_model_bundle(self, model_bundle, model_clust_thr, nb_pts=20,
                              select_randomly=500000):

        if self.verbose:
            t = time()
            logger.info('# Cluster model bundle using QBX')
            logger.info(f' Model bundle has {len(model_bundle)} streamlines')
            logger.info(f' Distance threshold {model_clust_thr:0.3f}')
        thresholds = self.start_thr + [model_clust_thr]

        model_cluster_map = qbx_and_merge(model_bundle, thresholds,
                                          nb_pts=nb_pts,
                                          select_randomly=select_randomly,
                                          rng=self.rng)
        model_centroids = model_cluster_map.centroids
        nb_model_centroids = len(model_centroids)
        if self.verbose:
            logger.info(f' Model bundle has {nb_model_centroids} centroids')
            logger.info(f' Duration {time() - t:0.3f} s\n')
        return model_centroids

    def _reduce_search_space(self, model_centroids,
                             reduction_thr=20, reduction_distance='mdf'):
        if self.verbose:
            t = time()
            logger.info('# Reduce search space')
            logger.info(f' Reduction threshold {reduction_thr:0.3f}')
            logger.info(f' Reduction distance {reduction_distance}')

        if reduction_distance.lower() == 'mdf':
            if self.verbose:
                logger.info(' Using MDF')
            centroid_matrix = bundles_distances_mdf(model_centroids,
                                                    self.centroids)
        elif reduction_distance.lower() == 'mam':
            if self.verbose:
                logger.info(' Using MAM')
            centroid_matrix = bundles_distances_mam(model_centroids,
                                                    self.centroids)
        else:
            raise ValueError('Given reduction distance not known')

        centroid_matrix[centroid_matrix > reduction_thr] = np.inf

        mins = np.min(centroid_matrix, axis=0)
        close_clusters_indices = list(np.where(mins != np.inf)[0])

        close_clusters = self.cluster_map[close_clusters_indices]

        neighb_indices = [cluster.indices for cluster in close_clusters]

        neighb_streamlines = Streamlines(chain(*close_clusters))

        nb_neighb_streamlines = len(neighb_streamlines)

        if nb_neighb_streamlines == 0:
            if self.verbose:
                logger.info('You have no neighbor streamlines... ' +
                            'No bundle recognition')
            return Streamlines([]), []
        if self.verbose:
            logger.info(f' Number of neighbor streamlines'
                        f' {nb_neighb_streamlines}')
            logger.info(f' Duration {time() - t:0.3f} s\n')

        return neighb_streamlines, neighb_indices

    def _register_neighb_to_model(self, model_bundle, neighb_streamlines,
                                  metric=None, x0=None, bounds=None,
                                  select_model=400, select_target=600,
                                  method='L-BFGS-B',
                                  nb_pts=20, num_threads=None):
        if self.verbose:
            logger.info('# Local SLR of neighb_streamlines to model')
            t = time()

        if metric is None or metric == 'symmetric':
            metric = BundleMinDistanceMetric(num_threads=num_threads)
        if metric == 'asymmetric':
            metric = BundleMinDistanceAsymmetricMetric()
        if metric == 'diagonal':
            metric = BundleSumDistanceMatrixMetric()

        if x0 is None:
            x0 = 'similarity'

        if bounds is None:
            bounds = [(-30, 30), (-30, 30), (-30, 30),
                      (-45, 45), (-45, 45), (-45, 45), (0.8, 1.2)]

        # TODO this can be speeded up by using directly the centroids
        static = select_random_set_of_streamlines(model_bundle,
                                                  select_model, rng=self.rng)
        moving = select_random_set_of_streamlines(neighb_streamlines,
                                                  select_target, rng=self.rng)

        static = set_number_of_points(static, nb_pts)
        moving = set_number_of_points(moving, nb_pts)

        slr = StreamlineLinearRegistration(metric=metric, x0=x0,
                                           bounds=bounds,
                                           method=method)
        slm = slr.optimize(static, moving)

        transf_streamlines = neighb_streamlines.copy()
        transf_streamlines._data = apply_affine(
            slm.matrix, transf_streamlines._data)

        transf_matrix = slm.matrix
        slr_bmd = slm.fopt
        slr_iterations = slm.iterations

        if self.verbose:
            logger.info(f' Square-root of BMD is {np.sqrt(slr_bmd):.3f}')
            if slr_iterations is not None:
                logger.info(f' Number of iterations {slr_iterations}')
            logger.info(f' Matrix size {slm.matrix.shape}')
            original = np.get_printoptions()
            np.set_printoptions(3, suppress=True)
            logger.info(transf_matrix)
            logger.info(slm.xopt)
            np.set_printoptions(**original)

            logger.info(f' Duration {time() - t:0.3f} s\n')

        return transf_streamlines, slr_bmd

    def _prune_what_not_in_model(self, model_centroids,
                                 transf_streamlines,
                                 neighb_indices,
                                 mdf_thr=5,
                                 pruning_thr=10,
                                 pruning_distance='mdf'):
        if self.verbose:
            if pruning_thr < 0:
                logger.info('Pruning_thr has to be greater or equal to 0')

            logger.info('# Prune streamlines using the MDF distance')
            logger.info(f' Pruning threshold {pruning_thr:0.3f}')
            logger.info(f' Pruning distance {pruning_distance}')
            t = time()

        thresholds = [40, 30, 20, 10, mdf_thr]
        rtransf_cluster_map = qbx_and_merge(transf_streamlines,
                                            thresholds, nb_pts=20,
                                            select_randomly=500000,
                                            rng=self.rng)
        if self.verbose:
            logger.info(f' QB Duration {time() - t:0.3f} s\n')

        rtransf_centroids = rtransf_cluster_map.centroids

        if pruning_distance.lower() == 'mdf':
            if self.verbose:
                logger.info(' Using MDF')
            dist_matrix = bundles_distances_mdf(model_centroids,
                                                rtransf_centroids)
        elif pruning_distance.lower() == 'mam':
            if self.verbose:
                logger.info(' Using MAM')
            dist_matrix = bundles_distances_mam(model_centroids,
                                                rtransf_centroids)
        else:
            raise ValueError('Given pruning distance is not available')
        dist_matrix[np.isnan(dist_matrix)] = np.inf
        dist_matrix[dist_matrix > pruning_thr] = np.inf

        pruning_matrix = dist_matrix.copy()
        if self.verbose:
            logger.info(' Pruning matrix size is (%d, %d)'
                        % pruning_matrix.shape)

        mins = np.min(pruning_matrix, axis=0)
        pruned_indices = [rtransf_cluster_map[i].indices
                          for i in np.where(mins != np.inf)[0]]
        pruned_indices = list(chain(*pruned_indices))
        idx = np.array(pruned_indices)
        if len(idx) == 0:
            if self.verbose:
                logger.info(' You have removed all streamlines')
            return Streamlines([]), []

        pruned_streamlines = transf_streamlines[idx]

        initial_indices = list(chain(*neighb_indices))
        final_indices = [initial_indices[i] for i in pruned_indices]
        labels = final_indices
        if self.verbose:
            logger.info(f' Number of centroids: {len(rtransf_centroids)}')
            logger.info(f' Number of streamlines after pruning:'
                        f' {len(pruned_streamlines)}')
            logger.info(f' Duration {time() - t:0.3f} s\n')

        return pruned_streamlines, labels
