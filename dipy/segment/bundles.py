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
        dtracks1 : Streamlines
        threshold: float
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
    A = len(pair12) / np.float(len(dtracks0))
    B = len(pair21) / np.float(len(dtracks1))
    res = 0.5 * (A + B)
    return res


class RecoBundles(object):

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
        rng : RandomState
            If None define RandomState in initialization function.
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
            logger.info("target brain streamlines length = %s" % len(streamlines))
            logger.info("After refining target brain streamlines" +
                        "length = %s" % len(self.streamlines))

        self.start_thr = [40, 25, 20]
        if rng is None:
            self.rng = np.random.RandomState()
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
                logger.info(' Streamlines have %d centroids'
                            % (self.nb_centroids,))
                logger.info(' Total loading duration %0.3f sec. \n'
                            % (time() - t,))

    def _cluster_streamlines(self, clust_thr, nb_pts):

        if self.verbose:
            t = time()
            logger.info('# Cluster streamlines using QBx')
            logger.info(' Tractogram has %d streamlines'
                        % (len(self.streamlines), ))
            logger.info(' Size is %0.3f MB' % (nbytes(self.streamlines),))
            logger.info(' Distance threshold %0.3f' % (clust_thr,))

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
            logger.info(' Streamlines have %d centroids'
                        % (self.nb_centroids,))
            logger.info(' Total duration %0.3f sec. \n' % (time() - t,))

    def recognize(self, model_bundle, model_clust_thr,
                  reduction_thr=10,
                  reduction_distance='mdf',
                  slr=True,
                  slr_num_threads=None,
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
        model_clust_thr : float
        reduction_thr : float
        reduction_distance : string
            mdf or mam (default mdf)
        slr : bool
            Use Streamline-based Linear Registration (SLR) locally
            (default True)
        slr_metric : BundleMinDistanceMetric
        slr_x0 : array
            (default None)
        slr_bounds : array
            (default None)
        slr_select : tuple
            Select the number of streamlines from model to neirborhood of
            model to perform the local SLR.
        slr_method : string
            Optimization method (default 'L-BFGS-B')
        pruning_thr : float
        pruning_distance : string
            MDF ('mdf') and MAM ('mam')

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
                num_threads=slr_num_threads)
        else:
            transf_streamlines = neighb_streamlines

        pruned_streamlines, labels = self._prune_what_not_in_model(
            model_centroids,
            transf_streamlines,
            neighb_indices,
            pruning_thr=pruning_thr,
            pruning_distance=pruning_distance)
        if self.verbose:
            logger.info('Total duration of recognition time is %0.3f sec.\n'
                        % (time()-t,))

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
        first ouput of recobundle by applying second local slr (optional),
        and second pruning. This method is useful when we are dealing with
        noisy data or when we want to extract small tracks from tractograms.

        Parameters
        ----------
        model_bundle : Streamlines
        pruned_streamlines : Streamlines
        model_clust_thr : float
        reduction_thr : float
        reduction_distance : string
            mdf or mam (default mam)
        slr : bool
            Use Streamline-based Linear Registration (SLR) locally
            (default True)
        slr_metric : BundleMinDistanceMetric
        slr_x0 : array
            (default None)
        slr_bounds : array
            (default None)
        slr_select : tuple
            Select the number of streamlines from model to neirborhood of
            model to perform the local SLR.
        slr_method : string
            Optimization method (default 'L-BFGS-B')
        pruning_thr : float
        pruning_distance : string
            MDF ('mdf') and MAM ('mam')

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
            logger.info('Total duration of recognition time is %0.3f sec.\n'
                        % (time()-t,))

        return pruned_streamlines, self.filtered_indices[labels]

    def evaluate_results(self, model_bundle, pruned_streamlines, slr_select):
        """ Compare the similiarity between two given bundles, model bundle,
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
            logger.info(' Model bundle has %d streamlines'
                        % (len(model_bundle), ))
            logger.info(' Distance threshold %0.3f' % (model_clust_thr,))
        thresholds = self.start_thr + [model_clust_thr]

        model_cluster_map = qbx_and_merge(model_bundle, thresholds,
                                          nb_pts=nb_pts,
                                          select_randomly=select_randomly,
                                          rng=self.rng)
        model_centroids = model_cluster_map.centroids
        nb_model_centroids = len(model_centroids)
        if self.verbose:
            logger.info(' Model bundle has %d centroids'
                        % (nb_model_centroids,))
            logger.info(' Duration %0.3f sec. \n' % (time() - t, ))
        return model_centroids

    def _reduce_search_space(self, model_centroids,
                             reduction_thr=20, reduction_distance='mdf'):
        if self.verbose:
            t = time()
            logger.info('# Reduce search space')
            logger.info(' Reduction threshold %0.3f' % (reduction_thr,))
            logger.info(' Reduction distance {}'.format(reduction_distance))

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
            logger.info(' Number of neighbor streamlines %d' %
                        (nb_neighb_streamlines,))
            logger.info(' Duration %0.3f sec. \n' % (time() - t,))

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
            logger.info(' Square-root of BMD is %.3f' % (np.sqrt(slr_bmd),))
            if slr_iterations is not None:
                logger.info(' Number of iterations %d' % (slr_iterations,))
            logger.info(' Matrix size {}'.format(slm.matrix.shape))
            original = np.get_printoptions()
            np.set_printoptions(3, suppress=True)
            logger.info(transf_matrix)
            logger.info(slm.xopt)
            np.set_printoptions(**original)

            logger.info(' Duration %0.3f sec. \n' % (time() - t,))

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
            logger.info(' Pruning threshold %0.3f' % (pruning_thr,))
            logger.info(' Pruning distance {}'.format(pruning_distance))
            t = time()

        thresholds = [40, 30, 20, 10, mdf_thr]
        rtransf_cluster_map = qbx_and_merge(transf_streamlines,
                                            thresholds, nb_pts=20,
                                            select_randomly=500000,
                                            rng=self.rng)
        if self.verbose:
            logger.info(' QB Duration %0.3f sec. \n' % (time() - t, ))

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
            msg = ' Number of centroids: %d'
            logger.info(msg % (len(rtransf_centroids),))
            msg = ' Number of streamlines after pruning: %d'
            logger.info(msg % (len(pruned_streamlines),))
            logger.info(' Duration %0.3f sec. \n' % (time() - t, ))

        return pruned_streamlines, labels
