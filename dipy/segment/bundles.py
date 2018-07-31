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
from time import time
from itertools import chain

from dipy.tracking.streamline import Streamlines
from nibabel.affines import apply_affine


class RecoBundles(object):

    def __init__(self, streamlines, cluster_map=None, clust_thr=15, nb_pts=20,
                 seed=42, verbose=True):
        """ Recognition of bundles

        Extract bundles from a participants' tractograms using model bundles
        segmented from a different subject or an atlas of bundles.
        See [Garyfallidis17]_ for the details.

        Parameters
        ----------
        streamlines : Streamlines
            The tractogram in which you want to recognize bundles.
        cluster_map : QB map
            Provide existing clustering to start RB faster (default None).
        clust_thr : float
            Distance threshold in mm for clustering `streamlines`
        seed : int
            Setup for random number generator (default 42).
        nb_pts : int
            Number of points per streamline (default 20)

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
        self.streamlines = streamlines

        self.nb_streamlines = len(self.streamlines)
        self.verbose = verbose

        self.start_thr = [40, 25, 20]

        if cluster_map is None:
            self._cluster_streamlines(clust_thr=clust_thr, nb_pts=nb_pts,
                                      seed=seed)
        else:
            if self.verbose:
                t = time()

            self.cluster_map = cluster_map
            self.cluster_map.refdata = self.streamlines
            self.centroids = self.cluster_map.centroids
            self.nb_centroids = len(self.centroids)
            self.indices = [cluster.indices for cluster in self.cluster_map]

            if self.verbose:
                print(' Streamlines have %d centroids'
                      % (self.nb_centroids,))
                print(' Total loading duration %0.3f sec. \n'
                      % (time() - t,))

    def _cluster_streamlines(self, clust_thr, nb_pts, seed):

        rng = np.random.RandomState(seed=seed)

        if self.verbose:
            t = time()
            print('# Cluster streamlines using QBx')
            print(' Tractogram has %d streamlines'
                  % (len(self.streamlines), ))
            print(' Size is %0.3f MB' % (nbytes(self.streamlines),))
            print(' Distance threshold %0.3f' % (clust_thr,))

        # TODO this needs to become a default parameter
        thresholds = self.start_thr + [clust_thr]

        merged_cluster_map = qbx_and_merge(self.streamlines, thresholds,
                                           nb_pts, None, rng, self.verbose)

        self.cluster_map = merged_cluster_map
        self.centroids = merged_cluster_map.centroids
        self.nb_centroids = len(self.centroids)
        self.indices = [cluster.indices for cluster in self.cluster_map]

        if self.verbose:
            print(' Streamlines have %d centroids'
                  % (self.nb_centroids,))
            print(' Total duration %0.3f sec. \n' % (time() - t,))

    def recognize(self, model_bundle, model_clust_thr,
                  reduction_thr=20,
                  reduction_distance='mdf',
                  slr=True,
                  slr_metric=None,
                  slr_x0=None,
                  slr_bounds=None,
                  slr_select=(400, 600),
                  slr_method='L-BFGS-B',
                  pruning_thr=10,
                  pruning_distance='mdf'):
        """ Recognize the model_bundle in self.streamlines

        Parameters
        ----------
        model_bundle : Streamlines
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
        recognized_bundle : Streamlines
            Recognized bundle in the space of the original tractogram

        References
        ----------
        .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
            bundles using local and global streamline-based registration and
            clustering, Neuroimage, 2017.
        """
        if self.verbose:
            t = time()
            print('## Recognize given bundle ## \n')

        model_centroids = self._cluster_model_bundle(
                model_bundle,
                model_clust_thr=model_clust_thr)
        neighb_streamlines, neighb_indices = self._reduce_search_space(
            model_centroids,
            reduction_thr=reduction_thr,
            reduction_distance=reduction_distance)
        if len(neighb_streamlines) == 0:
            return Streamlines([]), [], Streamlines([])
        if slr:
            transf_streamlines = self._register_neighb_to_model(
                model_bundle,
                neighb_streamlines,
                metric=slr_metric,
                x0=slr_x0,
                bounds=slr_bounds,
                select_model=slr_select[0],
                select_target=slr_select[1],
                method=slr_method)
        else:
            transf_streamlines = neighb_streamlines

        pruned_streamlines, labels = self._prune_what_not_in_model(
            model_centroids,
            transf_streamlines,
            neighb_indices,
            pruning_thr=pruning_thr,
            pruning_distance=pruning_distance)

        if self.verbose:
            print('Total duration of recognition time is %0.3f sec.\n'
                  % (time()-t,))
        # return recognized bundle in original streamlines, labels of
        # recognized bundle and transformed recognized bundle
        return pruned_streamlines, labels, self.streamlines[labels]

    def _cluster_model_bundle(self, model_bundle, model_clust_thr, nb_pts=20,
                              select_randomly=500000):

        if self.verbose:
            t = time()
            print('# Cluster model bundle using QBX')
            print(' Model bundle has %d streamlines'
                  % (len(model_bundle), ))
            print(' Distance threshold %0.3f' % (model_clust_thr,))
        thresholds = self.start_thr + [model_clust_thr]

        model_cluster_map = qbx_and_merge(model_bundle, thresholds,
                                          nb_pts=nb_pts,
                                          select_randomly=select_randomly,
                                          rng=None,
                                          verbose=self.verbose)
        model_centroids = model_cluster_map.centroids
        nb_model_centroids = len(model_centroids)

        if self.verbose:
            print(' Model bundle has %d centroids'
                  % (nb_model_centroids,))
            print(' Duration %0.3f sec. \n' % (time() - t, ))
        return model_centroids

    def _reduce_search_space(self, model_centroids,
                             reduction_thr=20, reduction_distance='mdf'):
        if self.verbose:
            t = time()
            print('# Reduce search space')
            print(' Reduction threshold %0.3f' % (reduction_thr,))
            print(' Reduction distance {}'.format(reduction_distance))

        if reduction_distance.lower() == 'mdf':
            if self.verbose:
                print(' Using MDF')
            centroid_matrix = bundles_distances_mdf(model_centroids,
                                                    self.centroids)
        elif reduction_distance.lower() == 'mam':
            if self.verbose:
                print(' Using MAM')
            centroid_matrix = bundles_distances_mdf(model_centroids,
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
            print(' You have no neighbor streamlines... No bundle recognition')
            return Streamlines([]), []

        if self.verbose:
            print(' Number of neighbor streamlines %d' %
                  (nb_neighb_streamlines,))
            print(' Duration %0.3f sec. \n' % (time() - t,))

        return neighb_streamlines, neighb_indices

    def _register_neighb_to_model(self, model_bundle, neighb_streamlines,
                                  metric=None, x0=None, bounds=None,
                                  select_model=400, select_target=600,
                                  method='L-BFGS-B',
                                  nb_pts=20):

        if self.verbose:
            print('# Local SLR of neighb_streamlines to model')
            t = time()

        if metric is None or metric == 'symmetric':
            metric = BundleMinDistanceMetric()
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
                                                  select_model)
        moving = select_random_set_of_streamlines(neighb_streamlines,
                                                  select_target)

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
            print(' Square-root of BMD is %.3f' % (np.sqrt(slr_bmd),))
            if slr_iterations is not None:
                print(' Number of iterations %d' % (slr_iterations,))
            print(' Matrix size {}'.format(slm.matrix.shape))
            original = np.get_printoptions()
            np.set_printoptions(3, suppress=True)
            print(transf_matrix)
            print(slm.xopt)
            np.set_printoptions(**original)

            print(' Duration %0.3f sec. \n' % (time() - t,))

        return transf_streamlines

    def _prune_what_not_in_model(self, model_centroids,
                                 transf_streamlines,
                                 neighb_indices,
                                 mdf_thr=5,
                                 pruning_thr=10,
                                 pruning_distance='mdf'):

        if pruning_thr < 0:
            print('Pruning_thr has to be greater or equal to 0')

        if self.verbose:
            print('# Prune streamlines using the MDF distance')
            print(' Pruning threshold %0.3f' % (pruning_thr,))
            print(' Pruning distance {}'.format(pruning_distance))
            t = time()

        thresholds = [40, 30, 20, 10, mdf_thr]
        rtransf_cluster_map = qbx_and_merge(transf_streamlines,
                                            thresholds, nb_pts=20,
                                            select_randomly=500000,
                                            rng=None,
                                            verbose=self.verbose)

        if self.verbose:
            print(' QB Duration %0.3f sec. \n' % (time() - t, ))

        rtransf_centroids = rtransf_cluster_map.centroids

        if pruning_distance.lower() == 'mdf':
            if self.verbose:
                print(' Using MDF')
            dist_matrix = bundles_distances_mdf(model_centroids,
                                                rtransf_centroids)
        elif pruning_distance.lower() == 'mam':
            if self.verbose:
                print(' Using MAM')
            dist_matrix = bundles_distances_mam(model_centroids,
                                                rtransf_centroids)
        else:
            raise ValueError('Given pruning distance is not available')
        dist_matrix[np.isnan(dist_matrix)] = np.inf
        dist_matrix[dist_matrix > pruning_thr] = np.inf

        pruning_matrix = dist_matrix.copy()

        if self.verbose:
            print(' Pruning matrix size is (%d, %d)'
                  % pruning_matrix.shape)

        mins = np.min(pruning_matrix, axis=0)
        pruned_indices = [rtransf_cluster_map[i].indices
                          for i in np.where(mins != np.inf)[0]]
        pruned_indices = list(chain(*pruned_indices))
        pruned_streamlines = [transf_streamlines[i]
                              for i in pruned_indices]

        initial_indices = list(chain(*neighb_indices))
        final_indices = [initial_indices[i] for i in pruned_indices]
        labels = final_indices

        if self.verbose:
            msg = ' Number of centroids: %d'
            print(msg % (len(rtransf_centroids),))
            msg = ' Number of streamlines after pruning: %d'
            print(msg % (len(pruned_streamlines),))

        if len(pruned_streamlines) == 0:
            print(' You have removed all streamlines')
            return Streamlines([]), []

        if self.verbose:
            print(' Duration %0.3f sec. \n' % (time() - t, ))

        return pruned_streamlines, labels
