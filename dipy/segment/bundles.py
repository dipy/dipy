import numpy as np
from dipy.tracking.streamline import (transform_streamlines,
                                      set_number_of_points, length,
                                      select_random_set_of_streamlines)
from dipy.segment.clustering import (QuickBundlesX,
                                     ClusterMapCentroid, ClusterCentroid,
                                     AveragePointwiseEuclideanMetric)
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from dipy.align.streamlinear import (StreamlineLinearRegistration,
                                     BundleMinDistanceMetric,
                                     BundleSumDistanceMatrixMetric,
                                     BundleMinDistanceStaticMetric)
from dipy.align.bundlemin import distance_matrix_mdf
from time import time
from itertools import chain

from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.affines import apply_affine


def check_range(streamline, gt, lt):
    length_s = length(streamline)
    if (length_s > gt) & (length_s < lt):
        return True
    else:
        return False


def nbytes(streamlines):
    return streamlines._data.nbytes / 1024. ** 2


def qbx_with_merge(streamlines, thresholds,
                   nb_pts=20, select_randomly=None, verbose=True):

    t = time()
    len_s = len(streamlines)
    if select_randomly is None:
        select_randomly = len_s
    indices = np.random.choice(len_s, min(select_randomly, len_s),
                               replace=False)
    sample_streamlines = set_number_of_points(streamlines, nb_pts)

    if verbose:
        print(' Resampled to {} points'.format(nb_pts))
        print(' Size is %0.3f MB' % (nbytes(sample_streamlines),))
        print(' Duration of resampling is %0.3f sec.' % (time() - t,))

    if verbose:
        print(' QBX phase starting...')

    qbx = QuickBundlesX(thresholds,
                        metric=AveragePointwiseEuclideanMetric())

    t1 = time()
    qbx_clusters = qbx.cluster(sample_streamlines, ordering=indices)

    if verbose:
        print(' Merging phase starting ...')

    qbx_merge = QuickBundlesX([thresholds[-1]],
                              metric=AveragePointwiseEuclideanMetric())

    final_level = len(thresholds)

    qbx_ordering_final = np.random.choice(
        len(qbx_clusters.get_clusters(final_level)),
        len(qbx_clusters.get_clusters(final_level)), replace=False)

    qbx_merged_cluster_map = qbx_merge.cluster(
        qbx_clusters.get_clusters(final_level).centroids,
        ordering=qbx_ordering_final).get_clusters(1)

    qbx_cluster_map = qbx_clusters.get_clusters(final_level)

    merged_cluster_map = ClusterMapCentroid()
    for cluster in qbx_merged_cluster_map:
        merged_cluster = ClusterCentroid(centroid=cluster.centroid)
        for i in cluster.indices:
            merged_cluster.indices.extend(qbx_cluster_map[i].indices)
        merged_cluster_map.add_cluster(merged_cluster)

    merged_cluster_map.refdata = streamlines

    if verbose:
        print(' QuickBundlesX time for %d random streamlines'
              % (select_randomly,))

        print(' Duration %0.3f sec. \n' % (time() - t1, ))

    return merged_cluster_map


class RecoBundles(object):

    def __init__(self, streamlines, cluster_map=None, clust_thr=15,
                 verbose=True):

        self.clust_thr = clust_thr
        self.streamlines = streamlines

        self.nb_streamlines = len(self.streamlines)
        self.verbose = verbose

        if cluster_map is None:
            self.cluster_streamlines(clust_thr=clust_thr)
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
                      % (time() - t, ))

    def cluster_streamlines(self, clust_thr=15, nb_pts=20):

        np.random.seed(42)

        if self.verbose:
            t = time()
            print('# Cluster streamlines using QBx')
            print(' Tractogram has %d streamlines'
                  % (len(self.streamlines), ))
            print(' Size is %0.3f MB' % (nbytes(self.streamlines),))
            print(' Distance threshold %0.3f' % (clust_thr,))

        # TODO this needs to become a default parameter
        thresholds = [40, 25, 20, clust_thr]

        merged_cluster_map = qbx_with_merge(self.streamlines, thresholds,
                                            nb_pts, None, self.verbose)

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

        self.reduction_thr = reduction_thr
        if self.verbose:
            t = time()
            print('## Recognize given bundle ## \n')

        self.model_bundle = model_bundle
        self.cluster_model_bundle(model_clust_thr=model_clust_thr)
        success = self.reduce_search_space(
            reduction_thr=reduction_thr,
            reduction_distance=reduction_distance)
        if not success:
            self.pruned_streamlines = None
            self.transf_streamlines = None
            self.transf_matrix = None
            self.labels = []
            return []
        if slr:
            self.register_neighb_to_model(metric=slr_metric,
                                          x0=slr_x0,
                                          bounds=slr_bounds,
                                          select_model=slr_select[0],
                                          select_target=slr_select[1],
                                          method=slr_method)
        else:
            self.transf_streamlines = self.neighb_streamlines
            self.transf_matrix = np.eye(4)
        self.prune_what_not_in_model(pruning_thr=pruning_thr,
                                     pruning_distance=pruning_distance)

        if self.verbose:
            print('Total duration of recognition time is %0.3f sec.\n'
                  % (time()-t,))
        return self.streamlines[self.labels], self.labels, self.pruned_streamlines

    def cluster_model_bundle(self, model_clust_thr, nb_pts=20):
        self.model_clust_thr = model_clust_thr
        
        if self.verbose:
            t = time()
            print('# Cluster model bundle using QBx')
            print(' Model bundle has %d streamlines'
                  % (len(self.model_bundle), ))
            print(' Distance threshold %0.3f' % (model_clust_thr,))
        thresholds = [40, 25, 20, model_clust_thr]

        self.model_cluster_map = qbx_with_merge(self.model_bundle, thresholds, nb_pts=nb_pts,
                                                select_randomly=500000, verbose=self.verbose)
        self.model_centroids = self.model_cluster_map.centroids
        self.nb_model_centroids = len(self.model_centroids)

        if self.verbose:
            print(' Model bundle has %d centroids'
                  % (self.nb_model_centroids,))
            print(' Duration %0.3f sec. \n' % (time() - t, ))

    def reduce_search_space(self, reduction_thr=20, reduction_distance='mdf'):
        if self.verbose:
            t = time()
            print('# Reduce search space')
            print(' Reduction threshold %0.3f' % (reduction_thr,))
            print(' Reduction distance {}'.format(reduction_distance))

        if reduction_distance.lower() == 'mdf':
            if self.verbose:
                print(' Using MDF')
            centroid_matrix = bundles_distances_mdf(self.model_centroids,
                                                    self.centroids)
        elif reduction_distance.lower() == 'mam':
            if self.verbose:
                print(' Using MAM')
            centroid_matrix = bundles_distances_mdf(self.model_centroids,
                                                    self.centroids)
        else:
            raise ValueError('Given reduction distance not known')

        centroid_matrix[centroid_matrix > reduction_thr] = np.inf

        mins = np.min(centroid_matrix, axis=0)
        close_clusters_indices = list(np.where(mins != np.inf)[0])

        close_clusters = self.cluster_map[close_clusters_indices]

        close_centroids = [self.centroids[i]
                           for i in close_clusters_indices]
        close_indices = [cluster.indices for cluster in close_clusters]

        close_streamlines = ArraySequence(chain(*close_clusters))
        self.centroid_matrix = centroid_matrix.copy()

        self.neighb_streamlines = close_streamlines
        self.neighb_clusters = close_clusters
        self.neighb_centroids = close_centroids
        self.neighb_indices = close_indices

        self.nb_neighb_streamlines = len(self.neighb_streamlines)

        if self.nb_neighb_streamlines == 0:
            print(' You have no neighbor streamlines... No bundle recognition')
            return False

        if self.verbose:
            print(' Number of neighbor streamlines %d' %
                  (self.nb_neighb_streamlines,))
            print(' Duration %0.3f sec. \n' % (time() - t,))

        return True

    def register_neighb_to_model(self, metric=None, x0=None, bounds=None,
                                 select_model=400, select_target=600,
                                 method='L-BFGS-B',
                                 nb_pts=20):

        if self.verbose:
            print('# Local SLR of neighb_streamlines to model')
            t = time()

        if metric is None or metric == 'symmetric':
            metric = BundleMinDistanceMetric()
        if metric == 'asymmetric':
            metric = BundleMinDistanceStaticMetric()
        if metric == 'diagonal':
            metric = BundleSumDistanceMatrixMetric()

        if x0 is None:
            x0 = 'similarity'

        if bounds is None:
            bounds = [(-30, 30), (-30, 30), (-30, 30),
                      (-45, 45), (-45, 45), (-45, 45), (0.8, 1.2)]

        # TODO this can be speeded up by using directly the centroids
        static = select_random_set_of_streamlines(self.model_bundle,
                                                  select_model)
        moving = select_random_set_of_streamlines(self.neighb_streamlines,
                                                  select_target)

        static = set_number_of_points(static, nb_pts)
        moving = set_number_of_points(moving, nb_pts)

        
        slr = StreamlineLinearRegistration(metric=metric, x0=x0,
                                           bounds=bounds,
                                           method=method)
        slm = slr.optimize(static, moving)

        self.transf_streamlines = self.neighb_streamlines.copy()
        self.transf_streamlines._data = apply_affine(slm.matrix, self.transf_streamlines._data)
        #self.transf_streamlines = transform_streamlines(
        #    self.neighb_streamlines, slm.matrix)

        self.transf_matrix = slm.matrix
        self.slr_bmd = slm.fopt
        self.slr_iterations = slm.iterations

        self.slr_initial_matrix = distance_matrix_mdf(
            static, moving)

        self.slr_final_matrix = distance_matrix_mdf(
            static, transform_streamlines(moving, slm.matrix))
        self.slr_xopt = slm.xopt

        if self.verbose:
            print(' Square-root of BMD is %.3f' % (np.sqrt(self.slr_bmd),))
            print(' Number of iterations %d' % (self.slr_iterations,))
            print(' Matrix size {}'.format(self.slr_final_matrix.shape))
            np.set_printoptions(3, suppress=True)
            print(self.transf_matrix)
            print(slm.xopt)
            np.set_printoptions()

            print(' Duration %0.3f sec. \n' % (time() - t,))

    def prune_what_not_in_model(self, mdf_thr=5, pruning_thr=10,
                                pruning_distance='mdf'):

        if pruning_thr < 0:
            print('Pruning_thr has to be greater or equal to 0')

        if self.verbose:
            print('# Prune streamlines using the MDF distance')
            print(' Pruning threshold %0.3f' % (pruning_thr,))
            print(' Pruning distance {}'.format(pruning_distance))
            t = time()

        thresholds = [40, 30, 20, 10, mdf_thr]
        self.rtransf_cluster_map = qbx_with_merge(self.transf_streamlines, thresholds, nb_pts=20,
                                                  select_randomly=500000, verbose=self.verbose)

        if self.verbose:
            print(' QB Duration %0.3f sec. \n' % (time() - t, ))

        self.rtransf_centroids = self.rtransf_cluster_map.centroids
        self.nb_rtransf_centroids = len(self.rtransf_centroids)

        if pruning_distance.lower() == 'mdf':
            if self.verbose:
                print(' Using MDF')
            dist_matrix = bundles_distances_mdf(self.model_centroids,
                                                self.rtransf_centroids)
        elif pruning_distance.lower() == 'mam':
            if self.verbose:
                print(' Using MAM')
            dist_matrix = bundles_distances_mam(self.model_centroids,
                                                self.rtransf_centroids)
        else:
            raise ValueError('Given pruning distance is not available')
        dist_matrix[np.isnan(dist_matrix)] = np.inf
        dist_matrix[dist_matrix > pruning_thr] = np.inf

        self.pruning_matrix = dist_matrix.copy()

        if self.verbose:
            print(' Pruning matrix size is (%d, %d)'
                  % self.pruning_matrix.shape)

        mins = np.min(self.pruning_matrix, axis=0)
        pruned_indices = [self.rtransf_cluster_map[i].indices
                          for i in np.where(mins != np.inf)[0]]
        pruned_indices = list(chain(*pruned_indices))
        pruned_streamlines = [self.transf_streamlines[i]
                              for i in pruned_indices]

        self.pruned_indices = pruned_indices
        self.pruned_streamlines = pruned_streamlines
        self.nb_pruned_streamlines = len(pruned_streamlines)

        initial_indices = list(chain(*self.neighb_indices))
        final_indices = [initial_indices[i] for i in pruned_indices]
        self.labels = final_indices
        self.labeled_streamlines = [self.streamlines[i]
                                    for i in final_indices]

        if self.verbose:
            msg = ' Number of centroids: %d'
            print(msg % (self.nb_rtransf_centroids,))
            msg = ' Number of streamlines after pruning: %d'
            print(msg % (self.nb_pruned_streamlines,))

        if self.nb_pruned_streamlines == 0:
            print(' You have removed all streamlines')

        if self.verbose:
            print(' Duration %0.3f sec. \n' % (time() - t, ))

