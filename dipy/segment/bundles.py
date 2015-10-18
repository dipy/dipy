import numpy as np
from dipy.tracking.streamline import (transform_streamlines,
                                      set_number_of_points, nbytes,
                                      select_random_set_of_streamlines)
from dipy.segment.clustering import (QuickBundles,
                                     AveragePointwiseEuclideanMetric)
from dipy.segment.metric import IdentityFeature, ResampleFeature
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from dipy.align.streamlinear import (StreamlineLinearRegistration,
                                     BundleMinDistanceMetric,
                                     BundleSumDistanceMatrixMetric,
                                     BundleMinDistanceStaticMetric)
from dipy.align.bundlemin import distance_matrix_mdf
from time import time
from itertools import chain
from scipy.spatial import cKDTree


class RecoBundles(object):

    def __init__(self, streamlines, mdf_thr=20, verbose=True):

        self.streamlines = streamlines
        self.nb_streamlines = len(streamlines)
        self.verbose = verbose
        self.cluster_streamlines(mdf_thr=mdf_thr)

    def cluster_streamlines(self, mdf_thr=20, nb_pts=20,
                            select_randomly=50000):

        t = time()
        if self.verbose:
            print('# Cluster streamlines using QuickBundles')
            print(' Streamlines has %d streamlines'
                  % (len(self.streamlines), ))
            print(' Size is %0.3f MB' % (nbytes(self.streamlines),))
            print(' Distance threshold %0.3f' % (mdf_thr,))

        len_s = len(self.streamlines)
        indices = np.random.choice(len_s, min(select_randomly, len_s),
                                   replace=False)
        sample_streamlines = [self.streamlines[i]
                              for i in indices]
        sample_streamlines = set_number_of_points(sample_streamlines, nb_pts)
        sample_streamlines = [s.astype('f4') for s in sample_streamlines]

        if self.verbose:
            print(' Resampled to {} points'.format(nb_pts))
            print(' Size is %0.3f MB' % (nbytes(sample_streamlines),))
            print(' Duration %0.3f sec. \n' % (time() - t, ))

        feature = IdentityFeature()
        metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(threshold=mdf_thr, metric=metric)

        t1 = time()
        initial_clusters = qb.cluster(sample_streamlines)

        if self.verbose:
            print(' QuickBundles time for %d random streamlines'
                  % (select_randomly,))

            print(' Duration %0.3f sec. \n' % (time() - t1, ))

        t2 = time()

        # change back to 2
        # but don't use set_number do it directly
        rstreamlines_2pt = set_number_of_points(self.streamlines, 3)
        rstreamlines_2pt = [s.astype('f4') for s in rstreamlines_2pt]

        if self.verbose:
            print(' Resampling to 2 points %d' % (self.nb_streamlines,))
            print(' Size is %0.3f MB' % (nbytes(rstreamlines_2pt),))
            print(' Duration %0.3f sec. \n' % (time() - t2, ))

        from ipdb import set_trace
        set_trace()

        t3 = time()

        initial_centroids = initial_clusters.centroids

        for cluster in initial_clusters:
            cluster.centroid = set_number_of_points(cluster.centroid, 3)

        cluster_map = qb.assign(initial_clusters, rstreamlines_2pt)

        if self.verbose:
            print(' Assignment time for %d streamlines' % (len_s,))
            print(' Duration %0.3f sec. \n' % (time() - t3, ))

        cluster_map.refdata = self.streamlines
        self.cluster_map = cluster_map
        self.centroids = initial_centroids
        self.nb_centroids = len(self.centroids)
        self.indices = [cluster.indices for cluster in self.cluster_map]

        if self.verbose:
            print(' Streamlines have %d centroids'
                  % (self.nb_centroids,))
            print(' Total duration %0.3f sec. \n' % (time() - t, ))

    def recognize(self, model_bundle, mdf_thr=10,
                  reduction_thr=20,
                  slr=True,
                  slr_metric=None,
                  slr_x0=None,
                  slr_bounds=None,
                  slr_select=(400, 600),
                  slr_method='L-BFGS-B',
                  slr_use_centroids=False,
                  slr_progressive=False,
                  pruning_thr=10):

        t = time()
        if self.verbose:
            print('## Recognize given bundle ## \n')

        self.model_bundle = model_bundle
        self.cluster_model_bundle(mdf_thr=mdf_thr)
        self.reduce_search_space(reduction_thr=reduction_thr)
        if slr:
            self.register_neighb_to_model(metric=slr_metric,
                                          x0=slr_x0,
                                          bounds=slr_bounds,
                                          select_model=slr_select[0],
                                          select_target=slr_select[1],
                                          method=slr_method,
                                          use_centroids=slr_use_centroids,
                                          progressive=slr_progressive
                                          )
        else:
            self.transf_streamlines = self.neighb_streamlines
            self.transf_matrix = np.eye(4)
        self.prune_what_not_in_model(pruning_thr=pruning_thr)

        """
        if slr and repeat_slr:
            self.neighb_streamlines = self.pruned_streamlines
            self.register_neighb_to_model(select_model=slr_select[0],
                                          select_target=slr_select[1],
                                          method=slr_method)
            self.prune_what_not_in_model(pruning_thr=pruning_thr)
        """

        if self.verbose:
            print('Total duration of recognition time is %0.3f sec.\n'
                  % (time()-t,))
        return self.pruned_streamlines

    def cluster_model_bundle(self, mdf_thr=20, nb_pts=20):
        t = time()
        if self.verbose:
            print('# Cluster model bundle using QuickBundles')
            print(' Model bundle has %d streamlines'
                  % (len(self.model_bundle), ))
            print(' Distance threshold %0.3f' % (mdf_thr,))

        rmodel_bundle = set_number_of_points(self.model_bundle, nb_pts)
        rmodel_bundle = [s.astype('f4') for s in rmodel_bundle]

        self.resampled_model_bundle = rmodel_bundle

        feature = IdentityFeature()
        metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(threshold=mdf_thr, metric=metric)

        self.model_cluster_map = qb.cluster(rmodel_bundle)
        self.model_centroids = self.model_cluster_map.centroids
        self.nb_model_centroids = len(self.model_centroids)

        if self.verbose:
            print(' Model bundle has %d centroids'
                  % (self.nb_model_centroids,))
            print(' Duration %0.3f sec. \n' % (time() - t, ))

    def reduce_search_space(self, reduction_thr=20):
        t = time()
        if self.verbose:
            print('# Reduce search space')
            print(' Reduction threshold %0.3f' % (reduction_thr,))

        centroid_matrix = bundles_distances_mdf(self.model_centroids,
                                                self.centroids)

        centroid_matrix[centroid_matrix > reduction_thr] = np.inf

        mins = np.min(centroid_matrix, axis=0)
        close_clusters = [self.cluster_map[i]
                          for i in np.where(mins != np.inf)[0]]
        close_centroids = [cluster.centroid for cluster in close_clusters]
        close_indices = [cluster.indices for cluster in close_clusters]

        close_streamlines = list(chain(*close_clusters))

        self.centroid_matrix = centroid_matrix.copy()

        self.neighb_streamlines = close_streamlines
        self.neighb_clusters = close_clusters
        self.neighb_centroids = close_centroids
        self.neighb_indices = close_indices

        self.nb_neighb_streamlines = len(self.neighb_streamlines)

        if self.nb_neighb_streamlines == 0:
            print(' You have no neighbor streamlines... No bundle recognition')

        if self.verbose:
            print(' Number of neighbor streamlines %d' %
                  (self.nb_neighb_streamlines,))
            print(' Duration %0.3f sec. \n' % (time() - t, ))

    def register_neighb_to_model(self, metric=None, x0=None, bounds=None,
                                 select_model=400, select_target=600,
                                 method='L-BFGS-B',
                                 use_centroids=False,
                                 progressive=False,
                                 nb_pts=20):

        if self.verbose:
            print('# Local SLR of neighb_streamlines to model')

        t = time()

        if metric is None:
            metric = BundleMinDistanceMetric()
        if metric == 'static':
            metric = BundleMinDistanceStaticMetric()
        if metric == 'sum':
            metric = BundleSumDistanceMatrixMetric()

        if x0 is None:
            x0 = np.array([0, 0, 0, 0, 0, 0, 1.])

        if bounds is None:
            bounds = [(-30, 30), (-30, 30), (-30, 30),
                      (-45, 45), (-45, 45), (-45, 45), (0.8, 1.2)]

        if not use_centroids:
            static = select_random_set_of_streamlines(self.model_bundle,
                                                      select_model)
            moving = select_random_set_of_streamlines(self.neighb_streamlines,
                                                      select_target)

            static = set_number_of_points(static, nb_pts)
            moving = set_number_of_points(moving, nb_pts)

        else:
            static = self.model_centroids

            moving_all = set_number_of_points(self.neighb_streamlines, nb_pts)
            feature = IdentityFeature()
            metric = AveragePointwiseEuclideanMetric(feature)
            qb = QuickBundles(threshold=5, metric=metric)
            cluster_map = qb.cluster(moving_all)
            moving = cluster_map.centroids

        if progressive is False:

            slr = StreamlineLinearRegistration(metric=metric, x0=x0,
                                               bounds=bounds,
                                               method=method)
            slm = slr.optimize(static, moving)

        if progressive is True:

            if self.verbose:
                print('Progressive Registration is Enabled')

            if x0 == 'translation' or x0 == 'rigid' or \
               x0 == 'similarity' or x0 == 'scaling':

                slr_t = StreamlineLinearRegistration(metric=metric,
                                                     x0='translation',
                                                     bounds=bounds[:3],
                                                     method=method)

                slm_t = slr_t.optimize(static, moving)

            if x0 == 'rigid' or x0 == 'similarity' or x0 == 'scaling':

                x_translation = slm_t.xopt
                x = np.zeros(6)
                x[:3] = x_translation

                slr_r = StreamlineLinearRegistration(metric=metric,
                                                     x0=x,
                                                     bounds=bounds[:6],
                                                     method=method)
                slm_r = slr_r.optimize(static, moving)

            if x0 == 'similarity' or x0 == 'scaling':

                x_rigid = slm_r.xopt
                x = np.zeros(7)
                x[:6] = x_rigid
                x[6] = 1.

                slr_s = StreamlineLinearRegistration(metric=metric,
                                                     x0=x,
                                                     bounds=bounds[:7],
                                                     method=method)
                slm_s = slr_s.optimize(static, moving)

            if x0 == 'scaling':

                x_similarity = slm_s.xopt
                x = np.zeros(9)
                x[:6] = x_similarity[:6]
                # from ipdb import set_trace
                # set_trace()
                x[6:] = np.array((x_similarity[6],) * 3)

                slr_c = StreamlineLinearRegistration(metric=metric,
                                                     x0=x,
                                                     bounds=bounds[:9],
                                                     method=method)
                slm_c = slr_c.optimize(static, moving)

            if x0 == 'translation':
                slm = slm_t
            if x0 == 'rigid':
                slm = slm_r
            if x0 == 'similarity':
                slm = slm_s
            if x0 == 'scaling':
                slm = slm_c

        self.transf_streamlines = transform_streamlines(
            self.neighb_streamlines, slm.matrix)

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

    def prune_what_not_in_model(self, mdf_thr=5, pruning_thr=10):

        if pruning_thr < 0:
            print('Pruning_thr has to be greater or equal to 0')

        if self.verbose:
            print('# Prune streamlines using the MDF distance')
            print(' Pruning threshold %0.3f' % (pruning_thr,))

        t = time()

        rtransf_streamlines = set_number_of_points(self.transf_streamlines, 20)

        feature = IdentityFeature()
        metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(threshold=mdf_thr, metric=metric)
        rtransf_cluster_map = qb.cluster(rtransf_streamlines)

        if self.verbose:
            print(' QB Duration %0.3f sec. \n' % (time() - t, ))

        self.rtransf_streamlines = rtransf_streamlines
        self.rtransf_cluster_map = rtransf_cluster_map
        self.rtransf_centroids = rtransf_cluster_map.centroids
        self.nb_rtransf_centroids = len(self.rtransf_centroids)

        dist_matrix = bundles_distances_mdf(self.model_centroids,
                                            self.rtransf_centroids)

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


class KDTreeBundles(object):

    def __init__(self, streamlines, labels):

        self.model_bundle = [streamlines[i] for i in labels]
        self.nb_streamlines = len(streamlines)
        self.labels = labels
        self.streamlines = streamlines
        self.rstreamlines = set_number_of_points(streamlines, 20)

    def build_kdtree(self, nb_pts=20, mdf_thr=10, mam_metric='min',
                     leaf_size=10):

        feature = ResampleFeature(nb_points=nb_pts)
        metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(threshold=mdf_thr, metric=metric)
        cluster_map = qb.cluster(self.model_bundle)

        self.kdtree_is_built = False

        search_labels = np.setdiff1d(np.arange(self.nb_streamlines),
                                     np.array(self.labels))

        search_rstreamlines = [self.rstreamlines[i] for i in search_labels]

        rlabeled_streamlines = set_number_of_points(self.model_bundle,
                                                    nb_pts)

        if mam_metric is not None:
            vectors = bundles_distances_mam(search_rstreamlines,
                                            cluster_map.centroids,
                                            metric=mam_metric)

            internal_vectors = bundles_distances_mam(rlabeled_streamlines,
                                                     cluster_map.centroids,
                                                     metric=mam_metric)
        else:
            vectors = bundles_distances_mdf(search_rstreamlines,
                                            cluster_map.centroids)

            internal_vectors = bundles_distances_mdf(rlabeled_streamlines,
                                                     cluster_map.centroids)

        self.search_labels = search_labels
        self.kd_vectors = vectors
        self.kd_internal_vectors = internal_vectors
        self.kdtree = cKDTree(vectors, leafsize=leaf_size)
        self.kdtree_internal = cKDTree(internal_vectors, leafsize=leaf_size)
        self.kdtree_is_built = True

    def expand(self, nb_nn, return_streamlines=False):
        """ Expand

        Parameters
        ----------
        nb_nn : int
            Number of nearest neighbors
        return_streamlines : bool
            Default is False.

        Returns
        -------
        dists : float
            Norm 2 distance from KDtree zero vector
        actual_indices : list
            Indices of new streamlines in the initial dataset
        new_streamlines : list of ndarrays (optional)
            Returned only if ``return_streamlines`` is True.
        """
        if not self.kdtree_is_built:
            if return_streamlines:
                return None, None, None
            else:
                return None, None

        dists, indices = self.kdtree.query(np.zeros(self.kd_vectors.shape[1]),
                                           nb_nn, p=2)
        actual_indices = [self.search_labels[i] for i in indices]
        if return_streamlines:
            new_streamlines = [self.streamlines[self.search_labels[i]]
                               for i in indices]
            return dists, actual_indices, new_streamlines
        return dists, actual_indices

    def reduce(self, nb_nn_reduced=1, return_streamlines=False):

        if not self.kdtree_is_built:
            if return_streamlines:
                return None, None, None
            else:
                return None, None

        cnt_max = self.kd_internal_vectors.shape[0]

        if nb_nn_reduced < cnt_max:
            cnt = cnt_max - nb_nn_reduced
        else:
            if return_streamlines:
                return None, None, None
            else:
                return None, None

        dists, indices = self.kdtree_internal.query(
            np.zeros(self.kd_internal_vectors.shape[1]), cnt, p=2)

        # from ipdb import set_trace
        # set_trace()
        if isinstance(indices, int):
            indices = [indices]

        actual_indices = [self.labels[i] for i in indices]
        if return_streamlines:
            new_streamlines = [self.streamlines[i]
                               for i in actual_indices]
            return dists, actual_indices, new_streamlines
        return dists, actual_indices


#def recognize_bundles(model_bundle, moved_streamlines,
#                      close_centroids_thr=20,
#                      clean_thr=7.,
#                      local_slr=True,
#                      expand_thr=None,
#                      scale_range=(0.8, 1.2),
#                      verbose=True,
#                      return_full=False):
#
#    if verbose:
#        print('# Centroids of model bundle')
#
#    t0 = time()
#
#    rmodel_bundle = set_number_of_points(model_bundle, 20)
#    rmodel_bundle = [s.astype('f4') for s in rmodel_bundle]
#
#    qb = QuickBundles(threshold=20)
#    model_cluster_map = qb.cluster(rmodel_bundle)
#    model_centroids = model_cluster_map.centroids
#
#    if verbose:
#        print('Duration %f ' % (time() - t0, ))
#
#    if verbose:
#        print('# Calculate centroids of moved_streamlines')
#
#    t = time()
#
#    rstreamlines = set_number_of_points(moved_streamlines, 20)
#    # qb.cluster had problem with f8
#    rstreamlines = [s.astype('f4') for s in rstreamlines]
#
#    cluster_map = qb.cluster(rstreamlines)
#    cluster_map.refdata = moved_streamlines
#
#    if verbose:
#        print('Duration %f ' % (time() - t, ))
#
#    if verbose:
#        print('# Find centroids which are close to the model_centroids')
#
#    t = time()
#
#    centroid_matrix = bundles_distances_mdf(model_centroids,
#                                            cluster_map.centroids)
#
#    centroid_matrix[centroid_matrix > close_centroids_thr] = np.inf
#
#    mins = np.min(centroid_matrix, axis=0)
#    close_clusters = [cluster_map[i] for i in np.where(mins != np.inf)[0]]
#
#    # close_centroids = [cluster.centroid for cluster in close_clusters]
#
#    close_streamlines = list(chain(*close_clusters))
#
#    if len(close_streamlines) > 20000:
#        print('Too many of close streamlines to process... subsampling to 20K')
#        close_streamlines = select_random_set_of_streamlines(close_streamlines,
#                                                             20000)
#
#    if verbose:
#        print('Duration %f secs.' % (time() - t, ))
#
#    out = []
#    if return_full:
#        # show_bundles(model_bundle, close_streamlines)
#        out.append(close_streamlines)
#
#    if local_slr:
#
#        if verbose:
#            print('# Local SLR of close_streamlines to model')
#
#        t = time()
#
#        x0 = np.array([0, 0, 0, 0, 0, 0, 1.])
#        bounds = [(-6, 6), (-6, 6), (-6, 6),
#                  (-5, 5), (-5, 5), (-5, 5), scale_range]
#
#        slr = StreamlineLinearRegistration(x0=x0, bounds=bounds)
#
#        static = select_random_set_of_streamlines(model_bundle, 400)
#
#        if verbose:
#            msg = 'Number of close streamlines: %d'
#            print(msg % (len(close_streamlines),))
#
#        if len(close_streamlines) == 0:
#            print('   You have no close streamlines... No bundle recognition')
#            if return_full:
#                return close_streamlines, None, None
#            return close_streamlines, None
#
#        moving = select_random_set_of_streamlines(close_streamlines, 600)
#
#        static = set_number_of_points(static, 20)
#        # static = [s.astype('f4') for s in static]
#        moving = set_number_of_points(moving, 20)
#        # moving = [m.astype('f4') for m in moving]
#
#        slm = slr.optimize(static, moving)
#
#        closer_streamlines = transform_streamlines(close_streamlines,
#                                                   slm.matrix)
#
#        if verbose:
#            print('Duration %f ' % (time() - t, ))
#
#        if return_full:
#            out.append(closer_streamlines)
#            # show_bundles(model_bundle, closer_streamlines)
#
#        matrix = slm.matrix
#    else:
#        closer_streamlines = close_streamlines
#        matrix = np.eye(4)
#
#    if verbose:
#        print('Number of closer streamlines: %d' % (len(closer_streamlines),))
#
#    if clean_thr > 0:
#        if verbose:
#            print('# Remove streamlines which are a bit far')
#
#        t = time()
#
#        rcloser_streamlines = set_number_of_points(closer_streamlines, 20)
#
#        # find the closer_streamlines that are closer than clean_thr
#        clean_matrix = bundles_distances_mdf(rmodel_bundle,
#                                             rcloser_streamlines)
#        clean_matrix[clean_matrix > clean_thr] = np.inf
#        mins = np.min(clean_matrix, axis=0)
#        close_clusters_clean = [closer_streamlines[i]
#                                for i in np.where(mins != np.inf)[0]]
#
#        if verbose:
#            msg = 'Number of streamlines after cleanup: %d'
#            print(msg % (len(close_clusters_clean),))
#
#        if len(close_clusters_clean) == 0:
#            print('   You have cleaned all your streamlines!')
#            if return_full:
#                return close_clusters_clean, None, None
#            return close_clusters_clean, None
#    else:
#        if verbose:
#            print('No cleaning up...')
#
#        close_clusters_clean = closer_streamlines
#
#    if verbose:
#        print('Duration %f ' % (time() - t, ))
#
#    if return_full:
#        out.append(close_clusters_clean)
#        # show_bundles(model_bundle, close_clusters_clean)
#
#    if expand_thr is not None:
#
#        if verbose:
#            print('# Start expansion strategy')
#
#        t = time()
#
#        rclose_clusters_clean = set_number_of_points(close_clusters_clean, 20)
#        expand_matrix = bundles_distances_mam(rclose_clusters_clean,
#                                              rcloser_streamlines)
#
#        expand_matrix[expand_matrix > expand_thr] = np.inf
#        mins = np.min(expand_matrix, axis=0)
#        expanded = [closer_streamlines[i]
#                    for i in np.where(mins != np.inf)[0]]
#
#        if verbose:
#            msg = 'Number of streamlines after expansion: %d'
#            print(msg % (len(expanded),))
#
#        print('Duration %f ' % (time() - t, ))
#
#        if return_full:
#            out.append(expanded)
#            return expanded, matrix, out
#        return expanded, matrix
#
#    msg = 'Total duration of bundle recognition is %0.4f seconds.'
#    print(msg % (time() - t0,))
#
#    if return_full:
#        return close_clusters_clean, matrix, out
#    return close_clusters_clean, matrix