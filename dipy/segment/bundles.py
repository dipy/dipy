import numpy as np
from dipy.tracking.streamline import (transform_streamlines,
                                      set_number_of_points,
                                      select_random_set_of_streamlines)
from dipy.segment.clustering import (QuickBundles,
                                     AveragePointwiseEuclideanMetric)
from dipy.segment.metric import IdentityFeature
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import time
from itertools import chain


class RecoBundles(object):

    def __init__(self, streamlines, mdf_thr=20, verbose=True):

        self.streamlines = streamlines
        self.verbose = verbose
        if self.verbose:
            print('## Recognition of Bundles ## \n')

        self.cluster_streamlines(mdf_thr=mdf_thr)

    def cluster_streamlines(self, mdf_thr=20, nb_pts=20):

        t = time()
        if self.verbose:
            print('# Starting clustering streamlines ...')
            print(' Streamlines has %d streamlines'
                  % (len(self.streamlines), ))
            print(' Algorithm used is QuickBundles')
            print(' Distance threshold %0.3f' % (mdf_thr,))

        rstreamlines = set_number_of_points(self.streamlines, nb_pts)
        rstreamlines = [s.astype('f4') for s in rstreamlines]

        feature = IdentityFeature()
        metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(threshold=mdf_thr, metric=metric)

        cluster_map = qb.cluster(rstreamlines)
        cluster_map.refdata = self.streamlines
        self.cluster_map = cluster_map
        self.centroids = self.cluster_map.centroids
        self.nb_centroids = len(self.centroids)
        self.indices = [cluster.indices for cluster in self.cluster_map]

        if self.verbose:
            print(' Streamlines have %d centroids'
                  % (self.nb_centroids,))
            print(' Duration %0.3f sec. \n' % (time() - t, ))

    def recognize(self, model_bundle, mdf_thr=10,
                  reduction_thr=20,
                  slr=True,
                  pruning_thr=10):

        t = time()

        self.model_bundle = model_bundle
        self.cluster_model_bundle(mdf_thr=mdf_thr)
        self.reduce_search_space(reduction_thr=reduction_thr)
        if slr:
            self.register_neighb_to_model()
        else:
            self.transf_streamlines = self.neighb_streamlines
            self.transf_matrix = np.eye(4)
        self.prune_what_not_in_model(pruning_thr=pruning_thr)

        if self.verbose:
            print('Total duration of recognition time is %0.3f sec.'
                  % (time()-t,))
        return self.pruned_streamlines

    def cluster_model_bundle(self, mdf_thr=20, nb_pts=20):
        t = time()
        if self.verbose:
            print('# Starting clustering model bundle ...')
            print(' Model bundle has %d streamlines'
                  % (len(self.model_bundle), ))
            print(' Algorithm used is QuickBundles')
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
            print('# Find centroids which are close to the model_centroids')

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
            print(' Duration %f sec. \n' % (time() - t, ))

    def register_neighb_to_model(self, x0=None, scale_range=(0.8, 1.2),
                                 select_model=400, select_target=600,
                                 nb_pts=20):

        if self.verbose:
            print('# Local SLR of neighb_streamlines to model')

        t = time()

        if x0 is None:
            x0 = np.array([0, 0, 0, 0, 0, 0, 1.])

        # if scale_range is None:
        bounds = [(-30, 30), (-30, 30), (-30, 30),
                  (-45, 45), (-45, 45), (-45, 45), scale_range]

        slr = StreamlineLinearRegistration(x0=x0, bounds=bounds)
        static = select_random_set_of_streamlines(self.model_bundle,
                                                  select_model)

        moving = select_random_set_of_streamlines(self.neighb_streamlines,
                                                  select_target)

        static = set_number_of_points(static, nb_pts)
        moving = set_number_of_points(moving, nb_pts)

        slm = slr.optimize(static, moving)

        self.transf_streamlines = transform_streamlines(
            self.neighb_streamlines, slm.matrix)

        self.transf_matrix = slm.matrix
        self.bmd = slm.fopt ** 2

        if self.verbose:
            print(' Squared BMD is %.3f' % (self.bmd,))
            print(' Duration %0.3f sec. \n' % (time() - t, ))

    def prune_what_not_in_model(self, mdf_thr=5, pruning_thr=10):

        if pruning_thr < 0:
            print('Pruning_thr has to be greater or equal to 0')

        if self.verbose:
            print('# Prune streamlines which have '
                  'different shapes in MDF terms')
            print(' Pruning threshold %0.3f' % (pruning_thr,))

        t = time()

        rtransf_streamlines = set_number_of_points(self.transf_streamlines, 20)

        feature = IdentityFeature()
        metric = AveragePointwiseEuclideanMetric(feature)
        qb = QuickBundles(threshold=mdf_thr, metric=metric)
        rtransf_cluster_map = qb.cluster(rtransf_streamlines)

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



# Think about this more carefuly
#        pruned_indices = [self.rtransf_cluster_map[i].indices
#                          for i in np.where(dist_matrix != np.inf)[1]]
#
#        pruned_indices = list(chain(*pruned_indices))
#        pruned_indices = list(np.unique(pruned_indices))
#
#        pruned_streamlines = [self.transf_streamlines[i]
#                              for i in pruned_indices]
#
#        self.pruned_indices = pruned_indices
        self.pruned_streamlines = pruned_streamlines
        self.nb_pruned_streamlines = len(pruned_streamlines)

        if self.verbose:
            msg = ' Number of centroids: %d'
            print(msg % (self.nb_rtransf_centroids,))
            msg = ' Number of streamlines after pruning: %d'
            print(msg % (self.nb_pruned_streamlines,))

        if self.nb_pruned_streamlines == 0:
            print(' You have removed all streamlines')

        if self.verbose:
            print(' Duration %0.3f sec. \n' % (time() - t, ))

    def shrink_with_shape_prior(self):
        pass

    def expand_with_shape_prior(self):
        pass


def recognize_bundles(model_bundle, moved_streamlines,
                      close_centroids_thr=20,
                      clean_thr=7.,
                      local_slr=True,
                      expand_thr=None,
                      scale_range=(0.8, 1.2),
                      verbose=True,
                      return_full=False):

    if verbose:
        print('# Centroids of model bundle')

    t0 = time()

    rmodel_bundle = set_number_of_points(model_bundle, 20)
    rmodel_bundle = [s.astype('f4') for s in rmodel_bundle]

    qb = QuickBundles(threshold=20)
    model_cluster_map = qb.cluster(rmodel_bundle)
    model_centroids = model_cluster_map.centroids

    if verbose:
        print('Duration %f ' % (time() - t0, ))

    if verbose:
        print('# Calculate centroids of moved_streamlines')

    t = time()

    rstreamlines = set_number_of_points(moved_streamlines, 20)
    # qb.cluster had problem with f8
    rstreamlines = [s.astype('f4') for s in rstreamlines]

    cluster_map = qb.cluster(rstreamlines)
    cluster_map.refdata = moved_streamlines

    if verbose:
        print('Duration %f ' % (time() - t, ))

    if verbose:
        print('# Find centroids which are close to the model_centroids')

    t = time()

    centroid_matrix = bundles_distances_mdf(model_centroids,
                                            cluster_map.centroids)

    centroid_matrix[centroid_matrix > close_centroids_thr] = np.inf

    mins = np.min(centroid_matrix, axis=0)
    close_clusters = [cluster_map[i] for i in np.where(mins != np.inf)[0]]

    # close_centroids = [cluster.centroid for cluster in close_clusters]

    close_streamlines = list(chain(*close_clusters))

    if len(close_streamlines) > 20000:
        print('Too many of close streamlines to process... subsampling to 20K')
        close_streamlines = select_random_set_of_streamlines(close_streamlines,
                                                             20000)

    if verbose:
        print('Duration %f secs.' % (time() - t, ))

    out = []
    if return_full:
        # show_bundles(model_bundle, close_streamlines)
        out.append(close_streamlines)

    if local_slr:

        if verbose:
            print('# Local SLR of close_streamlines to model')

        t = time()

        x0 = np.array([0, 0, 0, 0, 0, 0, 1.])
        bounds = [(-30, 30), (-30, 30), (-30, 30),
                  (-45, 45), (-45, 45), (-45, 45), scale_range]

        slr = StreamlineLinearRegistration(x0=x0, bounds=bounds)

        static = select_random_set_of_streamlines(model_bundle, 400)

        if verbose:
            msg = 'Number of close streamlines: %d'
            print(msg % (len(close_streamlines),))

        if len(close_streamlines) == 0:
            print('   You have no close streamlines... No bundle recognition')
            if return_full:
                return close_streamlines, None, None
            return close_streamlines, None

        moving = select_random_set_of_streamlines(close_streamlines, 600)

        static = set_number_of_points(static, 20)
        # static = [s.astype('f4') for s in static]
        moving = set_number_of_points(moving, 20)
        # moving = [m.astype('f4') for m in moving]

        slm = slr.optimize(static, moving)

        closer_streamlines = transform_streamlines(close_streamlines,
                                                   slm.matrix)

        if verbose:
            print('Duration %f ' % (time() - t, ))

        if return_full:
            out.append(closer_streamlines)
            # show_bundles(model_bundle, closer_streamlines)

        matrix = slm.matrix
    else:
        closer_streamlines = close_streamlines
        matrix = np.eye(4)

    if verbose:
        print('Number of closer streamlines: %d' % (len(closer_streamlines),))

    if clean_thr > 0:
        if verbose:
            print('# Remove streamlines which are a bit far')

        t = time()

        rcloser_streamlines = set_number_of_points(closer_streamlines, 20)

        # find the closer_streamlines that are closer than clean_thr
        clean_matrix = bundles_distances_mdf(rmodel_bundle,
                                             rcloser_streamlines)
        clean_matrix[clean_matrix > clean_thr] = np.inf
        mins = np.min(clean_matrix, axis=0)
        close_clusters_clean = [closer_streamlines[i]
                                for i in np.where(mins != np.inf)[0]]

        if verbose:
            msg = 'Number of streamlines after cleanup: %d'
            print(msg % (len(close_clusters_clean),))

        if len(close_clusters_clean) == 0:
            print('   You have cleaned all your streamlines!')
            if return_full:
                return close_clusters_clean, None, None
            return close_clusters_clean, None
    else:
        if verbose:
            print('No cleaning up...')

        close_clusters_clean = closer_streamlines

    if verbose:
        print('Duration %f ' % (time() - t, ))

    if return_full:
        out.append(close_clusters_clean)
        # show_bundles(model_bundle, close_clusters_clean)

    if expand_thr is not None:

        if verbose:
            print('# Start expansion strategy')

        t = time()

        rclose_clusters_clean = set_number_of_points(close_clusters_clean, 20)
        expand_matrix = bundles_distances_mam(rclose_clusters_clean,
                                              rcloser_streamlines)

        expand_matrix[expand_matrix > expand_thr] = np.inf
        mins = np.min(expand_matrix, axis=0)
        expanded = [closer_streamlines[i]
                    for i in np.where(mins != np.inf)[0]]

        if verbose:
            msg = 'Number of streamlines after expansion: %d'
            print(msg % (len(expanded),))

        print('Duration %f ' % (time() - t, ))

        if return_full:
            out.append(expanded)
            return expanded, matrix, out
        return expanded, matrix

    msg = 'Total duration of bundle recognition is %0.4f seconds.'
    print(msg % (time() - t0,))

    if return_full:
        return close_clusters_clean, matrix, out
    return close_clusters_clean, matrix