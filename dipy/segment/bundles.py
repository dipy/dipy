import numpy as np
from dipy.tracking.streamline import (transform_streamlines,
                                      set_number_of_points,
                                      select_random_set_of_streamlines)
from dipy.segment.clustering import QuickBundles
from dipy.tracking.distances import (bundles_distances_mdf,
                                     bundles_distances_mam)
from dipy.align.streamlinear import StreamlineLinearRegistration
from time import time
from itertools import chain


def recognize_bundles(model_bundle, moved_streamlines,
                      close_centroids_thr=20,
                      clean_thr=7.,
                      local_slr=True,
                      verbose=True, expand_thr=None,
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
                  (-45, 45), (-45, 45), (-45, 45), (0.8, 1.2)]

        slr = StreamlineLinearRegistration(x0=x0, bounds=bounds)

        static = select_random_set_of_streamlines(model_bundle, 400)
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
        print('# Remove streamlines which are a bit far')

    t = time()

    rcloser_streamlines = set_number_of_points(closer_streamlines, 20)

    clean_matrix = bundles_distances_mdf(rmodel_bundle, rcloser_streamlines)

    clean_matrix[clean_matrix > clean_thr] = np.inf

    mins = np.min(clean_matrix, axis=0)
    close_clusters_clean = [closer_streamlines[i]
                            for i in np.where(mins != np.inf)[0]]

    if verbose:
        print('Duration %f ' % (time() - t, ))

    msg = 'Total duration of automatic extraction %0.4f seconds.'
    print(msg % (time() - t0, ))

    if return_full:
        out.append(close_clusters_clean)
        # show_bundles(model_bundle, close_clusters_clean)

    if expand_thr is not None:
        rclose_clusters_clean = set_number_of_points(close_clusters_clean, 20)
        expand_matrix = bundles_distances_mam(rclose_clusters_clean,
                                              rcloser_streamlines)

        expand_matrix[expand_matrix > expand_thr] = np.inf
        mins = np.min(expand_matrix, axis=0)
        expanded = [closer_streamlines[i]
                    for i in np.where(mins != np.inf)[0]]

        if return_full:
            return expanded, matrix, out
        return expanded, matrix

    if return_full:
        return close_clusters_clean, matrix, out
    return close_clusters_clean, matrix