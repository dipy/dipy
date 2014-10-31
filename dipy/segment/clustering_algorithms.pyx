# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from cythonutils cimport Data2D
from metricspeed cimport Metric
from clusteringspeed cimport ClusterMapCentroid, Centroid


cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)
    void *memset(void *ptr, int value, size_t num)

DEF biggest_double = 1.7976931348623157e+308  # np.finfo('f8').max
DEF biggest_float = 3.4028235e+38  # np.finfo('f4').max
DEF biggest_int = 2147483647  # np.iinfo('i4').max


cpdef quickbundles(streamlines, Metric metric, double threshold=10., long max_nb_clusters=biggest_int, ordering=None):
    if len(streamlines) == 0:
        return ClusterMapCentroid((0, 0))

    if ordering is None:
        ordering = range(len(streamlines))

    # Threshold of np.inf is not supported, set it to 'biggest_double'
    threshold = min(threshold, biggest_double)
    # Threshold of -np.inf is not supported, set it to 0
    threshold = max(threshold, 0)

    dtype = streamlines[0].dtype
    features_shape = metric.feature.infer_shape(streamlines[0])
    cdef:
        int idx
        ClusterMapCentroid clusters = ClusterMapCentroid(features_shape)
        Data2D features_s_i = np.empty(features_shape, dtype=dtype)
        Data2D features_s_i_flip = np.empty(features_shape, dtype=dtype)

    for idx in ordering:
        streamline = streamlines[idx] if streamlines[idx].flags.writeable else streamlines[idx].astype(streamlines[idx].dtype)
        _quickbundles_step(streamline, idx, metric, clusters, features_s_i, features_s_i_flip, threshold, max_nb_clusters)

    return clusters


cdef void _quickbundles_step(Data2D s_i, int streamline_idx, Metric metric, ClusterMapCentroid clusters, Data2D features_s_i, Data2D features_s_i_flip, double threshold=10, int max_nb_clusters=biggest_int) nogil except *:
    cdef:
        Centroid* centroid
        int closest_cluster
        double dist, dist_min, dist_min_flip
        Data2D features_to_add = features_s_i

    # Find closest cluster to s_i
    metric.feature.c_extract(s_i, features_s_i)
    dist_min = biggest_double
    for k in range(clusters.c_size()):
        #centroid = clusters.c_get_centroid(k)
        #dist = metric.c_dist(centroid.features, features_s_i)
        dist = metric.c_dist(clusters._centroids[k].features, features_s_i)

        # Keep track of the closest cluster
        if dist < dist_min:
            dist_min = dist
            closest_cluster = k

    # Find closest cluster to s_i_flip if metric is not order invariant
    if not metric.feature.is_order_invariant:
        dist_min_flip = dist_min  # Initialize to the min distance not flipped.
        metric.feature.c_extract(s_i[::-1], features_s_i_flip)
        for k in range(clusters.c_size()):
            #centroid = clusters.c_get_centroid(k)
            #dist = metric.c_dist(centroid.features, features_s_i_flip)
            dist = metric.c_dist(clusters._centroids[k].features, features_s_i_flip)

            # Keep track of the closest cluster
            if dist < dist_min_flip:
                dist_min_flip = dist
                closest_cluster = k

        # If we found a lower distance using a flipped streamline,
        #  add the flipped version instead
        if dist_min_flip < dist_min:
            dist_min = dist_min_flip
            features_to_add = features_s_i_flip

    # Check if distance with the closest cluster is below some threshold
    # or if we already have the maximum number of clusters.
    # If the former or the latter is true, assign streamline to its closest cluster
    # otherwise create a new cluster and assign the streamline to it.
    if not (dist_min < threshold or clusters.c_size() >= max_nb_clusters):
        closest_cluster = clusters.c_create_cluster()

    clusters.c_add(closest_cluster, streamline_idx, features_to_add)
