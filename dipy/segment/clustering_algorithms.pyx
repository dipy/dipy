# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import itertools
import numpy as np
cimport numpy as np

from cythonutils cimport Data2D, shape2tuple
from metricspeed cimport Metric
from clusteringspeed cimport ClustersListCentroid, Centroid
from dipy.segment.clustering import ClusterMapCentroid, ClusterCentroid

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)
    void *memset(void *ptr, int value, size_t num)

DEF biggest_double = 1.7976931348623157e+308  # np.finfo('f8').max
DEF biggest_float = 3.4028235e+38  # np.finfo('f4').max
DEF biggest_int = 2147483647  # np.iinfo('i4').max


def clusterslist2clustermap(ClustersListCentroid clusters_list):
    clusters = ClusterMapCentroid()
    for i in range(clusters_list._nb_clusters):
        centroid = np.asarray(clusters_list._centroids[i].features)
        indices = np.asarray(<int[:clusters_list._clusters_size[i]]> clusters_list._clusters_indices[i]).tolist()
        clusters.add_cluster(ClusterCentroid(id=i, centroid=centroid, indices=indices))

    return clusters


cdef struct NearestCluster:
    int id
    double dist


def pick(A, indices):
    for idx in indices:
        yield A[idx]

def peek(iterable):
    iterable = iter(iterable)
    first = None
    rest = []
    try:
        first = iterable.next()
        rest = itertools.chain([first], iterable)
    except StopIteration:
        pass

    return rest, first


def quickbundles(streamlines, Metric metric, double threshold=10., long max_nb_clusters=biggest_int, ordering=None):
    # Threshold of np.inf is not supported, set it to 'biggest_double'
    threshold = min(threshold, biggest_double)
    # Threshold of -np.inf is not supported, set it to 0
    threshold = max(threshold, 0)

    if ordering is not None:
        ordering, ordering_bak = itertools.tee(ordering, 2)
        streamlines = pick(streamlines, ordering_bak)
    else:
        ordering = itertools.count()

    streamlines, streamline0 = peek(streamlines)

    if streamline0 is None:
        return ClusterMapCentroid()

    dtype = np.float32
    features_shape = shape2tuple(metric.feature.c_infer_shape(streamline0.astype(dtype)))
    cdef:
        int idx
        ClustersListCentroid clusters = ClustersListCentroid(features_shape)
        Data2D features_s_i = np.empty(features_shape, dtype=dtype)
        Data2D features_s_i_flip = np.empty(features_shape, dtype=dtype)

    for streamline, idx in itertools.izip(streamlines, ordering):
        if not streamline.flags.writeable or streamline.dtype != dtype:
            streamline = streamline.astype(dtype)

        cluster_id = _quickbundles_assignment_step(streamline, idx, metric, clusters, features_s_i, features_s_i_flip, threshold, max_nb_clusters)
        clusters.c_update(cluster_id)

    return clusterslist2clustermap(clusters)


cdef NearestCluster _find_nearest_cluster(Data2D features, Metric metric, ClustersListCentroid clusters) nogil except *:
    """ Finds the nearest cluster given a `features` vector. """
    cdef:
        np.npy_intp k
        double dist
        NearestCluster nearest_cluster

    nearest_cluster.id = -1
    nearest_cluster.dist = biggest_double

    for k in range(clusters.c_size()):
        dist = metric.c_dist(clusters._centroids[k].features, features)

        # Keep track of the nearest cluster
        if dist < nearest_cluster.dist:
            nearest_cluster.dist = dist
            nearest_cluster.id = k

    return nearest_cluster


cdef int _quickbundles_assignment_step(Data2D s_i, int streamline_idx, Metric metric, ClustersListCentroid clusters, Data2D features_s_i, Data2D features_s_i_flip, double threshold=10, int max_nb_clusters=biggest_int) nogil except -1:
    cdef:
        Data2D features_to_add = features_s_i
        NearestCluster nearest_cluster, nearest_cluster_flip

    # Check if streamline is compatible with the metric
    if not metric.c_compatible(metric.feature.c_infer_shape(s_i), clusters._features_shape):
        with gil:
            raise ValueError("Streamlines features' shapes must be compatible according to the metric used!")

    # Find nearest cluster to s_i
    metric.feature.c_extract(s_i, features_s_i)
    nearest_cluster = _find_nearest_cluster(features_s_i, metric, clusters)

    # Find nearest cluster to s_i_flip if metric is not order invariant
    if not metric.feature.is_order_invariant:
        metric.feature.c_extract(s_i[::-1], features_s_i_flip)
        nearest_cluster_flip = _find_nearest_cluster(features_s_i_flip, metric, clusters)

        # If we found a lower distance using a flipped streamline,
        #  add the flipped version instead
        if nearest_cluster_flip.dist < nearest_cluster.dist:
            nearest_cluster.id = nearest_cluster_flip.id
            nearest_cluster.dist = nearest_cluster_flip.dist
            features_to_add = features_s_i_flip

    # Check if distance with the nearest cluster is below some threshold
    # or if we already have the maximum number of clusters.
    # If the former or the latter is true, assign streamline to its nearest cluster
    # otherwise create a new cluster and assign the streamline to it.
    if not (nearest_cluster.dist < threshold or clusters.c_size() >= max_nb_clusters):
        nearest_cluster.id = clusters.c_create_cluster()

    clusters.c_assign(nearest_cluster.id, streamline_idx, features_to_add)
    return nearest_cluster.id


def kmeans(streamlines, Metric metric, int K, ordering=None, max_nb_iterations=biggest_int):
    if len(streamlines) == 0:
        return ClustersListCentroid((0, 0))

    if ordering is None:
        ordering = range(len(streamlines))

    streamline0 = streamlines[ordering[0]]

    dtype = streamline0.dtype
    features_shape = shape2tuple(metric.feature.c_infer_shape(streamline0))
    cdef:
        int idx
        ClustersListCentroid clusters = ClustersListCentroid(features_shape)
        Data2D features_s_i = np.empty(features_shape, dtype=dtype)
        Data2D features_s_i_flip = np.empty(features_shape, dtype=dtype)

    # Initialize K centroids (at random)
    random_indices = np.random.choice(ordering, size=K, replace=False)
    for i, idx in enumerate(random_indices):
        clusters.c_create_cluster()
        clusters.c_assign(i, idx, streamlines[idx])
        clusters.c_update(i)

    converge = False
    nb_itererations = 0
    while not converge and nb_itererations < max_nb_iterations:
        nb_itererations += 1
        clusters.c_clear()
        for idx in ordering:
            streamline = streamlines[idx] if streamlines[idx].flags.writeable else streamlines[idx].astype(dtype)
            _kmeans_assignment_step(streamline, idx, metric, clusters, features_s_i, features_s_i_flip)

        converge = _kmeans_update_step(clusters) == 1

    return clusters

cdef void _kmeans_assignment_step(Data2D s_i, int streamline_idx, Metric metric, ClustersListCentroid clusters, Data2D features_s_i, Data2D features_s_i_flip) nogil except *:
    threshold = biggest_double
    max_nb_clusters = 0
    _quickbundles_assignment_step(s_i, streamline_idx, metric, clusters, features_s_i, features_s_i_flip, threshold, max_nb_clusters)

cdef int _kmeans_update_step(ClustersListCentroid clusters) nogil except -1:
    cdef int cluster_id
    cdef int converged = 1
    for cluster_id in range(clusters.c_size()):
        converged &= clusters.c_update(cluster_id)

    return converged
