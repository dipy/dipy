# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False, initializedcheck=False

import itertools
import numpy as np

from cythonutils cimport Data2D, shape2tuple
from metricspeed cimport Metric
from clusteringspeed cimport ClustersCentroid, Centroid, QuickBundles
from clusteringspeed cimport QuickBundlesX
from dipy.segment.clustering import ClusterMapCentroid, ClusterCentroid
from dipy.tracking import Streamlines

DTYPE = np.float32
DEF BIGGEST_DOUBLE = 1.7976931348623157e+308  # np.finfo('f8').max
DEF BIGGEST_FLOAT = 3.4028235e+38  # np.finfo('f4').max
DEF BIGGEST_INT = 2147483647  # np.iinfo('i4').max


def clusters_centroid2clustermap_centroid(ClustersCentroid clusters_list):
    """ Converts a `ClustersCentroid` object (Cython) to a `ClusterMapCentroid`
    object (Python).

    Only basic functionalities are provided with a `Clusters` object. To have
    more flexibility, one should use `ClusterMap` object, hence this conversion
    function.

    Parameters
    ----------
    clusters_list : `ClustersCentroid` object
        Result of the clustering contained in a Cython's object.

    Returns
    -------
    `ClusterMapCentroid` object
        Result of the clustering contained in a Python's object.
    """
    clusters = ClusterMapCentroid()
    for i in range(clusters_list._nb_clusters):
        centroid = np.asarray(clusters_list.centroids[i].features)
        indices = np.asarray(<int[:clusters_list.clusters_size[i]]> clusters_list.clusters_indices[i]).tolist()
        clusters.add_cluster(ClusterCentroid(id=i, centroid=centroid, indices=indices))

    return clusters


def peek(iterable):
    """ Returns the first element of an iterable and the iterator. """
    iterable = iter(iterable)
    first = next(iterable, None)
    iterator = itertools.chain([first], iterable)
    return first, iterator


def quickbundles_compactlist(streamlines, Metric metric, double threshold, long max_nb_clusters=BIGGEST_INT, ordering=None, bvh=False):
    if not isinstance(streamlines, Streamlines):
        raise ValueError("`streamlines` must be a ``Streamlines`` object")

    # Threshold of np.inf is not supported, set it to 'biggest_double'
    threshold = min(threshold, BIGGEST_DOUBLE)
    # Threshold of -np.inf is not supported, set it to 0
    threshold = max(threshold, 0)

    if ordering is None:
        ordering = np.arange(len(streamlines))

    # Check if `ordering` or `streamlines` are empty
    first_idx = ordering[0] if len(ordering) > 0 else None
    if first_idx is None or len(streamlines) == 0:
        return ClusterMapCentroid()

    features_shape = shape2tuple(metric.feature.c_infer_shape(streamlines[first_idx].astype(DTYPE)))
    cdef QuickBundles qb = QuickBundles(features_shape, metric, threshold, max_nb_clusters, bvh)
    cdef int idx, i
    cdef int cluster_id
    cdef int nb_streamlines = len(streamlines)
    cdef long[:] c_ordering = np.asarray(ordering)
    cdef long[:] offsets = np.asarray(streamlines._offsets)
    cdef long[:] lengths = np.asarray(streamlines._lengths)
    cdef Data2D data = streamlines._data

    with nogil:
        for i in range(c_ordering.shape[0]):
            idx = c_ordering[i]
            cluster_id = qb.assignment_step(data[offsets[idx]:offsets[idx]+lengths[idx]], idx)
            # The update step is performed right after the assignement step instead
            # of after all streamlines have been assigned like k-means algorithm.
            qb.update_step(cluster_id)

    results = clusters_centroid2clustermap_centroid(qb.clusters)
    results.stats = qb.get_stats()
    return results


def quickbundles(streamlines, Metric metric, double threshold,
                 long max_nb_clusters=BIGGEST_INT, ordering=None, bvh=False):
    """ Clusters streamlines using QuickBundles.

    Parameters
    ----------
    streamlines : list of 2D arrays or `Streamlines` object
        List of streamlines to cluster.
    metric : `Metric` object
        Tells how to compute the distance between two streamlines.
    threshold : double
        The maximum distance from a cluster for a streamline to be still
        considered as part of it.
    max_nb_clusters : int, optional
        Limits the creation of bundles. (Default: inf)
    ordering : iterable of indices, optional
        Iterate through `data` using the given ordering.
    bvh : bool
        Boundary volume hierarchy

    Returns
    -------
    `ClusterMapCentroid` object
        Result of the clustering.

    References
    ----------
    .. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                        tractography simplification, Frontiers in Neuroscience,
                        vol 6, no 175, 2012.
    """

    if isinstance(streamlines, Streamlines):
        return quickbundles_compactlist(streamlines, metric, threshold, max_nb_clusters, ordering, bvh)

    # Threshold of np.inf is not supported, set it to 'biggest_double'
    threshold = min(threshold, BIGGEST_DOUBLE)
    # Threshold of -np.inf is not supported, set it to 0
    threshold = max(threshold, 0)



    if ordering is None:
        ordering = xrange(len(streamlines))

    # Check if `ordering` or `streamlines` are empty
    first_idx, ordering = peek(ordering)
    if first_idx is None or len(streamlines) == 0:
        return ClusterMapCentroid()

    features_shape = shape2tuple(metric.feature.c_infer_shape(streamlines[first_idx].astype(DTYPE)))
    cdef QuickBundles qb = QuickBundles(features_shape, metric, threshold,
                                        max_nb_clusters, bvh)
    cdef int idx

    for idx in ordering:
        streamline = streamlines[idx]
        if not streamline.flags.writeable or streamline.dtype != DTYPE:
            streamline = streamline.astype(DTYPE)

        cluster_id = qb.assignment_step(streamline, idx)
        # The update step is performed right after the assignement step instead
        # of after all streamlines have been assigned like k-means algorithm.
        qb.update_step(cluster_id)

    results = clusters_centroid2clustermap_centroid(qb.clusters)
    results.stats = qb.get_stats()
    return results


def quickbundles_online(features_shape, Metric metric, double threshold,
                        long max_nb_clusters=BIGGEST_INT, bvh=False):
    """ Clusters streamlines using QuickBundles in a online fashion.

    Parameters
    ----------
    features_shape : tuple
        Expected shape of the features (used to preallocate centroids).
    metric : `Metric` object
        Tells how to compute the distance between two streamlines.
    threshold : double
        The maximum distance from a cluster for a streamline to be still
        considered as part of it.
    max_nb_clusters : int, optional
        Limits the creation of bundles. (Default: inf)
    bvh : bool
        Boundary volume hierarchy

    Returns
    -------
    function(streamline, idx)
        When called this function cluster a `streamline` with id `idx` and returns
        the state of the QuickBundles Cython object and the cluster id where the
        streamine has been assigned to.
    """

    # Threshold of np.inf is not supported, set it to 'biggest_double'
    threshold = min(threshold, BIGGEST_DOUBLE)
    # Threshold of -np.inf is not supported, set it to 0
    threshold = max(threshold, 0)

    cdef QuickBundles qb = QuickBundles(features_shape, metric, threshold, max_nb_clusters, bvh)
    cdef int cluster_id

    def _step(streamline, int idx):
        if not streamline.flags.writeable or streamline.dtype != np.float32:
            streamline = streamline.astype(np.float32)

        cluster_id = qb.assignment_step(streamline, idx)
        # The update step is performed right after the assignement step instead
        # of after all streamlines have been assigned like k-means algorithm.
        qb.update_step(cluster_id)
        return qb, cluster_id

    return _step


def quickbundlesX(streamlines, Metric metric, thresholds, ordering=None):
    """ Clusters streamlines using QuickBundles.

    Parameters
    ----------
    streamlines : list of 2D arrays
        List of streamlines to cluster.
    metric : `Metric` object
        Tells how to compute the distance between two streamlines.
    thresholds : list of double
        Thresholds to use for each clustering layer. A threshold represents the
        maximum distance from a cluster for a streamline to be still considered
        as part of it.
    ordering : iterable of indices, optional
        Iterate through `data` using the given ordering.

    Returns
    -------
    `QuickBundlesX` object
        Result of the clustering.

    """
    if ordering is None:
        ordering = xrange(len(streamlines))

    # Check if `ordering` or `streamlines` are empty
    first_idx, ordering = peek(ordering)
    if first_idx is None or len(streamlines) == 0:
        return ClusterMapCentroid()

    features_shape = shape2tuple(metric.feature.c_infer_shape(streamlines[first_idx].astype(DTYPE)))
    cdef QuickBundlesX qbx = QuickBundlesX(features_shape, thresholds, metric)
    cdef int idx

    for idx in ordering:
        streamline = streamlines[idx]
        if not streamline.flags.writeable or streamline.dtype != DTYPE:
            streamline = streamline.astype(DTYPE)

        qbx.insert(streamline, idx)

    return qbx


def quickbundlesX_online(features_shape, Metric metric, thresholds):
    """ Clusters streamlines using QuickBundles.

    Parameters
    ----------
    features_shape : tuple
        Expected shape of the features (used to preallocate centroids).
    metric : `Metric` object
        Tells how to compute the distance between two streamlines.
    thresholds : list of double
        Thresholds to use for each clustering layer. A threshold represents the
        maximum distance from a cluster for a streamline to be still considered
        as part of it.

    Returns
    -------
    function(streamline, idx)
        When called this function cluster a `streamline` with id `idx` and returns
        the state of the QuickBundles Cython object and the cluster id where the
        streamine has been assigned to.
    """

    cdef QuickBundlesX qbx = QuickBundlesX(features_shape, thresholds, metric)
    cdef int cluster_id

    def _step(streamline, int idx):
        if not streamline.flags.writeable or streamline.dtype != np.float32:
            streamline = streamline.astype(np.float32)

        path = qbx.insert(streamline, idx)
        return qbx, path

    return _step
