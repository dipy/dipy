# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as cnp

from libc.math cimport fabs
from libc.stdlib cimport calloc, realloc, free
from libc.string cimport memset
from cythonutils cimport Data2D, Shape, shape2tuple, tuple2shape, same_shape


DTYPE = np.float32
DEF BIGGEST_DOUBLE = 1.7976931348623157e+308  # np.finfo('f8').max
DEF BIGGEST_INT = 2147483647  # np.iinfo('i4').max


cdef class Clusters:
    """ Provides Cython functionalities to interact with clustering outputs.

    This class allows to create clusters and assign elements to them.
    Assignements of a cluster are represented as a list of element indices.
    """
    def __init__(Clusters self):
        self._nb_clusters = 0
        self.clusters_indices = NULL
        self.clusters_size = NULL

    def __dealloc__(Clusters self):
        """ Deallocates memory created with `c_create_cluster` and `c_assign`. """
        for i in range(self._nb_clusters):
            free(self.clusters_indices[i])
            self.clusters_indices[i] = NULL

        free(self.clusters_indices)
        self.clusters_indices = NULL
        free(self.clusters_size)
        self.clusters_size = NULL

    cdef int c_size(Clusters self) nogil:
        """ Returns the number of clusters. """
        return self._nb_clusters

    cdef void c_assign(Clusters self, int id_cluster, int id_element, Data2D element) nogil except *:
        """ Assigns an element to a cluster.

        Parameters
        ----------
        id_cluster : int
            Index of the cluster to which the element will be assigned.
        id_element : int
            Index of the element to assign.
        element : 2d array (float)
            Data of the element to assign.
        """
        cdef cnp.npy_intp C = self.clusters_size[id_cluster]
        self.clusters_indices[id_cluster] = <int*> realloc(self.clusters_indices[id_cluster], (C+1)*sizeof(int))
        self.clusters_indices[id_cluster][C] = id_element
        self.clusters_size[id_cluster] += 1

    cdef int c_create_cluster(Clusters self) nogil except -1:
        """ Creates a cluster and adds it at the end of the list.

        Returns
        -------
        id_cluster : int
            Index of the new cluster.
        """
        self.clusters_indices = <int**> realloc(self.clusters_indices, (self._nb_clusters+1)*sizeof(int*))
        self.clusters_indices[self._nb_clusters] = <int*> calloc(0, sizeof(int))
        self.clusters_size = <int*> realloc(self.clusters_size, (self._nb_clusters+1)*sizeof(int))
        self.clusters_size[self._nb_clusters] = 0

        self._nb_clusters += 1
        return self._nb_clusters - 1


cdef class ClustersCentroid(Clusters):
    """ Provides Cython functionalities to interact with clustering outputs
    having the notion of cluster's centroid.

    This class allows to create clusters, assign elements to them and
    update their centroid.

    Parameters
    ----------
    centroid_shape : int, tuple of int
        Information about the shape of the centroid.
    eps : float, optional
        Consider the centroid has not changed if the changes per dimension
        are less than this epsilon. (Default: 1e-6)
    """
    def __init__(ClustersCentroid self, centroid_shape, float eps=1e-6, *args, **kwargs):
        Clusters.__init__(self, *args, **kwargs)
        if isinstance(centroid_shape, int):
            centroid_shape = (1, centroid_shape)

        if not isinstance(centroid_shape, tuple):
            raise ValueError("'centroid_shape' must be a tuple or a int.")

        self._centroid_shape = tuple2shape(centroid_shape)

        self.centroids = NULL
        self._updated_centroids = NULL
        self.eps = eps

    def __dealloc__(ClustersCentroid self):
        """ Deallocates memory created with `c_create_cluster` and `c_assign`.

        Notes
        -----
        The `__dealloc__` method of the superclass is automatically called:
        http://docs.cython.org/src/userguide/special_methods.html#finalization-method-dealloc
        """
        cdef cnp.npy_intp i
        for i in range(self._nb_clusters):
            free(&(self.centroids[i].features[0, 0]))
            free(&(self._updated_centroids[i].features[0, 0]))
            self.centroids[i].features = None  # Necessary to decrease refcount
            self._updated_centroids[i].features = None  # Necessary to decrease refcount

        free(self.centroids)
        self.centroids = NULL
        free(self._updated_centroids)
        self._updated_centroids = NULL

    cdef void c_assign(ClustersCentroid self, int id_cluster, int id_element, Data2D element) nogil except *:
        """ Assigns an element to a cluster.

        In addition of keeping element's index, an updated version of the
        cluster's centroid is computed. The centroid is the average of all
        elements in a cluster.

        Parameters
        ----------
        id_cluster : int
            Index of the cluster to which the element will be assigned.
        id_element : int
            Index of the element to assign.
        element : 2d array (float)
            Data of the element to assign.
        """
        cdef Data2D updated_centroid = self._updated_centroids[id_cluster].features
        cdef cnp.npy_intp C = self.clusters_size[id_cluster]
        cdef cnp.npy_intp n, d

        cdef cnp.npy_intp N = updated_centroid.shape[0], D = updated_centroid.shape[1]
        for n in range(N):
            for d in range(D):
                updated_centroid[n, d] = ((updated_centroid[n, d] * C) + element[n, d]) / (C+1)

        Clusters.c_assign(self, id_cluster, id_element, element)

    cdef int c_update(ClustersCentroid self, cnp.npy_intp id_cluster) nogil except -1:
        """ Update the centroid of a cluster.

        Parameters
        ----------
        id_cluster : int
            Index of the cluster of which its centroid will be updated.

        Returns
        -------
        int
            Tells whether the centroid has changed or not, i.e. converged.
        """
        cdef Data2D centroid = self.centroids[id_cluster].features
        cdef Data2D updated_centroid = self._updated_centroids[id_cluster].features
        cdef cnp.npy_intp N = updated_centroid.shape[0], D = centroid.shape[1]
        cdef cnp.npy_intp n, d
        cdef int converged = 1

        for n in range(N):
            for d in range(D):
                converged &= fabs(centroid[n, d] - updated_centroid[n, d]) < self.eps
                centroid[n, d] = updated_centroid[n, d]

        return converged

    cdef int c_create_cluster(ClustersCentroid self) nogil except -1:
        """ Creates a cluster and adds it at the end of the list.

        Returns
        -------
        id_cluster : int
            Index of the new cluster.
        """
        self.centroids = <Centroid*> realloc(self.centroids, (self._nb_clusters+1)*sizeof(Centroid))
        # Zero-initialize the Centroid structure
        memset(&self.centroids[self._nb_clusters], 0, sizeof(Centroid))

        self._updated_centroids = <Centroid*> realloc(self._updated_centroids, (self._nb_clusters+1)*sizeof(Centroid))
        # Zero-initialize the new Centroid structure
        memset(&self._updated_centroids[self._nb_clusters], 0, sizeof(Centroid))

        with gil:
            self.centroids[self._nb_clusters].features = <float[:self._centroid_shape.dims[0], :self._centroid_shape.dims[1]]> calloc(self._centroid_shape.size, sizeof(float))
            self._updated_centroids[self._nb_clusters].features = <float[:self._centroid_shape.dims[0], :self._centroid_shape.dims[1]]> calloc(self._centroid_shape.size, sizeof(float))

        return Clusters.c_create_cluster(self)


cdef class QuickBundles(object):
    def __init__(QuickBundles self, features_shape, Metric metric, double threshold, int max_nb_clusters=BIGGEST_INT):
        self.metric = metric
        self.features_shape = tuple2shape(features_shape)
        self.threshold = threshold
        self.max_nb_clusters = max_nb_clusters
        self.clusters = ClustersCentroid(features_shape)
        self.features = np.empty(features_shape, dtype=DTYPE)
        self.features_flip = np.empty(features_shape, dtype=DTYPE)

    cdef NearestCluster find_nearest_cluster(QuickBundles self, Data2D features) nogil except *:
        """ Finds the nearest cluster of a datum given its `features` vector.

        Parameters
        ----------
        features : 2D array
            Features of a datum.

        Returns
        -------
        `NearestCluster` object
            Nearest cluster to `features` according to the given metric.
        """
        cdef:
            cnp.npy_intp k
            double dist
            NearestCluster nearest_cluster

        nearest_cluster.id = -1
        nearest_cluster.dist = BIGGEST_DOUBLE

        for k in range(self.clusters.c_size()):
            dist = self.metric.c_dist(self.clusters.centroids[k].features, features)

            # Keep track of the nearest cluster
            if dist < nearest_cluster.dist:
                nearest_cluster.dist = dist
                nearest_cluster.id = k

        return nearest_cluster

    cdef int assignment_step(QuickBundles self, Data2D datum, int datum_id) nogil except -1:
        """ Compute the assignment step of the QuickBundles algorithm.

        It will assign a datum to its closest cluster according to a given
        metric. If the distance between the datum and its closest cluster is
        greater than the specified threshold, a new cluster is created and the
        datum is assigned to it.

        Parameters
        ----------
        datum : 2D array
            The datum to assign.
        datum_id : int
            ID of the datum, usually its index.

        Returns
        -------
        int
            Index of the cluster the datum has been assigned to.
        """
        cdef:
            Data2D features_to_add = self.features
            NearestCluster nearest_cluster, nearest_cluster_flip
            Shape features_shape = self.metric.feature.c_infer_shape(datum)

        # Check if datum is compatible with the metric
        if not same_shape(features_shape, self.features_shape):
            with gil:
                raise ValueError("All features do not have the same shape! QuickBundles requires this to compute centroids!")

        # Check if datum is compatible with the metric
        if not self.metric.c_are_compatible(features_shape, self.features_shape):
            with gil:
                raise ValueError("Data features' shapes must be compatible according to the metric used!")

        # Find nearest cluster to datum
        self.metric.feature.c_extract(datum, self.features)
        nearest_cluster = self.find_nearest_cluster(self.features)

        # Find nearest cluster to s_i_flip if metric is not order invariant
        if not self.metric.feature.is_order_invariant:
            self.metric.feature.c_extract(datum[::-1], self.features_flip)
            nearest_cluster_flip = self.find_nearest_cluster(self.features_flip)

            # If we found a lower distance using a flipped datum,
            #  add the flipped version instead
            if nearest_cluster_flip.dist < nearest_cluster.dist:
                nearest_cluster.id = nearest_cluster_flip.id
                nearest_cluster.dist = nearest_cluster_flip.dist
                features_to_add = self.features_flip

        # Check if distance with the nearest cluster is below some threshold
        # or if we already have the maximum number of clusters.
        # If the former or the latter is true, assign datum to its nearest cluster
        # otherwise create a new cluster and assign the datum to it.
        if not (nearest_cluster.dist < self.threshold or self.clusters.c_size() >= self.max_nb_clusters):
            nearest_cluster.id = self.clusters.c_create_cluster()

        self.clusters.c_assign(nearest_cluster.id, datum_id, features_to_add)
        return nearest_cluster.id

    cdef void update_step(QuickBundles self, int cluster_id) nogil except *:
        """ Compute the update step of the QuickBundles algorithm.

        It will update the centroid of a cluster given its index.

        Parameters
        ----------
        cluster_id : int
            ID of the cluster to update.

        """
        self.clusters.c_update(cluster_id)
