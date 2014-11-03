# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import operator

from dipy.segment.clustering import Cluster, ClusterCentroid
import numpy as np
cimport numpy as np

from cythonutils cimport Data2D, Shape, shape2tuple, tuple2shape


cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)
    void *memset(void *ptr, int value, size_t num)


cdef struct Centroid:
    Data2D features
    int size


cdef class Identity:
    def __getitem__(self, idx):
        return idx


cdef class ClusterMap:
    """ Provides functionalities to interact with clustering outputs.

    Useful container to create, remove, retrieve and filter clusters.
    If `refdata` is given, elements will be returned instead of their
    index when using `Cluster` objects.

    Parameters
    ----------
    refdata : list
        actual elements that clustered indices refer to
    """
    #cdef object refdata
    #cdef int _nb_clusters
    #cdef int** _clusters_indices
    #cdef int* _clusters_size

    def __init__(ClusterMap self, refdata=Identity()):
        self._nb_clusters = 0
        self._clusters_indices = NULL
        self._clusters_size = NULL
        self.refdata = refdata

    property refdata:
        def __get__(self):
            return self.refdata
        def __set__(self, refdata):
            self.refdata = refdata

    property clusters:
        def __get__(self):
            return list(self)

    def __len__(self):
        return self.c_size()

    def get_cluster(self, cluster_id):
        indices = []
        if self._clusters_size[cluster_id] > 0:
            indices = np.asarray(<int[:self._clusters_size[cluster_id]]> self._clusters_indices[cluster_id]).tolist()

        return Cluster(id=cluster_id, indices=indices, refdata=self.refdata)

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, np.integer):
            if idx < -len(self) or len(self) <= idx:
                raise IndexError("Index out of bound: idx={0}".format(idx))

            if idx < 0:
                idx += len(self)

            return self.get_cluster(idx)
        elif type(idx) is slice:
            return [self.get_cluster(i) for i in xrange(*idx.indices(len(self)))]
        elif type(idx) is list:
            return [self[i] for i in idx]
        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            return [self.get_cluster(i) for i in np.arange(len(self))[idx]]

        raise TypeError("Index must be a int or a slice! Not " + str(type(idx)))

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __str__(self):
        return "[" + ", ".join(map(str, self)) + "]"

    def __repr__(self):
        return "ClusterMap(" + str(self) + ")"

    def __dealloc__(ClusterMap self):
        for i in range(self._nb_clusters):
            free(self._clusters_indices[i])
            self._clusters_indices[i] = NULL

        free(self._clusters_indices)
        self._clusters_indices = NULL
        free(self._clusters_size)
        self._clusters_size = NULL

    def __richcmp__(self, other, op):
        # See http://docs.cython.org/src/userguide/special_methods.html#rich-comparisons
        # Comparisons operators are: {0: "<", 1: "<=", 2: "==", 3: "!=", 4: ">", 5: ">=""}

        if isinstance(other, ClusterMap):
            if op == 2:  # ==
                #return len(self) == len(other) and all([cluster1 == cluster2 for cluster1, cluster2 in zip(self.clusters, other.clusters)])
                return len(self) == len(other) and self.clusters == other.clusters
            elif op == 3:  # !=
                return not self == other
            else:
                return NotImplemented("ClusterMap does not support this type of comparison!")

        elif isinstance(other, int):
            if op == 0:  # <
                op = operator.lt
            elif op == 1:  # <=
                op = operator.le
            elif op == 2:  # ==
                op = operator.eq
            elif op == 3:  # !=
                op = operator.ne
            elif op == 4:  # >
                op = operator.gt
            elif op == 5:  # >=
                op = operator.ge

            return np.array([op(len(cluster), other) for cluster in self])

        else:
            return NotImplemented("ClusterMap does not support this type of comparison!")

    cdef void c_assign(ClusterMap self, int id_cluster, int id_data, Data2D data) nogil except *:
        cdef int C = self._clusters_size[id_cluster]
        self._clusters_indices[id_cluster] = <int*> realloc(self._clusters_indices[id_cluster], (C+1)*sizeof(int))
        self._clusters_indices[id_cluster][C] = id_data
        self._clusters_size[id_cluster] += 1

    cdef void c_remove_cluster(ClusterMap self, int id_cluster) nogil except *:
        # Overwrite cluster to be removed with last one
        free(self._clusters_indices[id_cluster])
        self._clusters_indices[id_cluster] = self._clusters_indices[self._nb_clusters-1]
        self._clusters_size[id_cluster] = self._clusters_size[self._nb_clusters-1]

        # Remove last cluster
        self._clusters_indices = <int**> realloc(self._clusters_indices, (self._nb_clusters-1)*sizeof(int*))
        self._clusters_size = <int*> realloc(self._clusters_size, (self._nb_clusters-1)*sizeof(int))
        self._nb_clusters -= 1

    cdef int c_create_cluster(ClusterMap self) nogil except -1:
        self._clusters_indices = <int**> realloc(self._clusters_indices, (self._nb_clusters+1)*sizeof(int*))
        self._clusters_indices[self._nb_clusters] = <int*> calloc(0, sizeof(int))
        self._clusters_size = <int*> realloc(self._clusters_size, (self._nb_clusters+1)*sizeof(int))
        self._clusters_size[self._nb_clusters] = 0

        self._nb_clusters += 1
        return self._nb_clusters - 1

    cdef void c_clear(ClusterMap self) nogil except *:
        cdef int i
        for i in range(self._nb_clusters):
            self._clusters_indices[i] = <int*> realloc(self._clusters_indices[i], 0*sizeof(int))
            self._clusters_size[i] = 0

    cdef int c_size(ClusterMap self) nogil:
        return self._nb_clusters

    def add_cluster(ClusterMap self, cluster):
        """ Adds a new cluster to this cluster map.

        Returns
        -------
        cluster : `Cluster` object
            newly created cluster
        """
        id_cluster = self.c_create_cluster()

        for i in cluster.indices:
            self.c_assign(id_cluster, i, None)

        return id_cluster

    def remove_cluster(ClusterMap self, index):
        """ Remove a cluster from this cluster map using its index.

        Parameters
        ----------
        index : int
            index of the cluster to remove
        """
        self.c_remove_cluster(index)


cdef class ClusterMapCentroid(ClusterMap):
    """ Provides functionalities to interact with clustering outputs.

    Useful container to create, remove, retrieve and filter clusters.
    It allows also to retrieve easely the centroid of every clusters.
    If `refdata` is given, elements will be returned instead of their
    index when using `ClusterCentroid` objects.

    Parameters
    ----------
    refdata : list
        actual elements that clustered indices refer to
    """
    #cdef Centroid* _centroids
    #cdef Shape _features_shape

    def __init__(ClusterMapCentroid self, feature_shape, *args, **kwargs):
        ClusterMap.__init__(self, *args, **kwargs)
        if isinstance(feature_shape, int):
            feature_shape = (1, feature_shape)

        if not isinstance(feature_shape, tuple):
            raise ValueError("'feature_shape' must be a tuple or a int.")

        self._features_shape = tuple2shape(feature_shape)

        self._centroids = NULL
        self._new_centroids = NULL

    property centroids:
        def __get__(self):
            shape = shape2tuple(self._features_shape)
            return [np.asarray(self._centroids[i].features) for i in range(self._nb_clusters)]

    def __dealloc__(ClusterMapCentroid self):
        # __dealloc__ method of the superclass is automatically called.
        # see: http://docs.cython.org/src/userguide/special_methods.html#finalization-method-dealloc
        for i in range(self._nb_clusters):
            free(&(self._centroids[i].features[0, 0]))

        free(self._centroids)
        self._centroids = NULL
        free(self._new_centroids)
        self._new_centroids = NULL

    def get_cluster(self, cluster_id):
        cluster = super(ClusterMapCentroid, self).get_cluster(cluster_id)
        centroid = np.asarray(self._centroids[cluster_id].features)

        return ClusterCentroid(id=cluster_id, centroid=centroid, indices=cluster.indices, refdata=self.refdata)

    cdef void c_assign(ClusterMapCentroid self, int id_cluster, int id_data, Data2D data) nogil:
        cdef Data2D new_centroid = self._new_centroids[id_cluster].features
        cdef int C = self._clusters_size[id_cluster]
        cdef int n, d

        cdef int N = new_centroid.shape[0], D = new_centroid.shape[1]
        for n in range(N):
            for d in range(D):
                new_centroid[n, d] = ((new_centroid[n, d] * C) + data[n, d]) / (C+1)

        ClusterMap.c_assign(self, id_cluster, id_data, data)

    cdef int c_update(ClusterMapCentroid self, int id_cluster) nogil:
        cdef Data2D centroid = self._centroids[id_cluster].features
        cdef Data2D new_centroid = self._new_centroids[id_cluster].features
        cdef int N = new_centroid.shape[0], D = centroid.shape[1]
        cdef int n, d
        cdef int converged = 1

        for n in range(N):
            for d in range(D):
                converged &= centroid[n, d] == new_centroid[n, d]
                centroid[n, d] = new_centroid[n, d]

        return converged

    cdef int c_create_cluster(ClusterMapCentroid self) nogil except -1:
        self._centroids = <Centroid*> realloc(self._centroids, (self._nb_clusters+1)*sizeof(Centroid))
        memset(&self._centroids[self._nb_clusters], 0, sizeof(Centroid))  # Zero-initialize the Centroid structure

        self._new_centroids = <Centroid*> realloc(self._new_centroids, (self._nb_clusters+1)*sizeof(Centroid))
        memset(&self._new_centroids[self._nb_clusters], 0, sizeof(Centroid))  # Zero-initialize the new Centroid structure

        with gil:
            self._centroids[self._nb_clusters].features = <float[:self._features_shape.dims[0], :self._features_shape.dims[1]]> calloc(self._features_shape.size, sizeof(float))
            self._new_centroids[self._nb_clusters].features = <float[:self._features_shape.dims[0], :self._features_shape.dims[1]]> calloc(self._features_shape.size, sizeof(float))

        return ClusterMap.c_create_cluster(self)

    cdef void c_remove_cluster(ClusterMapCentroid self, int id_cluster) nogil except *:
        # Overwrite cluster's centroid to be removed with last one
        free(&(self._centroids[id_cluster].features[0, 0]))
        free(&(self._new_centroids[id_cluster].features[0, 0]))
        self._centroids[id_cluster] = self._centroids[self._nb_clusters-1]
        self._new_centroids[id_cluster] = self._new_centroids[self._nb_clusters-1]

        # Remove last cluster's centroid
        self._centroids = <Centroid*> realloc(self._centroids, (self._nb_clusters-1)*sizeof(Centroid))
        self._new_centroids = <Centroid*> realloc(self._new_centroids, (self._nb_clusters-1)*sizeof(Centroid))

        ClusterMap.c_remove_cluster(self, id_cluster)

    def add_cluster(ClusterMapCentroid self, cluster):
        """ Adds a new cluster to this cluster map.

        Parameters
        ----------
        cluster : `ClusterCentroid` object
            cluster to add

        Returns
        -------
        id_cluster : int
            index of the newly added cluster
        """
        cdef int id_cluster = self.c_create_cluster()
        cdef int N = cluster.centroid.shape[0], D = cluster.centroid.shape[1]
        cdef int idx, n, d
        cdef Data2D new_centroid = self._new_centroids[id_cluster].features

        for idx in cluster.indices:
            ClusterMap.c_assign(self, id_cluster, idx, None)

        for n in range(N):
            for d in range(D):
                new_centroid[n, d] = cluster.centroid[n, d]

        self.c_update(id_cluster)
        return id_cluster
