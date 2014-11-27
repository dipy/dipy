# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

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


cdef class ClustersList:
    """ Provides functionalities to interact with clustering outputs.

    Useful container to create, remove, retrieve and filter clusters.
    If `refdata` is given, elements will be returned instead of their
    index when using `Cluster` objects.

    Parameters
    ----------
    refdata : list
        actual elements that clustered indices refer to
    """
    #cdef int _nb_clusters
    #cdef int** _clusters_indices
    #cdef int* _clusters_size

    def __init__(ClustersList self):
        self._nb_clusters = 0
        self._clusters_indices = NULL
        self._clusters_size = NULL

    def __dealloc__(ClustersList self):
        for i in range(self._nb_clusters):
            free(self._clusters_indices[i])
            self._clusters_indices[i] = NULL

        free(self._clusters_indices)
        self._clusters_indices = NULL
        free(self._clusters_size)
        self._clusters_size = NULL

    cdef int c_size(ClustersList self) nogil:
        return self._nb_clusters

    cdef void c_assign(ClustersList self, int id_cluster, int id_data, Data2D data) nogil except *:
        cdef int C = self._clusters_size[id_cluster]
        self._clusters_indices[id_cluster] = <int*> realloc(self._clusters_indices[id_cluster], (C+1)*sizeof(int))
        self._clusters_indices[id_cluster][C] = id_data
        self._clusters_size[id_cluster] += 1

    cdef int c_create_cluster(ClustersList self) nogil except -1:
        self._clusters_indices = <int**> realloc(self._clusters_indices, (self._nb_clusters+1)*sizeof(int*))
        self._clusters_indices[self._nb_clusters] = <int*> calloc(0, sizeof(int))
        self._clusters_size = <int*> realloc(self._clusters_size, (self._nb_clusters+1)*sizeof(int))
        self._clusters_size[self._nb_clusters] = 0

        self._nb_clusters += 1
        return self._nb_clusters - 1

    #cdef void c_remove_cluster(ClustersList self, int id_cluster) nogil except *:
    #    # Overwrite cluster to be removed with last one
    #    free(self._clusters_indices[id_cluster])
    #    self._clusters_indices[id_cluster] = self._clusters_indices[self._nb_clusters-1]
    #    self._clusters_size[id_cluster] = self._clusters_size[self._nb_clusters-1]

    #    # Remove last cluster
    #    self._clusters_indices = <int**> realloc(self._clusters_indices, (self._nb_clusters-1)*sizeof(int*))
    #    self._clusters_size = <int*> realloc(self._clusters_size, (self._nb_clusters-1)*sizeof(int))
    #    self._nb_clusters -= 1

    #cdef void c_clear(ClustersList self) nogil except *:
    #    cdef int i
    #    for i in range(self._nb_clusters):
    #        self._clusters_indices[i] = <int*> realloc(self._clusters_indices[i], 0*sizeof(int))
    #        self._clusters_size[i] = 0


cdef class ClustersListCentroid(ClustersList):
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

    def __init__(ClustersListCentroid self, feature_shape, *args, **kwargs):
        ClustersList.__init__(self, *args, **kwargs)
        if isinstance(feature_shape, int):
            feature_shape = (1, feature_shape)

        if not isinstance(feature_shape, tuple):
            raise ValueError("'feature_shape' must be a tuple or a int.")

        self._features_shape = tuple2shape(feature_shape)

        self._centroids = NULL
        self._new_centroids = NULL

    def __dealloc__(ClustersListCentroid self):
        # __dealloc__ method of the superclass is automatically called.
        # see: http://docs.cython.org/src/userguide/special_methods.html#finalization-method-dealloc
        for i in range(self._nb_clusters):
            free(&(self._centroids[i].features[0, 0]))

        free(self._centroids)
        self._centroids = NULL
        free(self._new_centroids)
        self._new_centroids = NULL

    cdef void c_assign(ClustersListCentroid self, int id_cluster, int id_data, Data2D data) nogil except *:
        cdef Data2D new_centroid = self._new_centroids[id_cluster].features
        cdef int C = self._clusters_size[id_cluster]
        cdef int n, d

        cdef int N = new_centroid.shape[0], D = new_centroid.shape[1]
        for n in range(N):
            for d in range(D):
                new_centroid[n, d] = ((new_centroid[n, d] * C) + data[n, d]) / (C+1)

        ClustersList.c_assign(self, id_cluster, id_data, data)

    cdef int c_update(ClustersListCentroid self, int id_cluster) nogil:
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

    cdef int c_create_cluster(ClustersListCentroid self) nogil except -1:
        self._centroids = <Centroid*> realloc(self._centroids, (self._nb_clusters+1)*sizeof(Centroid))
        memset(&self._centroids[self._nb_clusters], 0, sizeof(Centroid))  # Zero-initialize the Centroid structure

        self._new_centroids = <Centroid*> realloc(self._new_centroids, (self._nb_clusters+1)*sizeof(Centroid))
        memset(&self._new_centroids[self._nb_clusters], 0, sizeof(Centroid))  # Zero-initialize the new Centroid structure

        with gil:
            self._centroids[self._nb_clusters].features = <float[:self._features_shape.dims[0], :self._features_shape.dims[1]]> calloc(self._features_shape.size, sizeof(float))
            self._new_centroids[self._nb_clusters].features = <float[:self._features_shape.dims[0], :self._features_shape.dims[1]]> calloc(self._features_shape.size, sizeof(float))

        return ClustersList.c_create_cluster(self)

    #cdef void c_remove_cluster(ClustersListCentroid self, int id_cluster) nogil except *:
    #    # Overwrite cluster's centroid to be removed with last one
    #    free(&(self._centroids[id_cluster].features[0, 0]))
    #    free(&(self._new_centroids[id_cluster].features[0, 0]))
    #    self._centroids[id_cluster] = self._centroids[self._nb_clusters-1]
    #    self._new_centroids[id_cluster] = self._new_centroids[self._nb_clusters-1]

    #    # Remove last cluster's centroid
    #    self._centroids = <Centroid*> realloc(self._centroids, (self._nb_clusters-1)*sizeof(Centroid))
    #    self._new_centroids = <Centroid*> realloc(self._new_centroids, (self._nb_clusters-1)*sizeof(Centroid))

    #    ClustersList.c_remove_cluster(self, id_cluster)
