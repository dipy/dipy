# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

from cythonutils cimport Data2D, Shape, shape2tuple, tuple2shape


cdef struct Centroid:
    Data2D features
    int size


cdef class ClustersList:
    cdef int _nb_clusters
    cdef int** _clusters_indices
    cdef int* _clusters_size

    cdef void c_assign(ClustersList self, int id_cluster, int id_features, Data2D features) nogil except *
    cdef int c_create_cluster(ClustersList self) nogil except -1
    cdef int c_size(ClustersList self) nogil
    #cdef void c_remove_cluster(ClustersList self, int id_cluster) nogil except *
    #cdef void c_clear(ClustersList self) nogil except *


cdef class ClustersListCentroid(ClustersList):
    cdef Centroid* _centroids
    cdef Centroid* _new_centroids
    cdef Shape _features_shape

    cdef void c_assign(ClustersListCentroid self, int id_cluster, int id_features, Data2D features) nogil except *
    cdef int c_create_cluster(ClustersListCentroid self) nogil except -1
    cdef int c_update(ClustersListCentroid self, int id_cluster) nogil except -1
