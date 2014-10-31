# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

from cythonutils cimport Data2D, Shape, shape2tuple, tuple2shape


cdef struct Centroid:
    Data2D features
    int size


#cdef class Cluster:
#    cdef int _id
#    cdef ClusterMap _cluster_map


#cdef class ClusterCentroid(Cluster):
#    pass


cdef class ClusterMap:
    cdef object refdata
    cdef int _nb_clusters
    cdef int** _clusters_indices
    cdef int* _clusters_size

    cdef void c_add(ClusterMap self, int id_cluster, int id_features, Data2D features) nogil except *
    cdef int c_create_cluster(ClusterMap self) nogil except -1
    cdef int c_size(ClusterMap self) nogil


cdef class ClusterMapCentroid(ClusterMap):
    cdef Centroid* _centroids
    cdef Shape _features_shape

    cdef void c_add(ClusterMapCentroid self, int id_cluster, int id_features, Data2D features) nogil except *
    cdef Centroid* c_get_centroid(ClusterMapCentroid self, int id_cluster) nogil
