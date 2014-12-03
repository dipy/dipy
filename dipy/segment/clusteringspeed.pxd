from cythonutils cimport Data2D, Shape, shape2tuple, tuple2shape


cdef struct Centroid:
    Data2D features
    int size


cdef class Clusters:
    cdef int _nb_clusters
    cdef int** _clusters_indices
    cdef int* _clusters_size

    cdef void c_assign(Clusters self, int id_cluster, int id_features, Data2D features) nogil except *
    cdef int c_create_cluster(Clusters self) nogil except -1
    cdef int c_size(Clusters self) nogil


cdef class ClustersCentroid(Clusters):
    cdef Centroid* _centroids
    cdef Centroid* _new_centroids
    cdef Shape _features_shape

    cdef void c_assign(ClustersCentroid self, int id_cluster, int id_features, Data2D features) nogil except *
    cdef int c_create_cluster(ClustersCentroid self) nogil except -1
    cdef int c_update(ClustersCentroid self, int id_cluster) nogil except -1
