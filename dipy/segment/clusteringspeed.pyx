# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import numpy as np
cimport numpy as np

from metric cimport Metric, Streamline, Features

cdef extern from "stdlib.h" nogil:
    ctypedef unsigned long size_t
    void free(void *ptr)
    void *calloc(size_t nelem, size_t elsize)
    void *realloc(void *ptr, size_t elsize)

DEF biggest_float = 3.4028235e+38  # np.finfo('f4').max
DEF biggest_int = 2147483647  # np.iinfo('i4').max

cdef class CentroidClusters

cdef class Cluster:
    cdef int _id
    cdef CentroidClusters _clusters

    def __init__(Cluster self, CentroidClusters clusters, int id):
        self._id = id
        self._clusters = clusters

    property id:
        def __get__(self):
            return self._id

    property nb_features:
        def __get__(self):
            return self._clusters._nb_features

    property indices:
        def __get__(self):
            return np.array(<int[:self._clusters._clusters_size[self.id]]> self._clusters._clusters_indices[self.id])

    property centroid:
        def __get__(self):
            return np.array(<float[:self.nb_features]> self._clusters._centroids[self.id])

    cpdef add( self, int id_features, Features features):
        self._clusters.c_add(self.id, id_features, features)

    def __len__(self):
        return len(self.indices)

cdef class CentroidClusters:
    cdef int _nb_clusters
    cdef int** _clusters_indices
    cdef int* _clusters_size
    cdef float** _centroids
    cdef int _nb_features

    def __init__(CentroidClusters self, int nb_features):
        self._nb_clusters = 0
        self._nb_features = nb_features
        self._centroids = NULL
        self._clusters_indices = NULL
        self._clusters_size = NULL

    cdef void c_add(CentroidClusters self, int id_cluster, int id_features, Features features) nogil except *:
        if self._nb_features != features.shape[0]:
            with gil:
                raise ValueError("CentroidClusters requires all features having the same length!")

        cdef float* centroid = self._centroids[id_cluster]
        cdef int C = self._clusters_size[id_cluster]

        for i in range(self._nb_features):
            centroid[i] = ((centroid[i] * C) + features[i]) / (C+1)

        # Keep streamline's index in the given cluster
        self._clusters_indices[id_cluster] = <int*> realloc(self._clusters_indices[id_cluster], (C+1)*sizeof(int))
        self._clusters_indices[id_cluster][C] = id_features
        self._clusters_size[id_cluster] += 1

    cdef int c_create_cluster(CentroidClusters self) nogil except -1:
        self._centroids = <float**> realloc(self._centroids, (self._nb_clusters+1)*sizeof(float*))
        self._centroids[self._nb_clusters] = <float*> calloc(self._nb_features, sizeof(float))

        self._clusters_indices = <int**> realloc(self._clusters_indices, (self._nb_clusters+1)*sizeof(int*))
        self._clusters_indices[self._nb_clusters] = NULL

        self._clusters_size = <int*> realloc(self._clusters_size, (self._nb_clusters+1)*sizeof(int))
        self._clusters_size[self._nb_clusters] = 0

        self._nb_clusters += 1
        return self._nb_clusters - 1

    cdef int c_size(CentroidClusters self) nogil:
        #return self._centroids.size()
        return self._nb_clusters

    cdef float* c_get_centroid(CentroidClusters self, int idx) nogil:
        return self._centroids[idx]

    def __len__(self):
        return self.c_size()

    def __getitem__(self, int idx):
        if idx >= len(self):
            raise IndexError("Index out of bound: '{0}'".format(idx))

        return Cluster(self, idx)

    def get_clusters(self):
        clusters = []
        for i in range(self._nb_clusters):
            clusters.append(self[i].indices)

        return clusters

    cpdef Cluster create_cluster(CentroidClusters self):
        id_cluster = self.c_create_cluster()
        return self[id_cluster]

    cpdef add(CentroidClusters self, int id_cluster, int id_features, Features features):
        self.c_add(id_cluster, id_features, features)

    def __dealloc__(CentroidClusters self):
        for i in range(self._nb_clusters):
            free(self._centroids[i])
            self._centroids[i] = NULL
            free(self._clusters_indices[i])
            self._clusters_indices[i] = NULL

        free(self._centroids)
        self._centroids = NULL
        free(self._clusters_indices)
        self._clusters_indices = NULL
        free(self._clusters_size)
        self._clusters_size = NULL



cpdef quickbundles(streamlines, Metric metric, ordering=None, float threshold=10., int max_nb_clusters=biggest_int):
    if ordering is None:
        ordering = np.arange(len(streamlines))

    cdef:
        int nb_features = metric._nb_features(streamlines[0])
        CentroidClusters clusters = CentroidClusters(nb_features)
        Features features_s_i = <float[:nb_features]> calloc(nb_features, sizeof(float))
        Features features_s_i_flip = <float[:nb_features]> calloc(nb_features, sizeof(float))

    for idx, streamline in enumerate(streamlines):
        _quickbundles(streamline, idx, metric, clusters, features_s_i, features_s_i_flip, threshold, max_nb_clusters)

    free(&features_s_i[0])
    free(&features_s_i_flip[0])

    return clusters.get_clusters()


cdef void _quickbundles(Streamline s_i, int idx, Metric metric, CentroidClusters clusters, Features features_s_i, Features features_s_i_flip, float threshold=10, int max_nb_clusters=biggest_int) nogil except *:

    cdef:
        int nb_features = clusters._nb_features
        float* centroid
        long i_k=0, y=0, x=0
        long i, k
        Streamline s_i_flip
        float dist_min, dist_ck_si, dist_ck_si_flip, A1, A2
        int is_flip=0

    s_i_flip = s_i[::-1]

    metric._get_features(s_i, features_s_i)
    metric._get_features(s_i_flip, features_s_i_flip)

    # Find closest cluster
    dist_min = biggest_float
    for k in range(clusters.c_size()):
        centroid = clusters.c_get_centroid(k)
        dist_ck_si = metric._dist(centroid, features_s_i)
        dist_ck_si_flip = metric._dist(centroid, features_s_i_flip)

        # Keep track of the closest cluster
        if dist_ck_si < dist_min:
            dist_min = dist_ck_si
            i_k = k
            is_flip = False

        if dist_ck_si_flip < dist_min:
            dist_min = dist_ck_si_flip
            i_k = k
            is_flip = True

    # Check if distance with the closest cluster is below some threshold
    # or if we already have the maximum number of clusters.
    # If the latter is true, then assign remaining streamlines to its closest cluster
    if dist_min < threshold or clusters.c_size() >= max_nb_clusters:
        if is_flip:
            clusters.c_add(i_k, idx, features_s_i_flip)
        else:
            clusters.c_add(i_k, idx, features_s_i)

    else:  #  If not, add new cluster
        i_k = clusters.c_create_cluster()
        clusters.c_add(i_k, idx, features_s_i)
