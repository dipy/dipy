# distutils: language = c
# cython: wraparound=False, cdivision=True, boundscheck=False

import operator

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


cdef class Cluster:
    """ Provides functionalities to interact with a cluster.

    Useful container to retrieve index of elements grouped together. If
    a reference to the data is provided to `cluster_map`, elements will
    be returned instead of their index when applicable.

    Parameters
    ----------
    cluster_map : `ClusterMap` object
        reference to the set of clusters this cluster is being part of
    id : int
        id of this cluster in its associated `cluster_map`

    Notes
    -----
    A cluster does not contain actual data but instead knows how to
    retrieves them using its `ClusterMap` object.
    """
    #cdef int _id
    #cdef ClusterMap _cluster_map

    def __init__(Cluster self, ClusterMap cluster_map, int id):
        self._id = id
        self._cluster_map = cluster_map
        self.id  # Implicitly test if id and cluster_map are valid!

    property cluster_map:
        def __get__(self):
            if self._cluster_map is None:
                raise ValueError("This cluster is not linked with a cluster map.")

            return self._cluster_map

    property id:
        def __get__(self):
            if self._id < 0 or self._id >= len(self.cluster_map):
                raise ValueError("Cluster id {0} can't not be found in linked cluster map.".format(self._id))

            return self._id

    property indices:
        def __get__(self):
            cdef ClusterMap cluster_map = <ClusterMap> self.cluster_map
            if cluster_map._clusters_size[self.id] == 0:
                return np.array([], dtype="int32")

            return np.asarray(<int[:cluster_map._clusters_size[self.id]]> cluster_map._clusters_indices[self.id])

    def __len__(self):
        return (<ClusterMap>self.cluster_map)._clusters_size[self.id]

    def __getitem__(self, idx):
        cdef ClusterMap cluster_map = <ClusterMap> self.cluster_map
        cdef int* indices = cluster_map._clusters_indices[self.id]

        if isinstance(idx, int) or isinstance(idx, np.integer):
            if idx < -len(self) or len(self) <= idx:
                raise IndexError("Index out of bound: idx={0}".format(idx))

            if idx < 0:
                idx += len(self)

            return cluster_map.refdata[indices[idx]]
        elif type(idx) is slice:
            return [cluster_map.refdata[indices[i]] for i in xrange(*idx.indices(len(self)))]
        elif type(idx) is list:
            return [self[i] for i in idx]

        raise TypeError("Index must be a int or a slice! Not " + str(type(idx)))

        return cluster_map.refdata[idx]

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def __str__(self):
        return "[" + ", ".join(map(str, self.indices)) + "]"

    def __repr__(self):
        return "Cluster(" + str(self) + ")"

    def __richcmp__(self, other, op):
        # See http://docs.cython.org/src/userguide/special_methods.html#rich-comparisons
        if op == 2:
            return isinstance(other, Cluster) and self.id == other.id and self.cluster_map is other.cluster_map
        elif op == 3:
            return not self == other
        else:
            return NotImplemented("Cluster does not support this type of comparison!")

    def add(self, *indices):
        """ Adds indices to this cluster.

        Parameters
        ----------
        *indices : list of indices
            indices to add to this cluster
        """
        cdef ClusterMap cluster_map = <ClusterMap> self.cluster_map
        for id_data in indices:
            cluster_map.c_add(self.id, id_data, None)


cdef class ClusterCentroid(Cluster):
    """ Provides functionalities to interact with a cluster.

    Useful container to retrieve index of elements grouped together and
    the cluster's centroid. If a reference to the data is provided to
    `cluster_map`, elements will be returned instead of their index when
    applicable.

    Parameters
    ----------
    cluster_map : `ClusterMapCentroid` object
        reference to the set of clusters this cluster is being part of
    id : int
        id of this cluster in its associated `cluster_map`

    Notes
    -----
    A cluster does not contain actual data but instead knows how to
    retrieves them using its `ClusterMapCentroid` object.
    """
    def __init__(ClusterCentroid self, ClusterMapCentroid cluster_map, int id):
        super(ClusterCentroid, self).__init__(cluster_map, id)

    property centroid:
        def __get__(self):
            cdef ClusterMapCentroid cluster_map = <ClusterMapCentroid> self.cluster_map
            shape = shape2tuple(cluster_map._features_shape)
            return np.asarray(cluster_map._centroids[self.id].features)

    def add(self, id_data, data):
        cdef ClusterMapCentroid cluster_map = <ClusterMapCentroid> self.cluster_map
        if shape2tuple(cluster_map._features_shape) != data.shape:
            raise ValueError("The shape of the centroid and the data to add must be the same!")

        cluster_map.c_add(self.id, id_data, data)


cdef class ClusterMap:
    #cdef object _cluster_class
    #cdef object refdata
    #cdef int _nb_clusters
    #cdef int** _clusters_indices
    #cdef int* _clusters_size

    def __init__(ClusterMap self, refdata=Identity()):
        self._nb_clusters = 0
        self._clusters_indices = NULL
        self._clusters_size = NULL
        self.refdata = refdata
        self._cluster_class = Cluster

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

    def __getitem__(self, idx):
        if isinstance(idx, int) or isinstance(idx, np.integer):
            if idx < -len(self) or len(self) <= idx:
                raise IndexError("Index out of bound: idx={0}".format(idx))

            if idx < 0:
                idx += len(self)

            return self._cluster_class(self, idx)
        elif type(idx) is slice:
            return [self._cluster_class(self, i) for i in xrange(*idx.indices(len(self)))]
        elif type(idx) is list:
            return [self[i] for i in idx]
        elif isinstance(idx, np.ndarray) and idx.dtype == np.bool:
            return [self._cluster_class(self, i) for i in np.arange(len(self))[idx]]

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
            return NotImplemented("Cluster does not support this type of comparison!")

    cdef void c_add(ClusterMap self, int id_cluster, int id_data, Data2D data) nogil except *:
        # Keep streamline's index in the given cluster
        cdef int C = self._clusters_size[id_cluster]
        self._clusters_indices[id_cluster] = <int*> realloc(self._clusters_indices[id_cluster], (C+1)*sizeof(int))
        self._clusters_indices[id_cluster][C] = id_data
        self._clusters_size[id_cluster] += 1

    cdef int c_create_cluster(ClusterMap self) nogil except -1:
        self._clusters_indices = <int**> realloc(self._clusters_indices, (self._nb_clusters+1)*sizeof(int*))
        self._clusters_indices[self._nb_clusters] = <int*> calloc(0, sizeof(int))
        self._clusters_size = <int*> realloc(self._clusters_size, (self._nb_clusters+1)*sizeof(int))
        self._clusters_size[self._nb_clusters] = 0

        self._nb_clusters += 1
        return self._nb_clusters - 1

    cdef int c_size(ClusterMap self) nogil:
        return self._nb_clusters

    def add_cluster(ClusterMap self):
        id_cluster = self.c_create_cluster()
        return self[id_cluster]


cdef class ClusterMapCentroid(ClusterMap):
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
        self._cluster_class = ClusterCentroid

    property centroids:
        def __get__(self):
            shape = shape2tuple(self._features_shape)
            return [np.asarray(self.c_get_centroid(i).features) for i in range(self._nb_clusters)]

    def __dealloc__(ClusterMapCentroid self):
        # __dealloc__ method of the superclass is automatically called.
        # see: http://docs.cython.org/src/userguide/special_methods.html#finalization-method-dealloc
        for i in range(self._nb_clusters):
            free(&(self._centroids[i].features[0, 0]))

        free(self._centroids)
        self._centroids = NULL

    cdef void c_add(ClusterMapCentroid self, int id_cluster, int id_data, Data2D data) nogil:
        cdef Data2D centroid = self._centroids[id_cluster].features
        cdef int C = self._clusters_size[id_cluster]

        cdef int N = centroid.shape[0], D = centroid.shape[1]
        for n in range(N):
            for d in range(D):
                centroid[n, d] = ((centroid[n, d] * C) + data[n, d]) / (C+1)

        ClusterMap.c_add(self, id_cluster, id_data, data)

    cdef int c_create_cluster(ClusterMapCentroid self) nogil except -1:
        self._centroids = <Centroid*> realloc(self._centroids, (self._nb_clusters+1)*sizeof(Centroid))
        memset(&self._centroids[self._nb_clusters], 0, sizeof(Centroid))  # Zero-initialize the new centroid

        with gil:
            self._centroids[self._nb_clusters].features = <float[:self._features_shape.dims[0], :self._features_shape.dims[1]]> calloc(self._features_shape.size, sizeof(float))

        return ClusterMap.c_create_cluster(self)

    cdef Centroid* c_get_centroid(ClusterMapCentroid self, int idx) nogil:
        return &self._centroids[idx]

