
import numpy as np
import unittest
import itertools

from dipy.segment.clusteringspeed import Cluster, ClusterCentroid
from dipy.segment.clusteringspeed import ClusterMap, ClusterMapCentroid

from nose.tools import assert_equal
from numpy.testing import assert_array_equal, assert_raises


class TestCluster(unittest.TestCase):
    def test_attributes_and_constructor(self):
        clusters = ClusterMap()
        cluster = clusters.create_cluster()
        assert_equal(type(cluster), Cluster)

        assert_equal(cluster.id, 0)
        assert_raises(AttributeError, setattr, cluster, 'id', 0)
        assert_raises(AttributeError, setattr, cluster, 'indices', [])
        assert_array_equal(cluster.indices, [])
        assert_equal(len(cluster), 0)

        # Cannot create cluster that unexists in cluster map.
        assert_raises(ValueError, Cluster, None, 0)
        assert_raises(ValueError, Cluster, clusters, -1)
        assert_raises(ValueError, Cluster, clusters, 42)

        # Duplicate
        assert_equal(cluster, Cluster(clusters, cluster.id))

    def test_add(self):
        clusters = ClusterMap()
        cluster = clusters.create_cluster()

        indices = []
        for idx in range(1, 10):
            cluster.add(idx)
            indices.append(idx)
            assert_equal(len(cluster), idx)
            assert_array_equal(cluster.indices, indices)

        # Test __iter__
        assert_array_equal(list(cluster), indices)

        # Test __getitem__
        for i in range(len(indices)):
            assert_equal(cluster[i], indices[i])

        # Test index out of bound
        assert_raises(IndexError, cluster.__getitem__, len(cluster))
        assert_raises(IndexError, cluster.__getitem__, -len(cluster)-1)

        # Test slicing and negative indexing
        assert_equal(cluster[-1], indices[-1])
        assert_equal(cluster[::2], indices[::2])
        assert_raises(TypeError, cluster.__getitem__, [0])  # TODO: support list of indices


class TestClusterCentroid(unittest.TestCase):
    def setUp(self):
        self.features_shape = 10
        self.dtype = "float32"
        self.features = np.ones(self.features_shape, dtype=self.dtype)

    def test_attributes_and_constructor(self):
        clusters = ClusterMapCentroid(self.features_shape)
        cluster = clusters.create_cluster()
        assert_equal(type(cluster), ClusterCentroid)

        assert_equal(cluster.id, 0)
        assert_raises(AttributeError, setattr, cluster, 'id', 0)
        assert_raises(AttributeError, setattr, cluster, 'indices', [])
        assert_raises(AttributeError, setattr, cluster, 'centroid', [])
        assert_array_equal(cluster.indices, [])
        assert_array_equal(cluster.centroid, np.zeros(self.features_shape))
        assert_equal(len(cluster), 0)

        # Cannot create cluster that unexists in cluster map.
        assert_raises(ValueError, ClusterCentroid, None, 0)
        assert_raises(ValueError, ClusterCentroid, clusters, -1)
        assert_raises(ValueError, ClusterCentroid, clusters, 42)

        # Duplicate
        assert_equal(cluster, ClusterCentroid(clusters, cluster.id))

    def test_add(self):
        clusters = ClusterMapCentroid(self.features_shape)
        cluster = clusters.create_cluster()

        indices = []
        centroid = np.zeros(self.features_shape, dtype=self.dtype)
        for idx in range(1, 10):
            cluster.add(idx, (idx+1)*self.features)
            indices.append(idx)
            centroid = (centroid * (idx-1) + (idx+1)*self.features) / idx
            assert_equal(len(cluster), idx)
            assert_array_equal(cluster.indices, indices)
            assert_array_equal(cluster.centroid, centroid)


class TestClusterMap(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.data = [np.arange(3*5, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*10, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*15, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*17, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*20, dtype=self.dtype).reshape((-1, 3))]

        self.clusters = [[2, 4], [0, 3], [1]]

    def test_attributes_and_constructor(self):
        clusters = ClusterMap()
        assert_equal(len(clusters), 0)
        assert_array_equal(clusters.clusters, [])
        assert_array_equal(list(clusters), [])
        assert_raises(IndexError, clusters.__getitem__, 0)
        assert_raises(AttributeError, setattr, clusters, 'clusters', [])

    def test_add(self):
        clusters = ClusterMap()

        list_of_indices = []
        for i in range(3):
            list_of_indices.append([])
            id_cluster = clusters.create_cluster().id
            cluster = clusters[id_cluster]
            assert_equal(type(cluster), Cluster)
            assert_equal(len(clusters), i+1)

            for id_data in range(2*i):
                list_of_indices[-1].append(id_data)
                clusters.add(id_cluster, id_data)

        assert_array_equal(list(itertools.chain(*clusters)), list(itertools.chain(*list_of_indices)))

    def test_getitem_without_refdata(self):
        clusters = ClusterMap()
        for cluster in self.clusters:
            c = clusters.create_cluster()
            c.add(*cluster)

        #Test __iter__
        for id_cluster, (cluster, indices) in enumerate(zip(clusters, self.clusters)):
            assert_equal(type(cluster), Cluster)
            assert_equal(cluster.id, id_cluster)
            assert_array_equal(cluster, indices)
            assert_array_equal(cluster.indices, indices)

        #Test 'clusters' property
        for id_cluster, (cluster, indices) in enumerate(zip(clusters.clusters, self.clusters)):
            assert_equal(type(cluster), Cluster)
            assert_equal(cluster.id, id_cluster)
            assert_array_equal(cluster, indices)
            assert_array_equal(cluster.indices, indices)

        #Test __getitem__
        for id_cluster in range(len(clusters)):
            assert_array_equal(clusters[id_cluster].indices, self.clusters[id_cluster])

        # Test slicing
        for slice_obj in [np.s_[1:], np.s_[:-1], np.s_[::2], np.s_[::-1]]:
            for c1, c2 in zip(clusters[slice_obj], self.clusters[slice_obj]):
                assert_array_equal(c1, c2)

        # Test negative indexing
        assert_array_equal(clusters[-2], self.clusters[-2])

        # Test getting unexisting cluster
        assert_raises(IndexError, clusters.__getitem__, len(self.clusters))
        assert_raises(IndexError, clusters.__getitem__, -len(self.clusters)-1)
        assert_raises(TypeError, clusters.__getitem__, [0])  # TODO: support list of indices

    def test_getitem_with_refdata(self):
        clusters = ClusterMap(refdata=self.data)
        for cluster in self.clusters:
            c = clusters.create_cluster()
            c.add(*cluster)

        #Test __iter__
        for cluster, indices in zip(clusters, self.clusters):
            for i in range(len(indices)):
                assert_array_equal(cluster[i], self.data[indices[i]])

        # Test slicing
        for slice_obj in [np.s_[1:], np.s_[:-1], np.s_[::2], np.s_[::-1]]:
            for c1, c2 in zip(clusters[slice_obj], self.clusters[slice_obj]):
                for i in range(len(c2)):
                    assert_array_equal(c1[i], self.data[c2[i]])

        # Test negative indexing
        for i in range(len(self.clusters[-2])):
            assert_array_equal(clusters[-2][i], self.data[self.clusters[-2][i]])


class TestClusterMapCentroid(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.data = [np.arange(3*5, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*10, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*15, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*17, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*20, dtype=self.dtype).reshape((-1, 3))]

        self.clusters = [[2, 4], [0, 3], [1]]
        self.features_shape = 7
        self.features = np.ones(self.features_shape, dtype=self.dtype)

    def test_attributes_and_constructor(self):
        clusters = ClusterMapCentroid(self.features_shape)
        assert_array_equal(clusters.centroids, [])
        assert_raises(AttributeError, setattr, clusters, 'centroids', [])

    def test_add(self):
        clusters = ClusterMapCentroid(self.features_shape)

        centroids = []
        for i in range(3):
            id_cluster = clusters.create_cluster().id
            cluster = clusters[id_cluster]
            assert_array_equal(cluster.centroid, np.zeros_like(self.features))
            assert_equal(type(cluster), ClusterCentroid)

            centroids.append(np.zeros_like(self.features))
            for id_data in range(2*i):
                centroids[-1] = (centroids[-1]*id_data + (id_data+1)*self.features) / (id_data+1)
                clusters.add(id_cluster, id_data, (id_data+1)*self.features)

        assert_array_equal(list(itertools.chain(*clusters.centroids)), list(itertools.chain(*centroids)))

        # Check adding features of different sizes (shorter and longer)
        features_too_short = np.ones(self.features_shape-3, dtype=self.dtype)
        cluster = clusters.create_cluster()
        assert_raises(ValueError, cluster.add, 123, features_too_short)

        features_too_long = np.ones(self.features_shape+3, dtype=self.dtype)
        cluster = clusters.create_cluster()
        assert_raises(ValueError, cluster.add, 123, features_too_long)

    def test_getitem_without_refdata(self):
        clusters = ClusterMapCentroid(self.features_shape)
        for cluster in self.clusters:
            c = clusters.create_cluster()
            for i in cluster:
                c.add(i, self.features)

        #Test __iter__
        for id_cluster, (cluster, indices) in enumerate(zip(clusters, self.clusters)):
            assert_equal(type(cluster), ClusterCentroid)
            assert_equal(cluster.id, id_cluster)
            assert_array_equal(cluster, indices)
            assert_array_equal(cluster.indices, indices)

        #Test 'clusters' property
        for id_cluster, (cluster, indices) in enumerate(zip(clusters.clusters, self.clusters)):
            assert_equal(type(cluster), ClusterCentroid)
            assert_equal(cluster.id, id_cluster)
            assert_array_equal(cluster, indices)
            assert_array_equal(cluster.indices, indices)

        #Test __getitem__
        for id_cluster in range(len(clusters)):
            assert_array_equal(clusters[id_cluster].indices, self.clusters[id_cluster])

        # Test slicing
        for slice_obj in [np.s_[1:], np.s_[:-1], np.s_[::2], np.s_[::-1]]:
            for c1, c2 in zip(clusters[slice_obj], self.clusters[slice_obj]):
                assert_array_equal(c1, c2)

        # Test negative indexing
        assert_array_equal(clusters[-2], self.clusters[-2])

        # Test getting unexisting cluster
        assert_raises(IndexError, clusters.__getitem__, len(self.clusters))
        assert_raises(IndexError, clusters.__getitem__, -len(self.clusters)-1)
        assert_raises(TypeError, clusters.__getitem__, [0])  # TODO: support list of indices

    def test_getitem_with_refdata(self):
        clusters = ClusterMapCentroid(self.features_shape, refdata=self.data)
        for cluster in self.clusters:
            c = clusters.create_cluster()
            for i in cluster:
                c.add(i, self.features)

        #Test __iter__
        for cluster, indices in zip(clusters, self.clusters):
            for i in range(len(indices)):
                assert_array_equal(cluster[i], self.data[indices[i]])

        # Test slicing
        for slice_obj in [np.s_[1:], np.s_[:-1], np.s_[::2], np.s_[::-1]]:
            for c1, c2 in zip(clusters[slice_obj], self.clusters[slice_obj]):
                for i in range(len(c2)):
                    assert_array_equal(c1[i], self.data[c2[i]])

        # Test negative indexing
        for i in range(len(self.clusters[-2])):
            assert_array_equal(clusters[-2][i], self.data[self.clusters[-2][i]])
