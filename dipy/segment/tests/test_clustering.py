
import numpy as np
import unittest
from dipy.segment.clustering import ClusterMap
from dipy.segment.quickbundles import QuickBundles2

import dipy.segment.metric as dipymetric
from dipy.segment.clustering import CentroidClusters
from dipy.segment.clusteringspeed import quickbundles as cython_quickbundles

import nose
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_raises


class TestClusterMap(unittest.TestCase):
    def setUp(self):
        self.data = [np.arange(3*5).reshape((-1, 3)),
                     np.arange(3*10).reshape((-1, 3)),
                     np.arange(3*15).reshape((-1, 3)),
                     np.arange(3*17).reshape((-1, 3)),
                     np.arange(3*20).reshape((-1, 3))]

        self.clusters = [[2, 4], [0, 3], [1]]
        #self.cluster_map = ClusterMap(self.clusters, data=self.data)

    def test_iteration(self):
        # Test without passing a reference to the data
        cluster_map = ClusterMap(self.clusters)
        assert_equal(len(cluster_map), len(self.clusters))

        for cluster_data, cluster_idx in zip(cluster_map, self.clusters):
            assert_array_equal(cluster_data, cluster_data)

        # Test passing a reference to the data
        cluster_map = ClusterMap(self.clusters, data=self.data)
        assert_equal(len(cluster_map), len(self.clusters))

        for cluster_data, cluster_idx in zip(cluster_map, self.clusters):
            for element, element_id in zip(cluster_data, cluster_idx):
                assert_array_equal(element, self.data[element_id])

    def test_getitem(self):
        # Test without passing a reference to the data
        cluster_map = ClusterMap(self.clusters)
        assert_equal(len(cluster_map), len(self.clusters))

        for i in range(len(self.clusters)):
            assert_array_equal(cluster_map[i], self.clusters[i])

        # Test passing a reference to the data
        cluster_map = ClusterMap(self.clusters, data=self.data)
        assert_equal(len(cluster_map), len(self.clusters))

        for i in range(len(self.clusters)):
            for element, element_id in zip(cluster_map[i], self.clusters[i]):
                assert_array_equal(element, self.data[element_id])

    def test_centroids(self):
        raise nose.SkipTest()

    def test_medoids(self):
        raise nose.SkipTest()


class TestCentroidClusters(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"

    def test_add(self):
        features = np.ones(10, dtype=self.dtype)

        clusters = CentroidClusters(nb_features=len(features))
        assert_equal(len(clusters), 0)

        cluster = clusters.create_cluster()
        assert_equal(cluster.id, 0)
        assert_equal(len(clusters), 1)

        cluster = clusters[cluster.id]
        assert_array_equal(cluster.centroid, np.zeros(len(features), dtype=self.dtype))
        assert_equal(len(cluster), 0)

        # Add a streamline to a specific cluster using CentroidClusters interface.
        idx = 42
        clusters.add(cluster.id, idx, features)
        assert_array_equal(cluster.centroid, np.ones(len(features), dtype=self.dtype))

        assert_equal(len(cluster), 1)
        assert_equal(cluster.indices[0], idx)

        # Add a streamline to a specific cluster using Cluster interface.
        idx = 21
        cluster = clusters.create_cluster()
        assert_equal(cluster.id, 1)
        assert_equal(len(clusters), 2)

        cluster.add(idx, features)
        assert_array_equal(cluster.centroid, np.ones(len(features), dtype=self.dtype))

        assert_equal(len(cluster), 1)
        assert_equal(cluster.indices[0], idx)

        # Check centroid after adding several features vectors.
        features = np.ones(10, dtype=self.dtype)
        M = 11
        clusters = CentroidClusters(nb_features=len(features))
        cluster = clusters.create_cluster()
        for i in range(M):
            cluster.add(i, np.arange(10, dtype=self.dtype) * i)

        expected_centroid = np.arange(10, dtype=self.dtype) * ((M*(M-1))/2.) / M

        assert_array_equal(cluster.centroid, expected_centroid)

        # Check adding features of different sizes (shorter and longer)
        features_short = np.ones(len(features)-3, dtype=self.dtype)
        cluster = clusters.create_cluster()
        assert_raises(ValueError, cluster.add, 123, features_short)

        features_long = np.ones(len(features)+3, dtype=self.dtype)
        cluster = clusters.create_cluster()
        assert_raises(ValueError, cluster.add, 123, features_long)


class TestQuickBundles(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.data = [np.arange(3*5, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*10, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*15, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*17, dtype=self.dtype).reshape((-1, 3)),
                     np.arange(3*20, dtype=self.dtype).reshape((-1, 3))]

        self.clusters = [[2, 4], [0, 3], [1]]
        #self.cluster_map = ClusterMap(self.clusters, data=self.data)

    def test_clustering(self):
        qb = QuickBundles2(threshold=11, metric=dipymetric.Spatial())

        clusters = qb.cluster(self.data)
        from ipdb import set_trace as dbg
        dbg()
        self.assertSequenceEqual(clusters.clusters, self.clusters)

        # TODO: move this test into test_metric.
        # MDF required streamlines to have the same length
        qb = QuickBundles2(threshold=10, metric=dipymetric.MDF())
        assert_raises(ValueError, qb.cluster, self.data)

    def test_memory_leak(self):
        import resource

        NB_LOOPS = 20
        NB_DATA = 1000
        NB_POINTS = 10
        data = []

        for i in range(NB_DATA):
            data.append(i * np.ones((NB_POINTS, 3), dtype=self.dtype))

        metric = dipymetric.MDF()

        ram_usages = np.zeros(NB_LOOPS)
        for i in range(NB_LOOPS):
            cython_quickbundles(data, metric, threshold=10)
            ram_usages[i] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        print (["{0:.2f}Mo".format(ram/1024.) for ram in np.diff(ram_usages)])
