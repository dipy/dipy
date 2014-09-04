import numpy as np
import unittest
import itertools

from dipy.segment.clustering import QuickBundles

import dipy.segment.metric as dipymetric
from dipy.segment.clusteringspeed import quickbundles

import nose
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_raises


# TODO: WIP
class TestQuickBundles(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.threshold = 7
        self.data = [np.arange(3*05, dtype=self.dtype).reshape((-1, 3)) + 2*self.threshold,
                     np.arange(3*10, dtype=self.dtype).reshape((-1, 3)) + 0*self.threshold,
                     np.arange(3*15, dtype=self.dtype).reshape((-1, 3)) + 8*self.threshold,
                     np.arange(3*17, dtype=self.dtype).reshape((-1, 3)) + 2*self.threshold,
                     np.arange(3*20, dtype=self.dtype).reshape((-1, 3)) + 8*self.threshold]

        self.clusters = [[0, 1], [2, 4], [3]]

    def test_2D_clustering(self):
        # Test quickbundles clustering using 2D points and the Eulidean metric.
        rng = np.random.RandomState(42)
        data = []
        data += [rng.randn(1, 2) + np.array([0, 0]) for i in range(1)]
        data += [rng.randn(1, 2) + np.array([10, 10]) for i in range(2)]
        data += [rng.randn(1, 2) + np.array([-10, 10]) for i in range(3)]
        data += [rng.randn(1, 2) + np.array([10, -10]) for i in range(4)]
        data += [rng.randn(1, 2) + np.array([-10, -10]) for i in range(5)]
        data = np.array(data, dtype=self.dtype)

        clusters_truth = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13, 14]]

        # Uncomment the following to visualize this test
        # import pylab as plt
        # plt.plot(*zip(*data[0:1, 0]), linestyle='None', marker='s')
        # plt.plot(*zip(*data[1:3, 0]), linestyle='None', marker='o')
        # plt.plot(*zip(*data[3:6, 0]), linestyle='None', marker='+')
        # plt.plot(*zip(*data[6:10, 0]), linestyle='None', marker='.')
        # plt.plot(*zip(*data[10:, 0]), linestyle='None', marker='*')
        # plt.show()

        # Theorically using a threshold above the following value will not
        # produce expected results.
        threshold = np.sqrt(2*(10**2))-np.sqrt(2)
        metric = dipymetric.Euclidean(dipymetric.CenterOfMass())
        ordering = np.arange(len(data))
        for i in range(100):
            rng.shuffle(ordering)

            clusters = quickbundles(data, metric, threshold, ordering=ordering)

            # Check if clusters are the same as 'clusters_truth'
            for cluster in clusters:
                # Find the corresponding cluster in 'clusters_truth'
                for cluster_truth in clusters_truth:
                    if cluster_truth[0] in cluster.indices:
                        self.assertItemsEqual(cluster.indices, cluster_truth)

        # A very large threshold should produce only 1 cluster
        clusters = quickbundles(data, metric, threshold=np.inf)
        assert_equal(len(clusters), 1)
        assert_equal(len(clusters[0]), len(data))
        assert_array_equal(clusters[0].indices, range(len(data)))

        # A very small threshold should produce only N clusters where N=len(data)
        clusters = quickbundles(data, metric, threshold=0)
        assert_equal(len(clusters), len(data))
        assert_array_equal(map(len, clusters), np.ones(len(data)))
        assert_array_equal([idx for cluster in clusters for idx in cluster.indices], range(len(data)))

    def test_clustering(self):
        metric = dipymetric.Euclidean(dipymetric.CenterOfMass())
        qb = QuickBundles(threshold=2*self.threshold, metric=metric)

        clusters = qb.cluster(self.data)
        assert_array_equal(list(itertools.chain(*clusters)), list(itertools.chain(*self.clusters)))

        # TODO: move this test into test_metric.
        # MDF required streamlines to have the same length
        # qb = QuickBundles(threshold=10, metric=dipymetric.MDF())
        # assert_raises(ValueError, qb.cluster, self.data)

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
            quickbundles(data, metric, threshold=10)
            ram_usages[i] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        print (["{0:.2f}Mo".format(ram/1024.) for ram in np.diff(ram_usages)])
