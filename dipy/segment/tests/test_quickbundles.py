import numpy as np
import unittest

from dipy.segment.quickbundles import QuickBundles

import dipy.segment.metric as dipymetric
from dipy.segment.clusteringspeed import quickbundles as cython_quickbundles

import nose
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_raises


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
        qb = QuickBundles(threshold=11, metric=dipymetric.Spatial())

        clusters = qb.cluster(self.data)
        self.assertSequenceEqual(clusters.clusters, self.clusters)

        # TODO: move this test into test_metric.
        # MDF required streamlines to have the same length
        qb = QuickBundles(threshold=10, metric=dipymetric.MDF())
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
