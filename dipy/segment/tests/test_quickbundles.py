import numpy as np
import unittest
import itertools

from dipy.segment.clustering import QuickBundles

import dipy.segment.metric as dipymetric
from dipy.segment.clusteringspeed import quickbundles as cython_quickbundles

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
            cython_quickbundles(data, metric, threshold=10)
            ram_usages[i] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        print (["{0:.2f}Mo".format(ram/1024.) for ram in np.diff(ram_usages)])
