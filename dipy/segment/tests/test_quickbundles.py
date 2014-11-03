import numpy as np
import itertools

from dipy.segment.clustering import QuickBundles

import dipy.segment.metric as dipymetric
from dipy.segment.clustering_algorithms import quickbundles

from nose.tools import assert_equal, assert_items_equal
from numpy.testing import assert_array_equal, run_module_suite

dtype = "float32"
threshold = 7
data = [np.arange(3*05, dtype=dtype).reshape((-1, 3)) + 2*threshold,
        np.arange(3*10, dtype=dtype).reshape((-1, 3)) + 0*threshold,
        np.arange(3*15, dtype=dtype).reshape((-1, 3)) + 8*threshold,
        np.arange(3*17, dtype=dtype).reshape((-1, 3)) + 2*threshold,
        np.arange(3*20, dtype=dtype).reshape((-1, 3)) + 8*threshold]

clusters_truth = [[0, 1], [2, 4], [3]]


def test_quickbundles_empty_data():
    data = []
    threshold = 10
    metric = dipymetric.SumPointwiseEuclideanMetric()
    clusters = quickbundles(data, metric, threshold)
    assert_equal(len(clusters), 0)
    assert_equal(len(clusters.centroids), 0)


def test_quickbundles_2D():
    # Test quickbundles clustering using 2D points and the Eulidean metric.
    rng = np.random.RandomState(42)
    data = []
    data += [rng.randn(1, 2) + np.array([0, 0]) for i in range(1)]
    data += [rng.randn(1, 2) + np.array([10, 10]) for i in range(2)]
    data += [rng.randn(1, 2) + np.array([-10, 10]) for i in range(3)]
    data += [rng.randn(1, 2) + np.array([10, -10]) for i in range(4)]
    data += [rng.randn(1, 2) + np.array([-10, -10]) for i in range(5)]
    data = np.array(data, dtype=dtype)

    clusters_truth = [[0], [1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12, 13, 14]]

    # # Uncomment the following to visualize this test
    # import pylab as plt
    # plt.plot(*zip(*data[0:1, 0]), linestyle='None', marker='s')
    # plt.plot(*zip(*data[1:3, 0]), linestyle='None', marker='o')
    # plt.plot(*zip(*data[3:6, 0]), linestyle='None', marker='+')
    # plt.plot(*zip(*data[6:10, 0]), linestyle='None', marker='.')
    # plt.plot(*zip(*data[10:, 0]), linestyle='None', marker='*')

    # from dipy.segment.metric import distance_matrix
    # metric = dipymetric.Euclidean(dipymetric.CenterOfMass())
    # dM = distance_matrix(metric, data, data)
    # plt.figure()
    # plt.imshow(dM, interpolation="nearest")
    # plt.colorbar()

    # plt.show(False)

    # Theorically using a threshold above the following value will not
    # produce expected results.
    threshold = np.sqrt(2*(10**2))-np.sqrt(2)
    metric = dipymetric.SumPointwiseEuclideanMetric(dipymetric.CenterOfMassFeature())
    ordering = np.arange(len(data))
    for i in range(100):
        rng.shuffle(ordering)
        clusters = quickbundles(data, metric, threshold, ordering=ordering)

        # Check if clusters are the same as 'clusters_truth'
        for cluster in clusters:
            # Find the corresponding cluster in 'clusters_truth'
            for cluster_truth in clusters_truth:
                if cluster_truth[0] in cluster.indices:
                    assert_items_equal(cluster.indices, cluster_truth)

    # Cluster each cluster again using a small threshold
    for cluster in clusters:
        subclusters = quickbundles(data, metric, threshold=0, ordering=cluster.indices)
        assert_equal(len(subclusters), len(cluster))
        assert_items_equal(itertools.chain(*subclusters), cluster.indices)

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


def test_quickbundles_streamlines():
    metric = dipymetric.SumPointwiseEuclideanMetric(dipymetric.CenterOfMassFeature())
    qb = QuickBundles(threshold=2*threshold, metric=metric)

    clusters = qb.cluster(data)
    assert_array_equal(list(itertools.chain(*clusters)), list(itertools.chain(*clusters_truth)))

    # Cluster from a generator
    clusters = qb.cluster(iter(data))
    assert_array_equal(list(itertools.chain(*clusters)), list(itertools.chain(*clusters_truth)))

    # Cluster read-only data
    for datum in data:
        datum.setflags(write=False)

    clusters = qb.cluster(data)


def test_quickbundles_with_not_order_invariant_metric():
    metric = dipymetric.AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=np.inf, metric=metric)

    streamline = np.arange(10*3, dtype=dtype).reshape((-1, 3))
    streamlines = [streamline, streamline[::-1]]

    clusters = qb.cluster(streamlines)
    assert_equal(len(clusters), 1)
    assert_array_equal(clusters[0].centroid, streamline)


if __name__ == '__main__':
    run_module_suite()
