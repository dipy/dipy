import numpy as np
import itertools

from numpy.testing import assert_array_equal, assert_equal, assert_raises
from dipy.testing.memory import get_type_refcount
from dipy.testing import assert_arrays_equal

from dipy.segment.clustering import QuickBundles

import dipy.segment.featurespeed as dipysfeature
import dipy.segment.metricspeed as dipysmetric
from dipy.segment.clustering_algorithms import quickbundles
import dipy.tracking.streamline as streamline_utils
from dipy.testing.decorators import set_random_number_generator


dtype = "float32"
threshold = 7
data = [np.arange(3 * 5, dtype=dtype).reshape((-1, 3)) + 2 * threshold,
        np.arange(3 * 10, dtype=dtype).reshape((-1, 3)) + 0 * threshold,
        np.arange(3 * 15, dtype=dtype).reshape((-1, 3)) + 8 * threshold,
        np.arange(3 * 17, dtype=dtype).reshape((-1, 3)) + 2 * threshold,
        np.arange(3 * 20, dtype=dtype).reshape((-1, 3)) + 8 * threshold]

clusters_truth = [[0, 1], [2, 4], [3]]


def test_quickbundles_empty_data():
    threshold = 10
    metric = dipysmetric.SumPointwiseEuclideanMetric()
    clusters = quickbundles([], metric, threshold)
    assert_equal(len(clusters), 0)
    assert_equal(len(clusters.centroids), 0)

    clusters = quickbundles([], metric, threshold, ordering=[])
    assert_equal(len(clusters), 0)
    assert_equal(len(clusters.centroids), 0)


def test_quickbundles_wrong_metric():
    assert_raises(ValueError, QuickBundles,
                  threshold=10., metric="WrongMetric")


def test_quickbundles_shape_incompatibility():
    # QuickBundles' old default metric (AveragePointwiseEuclideanMetric,
    #  aka MDF) requires that all streamlines have the same number of points.
    metric = dipysmetric.AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=20., metric=metric)
    assert_raises(ValueError, qb.cluster, data)

    # QuickBundles' new default metric (AveragePointwiseEuclideanMetric,
    # aka MDF combined with ResampleFeature) will automatically resample
    # streamlines so they all have 18 points.
    qb = QuickBundles(threshold=20.)
    clusters1 = qb.cluster(data)

    feature = dipysfeature.ResampleFeature(nb_points=18)
    metric = dipysmetric.AveragePointwiseEuclideanMetric(feature)
    qb = QuickBundles(threshold=20., metric=metric)
    clusters2 = qb.cluster(data)

    assert_arrays_equal(list(itertools.chain(*clusters1)),
                        list(itertools.chain(*clusters2)))


@set_random_number_generator(7)
def test_quickbundles_2D(rng):
    # Test quickbundles clustering using 2D points and the Eulidean metric.
    data = []
    data += \
        [rng.standard_normal((1, 2)) + np.array([0, 0]) for _ in range(1)]
    data += \
        [rng.standard_normal((1, 2)) + np.array([10, 10]) for _ in range(2)]
    data += \
        [rng.standard_normal((1, 2)) + np.array([-10, 10]) for _ in range(3)]
    data += \
        [rng.standard_normal((1, 2)) + np.array([10, -10]) for _ in range(4)]
    data += \
        [rng.standard_normal((1, 2)) + np.array([-10, -10]) for _ in range(5)]
    data = np.array(data, dtype=dtype)

    clusters_truth = [[0], [1, 2], [3, 4, 5],
                      [6, 7, 8, 9], [10, 11, 12, 13, 14]]

    # # Uncomment the following to visualize this test
    # import pylab as plt
    # plt.plot(*zip(*data[0:1, 0]), linestyle='None', marker='s')
    # plt.plot(*zip(*data[1:3, 0]), linestyle='None', marker='o')
    # plt.plot(*zip(*data[3:6, 0]), linestyle='None', marker='+')
    # plt.plot(*zip(*data[6:10, 0]), linestyle='None', marker='.')
    # plt.plot(*zip(*data[10:, 0]), linestyle='None', marker='*')
    # plt.show()

    # Theoretically, using a threshold above the following value will not
    # produce expected results.
    threshold = np.sqrt(2*(10**2))-np.sqrt(2)
    metric = dipysmetric.SumPointwiseEuclideanMetric()
    ordering = np.arange(len(data))
    for i in range(100):
        rng.shuffle(ordering)
        clusters = quickbundles(data, metric, threshold, ordering=ordering)

        # Check if clusters are the same as 'clusters_truth'
        for cluster in clusters:
            # Find the corresponding cluster in 'clusters_truth'
            for cluster_truth in clusters_truth:
                if cluster_truth[0] in cluster.indices:
                    assert_equal(sorted(cluster.indices),
                                 sorted(cluster_truth))

    # Cluster each cluster again using a small threshold
    for cluster in clusters:
        subclusters = quickbundles(data, metric, threshold=0,
                                   ordering=cluster.indices)
        assert_equal(len(subclusters), len(cluster))
        assert_equal(sorted(itertools.chain(*subclusters)),
                     sorted(cluster.indices))

    # A very large threshold should produce only 1 cluster
    clusters = quickbundles(data, metric, threshold=np.inf)
    assert_equal(len(clusters), 1)
    assert_equal(len(clusters[0]), len(data))
    assert_array_equal(clusters[0].indices, range(len(data)))

    # A very small threshold should produce only N clusters where N=len(data)
    clusters = quickbundles(data, metric, threshold=0)
    assert_equal(len(clusters), len(data))
    assert_array_equal(list(map(len, clusters)), np.ones(len(data)))
    assert_array_equal([idx for cluster in clusters
                        for idx in cluster.indices], range(len(data)))


def test_quickbundles_streamlines():
    rdata = streamline_utils.set_number_of_points(data, 10)
    qb = QuickBundles(threshold=2*threshold)

    clusters = qb.cluster(rdata)
    # By default `refdata` refers to data being clustered.
    assert_equal(clusters.refdata, rdata)
    # Set `refdata` to return indices instead of actual data points.
    clusters.refdata = None
    assert_array_equal(list(itertools.chain(*clusters)),
                       list(itertools.chain(*clusters_truth)))

    # Cluster read-only data
    for datum in rdata:
        datum.setflags(write=False)

    # Cluster data with different dtype (should be converted into float32)
    for datatype in [np.float64, np.int32, np.int64]:
        newdata = [datum.astype(datatype) for datum in rdata]
        clusters = qb.cluster(newdata)
        assert_equal(clusters.centroids[0].dtype, np.float32)


def test_quickbundles_with_python_metric():

    class MDFpy(dipysmetric.Metric):
        def are_compatible(self, shape1, shape2):
            return shape1 == shape2

        def dist(self, features1, features2):
            dist = np.sqrt(np.sum((features1 - features2)**2, axis=1))
            dist = np.sum(dist / len(features1))
            return dist

    rdata = streamline_utils.set_number_of_points(data, 10)
    qb = QuickBundles(threshold=2 * threshold, metric=MDFpy())

    clusters = qb.cluster(rdata)

    # By default `refdata` refers to data being clustered.
    assert_equal(clusters.refdata, rdata)
    # Set `refdata` to return indices instead of actual data points.
    clusters.refdata = None
    assert_array_equal(list(itertools.chain(*clusters)),
                       list(itertools.chain(*clusters_truth)))

    # Cluster read-only data
    for datum in rdata:
        datum.setflags(write=False)

    # Cluster data with different dtype (should be converted into float32)
    for datatype in [np.float64, np.int32, np.int64]:
        newdata = [datum.astype(datatype) for datum in rdata]
        clusters = qb.cluster(newdata)
        assert_equal(clusters.centroids[0].dtype, np.float32)


def test_quickbundles_with_not_order_invariant_metric():
    metric = dipysmetric.AveragePointwiseEuclideanMetric()
    qb = QuickBundles(threshold=np.inf, metric=metric)

    streamline = np.arange(10*3, dtype=dtype).reshape((-1, 3))
    streamlines = [streamline, streamline[::-1]]

    clusters = qb.cluster(streamlines)
    assert_equal(len(clusters), 1)
    assert_array_equal(clusters[0].centroid, streamline)


def test_quickbundles_memory_leaks():
    qb = QuickBundles(threshold=2*threshold)

    type_name_pattern = "memoryview"
    initial_types_refcount = get_type_refcount(type_name_pattern)

    qb.cluster(data)
    # At this point, all memoryviews created during clustering should be freed.
    assert_equal(get_type_refcount(type_name_pattern), initial_types_refcount)
