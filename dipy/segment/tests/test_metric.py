
import numpy as np
import dipy.segment.metric as dipymetric

from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_raises, run_module_suite


def norm_L2(x):
    return np.sqrt(np.sum(x**2, axis=1, dtype="float"))


def average_euclidean(x, y):
    return np.mean(norm_L2(x-y))


def mdf_distance(x, y):
    dist_direct = average_euclidean(x, y)
    dist_flipped = average_euclidean(x, y[::-1])
    return np.min(np.array([dist_direct, dist_flipped]))


dtype = "float32"
s1 = np.array([np.arange(10, dtype=dtype)]*3).T  # 10x3
s2 = np.arange(3*10, dtype=dtype).reshape((-1, 3))[::-1]  # 10x3
s3 = np.array([np.arange(5, dtype=dtype)]*4)  # 5x4
s4 = np.array([np.arange(5, dtype=dtype)]*3)  # 5x3


def test_metric_minimum_average_direct_flip():
    metric = dipymetric.MinimumAverageDirectFlipMetric()

    assert_equal(metric.feature.infer_shape(s1), s1.shape)

    features = metric.feature.extract(s1)
    assert_equal(features.shape, s1.shape)
    assert_array_equal(features, s1)

    # Test dist()
    features1 = metric.feature.extract(s1)
    assert_true(metric.are_compatible(features1.shape, features1.shape))
    dist = metric.dist(features1, features1)
    assert_equal(dist, 0.0)
    assert_equal(dipymetric.mdf(s1, s1), 0.0)

    # Features 1 and 2 do have the same number of points and dimensions
    features2 = metric.feature.extract(s2)
    assert_true(metric.are_compatible(features1.shape, features2.shape))
    dist = metric.dist(features1, features2)
    ground_truth = mdf_distance(s1, s2)
    assert_almost_equal(dist, ground_truth)
    assert_almost_equal(dipymetric.mdf(s1, s2), ground_truth)

    # Features 1 and 3 do not have the same number of dimensions
    features3 = metric.feature.extract(s3)
    assert_false(metric.are_compatible(features1.shape, features3.shape))
    assert_raises(ValueError, metric.dist, features1, features3)
    assert_raises(ValueError, dipymetric.mdf, s1, s3)

    # Features 1 and 4 do not have the same number of points
    features4 = metric.feature.extract(s4)
    assert_false(metric.are_compatible(features1.shape, features4.shape))
    assert_raises(ValueError, metric.dist, features1, features4)
    assert_raises(ValueError, dipymetric.dist, metric, s1, s4)


def test_subclassing_metric():
    class EmptyMetric(dipymetric.Metric):
        pass

    metric = EmptyMetric()
    assert_raises(NotImplementedError, metric.are_compatible, None, None)
    assert_raises(NotImplementedError, metric.dist, None, None)

    class MDF(dipymetric.Metric):
        def are_compatible(self, shape1, shape2):
            return shape1 == shape2

        def dist(self, features1, features2):
            return mdf_distance(features1, features2)

    metric = MDF()
    d1 = dipymetric.dist(metric, s1, s2)
    d2 = dipymetric.mdf(s1, s2)
    assert_almost_equal(d1, d2)

    features1 = metric.feature.extract(s1)
    features2 = metric.feature.extract(s2)
    d3 = metric.dist(features1, features2)
    assert_almost_equal(d1, d3)


def test_distance_matrix():
    metric = dipymetric.SumPointwiseEuclideanMetric()

    for dtype in [np.int32, np.int64, np.float32, np.float64]:
        # Compute distances of all tuples spawn by the Cartesian product
        # of `data` with itself.
        data = (np.random.rand(4, 10, 3)*10).astype(dtype)
        D = dipymetric.distance_matrix(metric, data)
        assert_equal(D.shape, (len(data), len(data)))
        assert_array_equal(np.diag(D), np.zeros(len(data)))

        if metric.is_order_invariant:
            # Distance matrix should be symmetric
            assert_array_equal(D, D.T)

        for i in range(len(data)):
            for j in range(len(data)):
                assert_equal(D[i, j], dipymetric.dist(metric, data[i], data[j]))

        # Compute distances of all tuples spawn by the Cartesian product
        # of `data` with `data2`.
        data2 = (np.random.rand(3, 10, 3)*10).astype(dtype)
        D = dipymetric.distance_matrix(metric, data, data2)
        assert_equal(D.shape, (len(data), len(data2)))

        for i in range(len(data)):
            for j in range(len(data2)):
                assert_equal(D[i, j], dipymetric.dist(metric, data[i], data2[j]))


if __name__ == '__main__':
    run_module_suite()
