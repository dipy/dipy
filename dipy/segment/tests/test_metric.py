import numpy as np
import dipy.segment.metric as dipymetric
import itertools

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
streamlines = [s1, s2, s3, s4]


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


def test_cosine_metric():
    feature = dipymetric.VectorBetweenEndpointsFeature()

    class CosineMetric(dipymetric.Metric):
        def __init__(self, feature):
            super(CosineMetric, self).__init__(feature=feature)

        def are_compatible(self, shape1, shape2):
            return shape1 == shape2 and shape1[0] == 1

        def dist(self, v1, v2):
            norm = lambda x: np.sqrt(np.sum(x**2, dtype=np.float64))

            # Check if we have null vectors
            if norm(v1) == 0:
                return 0. if norm(v2) == 0 else 1.

            v1_normed = v1.astype(np.float64) / norm(v1)
            v2_normed = v2.astype(np.float64) / norm(v2)
            cos_theta = np.dot(v1_normed, v2_normed.T)
            # Make sure it's in [-1, 1], i.e. within domain of arccosine
            cos_theta = np.minimum(cos_theta, 1.)
            cos_theta = np.maximum(cos_theta, -1.)
            return np.arccos(cos_theta) / np.pi  # Normalized cosine distance

    for metric in [CosineMetric(feature), dipymetric.CosineMetric(feature)]:
        # Test special cases of the cosine distance.
        v0 = np.array([[0, 0, 0]], dtype=np.float32)
        v1 = np.array([[1, 2, 3]], dtype=np.float32)
        v2 = np.array([[1, -1./2, 0]], dtype=np.float32)
        v3 = np.array([[-1, -2, -3]], dtype=np.float32)

        assert_equal(metric.dist(v0, v0), 0.)   # dot-dot
        assert_equal(metric.dist(v0, v1), 1.)   # dot-line
        assert_equal(metric.dist(v1, v1), 0.)   # collinear
        assert_equal(metric.dist(v1, v2), 0.5)  # orthogonal
        assert_equal(metric.dist(v1, v3), 1.)   # opposite

        for s1, s2 in itertools.product(*[streamlines]*2):  # All possible pairs
            # Extract features since metric doesn't work directly on streamlines
            f1 = metric.feature.extract(s1)
            f2 = metric.feature.extract(s2)

            # Test method are_compatible
            are_vectors = f1.shape[0] == 1 and f2.shape[0] == 1
            same_dimension = f1.shape[1] == f2.shape[1]
            assert_equal(metric.are_compatible(f1.shape, f2.shape), are_vectors and same_dimension)

            # Test method dist if features are compatible
            if metric.are_compatible(f1.shape, f2.shape):
                distance = metric.dist(f1, f2)
                if np.all(f1 == f2):
                    assert_equal(distance, 0.)

                assert_almost_equal(distance, dipymetric.dist(metric, s1, s2))
                assert_true(distance >= 0.)
                assert_true(distance <= 1.)

        # This metric type is not order invariant
        assert_false(metric.is_order_invariant)
        for s1, s2 in itertools.product(*[streamlines]*2):  # All possible pairs
            f1 = metric.feature.extract(s1)
            f2 = metric.feature.extract(s2)

            if not metric.are_compatible(f1.shape, f2.shape):
                continue

            f1_flip = metric.feature.extract(s1[::-1])
            f2_flip = metric.feature.extract(s2[::-1])

            distance = metric.dist(f1, f2)
            assert_almost_equal(metric.dist(f1_flip, f2_flip), distance)

            if not np.all(f1_flip == f2_flip):
                assert_false(metric.dist(f1, f2_flip) == distance)
                assert_false(metric.dist(f1_flip, f2) == distance)


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
