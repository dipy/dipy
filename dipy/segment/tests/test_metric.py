import numpy as np
import dipy.segment.metric as dipymetric
import itertools

from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_raises, run_module_suite


def norm(x, ord=None, axis=None):
    if axis is not None:
        return np.apply_along_axis(np.linalg.norm, axis, x.astype(np.float64), ord)

    return np.linalg.norm(x.astype(np.float64), ord=ord)


dtype = "float32"

# Create wiggling streamline
nb_points = 18
rng = np.random.RandomState(42)
x = np.linspace(0, 10, nb_points)
y = rng.rand(nb_points)
z = np.sin(np.linspace(0, np.pi, nb_points))  # Bending
s = np.array([x, y, z], dtype=dtype).T

# Create trivial streamlines
s1 = np.array([np.arange(10, dtype=dtype)]*3).T  # 10x3
s2 = np.arange(3*10, dtype=dtype).reshape((-1, 3))[::-1]  # 10x3
s3 = np.array([np.arange(5, dtype=dtype)]*4)  # 5x4
s4 = np.array([np.arange(5, dtype=dtype)]*3)  # 5x3
streamlines = [s, s1, s2, s3, s4]


def test_metric_minimum_average_direct_flip():
    feature = dipymetric.IdentityFeature()

    class MinimumAverageDirectFlipMetric(dipymetric.Metric):
        def __init__(self, feature):
            super(MinimumAverageDirectFlipMetric, self).__init__(feature=feature)

        @property
        def is_order_invariant(self):
            return True  # Ordering is handled in the distance computation

        def are_compatible(self, shape1, shape2):
            return shape1[0] == shape2[0]

        def dist(self, v1, v2):
            average_euclidean = lambda x, y: np.mean(norm(x-y, axis=1))
            dist_direct = average_euclidean(v1, v2)
            dist_flipped = average_euclidean(v1, v2[::-1])
            return min(dist_direct, dist_flipped)

    for metric in [MinimumAverageDirectFlipMetric(feature),
                   dipymetric.MinimumAverageDirectFlipMetric(feature)]:

        # Test special cases of the MDF distance.
        assert_equal(metric.dist(s, s), 0.)
        assert_equal(metric.dist(s, s[::-1]), 0.)

        # Translation
        offset = np.array([0.8, 1.3, 5], dtype=dtype)
        assert_almost_equal(metric.dist(s, s+offset), norm(offset), 5)

        # Scaling
        M_scaling = np.diag([1.2, 2.8, 3]).astype(dtype)
        s_mean = np.mean(s, axis=0)
        s_zero_mean = s - s_mean
        s_scaled = np.dot(M_scaling, s_zero_mean.T).T + s_mean
        d = np.mean(norm((np.diag(M_scaling)-1)*s_zero_mean, axis=1))
        assert_almost_equal(metric.dist(s, s_scaled), d, 5)

        # Rotation
        from dipy.core.geometry import rodrigues_axis_rotation
        rot_axis = np.array([1, 2, 3], dtype=dtype)
        M_rotation = rodrigues_axis_rotation(rot_axis, 60.).astype(dtype)
        s_mean = np.mean(s, axis=0)
        s_zero_mean = s - s_mean
        s_rotated = np.dot(M_rotation, s_zero_mean.T).T + s_mean

        opposite = norm(np.cross(rot_axis, s_zero_mean), axis=1) / norm(rot_axis)
        distances = np.sqrt(2*opposite**2 * (1 - np.cos(60.*np.pi/180.))).astype(dtype)
        d = np.mean(distances)
        assert_almost_equal(metric.dist(s, s_rotated), d, 5)

        for s1, s2 in itertools.product(*[streamlines]*2):  # All possible pairs
            # Extract features since metric doesn't work directly on streamlines
            f1 = metric.feature.extract(s1)
            f2 = metric.feature.extract(s2)

            # Test method are_compatible
            same_nb_points = f1.shape[0] == f2.shape[0]
            assert_equal(metric.are_compatible(f1.shape, f2.shape), same_nb_points)

            # Test method dist if features are compatible
            if metric.are_compatible(f1.shape, f2.shape):
                distance = metric.dist(f1, f2)
                if np.all(f1 == f2):
                    assert_equal(distance, 0.)

                assert_almost_equal(distance, dipymetric.dist(metric, s1, s2))
                assert_almost_equal(distance, dipymetric.mdf(s1, s2))
                assert_true(distance >= 0.)

        # This metric type is order invariant
        assert_true(metric.is_order_invariant)
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
                assert_true(np.allclose(metric.dist(f1, f2_flip), distance))
                assert_true(np.allclose(metric.dist(f1_flip, f2), distance))


def test_metric_cosine():
    feature = dipymetric.VectorOfEndpointsFeature()

    class CosineMetric(dipymetric.Metric):
        def __init__(self, feature):
            super(CosineMetric, self).__init__(feature=feature)

        def are_compatible(self, shape1, shape2):
            # Cosine metric works on vectors.
            return shape1 == shape2 and shape1[0] == 1

        def dist(self, v1, v2):
            # Check if we have null vectors
            if norm(v1) == 0:
                return 0. if norm(v2) == 0 else 1.

            v1_normed = v1.astype(np.float64) / norm(v1.astype(np.float64))
            v2_normed = v2.astype(np.float64) / norm(v2.astype(np.float64))
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
            assert_equal(metric.are_compatible(f1.shape, f2.shape),
                         are_vectors and same_dimension)

            # Test method dist if features are compatible
            if metric.are_compatible(f1.shape, f2.shape):
                distance = metric.dist(f1, f2)
                if np.all(f1 == f2):
                    assert_almost_equal(distance, 0.)

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
