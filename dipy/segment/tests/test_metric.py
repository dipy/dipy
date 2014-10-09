
import numpy as np
import dipy.segment.metric as dipymetric
from dipy.segment.metric import Metric
import dipy.segment.metricspeed as dipymetricspeed
from dipy.tracking.streamline import length

from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_raises, run_module_suite


def norm_L2(x):
    return np.sqrt(np.sum(x**2, axis=1, dtype="float"))


def average_euclidean(x, y):
    return np.mean(norm_L2(x-y))


def MDF_distance(x, y):
    dist_direct = average_euclidean(x, y)
    dist_flipped = average_euclidean(x, y[::-1])
    return np.min(np.array([dist_direct, dist_flipped]))


dtype = "float32"
s1 = np.array([np.arange(10, dtype=dtype)]*3).T  # 10x3
s2 = np.arange(3*10, dtype=dtype).reshape((-1, 3))[::-1]  # 10x3
s3 = np.array([np.arange(5, dtype=dtype)]*4)  # 5x4
s4 = np.array([np.arange(5, dtype=dtype)]*3)  # 5x3


def test_feature_type_identity():
    feature_type = dipymetric.IdentityFeature()
    assert_equal(feature_type.infer_shape(s1), s1.shape)

    features = feature_type.extract(s1)
    assert_equal(features.shape, s1.shape)
    assert_array_equal(features, s1)

    # This feature type is not order invariant
    assert_false(feature_type.is_order_invariant)
    for s in [s1, s2]:
        features = feature_type.extract(s)
        features_flip = feature_type.extract(s[::-1])
        assert_array_equal(features_flip, s[::-1])
        assert_true(np.any(np.not_equal(features, features_flip)))


def test_feature_type_midpoint():
    FEATURE_TYPE_SHAPE = (1, 3)  # One N-dimensional point
    feature_type = dipymetric.MidpointFeature()

    assert_equal(feature_type.infer_shape(s1), FEATURE_TYPE_SHAPE)

    features = feature_type.extract(s1)
    assert_equal(features.shape, FEATURE_TYPE_SHAPE)
    assert_array_equal(features, s1[[len(s1)//2], :])

    # This feature type is not order invariant
    assert_false(feature_type.is_order_invariant)
    for s in [s1, s2]:
        features = feature_type.extract(s)
        features_flip = feature_type.extract(s[::-1])
        assert_array_equal(features_flip, s[::-1][[len(s)//2], :])
        assert_true(np.any(np.not_equal(features, features_flip)))


def test_feature_type_center_of_mass():
    FEATURE_TYPE_SHAPE = (1, 3)
    feature_type = dipymetric.CenterOfMassFeature()

    assert_equal(feature_type.infer_shape(s1), FEATURE_TYPE_SHAPE)

    assert_true(feature_type.is_order_invariant)
    for s in [s1, s2]:
        features = feature_type.extract(s)
        assert_equal(features.shape, FEATURE_TYPE_SHAPE)
        assert_array_equal(features, np.mean(s, axis=0, keepdims=True))


def test_metric_pointwise_euclidean():
    metric_classes = [dipymetric.SumPointwiseEuclideanMetric,
                      dipymetric.AveragePointwiseEuclideanMetric,
                      dipymetric.MinimumPointwiseEuclideanMetric,
                      dipymetric.MaximumPointwiseEuclideanMetric]

    reducers = [np.sum, np.mean, np.min, np.max]

    for metric_class, reducer in zip(metric_classes, reducers):
        for feature_type_class in [dipymetric.IdentityFeature, dipymetric.CenterOfMassFeature, dipymetric.MidpointFeature]:
            print "Testing {0} with {1}".format(metric_class.__name__, feature_type_class.__name__)
            metric = metric_class(feature_type_class())

            features1 = metric.feature_type.extract(s1)
            dist = metric.dist(features1, features1)
            assert_equal(dist, 0.0)

            features2 = metric.feature_type.extract(s2)
            assert_almost_equal(metric.dist(features1, features2), reducer(norm_L2(features1-features2)))
            assert_almost_equal(dipymetric.dist(metric, s1, s2), reducer(norm_L2(features1-features2)))

            # Features 1 and 3 do not have the same number of dimensions
            features3 = metric.feature_type.extract(s3)
            if not metric.compatible(features1.shape, features3.shape):
                assert_raises(ValueError, metric.dist, features1, features3)
                assert_raises(ValueError, dipymetric.dist, metric, s1, s3)

            # Features 1 and 4 do not have the same number of points
            features4 = metric.feature_type.extract(s4)
            if not metric.compatible(features1.shape, features4.shape):
                assert_raises(ValueError, metric.dist, features1, features4)
                assert_raises(ValueError, dipymetric.dist, metric, s1, s4)


def test_metric_mdf():
    metric = dipymetric.MDF()

    assert_equal(metric.feature_type.infer_shape(s1), s1.shape)

    features = metric.feature_type.extract(s1)
    assert_equal(features.shape, s1.shape)
    assert_array_equal(features, s1)

    # Test dist()
    features1 = metric.feature_type.extract(s1)
    assert_true(metric.compatible(features1.shape, features1.shape))
    dist = metric.dist(features1, features1)
    assert_equal(dist, 0.0)
    assert_equal(dipymetric.mdf(s1, s1), 0.0)

    # Features 1 and 2 do have the same number of points and dimensions
    features2 = metric.feature_type.extract(s2)
    assert_true(metric.compatible(features1.shape, features2.shape))
    dist = metric.dist(features1, features2)
    ground_truth = MDF_distance(s1, s2)
    assert_almost_equal(dist, ground_truth)
    assert_almost_equal(dipymetric.mdf(s1, s2), ground_truth)

    # Features 1 and 3 do not have the same number of dimensions
    features3 = metric.feature_type.extract(s3)
    assert_false(metric.compatible(features1.shape, features3.shape))
    assert_raises(ValueError, metric.dist, features1, features3)
    assert_raises(ValueError, dipymetric.mdf, s1, s3)

    # Features 1 and 4 do not have the same number of points
    features4 = metric.feature_type.extract(s4)
    assert_false(metric.compatible(features1.shape, features4.shape))
    assert_raises(ValueError, metric.dist, features1, features4)
    assert_raises(ValueError, dipymetric.dist, metric, s1, s4)


def test_subclassing_feature_type():
    class EmptyFeatureType(dipymetric.FeatureType):
        pass

    feature_type = EmptyFeatureType()
    assert_raises(NotImplementedError, feature_type.infer_shape, None)
    assert_raises(NotImplementedError, feature_type.extract, None)

    class CenterOfMass(dipymetric.FeatureType):
        def infer_shape(self, streamline):
            return (1, streamline.shape[1])

        def extract(self, streamline):
            return np.mean(streamline, axis=0, keepdims=True)

    feature_type = CenterOfMass()
    assert_equal(feature_type.infer_shape(s1), np.mean(s1, axis=0, keepdims=True).shape)
    assert_array_equal(feature_type.extract(s1), np.mean(s1, axis=0, keepdims=True))


def test_subclassing_metric():
    class EmptyMetric(dipymetric.Metric):
        pass

    metric = EmptyMetric()
    assert_raises(NotImplementedError, metric.compatible, None, None)
    assert_raises(NotImplementedError, metric.dist, None, None)

    class MDF(dipymetric.Metric):
        def compatible(self, shape1, shape2):
            return shape1 == shape2

        def dist(self, features1, features2):
            return MDF_distance(features1, features2)

    metric = MDF()
    d1 = dipymetric.dist(metric, s1, s2)
    d2 = dipymetric.mdf(s1, s2)
    assert_equal(d1, d2)

    features1 = metric.feature_type.extract(s1)
    features2 = metric.feature_type.extract(s2)
    d3 = metric.dist(features1, features2)
    assert_equal(d1, d3)


def test_subclassing_metric_and_feature_type():
    class Identity(dipymetric.FeatureType):
        def infer_shape(self, streamline):
            return streamline.shape

        def extract(self, streamline):
            return streamline

    class MDF(dipymetric.Metric):
        def compatible(self, shape1, shape2):
            return shape1 == shape2

        def dist(self, features1, features2):
            return MDF_distance(features1, features2)

    # Test using Python FeatureType with Cython Metric
    feature_type = Identity()
    metric = dipymetric.AveragePointwiseEuclideanMetric(feature_type)
    d1 = dipymetric.dist(metric, s1, s2)

    features1 = metric.feature_type.extract(s1)
    features2 = metric.feature_type.extract(s2)
    d2 = metric.dist(features1, features2)
    assert_equal(d1, d2)

    # Test using Cython FeatureType with Python Metric
    feature_type = Identity()
    metric = MDF(dipymetric.IdentityFeature())
    d1 = dipymetric.dist(metric, s1, s2)

    features1 = metric.feature_type.extract(s1)
    features2 = metric.feature_type.extract(s2)
    d2 = metric.dist(features1, features2)
    assert_equal(d1, d2)

    # Test using Python FeatureType with Python Metric
    feature_type = Identity()
    metric = MDF(feature_type)
    d1 = dipymetric.dist(metric, s1, s2)

    features1 = metric.feature_type.extract(s1)
    features2 = metric.feature_type.extract(s2)
    d2 = metric.dist(features1, features2)
    assert_equal(d1, d2)


def test_metric_arclength():
    metric = dipymetric.ArcLengthMetric()
    shape = metric.feature_type.infer_shape(s1)
    features = metric.feature_type.extract(s1)

    assert_equal(shape, (1, 1))
    assert_equal(np.float32(length(s1)), features[0, 0])
    assert_equal(metric.dist(features, features), 0)

    f1 = metric.feature_type.extract(s1)
    f2 = metric.feature_type.extract(2 * s1)

    dist = metric.dist(f1, f2)
    assert_equal(f1, f2 - f1)

    compatibility = metric.compatible(f1.shape, f2.shape)
    assert_true(compatibility)

    class ArcLengthFeature(dipymetric.FeatureType):
        def infer_shape(self, streamline):
            return (1, 1)

        def extract(self, streamline):
            length_ = length(streamline).astype('f4')
            return np.array([[length_]])

    class ArcLengthMetric(dipymetric.Metric):
        def __init__(self):
            dipymetric.Metric.__init__(self, ArcLengthFeature())

        def dist(self, features1, features2):
            return np.abs(features1 - features2)[0, 0]

    metric_python = ArcLengthMetric()
    dist_python = metric.dist(metric_python.feature_type.extract(s1),
                              metric_python.feature_type.extract(2 * s1))

    assert_equal(dist, dist_python)

if __name__ == '__main__':

    run_module_suite()
