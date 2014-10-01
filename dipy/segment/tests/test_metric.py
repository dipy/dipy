
import numpy as np
import dipy.segment.metric as dipymetric
from dipy.segment.metric import Metric
import dipy.segment.metricspeed as dipymetricspeed
from dipy.tracking.streamline import length

from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_raises, run_module_suite


def L2_norm(x):
    return np.sqrt(np.sum((x)**2, axis=1))


def MDF_distance(x, y):
    return np.sum(L2_norm(x-y)/len(x))


dtype = "float32"
s1 = np.array([np.arange(10, dtype=dtype)]*3).T  # 10x3
s2 = np.arange(3*10, dtype=dtype).reshape((-1, 3))  # 10x3
s3 = np.array([np.arange(5, dtype=dtype)]*4)  # 5x4


def test_feature_type_midpoint():
    FEATURE_TYPE_SHAPE = (1, 3)
    feature_type = dipymetric.Midpoint()

    assert_equal(feature_type.infer_shape(s1), FEATURE_TYPE_SHAPE)

    features = feature_type.extract(s1)
    assert_equal(features.shape, FEATURE_TYPE_SHAPE)
    assert_array_equal(features, s1[[len(s1)//2], :])

    # This feature type is not order invariant
    for s in [s1, s2]:
        features = feature_type.extract(s)
        features_flip = feature_type.extract(s[::-1])
        assert_array_equal(features_flip, s[::-1][[len(s)//2], :])
        assert_true(np.any(np.not_equal(features, features_flip)))
        assert_false(feature_type.is_order_invariant)


def test_feature_type_center_of_mass():
    FEATURE_TYPE_SHAPE = (1, 3)
    feature_type = dipymetric.CenterOfMass()

    assert_equal(feature_type.infer_shape(s1), FEATURE_TYPE_SHAPE)

    for s in [s1, s2]:
        features = feature_type.extract(s)
        assert_equal(features.shape, FEATURE_TYPE_SHAPE)
        assert_array_equal(features, np.mean(s, axis=0, keepdims=True))


def test_metric_euclidean():
    # Should specify a FeatureType on which Euclidean distance will be computed
    assert_raises(TypeError, dipymetric.Euclidean)

    for feature_type in [dipymetric.CenterOfMass(), dipymetric.Midpoint()]:
        metric = dipymetric.Euclidean(feature_type)

        assert_equal(metric.infer_features_shape(s1), feature_type.infer_shape(s1))

        features1 = metric.extract_features(s1)
        dist = metric.dist(features1, features1)
        assert_equal(dist, 0.0)
        assert_equal(dipymetric.euclidean(s1, s1, feature_type), 0.0)

        # Features 1 and 2 do not have the same number of points
        features2 = metric.extract_features(s2)
        assert_equal(metric.dist(features1, features2), L2_norm(features1-features2))
        assert_equal(dipymetric.dist(metric, s1, s2), L2_norm(features1-features2))
        assert_equal(dipymetric.euclidean(s1, s2, feature_type), L2_norm(features1-features2))

        # Features 1 and 3 do not have the same number of dimensions
        assert_true(metric.infer_features_shape(s3) > metric.infer_features_shape(s1))
        features3 = metric.extract_features(s3)
        assert_raises(ValueError, metric.dist, features1, features3)
        assert_raises(ValueError, dipymetric.dist, metric, s1, s3)
        assert_raises(ValueError, dipymetric.euclidean, s1, s3, feature_type)


def test_metric_mdf():
    metric = dipymetric.MDF()

    # Test infer_features_shape() and extract_features()
    metric = dipymetric.MDF()

    assert_equal(metric.infer_features_shape(s1), s1.shape)

    features = metric.extract_features(s1)
    assert_equal(features.shape, s1.shape)
    assert_array_equal(features, s1)

    # Test dist()
    features1 = metric.extract_features(s1)
    dist = metric.dist(features1, features1)
    assert_equal(dist, 0.0)
    assert_equal(dipymetric.mdf(s1, s1), 0.0)

    # Features 1 and 2 do not have the same number of points
    features2 = metric.extract_features(s2)
    dist = metric.dist(features1, features2)
    ground_truth = MDF_distance(s1, s2)
    assert_almost_equal(dist, ground_truth)
    assert_equal(dipymetric.mdf(s1, s2), ground_truth)

    # Features 1 and 3 do not have the same number of dimensions
    assert_true(metric.infer_features_shape(s3) != metric.infer_features_shape(s1))
    features3 = metric.extract_features(s3)
    assert_raises(ValueError, metric.dist, features1, features3)
    assert_raises(ValueError, dipymetric.mdf, s1, s3)


def test_subclassing_metric():
    class EmptyMetric(dipymetric.Metric):
        pass

    metric = EmptyMetric()
    assert_raises(NotImplementedError, metric.infer_features_shape, None)
    assert_raises(NotImplementedError, metric.extract_features, None)
    assert_raises(NotImplementedError, metric.compatible, None, None)
    assert_raises(NotImplementedError, metric.dist, None, None)

    class MDFpy(dipymetric.Metric):
        def infer_features_shape(self, streamline):
            return streamline.shape

        def extract_features(self, streamline):
            return streamline.copy()

        def compatible(self, shape1, shape2):
            return shape1 == shape2

        def dist(self, features1, features2):
            return MDF_distance(features1, features2)

    metric = MDFpy()
    d1 = dipymetric.dist(metric, s1, s2)
    d2 = dipymetric.mdf(s1, s2)
    assert_equal(d1, d2)

    features1 = metric.extract_features(s1)
    features2 = metric.extract_features(s2)
    d3 = metric.dist(features1, features2)
    assert_equal(d1, d3)


def test_metric_arclength():

    metric = dipymetricspeed.ArcLength()
    shape = metric.infer_features_shape(s1)
    feature = metric.extract_features(s1)

    assert_equal(np.float32(length(s1)), feature[0, 0])

    assert_equal(0, metric.dist(feature, feature))

    f1 = metric.extract_features(s1)
    f2 = metric.extract_features(2 * s1)

    dist = metric.dist(f1, f2)

    assert_equal(f1, f2 - f1)

    compatibility = metric.compatible(f1.shape, f2.shape)

    assert_true(compatibility)

    class ArcLengthPython(Metric):
        def infer_features_shape(self, streamline):
            return (1, 1)

        def extract_features(self, streamline):
            length_ = length(streamline).astype('f4')
            return np.array([[length_]])

        def dist(self, features1, features2):
            return np.abs(features1 - features2)[0, 0]

    metric_python = ArcLengthPython()
    dist_python = metric.dist(metric_python.extract_features(s1),
                              metric_python.extract_features(2* s1))

    assert_equal(dist, dist_python)


if __name__ == '__main__':

    run_module_suite()
