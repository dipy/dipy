
import numpy as np
import dipy.segment.metric as dipymetric
from dipy.tracking.streamline import length
from dipy.segment.featurespeed import extract

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


def test_feature_identity():
    feature = dipymetric.IdentityFeature()
    assert_equal(feature.infer_shape(s1), s1.shape)

    features = feature.extract(s1)
    assert_equal(features.shape, s1.shape)
    assert_array_equal(features, s1)

    # This feature type is not order invariant
    assert_false(feature.is_order_invariant)
    for s in [s1, s2]:
        features = feature.extract(s)
        features_flip = feature.extract(s[::-1])
        assert_array_equal(features_flip, s[::-1])
        assert_true(np.any(np.not_equal(features, features_flip)))


def test_feature_midpoint():
    feature_SHAPE = (1, 3)  # One N-dimensional point
    feature = dipymetric.MidpointFeature()

    assert_equal(feature.infer_shape(s1), feature_SHAPE)

    features = feature.extract(s1)
    assert_equal(features.shape, feature_SHAPE)
    assert_array_equal(features, s1[[len(s1)//2], :])

    # This feature type is not order invariant
    assert_false(feature.is_order_invariant)
    for s in [s1, s2]:
        features = feature.extract(s)
        features_flip = feature.extract(s[::-1])
        assert_array_equal(features_flip, s[::-1][[len(s)//2], :])
        assert_true(np.any(np.not_equal(features, features_flip)))


def test_feature_center_of_mass():
    feature_SHAPE = (1, 3)
    feature = dipymetric.CenterOfMassFeature()

    assert_equal(feature.infer_shape(s1), feature_SHAPE)

    assert_true(feature.is_order_invariant)
    for s in [s1, s2]:
        features = feature.extract(s)
        assert_equal(features.shape, feature_SHAPE)
        assert_array_equal(features, np.mean(s, axis=0, keepdims=True))


def test_metric_pointwise_euclidean():
    metric_classes = [dipymetric.SumPointwiseEuclideanMetric,
                      dipymetric.AveragePointwiseEuclideanMetric,
                      dipymetric.MinimumPointwiseEuclideanMetric,
                      dipymetric.MaximumPointwiseEuclideanMetric]

    reducers = [np.sum, np.mean, np.min, np.max]

    for metric_class, reducer in zip(metric_classes, reducers):
        for feature_class in [dipymetric.IdentityFeature, dipymetric.CenterOfMassFeature, dipymetric.MidpointFeature]:
            print "Testing {0} with {1}".format(metric_class.__name__, feature_class.__name__)
            metric = metric_class(feature_class())

            features1 = metric.feature.extract(s1)
            dist = metric.dist(features1, features1)
            assert_equal(dist, 0.0)

            features2 = metric.feature.extract(s2)
            assert_almost_equal(metric.dist(features1, features2), reducer(norm_L2(features1-features2)))
            assert_almost_equal(dipymetric.dist(metric, s1, s2), reducer(norm_L2(features1-features2)))

            # Features 1 and 3 do not have the same number of dimensions
            features3 = metric.feature.extract(s3)
            if not metric.compatible(features1.shape, features3.shape):
                assert_raises(ValueError, metric.dist, features1, features3)
                assert_raises(ValueError, dipymetric.dist, metric, s1, s3)

            # Features 1 and 4 do not have the same number of points
            features4 = metric.feature.extract(s4)
            if not metric.compatible(features1.shape, features4.shape):
                assert_raises(ValueError, metric.dist, features1, features4)
                assert_raises(ValueError, dipymetric.dist, metric, s1, s4)


def test_metric_minimum_average_direct_flip():
    metric = dipymetric.MinimumAverageDirectFlipMetric()

    assert_equal(metric.feature.infer_shape(s1), s1.shape)

    features = metric.feature.extract(s1)
    assert_equal(features.shape, s1.shape)
    assert_array_equal(features, s1)

    # Test dist()
    features1 = metric.feature.extract(s1)
    assert_true(metric.compatible(features1.shape, features1.shape))
    dist = metric.dist(features1, features1)
    assert_equal(dist, 0.0)
    assert_equal(dipymetric.mdf(s1, s1), 0.0)

    # Features 1 and 2 do have the same number of points and dimensions
    features2 = metric.feature.extract(s2)
    assert_true(metric.compatible(features1.shape, features2.shape))
    dist = metric.dist(features1, features2)
    ground_truth = MDF_distance(s1, s2)
    assert_almost_equal(dist, ground_truth)
    assert_almost_equal(dipymetric.mdf(s1, s2), ground_truth)

    # Features 1 and 3 do not have the same number of dimensions
    features3 = metric.feature.extract(s3)
    assert_false(metric.compatible(features1.shape, features3.shape))
    assert_raises(ValueError, metric.dist, features1, features3)
    assert_raises(ValueError, dipymetric.mdf, s1, s3)

    # Features 1 and 4 do not have the same number of points
    features4 = metric.feature.extract(s4)
    assert_false(metric.compatible(features1.shape, features4.shape))
    assert_raises(ValueError, metric.dist, features1, features4)
    assert_raises(ValueError, dipymetric.dist, metric, s1, s4)


def test_metric_hausdorff():
    metric = dipymetric.HausdorffMetric()

    assert_equal(metric.feature.infer_shape(s1), s1.shape)

    features = metric.feature.extract(s1)
    assert_equal(features.shape, s1.shape)
    assert_array_equal(features, s1)

    # Test dist()
    features1 = metric.feature.extract(s1)
    assert_true(metric.compatible(features1.shape, features1.shape))
    dist = metric.dist(features1, features1)
    assert_equal(dist, 0.0)

    # Features 1 and 2 do have the same number of points and dimensions
    featuresA = np.array([[-1, -1], [0, 0], [1, 1]]).astype(dtype)
    featuresB = np.array([[-1, 1], [0, 0], [1, -1]]).astype(dtype)
    assert_true(metric.compatible(featuresA.shape, featuresB.shape))
    dist = metric.dist(featuresA, featuresB)
    assert_equal(dist, np.sqrt(2))

    # Features 1 and 2 do have the same number of points and dimensions
    featuresA = np.array([[-1, -1], [0, 0]]).astype(dtype)
    featuresB = np.array([[-1, 1], [0, 0], [1, -1]]).astype(dtype)
    assert_true(metric.compatible(featuresA.shape, featuresB.shape))
    dist = metric.dist(featuresA, featuresB)
    assert_equal(dist, np.sqrt(2))


def test_subclassing_feature():
    class EmptyFeature(dipymetric.Feature):
        pass

    feature = EmptyFeature()
    assert_raises(NotImplementedError, feature.infer_shape, None)
    assert_raises(NotImplementedError, feature.extract, None)

    class CenterOfMass(dipymetric.Feature):
        def infer_shape(self, streamline):
            return (1, streamline.shape[1])

        def extract(self, streamline):
            return np.mean(streamline, axis=0, keepdims=True)

    feature = CenterOfMass()
    assert_equal(feature.infer_shape(s1), np.mean(s1, axis=0, keepdims=True).shape)
    assert_array_equal(feature.extract(s1), np.mean(s1, axis=0, keepdims=True))

    # # Test that features are automatically cast into float32 when coming from Python space
    # class CenterOfMass64bit(dipymetric.Feature):
    #     def infer_shape(self, streamline):
    #         return (1, streamline.shape[1])

    #     def extract(self, streamline):
    #         return np.mean(streamline.astype(np.float64), axis=0, keepdims=True)

    # assert_equal(CenterOfMass64bit().extract(s1).dtype, np.float32)


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

    features1 = metric.feature.extract(s1)
    features2 = metric.feature.extract(s2)
    d3 = metric.dist(features1, features2)
    assert_equal(d1, d3)


def test_subclassing_metric_and_feature():
    class Identity(dipymetric.Feature):
        def infer_shape(self, streamline):
            return streamline.shape

        def extract(self, streamline):
            return streamline

    class MDF(dipymetric.Metric):
        def compatible(self, shape1, shape2):
            return shape1 == shape2

        def dist(self, features1, features2):
            return MDF_distance(features1, features2)

    # Test using Python Feature with Cython Metric
    feature = Identity()
    metric = dipymetric.AveragePointwiseEuclideanMetric(feature)
    d1 = dipymetric.dist(metric, s1, s2)

    features1 = metric.feature.extract(s1)
    features2 = metric.feature.extract(s2)
    d2 = metric.dist(features1, features2)
    assert_equal(d1, d2)

    # Test using Cython Feature with Python Metric
    feature = Identity()
    metric = MDF(dipymetric.IdentityFeature())
    d1 = dipymetric.dist(metric, s1, s2)

    features1 = metric.feature.extract(s1)
    features2 = metric.feature.extract(s2)
    d2 = metric.dist(features1, features2)
    assert_equal(d1, d2)

    # Test using Python Feature with Python Metric
    feature = Identity()
    metric = MDF(feature)
    d1 = dipymetric.dist(metric, s1, s2)

    features1 = metric.feature.extract(s1)
    features2 = metric.feature.extract(s2)
    d2 = metric.dist(features1, features2)
    assert_equal(d1, d2)


def test_metric_arclength():
    metric = dipymetric.ArcLengthMetric()
    shape = metric.feature.infer_shape(s1)
    features = metric.feature.extract(s1)

    assert_equal(shape, (1, 1))
    assert_equal(np.float32(length(s1)), features[0, 0])
    assert_equal(metric.dist(features, features), 0)

    f1 = metric.feature.extract(s1)
    f2 = metric.feature.extract(2 * s1)

    dist = metric.dist(f1, f2)
    assert_equal(f1, f2 - f1)

    compatibility = metric.compatible(f1.shape, f2.shape)
    assert_true(compatibility)

    class ArcLengthFeature(dipymetric.Feature):
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
    dist_python = metric.dist(metric_python.feature.extract(s1),
                              metric_python.feature.extract(2 * s1))

    assert_equal(dist, dist_python)


def test_weighted_sum_metric():
    class WeightedSumFeature(dipymetric.Feature):
        def __init__(self):
            self.features = []
            self.shapes = []

        def add_feature(self, feature):
            self.features.append(feature)
            self.is_order_invariant &= feature.is_order_invariant

        def infer_shape(self, streamline):
            nb_scalars = 0
            for feature in self.features:
                feature_shape = feature.infer_shape(streamline)
                nb_scalars += feature_shape[0] * feature_shape[1]

            return (1, nb_scalars)

        def extract(self, streamline):
            features_list = []
            self.shapes = []
            for feature in self.features:
                features = feature.extract(streamline)
                features_list.append(features.flatten())
                self.shapes.append(features.shape)

            return np.concatenate(features_list)

    class WeightedSumMetric(dipymetric.Metric):
        def __init__(self):
            super(WeightedSumMetric, self).__init__(WeightedSumFeature())
            self.metrics = []
            self.weights = []
            self.shapes = []

        def add_metric(self, metric, weight):
            self.metrics.append(metric)
            self.weights.append(weight)
            self.feature.add_feature(metric.feature)

        def dist(self, features1, features2):
            d = 0.0
            features_offset = 0
            for shape, metric, weight in zip(self.feature.shapes, self.metrics, self.weights):
                nb_scalars = shape[0] * shape[1]
                features1_reshaped = features1[features_offset:features_offset+nb_scalars].reshape(shape)
                features2_reshaped = features2[features_offset:features_offset+nb_scalars].reshape(shape)
                features_offset += nb_scalars

                d += weight * metric.dist(features1_reshaped, features2_reshaped)

            return d

    streamline1, streamline2 = s1, s2

    weight1 = 0.5
    weight2 = 0.25
    metric1 = dipymetric.AveragePointwiseEuclideanMetric()
    metric2 = dipymetric.ArcLengthMetric()

    metric = WeightedSumMetric()
    metric.add_metric(metric1, weight1)
    metric.add_metric(metric2, weight2)
    #metric.add_metric(dipymetric.SpatialMetric())
    #metric.add_metric(dipymetric.OrientationMetric())
    #metric.add_metric(dipymetric.ShapeMetric())

    metric1_features1 = metric1.feature.extract(streamline1)
    metric1_features2 = metric1.feature.extract(streamline2)
    metric2_features1 = metric2.feature.extract(streamline1)
    metric2_features2 = metric2.feature.extract(streamline2)
    features1 = metric.feature.extract(streamline1)
    features2 = metric.feature.extract(streamline2)

    metric1_dist = metric1.dist(metric1_features1, metric1_features2)
    metric2_dist = metric2.dist(metric2_features1, metric2_features2)
    metric_dist = metric.dist(features1, features2)
    assert_equal(metric_dist, weight1*metric1_dist + weight2*metric2_dist)

    # d1 = metric.dist_between_features(features1, features2)
    # d2 = metric.dist(streamline1, streamline2)
    # assert_equal(d1, d2)


def test_feature_extract():
    # Test that features are automatically cast into float32 when coming from Python space
    class CenterOfMass64bit(dipymetric.Feature):
        def infer_shape(self, streamline):
            return (1, streamline.shape[1])

        def extract(self, streamline):
            return np.mean(streamline.astype(np.float64), axis=0, keepdims=True)

    nb_streamlines = 100
    feature_SHAPE = (1, 3)  # One N-dimensional point
    #feature = dipymetric.MidpointFeature()
    feature = CenterOfMass64bit()

    streamlines = [np.arange(np.random.randint(20, 30) * 3).reshape((-1, 3)).astype(np.float32) for i in range(nb_streamlines)]
    features = extract(feature, streamlines)


if __name__ == '__main__':
    run_module_suite()
