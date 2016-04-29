import sys
import numpy as np
import dipy.segment.metric as dipymetric
from dipy.segment.featurespeed import extract

from nose.tools import assert_true, assert_false, assert_equal
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_raises, run_module_suite)


dtype = "float32"
s1 = np.array([np.arange(10, dtype=dtype)]*3).T  # 10x3
s2 = np.arange(3*10, dtype=dtype).reshape((-1, 3))[::-1]  # 10x3
s3 = np.random.rand(5, 4).astype(dtype)  # 5x4
s4 = np.random.rand(5, 3).astype(dtype)  # 5x3


def test_identity_feature():
    # Test subclassing Feature
    class IdentityFeature(dipymetric.Feature):
        def __init__(self):
            super(IdentityFeature, self).__init__(is_order_invariant=False)

        def infer_shape(self, streamline):
            return streamline.shape

        def extract(self, streamline):
            return streamline

    for feature in [dipymetric.IdentityFeature(), IdentityFeature()]:
        for s in [s1, s2, s3, s4]:
            # Test method infer_shape
            assert_equal(feature.infer_shape(s), s.shape)

            # Test method extract
            features = feature.extract(s)
            assert_equal(features.shape, s.shape)
            assert_array_equal(features, s)

        # This feature type is not order invariant
        assert_false(feature.is_order_invariant)
        for s in [s1, s2, s3, s4]:
            features = feature.extract(s)
            features_flip = feature.extract(s[::-1])
            assert_array_equal(features_flip, s[::-1])
            assert_true(np.any(np.not_equal(features, features_flip)))


def test_feature_resample():
    from dipy.tracking.streamline import set_number_of_points

    # Test subclassing Feature
    class ResampleFeature(dipymetric.Feature):
        def __init__(self, nb_points):
            super(ResampleFeature, self).__init__(is_order_invariant=False)
            self.nb_points = nb_points
            if nb_points <= 0:
                raise ValueError("ResampleFeature: `nb_points` must be strictly positive: {0}".format(nb_points))

        def infer_shape(self, streamline):
            return (self.nb_points, streamline.shape[1])

        def extract(self, streamline):
            return set_number_of_points(streamline, self.nb_points)

    assert_raises(ValueError, dipymetric.ResampleFeature, nb_points=0)
    assert_raises(ValueError, ResampleFeature, nb_points=0)

    max_points = max(map(len, [s1, s2, s3, s4]))
    for nb_points in [2, 5, 2*max_points]:
        for feature in [dipymetric.ResampleFeature(nb_points), ResampleFeature(nb_points)]:
            for s in [s1, s2, s3, s4]:
                # Test method infer_shape
                assert_equal(feature.infer_shape(s), (nb_points, s.shape[1]))

                # Test method extract
                features = feature.extract(s)
                assert_equal(features.shape, (nb_points, s.shape[1]))
                assert_array_almost_equal(features, set_number_of_points(s, nb_points))

            # This feature type is not order invariant
            assert_false(feature.is_order_invariant)
            for s in [s1, s2, s3, s4]:
                features = feature.extract(s)
                features_flip = feature.extract(s[::-1])
                assert_array_equal(features_flip, set_number_of_points(s[::-1], nb_points))
                assert_true(np.any(np.not_equal(features, features_flip)))


def test_feature_center_of_mass():
    # Test subclassing Feature
    class CenterOfMassFeature(dipymetric.Feature):
        def __init__(self):
            super(CenterOfMassFeature, self).__init__(is_order_invariant=True)

        def infer_shape(self, streamline):
            return (1, streamline.shape[1])

        def extract(self, streamline):
            return np.mean(streamline, axis=0)[None, :]

    for feature in [dipymetric.CenterOfMassFeature(), CenterOfMassFeature()]:
        for s in [s1, s2, s3, s4]:
            # Test method infer_shape
            assert_equal(feature.infer_shape(s), (1, s.shape[1]))

            # Test method extract
            features = feature.extract(s)
            assert_equal(features.shape, (1, s.shape[1]))
            assert_array_almost_equal(features, np.mean(s, axis=0)[None, :])

        # This feature type is order invariant
        assert_true(feature.is_order_invariant)
        for s in [s1, s2, s3, s4]:
            features = feature.extract(s)
            features_flip = feature.extract(s[::-1])
            assert_array_almost_equal(features, features_flip)


def test_feature_midpoint():
    # Test subclassing Feature
    class MidpointFeature(dipymetric.Feature):
        def __init__(self):
            super(MidpointFeature, self).__init__(is_order_invariant=False)

        def infer_shape(self, streamline):
            return (1, streamline.shape[1])

        def extract(self, streamline):
            return streamline[[len(streamline)//2]]

    for feature in [dipymetric.MidpointFeature(), MidpointFeature()]:
        for s in [s1, s2, s3, s4]:
            # Test method infer_shape
            assert_equal(feature.infer_shape(s), (1, s.shape[1]))

            # Test method extract
            features = feature.extract(s)
            assert_equal(features.shape, (1, s.shape[1]))
            assert_array_almost_equal(features, s[len(s)//2][None, :])

        # This feature type is not order invariant
        assert_false(feature.is_order_invariant)
        for s in [s1, s2, s3, s4]:
            features = feature.extract(s)
            features_flip = feature.extract(s[::-1])
            if len(s) % 2 == 0:
                assert_true(np.any(np.not_equal(features, features_flip)))
            else:
                assert_array_equal(features, features_flip)


def test_feature_arclength():
    from dipy.tracking.streamline import length

    # Test subclassing Feature
    class ArcLengthFeature(dipymetric.Feature):
        def __init__(self):
            super(ArcLengthFeature, self).__init__(is_order_invariant=True)

        def infer_shape(self, streamline):
            return (1, 1)

        def extract(self, streamline):
            return length(streamline)[None, None]

    for feature in [dipymetric.ArcLengthFeature(), ArcLengthFeature()]:
        for s in [s1, s2, s3, s4]:
            # Test method infer_shape
            assert_equal(feature.infer_shape(s), (1, 1))

            # Test method extract
            features = feature.extract(s)
            assert_equal(features.shape, (1, 1))
            assert_array_almost_equal(features, length(s)[None, None])

        # This feature type is order invariant
        assert_true(feature.is_order_invariant)
        for s in [s1, s2, s3, s4]:
            features = feature.extract(s)
            features_flip = feature.extract(s[::-1])
            assert_array_almost_equal(features, features_flip)


def test_feature_vector_of_endpoints():
    # Test subclassing Feature
    class VectorOfEndpointsFeature(dipymetric.Feature):
        def __init__(self):
            super(VectorOfEndpointsFeature, self).__init__(False)

        def infer_shape(self, streamline):
            return (1, streamline.shape[1])

        def extract(self, streamline):
            return streamline[[-1]] - streamline[[0]]

    feature_types = [dipymetric.VectorOfEndpointsFeature(),
                     VectorOfEndpointsFeature()]
    for feature in feature_types:
        for s in [s1, s2, s3, s4]:
            # Test method infer_shape
            assert_equal(feature.infer_shape(s), (1, s.shape[1]))

            # Test method extract
            features = feature.extract(s)
            assert_equal(features.shape, (1, s.shape[1]))
            assert_array_almost_equal(features, s[[-1]] - s[[0]])

        # This feature type is not order invariant
        assert_false(feature.is_order_invariant)
        for s in [s1, s2, s3, s4]:
            features = feature.extract(s)
            features_flip = feature.extract(s[::-1])
            # The flip features are simply the negative of the features.
            assert_array_almost_equal(features, -features_flip)


def test_feature_extract():
    # Test that features are automatically cast into float32 when coming from Python space
    class CenterOfMass64bit(dipymetric.Feature):
        def infer_shape(self, streamline):
            return streamline.shape[1]

        def extract(self, streamline):
            return np.mean(streamline.astype(np.float64), axis=0)

    nb_streamlines = 100
    feature_shape = (1, 3)  # One N-dimensional point
    feature = CenterOfMass64bit()

    streamlines = [np.arange(np.random.randint(20, 30) * 3).reshape((-1, 3)).astype(np.float32) for i in range(nb_streamlines)]
    features = extract(feature, streamlines)

    assert_equal(len(features), len(streamlines))
    assert_equal(features[0].shape, feature_shape)

    # Test that scalar features
    class ArcLengthFeature(dipymetric.Feature):
        def infer_shape(self, streamline):
            return 1

        def extract(self, streamline):
            return np.sum(np.sqrt(np.sum((streamline[1:] - streamline[:-1]) ** 2)))

    nb_streamlines = 100
    feature_shape = (1, 1)  # One scalar represented as a 2D array
    feature = ArcLengthFeature()

    streamlines = [np.arange(np.random.randint(20, 30) * 3).reshape((-1, 3)).astype(np.float32) for i in range(nb_streamlines)]
    features = extract(feature, streamlines)

    assert_equal(len(features), len(streamlines))
    assert_equal(features[0].shape, feature_shape)

    # Try if streamlines are readonly
    for s in streamlines:
        s.setflags(write=False)
    features = extract(feature, streamlines)


def test_subclassing_feature():
    class EmptyFeature(dipymetric.Feature):
        pass

    feature = EmptyFeature()
    assert_raises(NotImplementedError, feature.infer_shape, None)
    assert_raises(NotImplementedError, feature.extract, None)


def test_using_python_feature_with_cython_metric():
    class Identity(dipymetric.Feature):
        def infer_shape(self, streamline):
            return streamline.shape

        def extract(self, streamline):
            return streamline

    # Test using Python Feature with Cython Metric
    feature = Identity()
    metric = dipymetric.AveragePointwiseEuclideanMetric(feature)
    d1 = dipymetric.dist(metric, s1, s2)

    features1 = metric.feature.extract(s1)
    features2 = metric.feature.extract(s2)
    d2 = metric.dist(features1, features2)
    assert_equal(d1, d2)

    # Python 2.7 on Windows 64 bits uses long type instead of int for
    # constants integer. We make sure the code is robust to such behaviour
    # by explicitly testing it.
    class ArcLengthFeature(dipymetric.Feature):
        def infer_shape(self, streamline):
            if sys.version_info > (3,):
                return 1  # In Python 3, constant integer are of type long.

            return long(1)

        def extract(self, streamline):
            return np.sum(np.sqrt(np.sum((streamline[1:] - streamline[:-1]) ** 2)))

    # Test using Python Feature with Cython Metric
    feature = ArcLengthFeature()
    metric = dipymetric.EuclideanMetric(feature)
    d1 = dipymetric.dist(metric, s1, s2)

    features1 = metric.feature.extract(s1)
    features2 = metric.feature.extract(s2)
    d2 = metric.dist(features1, features2)
    assert_equal(d1, d2)


if __name__ == '__main__':
    run_module_suite()
