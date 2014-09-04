
import numpy as np
import unittest
import dipy.segment.metric as dipymetric

import nose
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_raises


class TestFeatureType(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.streamline = np.array([np.arange(10, dtype=self.dtype)]*3).T

    def test_Midpoint(self):
        FEATURE_TYPE_SHAPE = 3
        feature_type = dipymetric.Midpoint()

        assert_equal(feature_type.infer_shape(self.streamline), FEATURE_TYPE_SHAPE)

        features = feature_type.extract(self.streamline)
        assert_equal(len(features), 3)
        assert_array_equal(features, self.streamline[len(self.streamline)//2, :])

        # This feature type is not order invariant
        features_flip = feature_type.extract(self.streamline[::-1])
        assert_array_equal(features_flip, self.streamline[::-1][len(self.streamline)//2, :])
        assert_true(np.any(np.not_equal(features, features_flip)))
        assert_false(feature_type.is_order_invariant)

    def test_CenterOfMass(self):
        FEATURE_TYPE_SHAPE = 3
        feature_type = dipymetric.CenterOfMass()

        assert_equal(feature_type.infer_shape(self.streamline), FEATURE_TYPE_SHAPE)

        features = feature_type.extract(self.streamline)
        assert_equal(len(features), 3)
        assert_array_equal(features, np.mean(self.streamline, axis=0))


class TestMetric(unittest.TestCase):
    def setUp(self):
        self.s1 = np.array([np.arange(10, dtype="float32")]*3).T
        self.s2 = np.arange(3*10, dtype="float32").reshape((-1, 3))
        self.s3 = np.array([np.arange(5, dtype="float32")]*3)
        self.s4 = np.array([np.arange(10, dtype="float32")]*3)

    def test_subclassing(self):
        class EmptyMetric(dipymetric.Metric):
            pass

        metric = EmptyMetric()
        assert_raises(NotImplementedError, metric.infer_features_shape, None)
        assert_raises(NotImplementedError, metric.extract_features, None)
        assert_raises(NotImplementedError, metric.dist, None, None)

        class MDFpy(dipymetric.Metric):
            def infer_features_shape(self, streamline):
                return streamline.shape[0] * streamline.shape[1]

            def extract_features(self, streamline):
                N, D = streamline.shape

                features = np.empty(N*D, dtype=streamline.base.dtype)
                for y in range(N):
                    i = y*D
                    features[i+0] = streamline[y, 0]
                    features[i+1] = streamline[y, 1]
                    features[i+2] = streamline[y, 2]

                return features

            def dist(self, features1, features2):
                D = 3
                N = features2.shape[0] // D

                d = 0.0
                for y in range(N):
                    i = y*D
                    dx = features1[i+0] - features2[i+0]
                    dy = features1[i+1] - features2[i+1]
                    dz = features1[i+2] - features2[i+2]
                    d += np.sqrt(dx*dx + dy*dy + dz*dz)

                return d / N

        d1 = dipymetric.dist(dipymetric.MDF(), self.s1, self.s2)
        d2 = dipymetric.mdf(self.s1, self.s2)
        assert_equal(d1, d2)

    def test_Euclidean(self):
        # Should specify a FeatureType on which Euclidean distance will be computed
        assert_raises(TypeError, dipymetric.Euclidean)

        for feature_type in [dipymetric.CenterOfMass(), dipymetric.Midpoint()]:
            metric = dipymetric.Euclidean(feature_type)

            assert_equal(metric.infer_features_shape(self.s1), feature_type.infer_shape(self.s1))

            features1 = metric.extract_features(self.s1)
            dist = metric.dist(features1, features1)
            assert_equal(dist, 0.0)
            assert_equal(dipymetric.euclidean(self.s1, self.s1, feature_type), 0.0)

            L2_norm = lambda x, y: np.sqrt(np.sum((np.asarray(x) - np.asarray(y))**2))

            # Features 1 and 2 do not have the same number of points
            features2 = metric.extract_features(self.s2)
            dist = metric.dist(features1, features2)
            assert_equal(dist, L2_norm(features1, features2))
            assert_equal(dipymetric.euclidean(self.s1, self.s2, feature_type), L2_norm(features1, features2))

            # Features 1 and 3 do not have the same number of dimensions
            assert_true(metric.infer_features_shape(self.s3) > metric.infer_features_shape(self.s1))
            features3 = metric.extract_features(self.s3)
            assert_raises(ValueError, metric.dist, features1, features3)
            assert_raises(ValueError, dipymetric.euclidean, self.s1, self.s3, feature_type)

    def test_MDF(self):
        metric = dipymetric.MDF()

        # Test infer_features_shape() and extract_features()
        shape = np.prod(self.s1.shape)
        metric = dipymetric.MDF()

        assert_equal(metric.infer_features_shape(self.s1), shape)

        features = metric.extract_features(self.s1)
        assert_equal(len(features), shape)
        assert_array_equal(features, self.s1.flatten())

        # Test dist()
        features1 = metric.extract_features(self.s1)
        dist = metric.dist(features1, features1)
        assert_equal(dist, 0.0)
        assert_equal(dipymetric.mdf(self.s1, self.s1), 0.0)

        L2_norm = lambda x, y: np.sqrt(np.sum((x-y)**2, axis=1))
        MDF_distance = lambda x, y: np.sum(L2_norm(x, y)/len(x))

        # Features 1 and 2 do not have the same number of points
        features2 = metric.extract_features(self.s2)
        dist = metric.dist(features1, features2)
        ground_truth = MDF_distance(self.s1, self.s2)
        assert_almost_equal(dist, ground_truth)
        assert_equal(dipymetric.mdf(self.s1, self.s2), ground_truth)

        # Features 1 and 3 do not have the same number of dimensions
        assert_true(metric.infer_features_shape(self.s3) != metric.infer_features_shape(self.s1))
        features3 = metric.extract_features(self.s3)
        assert_raises(ValueError, metric.dist, features1, features3)
        assert_raises(ValueError, dipymetric.mdf, self.s1, self.s3)
