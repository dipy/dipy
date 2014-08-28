
import numpy as np
import unittest
import dipy.segment.metric as dipymetric

import nose
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_raises


class TestCluster(unittest.TestCase):
    def test_add_python_space(self):
        dtype = "float32"
        features = np.ones(10, dtype=dtype)

        cluster = dipymetric.Cluster(nb_features=len(features))
        assert_array_equal(cluster.centroid, np.zeros(len(features), dtype=dtype))
        assert_equal(len(cluster.indices), 0)

        idx = 42
        cluster.add(idx, features)
        assert_array_equal(cluster.centroid, np.ones(len(features), dtype=dtype))

        assert_equal(len(cluster.indices), 1)
        assert_equal(cluster.indices[0], idx)

        # Check centroid after adding several features vectors.
        M = 11
        cluster = dipymetric.Cluster(nb_features=len(features))
        for i in range(M):
            cluster.add(i, np.arange(10, dtype=dtype) * i)

        expected_centroid = np.arange(10, dtype=dtype) * ((M*(M-1))/2.) / M

        assert_array_equal(cluster.centroid, expected_centroid)


class TestFeatureType(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.streamline = np.array([np.arange(10, dtype=self.dtype)]*3).T

    def test_Midpoint(self):
        FEATURE_TYPE_SHAPE = 3
        feature_type = dipymetric.Midpoint()

        assert_equal(feature_type.shape(self.streamline), FEATURE_TYPE_SHAPE)

        features = feature_type.extract(self.streamline)
        assert_equal(len(features), 3)
        assert_array_equal(features, self.streamline[len(self.streamline)//2, :])

    def test_CenterOfMass(self):
        FEATURE_TYPE_SHAPE = 3
        feature_type = dipymetric.CenterOfMass()

        assert_equal(feature_type.shape(self.streamline), FEATURE_TYPE_SHAPE)

        features = feature_type.extract(self.streamline)
        assert_equal(len(features), 3)
        assert_array_equal(features, np.mean(self.streamline, axis=0))


class TestFeatures2(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.streamline = np.array([np.arange(10, dtype=self.dtype)]*3).T

    def test_MDF(self):
        NB_FEATURES = np.prod(self.streamline.shape)
        metric = dipymetric.MDF()

        assert_equal(metric.nb_features(self.streamline), NB_FEATURES)

        features = metric.get_features(self.streamline)
        assert_equal(len(features), np.prod(self.streamline.shape))
        assert_array_equal(features, self.streamline.flatten())


class TestMetric(unittest.TestCase):
    def setUp(self):
        self.streamline1 = np.array([np.arange(10, dtype="float32")]*3).T
        self.streamline2 = np.arange(3*10, dtype="float32").reshape((-1, 3))
        self.streamline3 = np.array([np.arange(5, dtype="float32")]*3)

    def test_subclassing(self):
        class EmptyMetric(dipymetric.Metric):
            pass

        metric = EmptyMetric()
        assert_raises(NotImplementedError, metric.nb_features, None)
        assert_raises(NotImplementedError, metric.get_features, None)
        assert_raises(NotImplementedError, metric.dist, None, None)

    def test_Euclidean(self):
        # Should specify a FeatureType on which Euclidean distance will be computed
        assert_raises(TypeError, dipymetric.Euclidean)

        for feature_type in [dipymetric.CenterOfMass(), dipymetric.Midpoint()]:
            metric = dipymetric.Euclidean(feature_type)

            assert_equal(metric.nb_features(self.streamline1), feature_type.shape(self.streamline1))

            features1 = metric.get_features(self.streamline1)
            dist = metric.dist(features1, features1)
            assert_equal(dist, 0.0)

            L2_norm = lambda x, y: np.sqrt(np.sum((np.asarray(x) - np.asarray(y))**2))

            # Features 1 and 2 do not have the same number of points
            features2 = metric.get_features(self.streamline2)
            dist = metric.dist(features1, features2)
            assert_equal(dist, L2_norm(features1, features2))

            # Features 1 and 3 do not have the same number of dimensions
            assert_true(metric.nb_features(self.streamline3) > metric.nb_features(self.streamline1))

    def test_MDF(self):
        metric = dipymetric.MDF()

        features1 = metric.get_features(self.streamline1)
        dist = metric.dist(features1, features1)
        assert_equal(dist, 0.0)

        L2_norm = lambda x, y: np.sqrt(np.sum((x-y)**2, axis=1))
        MDF_distance = lambda x, y: np.sum(L2_norm(x, y)/len(x))

        features2 = metric.get_features(self.streamline2)
        dist = metric.dist(features1, features2)
        assert_almost_equal(dist, MDF_distance(self.streamline1, self.streamline2))

        # TODO: Trying to compare streamlines of different lengths, should it raise an exception?
        #features3 = metric.get_features(self.streamline3)
        #dist = metric.dist(features1, features3)
        #assert_almost_equal(dist, MDF_distance(self.streamline1, self.streamline3))
