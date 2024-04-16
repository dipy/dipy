""" Benchmarks for ``dipy.segment`` module."""

import numpy as np

from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles as QB_New
from dipy.segment.mask import bounding_box
from dipy.segment.metricspeed import Metric
from dipy.tracking.streamline import Streamlines, set_number_of_points


class BenchMask:

    def setup(self):
        self.dense_vol = np.zeros((100, 100, 100))
        self.dense_vol[:] = 10
        self.sparse_vol = np.zeros((100, 100, 100))
        self.sparse_vol[0, 0, 0] = 1

    def time_bounding_box_sparse(self):
        bounding_box(self.sparse_vol)

    def time_bounding_box_dense(self):
        bounding_box(self.dense_vol)


class BenchQuickbundles:

    def setup(self):
        dtype = "float32"
        nb_points = 12
        # The expected number of clusters of the fornix using threshold=10
        # is 4.
        self.basic_parameters = {"threshold": 10,
                                 "expected_nb_clusters": 4 * 8,
                                 }

        fname = get_fnames('fornix')

        fornix = load_tractogram(fname, 'same',
                                 bbox_valid_check=False).streamlines

        fornix_streamlines = Streamlines(fornix)
        fornix_streamlines = set_number_of_points(fornix_streamlines,
                                                  nb_points)

        # Create eight copies of the fornix to be clustered (one in
        # each octant).
        self.streamlines = []
        self.streamlines += [s + np.array([100, 100, 100], dtype)
                             for s in fornix_streamlines]
        self.streamlines += [s + np.array([100, -100, 100], dtype)
                             for s in fornix_streamlines]
        self.streamlines += [s + np.array([100, 100, -100], dtype)
                             for s in fornix_streamlines]
        self.streamlines += [s + np.array([100, -100, -100], dtype)
                             for s in fornix_streamlines]
        self.streamlines += [s + np.array([-100, 100, 100], dtype)
                             for s in fornix_streamlines]
        self.streamlines += [s + np.array([-100, -100, 100], dtype)
                             for s in fornix_streamlines]
        self.streamlines += [s + np.array([-100, 100, -100], dtype)
                             for s in fornix_streamlines]
        self.streamlines += [s + np.array([-100, -100, -100], dtype)
                             for s in fornix_streamlines]

        class MDFpy(Metric):
            def are_compatible(self, shape1, shape2):
                return shape1 == shape2

            def dist(self, features1, features2):
                dist = np.sqrt(np.sum((features1 - features2)**2, axis=1))
                dist = np.sum(dist / len(features1))
                return dist

        self.custom_metric = MDFpy()

    def time_quickbundles(self):
        qb2 = QB_New(self.basic_parameters.get('threshold', 10))
        _ = qb2.cluster(self.streamlines)

    def time_quickbundles_metric(self):
        qb = QB_New(self.basic_parameters.get('threshold', 10),
                    metric=self.custom_metric)
        _ = qb.cluster(self.streamlines)
