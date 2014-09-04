""" Benchmarks for QuickBundles

Run all benchmarks with::

    import dipy.segment as dipysegment
    dipysegment.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will also run
the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' /path/to/bench_quickbundles.py
"""
import numpy as np
import nibabel as nib

from dipy.data import get_data

import dipy.tracking.streamline as streamline_utils
from dipy.segment.metric import Metric
from dipy.segment.quickbundles import QuickBundles as QuickBundlesOld
from dipy.segment.clustering import QuickBundles as QuickBundlesNew
from nose.tools import assert_equal

from numpy.testing import measure


class MDFpy(Metric):
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


def bench_quickbundles():
    dtype = "float32"
    repeat = 10
    nb_points_per_streamline = 12

    streams, hdr = nib.trackvis.read(get_data('fornix'))
    fornix = [s[0].astype(dtype) for s in streams]
    for s in fornix:
        s.setflags(write=True)
    fornix = streamline_utils.set_number_of_points(fornix, nb_points_per_streamline)

    #Create eight copies of the fornix to be clustered (one in each octant).
    streamlines = []
    streamlines += [streamline + np.array([100, 100, 100], dtype) for streamline in fornix]
    streamlines += [streamline + np.array([100, -100, 100], dtype) for streamline in fornix]
    streamlines += [streamline + np.array([100, 100, -100], dtype) for streamline in fornix]
    streamlines += [streamline + np.array([100, -100, -100], dtype) for streamline in fornix]
    streamlines += [streamline + np.array([-100, 100, 100], dtype) for streamline in fornix]
    streamlines += [streamline + np.array([-100, -100, 100], dtype) for streamline in fornix]
    streamlines += [streamline + np.array([-100, 100, -100], dtype) for streamline in fornix]
    streamlines += [streamline + np.array([-100, -100, -100], dtype) for streamline in fornix]

    # The expected number of clusters of the fornix using threshold=10 is 4.
    threshold = 10.
    expected_nb_clusters = 4*8

    print("Timing QuickBundles 1.0 vs. 2.0")

    qb = QuickBundlesOld(streamlines, threshold, pts=None)
    qb1_time = measure("QuickBundlesOld(streamlines, threshold, nb_points_per_streamline)", repeat)
    print("QuickBundles 1.0 time: {0:.4}sec".format(qb1_time))
    assert_equal(qb.total_clusters, expected_nb_clusters)

    qb = QuickBundlesNew(threshold)
    qb2_time = measure("clusters = qb.cluster(streamlines)", repeat)
    print("QuickBundles 2.0 time: {0:.4}sec".format(qb2_time))
    print("Speed up of {0}x".format(qb1_time/qb2_time))
    clusters = qb.cluster(streamlines)
    assert_equal(len(clusters), expected_nb_clusters)

    qb = QuickBundlesNew(threshold, metric=MDFpy())
    qb4_time = measure("clusters = qb.cluster(streamlines)", repeat)
    print("QuickBundles 2.0 time: {0:.4}sec".format(qb4_time))
    print("Speed up of {0}x".format(qb1_time/qb4_time))
    clusters = qb.cluster(streamlines)
    assert_equal(len(clusters), expected_nb_clusters)
