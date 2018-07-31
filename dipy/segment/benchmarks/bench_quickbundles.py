""" Benchmarks for QuickBundles

Run all benchmarks with::

    import dipy.segment as dipysegment
    dipysegment.bench()

If you have doctests enabled by default in nose (with a noserc file or
environment variable), and you have a numpy version <= 1.6.1, this will
also run the doctests, let's hope they pass.

Run this benchmark with:

    nosetests -s --match '(?:^|[\\b_\\.//-])[Bb]ench' bench_quickbundles.py
"""
import numpy as np
import nibabel as nib

from dipy.data import get_data

import dipy.tracking.streamline as streamline_utils
from dipy.segment.metric import Metric
from dipy.segment.quickbundles import QuickBundles as QB_Old
from dipy.segment.clustering import QuickBundles as QB_New
from nose.tools import assert_equal

from dipy.testing import assert_arrays_equal
from numpy.testing import assert_array_equal, measure


class MDFpy(Metric):
    def are_compatible(self, shape1, shape2):
        return shape1 == shape2

    def dist(self, features1, features2):
        dist = np.sqrt(np.sum((features1 - features2)**2, axis=1))
        dist = np.sum(dist / len(features1))
        return dist


def bench_quickbundles():
    dtype = "float32"
    repeat = 10
    nb_points = 12

    streams, hdr = nib.trackvis.read(get_data('fornix'))
    fornix = [s[0].astype(dtype) for s in streams]
    fornix = streamline_utils.set_number_of_points(fornix, nb_points)

    # Create eight copies of the fornix to be clustered (one in each octant).
    streamlines = []
    streamlines += [s + np.array([100, 100, 100], dtype) for s in fornix]
    streamlines += [s + np.array([100, -100, 100], dtype) for s in fornix]
    streamlines += [s + np.array([100, 100, -100], dtype) for s in fornix]
    streamlines += [s + np.array([100, -100, -100], dtype) for s in fornix]
    streamlines += [s + np.array([-100, 100, 100], dtype) for s in fornix]
    streamlines += [s + np.array([-100, -100, 100], dtype) for s in fornix]
    streamlines += [s + np.array([-100, 100, -100], dtype) for s in fornix]
    streamlines += [s + np.array([-100, -100, -100], dtype) for s in fornix]

    # The expected number of clusters of the fornix using threshold=10 is 4.
    threshold = 10.
    expected_nb_clusters = 4 * 8

    print("Timing QuickBundles 1.0 vs. 2.0")

    qb = QB_Old(streamlines, threshold, pts=None)
    qb1_time = measure("QB_Old(streamlines, threshold, nb_points)", repeat)
    print("QuickBundles time: {0:.4}sec".format(qb1_time))
    assert_equal(qb.total_clusters, expected_nb_clusters)
    sizes1 = [qb.partitions()[i]['N'] for i in range(qb.total_clusters)]
    indices1 = [qb.partitions()[i]['indices']
                for i in range(qb.total_clusters)]

    qb2 = QB_New(threshold)
    qb2_time = measure("clusters = qb2.cluster(streamlines)", repeat)
    print("QuickBundles2 time: {0:.4}sec".format(qb2_time))
    print("Speed up of {0}x".format(qb1_time / qb2_time))
    clusters = qb2.cluster(streamlines)
    sizes2 = map(len, clusters)
    indices2 = map(lambda c: c.indices, clusters)
    assert_equal(len(clusters), expected_nb_clusters)
    assert_array_equal(list(sizes2), sizes1)
    assert_arrays_equal(indices2, indices1)

    qb = QB_New(threshold, metric=MDFpy())
    qb3_time = measure("clusters = qb.cluster(streamlines)", repeat)
    print("QuickBundles2_python time: {0:.4}sec".format(qb3_time))
    print("Speed up of {0}x".format(qb1_time / qb3_time))
    clusters = qb.cluster(streamlines)
    sizes3 = map(len, clusters)
    indices3 = map(lambda c: c.indices, clusters)
    assert_equal(len(clusters), expected_nb_clusters)
    assert_array_equal(list(sizes3), sizes1)
    assert_arrays_equal(indices3, indices1)
