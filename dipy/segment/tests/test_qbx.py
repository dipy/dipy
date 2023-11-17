import itertools

import numpy as np
from numpy.testing import assert_array_equal, assert_equal, assert_raises

from dipy.segment.clustering import QuickBundlesX, QuickBundles, qbx_and_merge
from dipy.segment.featurespeed import ResampleFeature
from dipy.segment.metricspeed import (
    AveragePointwiseEuclideanMetric, MinimumAverageDirectFlipMetric,
)
from dipy.tracking.streamline import set_number_of_points
from dipy.tracking.streamline import Streamlines
from dipy.testing.decorators import set_random_number_generator


def straight_bundle(nb_streamlines=1, nb_pts=30, step_size=1,
                    radius=1, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    bundle = []

    bundle_length = step_size * nb_pts

    Z = -np.linspace(0, bundle_length, nb_pts)
    for k in range(nb_streamlines):
        theta = rng.random() * (2*np.pi)
        r = radius * rng.random()

        Xk = np.ones(nb_pts) * (r * np.cos(theta))
        Yk = np.ones(nb_pts) * (r * np.sin(theta))
        Zk = Z.copy()

        bundle.append(np.c_[Xk, Yk, Zk])

    return bundle


def bearing_bundles(nb_balls=6, bearing_radius=2):
    bundles = []

    for theta in np.linspace(0, 2*np.pi, nb_balls, endpoint=False):
        x = bearing_radius * np.cos(theta)
        y = bearing_radius * np.sin(theta)

        bundle = np.array(straight_bundle(nb_streamlines=100))
        bundle += (x, y, 0)
        bundles.append(bundle)

    return bundles


def streamlines_in_circle(nb_streamlines=1, nb_pts=30, step_size=1,
                          radius=1):
    bundle = []

    bundle_length = step_size * nb_pts

    Z = np.linspace(0, bundle_length, nb_pts)
    for theta in np.linspace(0, 2*np.pi, nb_streamlines, endpoint=False):
        Xk = np.ones(nb_pts) * (radius * np.cos(theta))
        Yk = np.ones(nb_pts) * (radius * np.sin(theta))
        Zk = Z.copy()

        bundle.append(np.c_[Xk, Yk, Zk])

    return bundle


def streamlines_parallel(nb_streamlines=1, nb_pts=30, step_size=1,
                         delta=1):
    bundle = []

    bundle_length = step_size * nb_pts

    Z = np.linspace(0, bundle_length, nb_pts)
    for x in delta*np.arange(0, nb_streamlines):
        Xk = np.ones(nb_pts) * x
        Yk = np.zeros(nb_pts)
        Zk = Z.copy()

        bundle.append(np.c_[Xk, Yk, Zk])

    return bundle


def simulated_bundle(no_streamlines=10, waves=False, no_pts=12):
    t = np.linspace(-10, 10, 200)
    # parallel waves or parallel lines
    bundle = []
    for i in np.linspace(-5, 5, no_streamlines):
        if waves:
            pts = np.vstack((np.cos(t), t, i * np.ones(t.shape))).T
        else:
            pts = np.vstack((np.zeros(t.shape), t, i * np.ones(t.shape))).T
        pts = set_number_of_points(pts, no_pts)
        bundle.append(pts)

    return bundle


def test_3D_points():

    points = np.array([[[1, 0, 0]],
                       [[3, 0, 0]],
                       [[2, 0, 0]],
                       [[5, 0, 0]],
                       [[5.5, 0, 0]]], dtype="f4")

    thresholds = [4, 2, 1]
    metric = AveragePointwiseEuclideanMetric()
    qbx = QuickBundlesX(thresholds, metric=metric)
    tree = qbx.cluster(points)
    clusters_2 = tree.get_clusters(2)
    assert_array_equal(clusters_2.clusters_sizes(), [3, 2])
    clusters_0 = tree.get_clusters(0)
    assert_array_equal(clusters_0.clusters_sizes(), [5])


def test_3D_segments():
    points = np.array([[[1, 0, 0],
                        [1, 1, 0]],
                       [[3, 1, 0],
                        [3, 0, 0]],
                       [[2, 0, 0],
                        [2, 1, 0]],
                       [[5, 1, 0],
                        [5, 0, 0]],
                       [[5.5, 0, 0],
                        [5.5, 1, 0]]], dtype="f4")

    thresholds = [4, 2, 1]

    feature = ResampleFeature(nb_points=20)
    metric = AveragePointwiseEuclideanMetric(feature)
    qbx = QuickBundlesX(thresholds, metric=metric)
    tree = qbx.cluster(points)
    clusters_0 = tree.get_clusters(0)
    clusters_1 = tree.get_clusters(1)
    clusters_2 = tree.get_clusters(2)

    assert_equal(len(clusters_0.centroids), len(clusters_1.centroids))
    assert_equal(len(clusters_2.centroids) > len(clusters_1.centroids), True)

    assert_array_equal(clusters_2[1].indices, np.array([3, 4], dtype=np.int32))


def test_with_simulated_bundles():

    streamlines = simulated_bundle(3, False, 2)
    thresholds = [10, 3, 1]
    qbx_class = QuickBundlesX(thresholds)
    tree = qbx_class.cluster(streamlines)
    for level in range(len(thresholds) + 1):
        clusters = tree.get_clusters(level)

    assert_equal(tree.leaves[0].indices[0], 0)
    assert_equal(tree.leaves[2][0], 2)
    clusters.refdata = streamlines

    assert_array_equal(clusters[0][0],
                       np.array([[0., -10., -5.],
                                 [0., 10., -5.]]))


def test_with_simulated_bundles2():

    # Generate synthetic streamlines
    bundles = bearing_bundles(4, 2)
    bundles.append(straight_bundle(1))
    streamlines = list(itertools.chain(*bundles))

    thresholds = [10, 2, 1]
    qbx_class = QuickBundlesX(thresholds)
    tree = qbx_class.cluster(streamlines)
    # By default `refdata` refers to data being clustered.
    assert_equal(tree.refdata, streamlines)


def test_circle_parallel_fornix():

    circle = streamlines_in_circle(100, step_size=2)

    parallel = streamlines_parallel(100)

    thresholds = [1, 0.1]

    qbx_class = QuickBundlesX(thresholds)
    tree = qbx_class.cluster(circle)

    clusters = tree.get_clusters(0)
    assert_equal(len(clusters), 1)

    clusters = tree.get_clusters(1)
    assert_equal(len(clusters), 3)

    clusters = tree.get_clusters(2)
    assert_equal(len(clusters), 34)

    thresholds = [.5]

    qbx_class = QuickBundlesX(thresholds)
    tree = qbx_class.cluster(parallel)

    clusters = tree.get_clusters(0)
    assert_equal(len(clusters), 1)

    clusters = tree.get_clusters(1)
    assert_equal(len(clusters), 100)


def test_raise_mdf():

    thresholds = [1, 0.1]

    metric = MinimumAverageDirectFlipMetric()

    assert_raises(ValueError, QuickBundlesX, thresholds, metric=metric)
    assert_raises(ValueError, QuickBundles, thresholds[1], metric=metric)


@set_random_number_generator(42)
def test_qbx_and_merge(rng):

    # Generate synthetic streamlines
    bundles = bearing_bundles(4, 2)
    bundles.append(straight_bundle(1, rng=rng))


    streamlines = Streamlines(list(itertools.chain(*bundles)))

    thresholds = [10, 2, 1]

    qbxm = qbx_and_merge(streamlines, thresholds, rng=rng)

    qbxm_centroids = qbxm.centroids

    qbxm_clusters = qbxm.clusters

    qbx = QuickBundlesX(thresholds)
    tree = qbx.cluster(streamlines)
    qbx_centroids = tree.get_clusters(3).centroids

    assert_equal(len(qbx_centroids) > len(qbxm_centroids), True)

    # check that refdata clusters return streamlines in qbx_and_merge
    streamline_idx =qbxm_clusters[0].indices[0]
    assert_array_equal(qbxm_clusters[0][0], streamlines[streamline_idx])

