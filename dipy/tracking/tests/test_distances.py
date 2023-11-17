import warnings

import numpy as np
from numpy.testing import (assert_array_almost_equal,
                           assert_equal, assert_almost_equal,
                           assert_array_equal)

from dipy.testing import assert_true
from dipy.testing.decorators import set_random_number_generator
from dipy.tracking import distances as pf
from dipy.tracking.streamline import set_number_of_points
from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram


@set_random_number_generator()
def test_LSCv2(verbose=False, rng=None):
    xyz1 = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype='float32')
    xyz2 = np.array([[1, 0, 0], [1, 2, 0], [1, 3, 0]], dtype='float32')
    xyz3 = np.array([[1.1, 0, 0], [1, 2, 0], [1, 3, 0]], dtype='float32')
    xyz4 = np.array([[1, 0, 0], [2.1, 0, 0], [3, 0, 0]], dtype='float32')

    xyz5 = np.array([[100, 0, 0], [200, 0, 0], [300, 0, 0]], dtype='float32')
    xyz6 = np.array([[0, 20, 0], [0, 40, 0], [300, 50, 0]], dtype='float32')

    T = [xyz1, xyz2, xyz3, xyz4, xyz5, xyz6]
    pf.local_skeleton_clustering(T, 0.2)

    pf.local_skeleton_clustering_3pts(T, 0.2)

    for i in range(40):
        xyz = rng.random((3, 3), dtype='f4')
        T.append(xyz)

    from time import time
    t1 = time()
    C3 = pf.local_skeleton_clustering(T, .5)
    t2 = time()
    if verbose:
        print(t2-t1)
        print(len(C3))

    t1 = time()
    C4 = pf.local_skeleton_clustering_3pts(T, .5)
    t2 = time()
    if verbose:
        print(t2-t1)
        print(len(C4))

    for c in C3:
        assert_equal(np.sum(C3[c]['hidden']-C4[c]['hidden']), 0)

    T2 = []
    for i in range(10**4):
        xyz = rng.random((10, 3), dtype='f4')
        T2.append(xyz)
    t1 = time()
    C5 = pf.local_skeleton_clustering(T2, .5)
    t2 = time()
    if verbose:
        print(t2-t1)
        print(len(C5))

    fname = get_fnames('fornix')
    fornix = load_tractogram(fname, 'same',
                             bbox_valid_check=False).streamlines

    T3 = set_number_of_points(fornix, 6)

    if verbose:
        print('lenT3', len(T3))

    C = pf.local_skeleton_clustering(T3, 10.)

    if verbose:
        print('lenC', len(C))

    """
    try:
        from dipy.viz import window, actor
    except ImportError as e:
        raise pytest.skip('Fails to import dipy.viz due to %s' % str(e))

    scene = window.Scene()
    colors = np.zeros((len(C), 3))
    for c in C:
        color = np.random.rand(3)
        for i in C[c]['indices']:
            scene.add(actor.line(T3[i], color))
        colors[c] = color
    window.show(scene)
    scene.clear()
    skeleton = []

    def width(w):
        if w<1:
            return 1
        else:
            return w

    for c in C:

        bundle = [T3[i] for i in C[c]['indices']]
        si,s = pf.most_similar_track_mam(bundle, 'avg')
        skeleton.append(bundle[si])
        actor.label(r,text = str(len(bundle)), pos=(bundle[si][-1]),
                    scale=(2, 2, 2))
        scene.add(actor.line(skeleton, colors, opacity=1,
                         linewidth = width(len(bundle)/10.)))

    window.show(scene)

    """


def test_bundles_distances_mam():
    xyz1A = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
                     dtype='float32')
    xyz2A = np.array([[0, 1, 1], [1, 0, 1], [2, 3, -2]], dtype='float32')
    xyz1B = np.array([[-1, 0, 0], [2, 0, 0], [2, 3, 0], [3, 0, 0]],
                     dtype='float32')
    tracksA = [xyz1A, xyz2A]
    tracksB = [xyz1B, xyz1A, xyz2A]

    for metric in ('avg', 'min', 'max'):
        pf.bundles_distances_mam(tracksA, tracksB, metric=metric)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        tracksC = [xyz2A, xyz1A]
        _ = pf.bundles_distances_mam(tracksA, tracksC)
        print(w)
        assert_true(len(w) == 1)
        assert_true(issubclass(w[0].category, UserWarning))
        assert_true("not have the same number of points" in str(w[0].message))


def test_bundles_distances_mdf(verbose=False):
    xyz1A = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype='float32')
    xyz2A = np.array([[0, 1, 1], [1, 0, 1], [2, 3, -2]], dtype='float32')
    xyz3A = np.array([[0, 0, 0], [1, 0, 0], [3, 0, 0]], dtype='float32')
    xyz1B = np.array([[-1, 0, 0], [2, 0, 0], [2, 3, 0]], dtype='float32')
    xyz1C = np.array([[-1, 0, 0], [2, 0, 0], [2, 3, 0], [3, 0, 0]],
                     dtype='float32')

    tracksA = [xyz1A, xyz2A]
    tracksB = [xyz1B, xyz1A, xyz2A]

    dist = pf.bundles_distances_mdf(tracksA, tracksA)
    assert_equal(dist[0, 0], 0)
    assert_equal(dist[1, 1], 0)
    assert_equal(dist[1, 0], dist[0, 1])

    pf.bundles_distances_mdf(tracksA, tracksB)

    tracksA = [xyz1A, xyz1A]
    tracksB = [xyz1A, xyz1A]

    DM2 = pf.bundles_distances_mdf(tracksA, tracksB)
    assert_array_almost_equal(DM2, np.zeros((2, 2)))

    tracksA = [xyz1A, xyz3A]
    tracksB = [xyz2A]

    DM2 = pf.bundles_distances_mdf(tracksA, tracksB)
    if verbose:
        print(DM2)

    # assert_array_almost_equal(DM2,np.zeros((2,2)))
    DM = np.zeros(DM2.shape)
    for (a, ta) in enumerate(tracksA):
        for (b, tb) in enumerate(tracksB):
            md = np.sum(np.sqrt(np.sum((ta-tb)**2, axis=1)))/3.
            md2 = np.sum(np.sqrt(np.sum((ta-tb[::-1])**2, axis=1)))/3.
            DM[a, b] = np.min((md, md2))

    if verbose:
        print(DM)

        print('--------------')
        for t in tracksA:
            print(t)
        print('--------------')
        for t in tracksB:
            print(t)

    assert_array_almost_equal(DM, DM2, 4)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        tracksC = [xyz1C, xyz1A]
        _ = pf.bundles_distances_mdf(tracksA, tracksC)
        print(w)
        assert_true(len(w) == 1)
        assert_true(issubclass(w[0].category, UserWarning))
        assert_true("not have the same number of points" in str(w[0].message))


def test_mam_distances():
    xyz1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]])
    xyz2 = np.array([[0, 1, 1], [1, 0, 1], [2, 3, -2]])
    # dm=array([[ 2,  2, 17], [ 3,  1, 14], [6,  2, 13], [11,  5, 14]])
    # this is the distance matrix between points of xyz1
    # and points of xyz2
    xyz1 = xyz1.astype('float32')
    xyz2 = xyz2.astype('float32')
    zd2 = pf.mam_distances(xyz1, xyz2)
    assert_almost_equal(zd2[0], 1.76135602742)


def test_approx_ei_traj():

    segs = 100
    t = np.linspace(0, 1.75*2*np.pi, segs)
    x = t
    y = 5*np.sin(5*t)
    z = np.zeros(x.shape)
    xyz = np.vstack((x, y, z)).T

    xyza = pf.approx_polygon_track(xyz)
    assert_equal(len(xyza), 27)

    # test repeated point
    track = np.array([[1., 0., 0.], [1., 0., 0.], [3., 0., 0.], [4., 0., 0.]])
    xyza = pf.approx_polygon_track(track)
    assert_array_equal(xyza, np.array([[1., 0., 0.], [4., 0., 0.]]))


def test_approx_mdl_traj():

    t = np.linspace(0, 1.75*2*np.pi, 100)
    x = np.sin(t)
    y = np.cos(t)
    z = t
    xyz = np.vstack((x, y, z)).T
    xyza1 = pf.approximate_mdl_trajectory(xyz, alpha=1.)
    xyza2 = pf.approximate_mdl_trajectory(xyz, alpha=2.)
    assert_equal(len(xyza1), 10)
    assert_equal(len(xyza2), 8)
    assert_array_almost_equal(
        xyza1,
        np.array([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                  [9.39692621e-01, 3.42020143e-01, 1.22173048e+00],
                  [6.42787610e-01, -7.66044443e-01, 2.44346095e+00],
                  [-5.00000000e-01, -8.66025404e-01, 3.66519143e+00],
                  [-9.84807753e-01, 1.73648178e-01, 4.88692191e+00],
                  [-1.73648178e-01, 9.84807753e-01, 6.10865238e+00],
                  [8.66025404e-01, 5.00000000e-01, 7.33038286e+00],
                  [7.66044443e-01, -6.42787610e-01, 8.55211333e+00],
                  [-3.42020143e-01, -9.39692621e-01, 9.77384381e+00],
                  [-1.00000000e+00, -4.28626380e-16, 1.09955743e+01]]))

    assert_array_almost_equal(
        xyza2,
        np.array([[0.00000000e+00, 1.00000000e+00, 0.00000000e+00],
                  [9.95471923e-01, -9.50560433e-02, 1.66599610e+00],
                  [-1.89251244e-01, -9.81928697e-01, 3.33199221e+00],
                  [-9.59492974e-01, 2.81732557e-01, 4.99798831e+00],
                  [3.71662456e-01, 9.28367933e-01, 6.66398442e+00],
                  [8.88835449e-01, -4.58226522e-01, 8.32998052e+00],
                  [-5.40640817e-01, -8.41253533e-01, 9.99597663e+00],
                  [-1.00000000e+00, -4.28626380e-16, 1.09955743e+01]]))


def test_point_track_sq_distance():

    t = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype='f4')
    p = np.array([-1, -1., -1], dtype='f4')
    assert_equal(pf.point_track_sq_distance_check(t, p, .2**2), False)
    pf.point_track_sq_distance_check(t, p, 2**2), True
    t = np.array([[0, 0, 0], [1, 0, 0], [2, 2, 0]], dtype='f4')
    p = np.array([.5, 0, 0], dtype='f4')
    assert_equal(pf.point_track_sq_distance_check(t, p, .2**2), True)
    p = np.array([.5, 1, 0], dtype='f4')
    assert_equal(pf.point_track_sq_distance_check(t, p, .2**2), False)


def test_track_roi_intersection_check():
    roi = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype='f4')
    t = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype='f4')
    assert_equal(pf.track_roi_intersection_check(t, roi, 1), True)
    t = np.array([[0, 0, 0], [1, 0, 0], [2, 2, 2]], dtype='f4')
    assert_equal(pf.track_roi_intersection_check(t, roi, 1), True)
    t = np.array([[1, 1, 0], [1, 0, 0], [1, -1, 0]], dtype='f4')
    assert_equal(pf.track_roi_intersection_check(t, roi, 1), True)
    t = np.array([[4, 0, 0], [4, 1, 1], [4, 2, 0]], dtype='f4')
    assert_equal(pf.track_roi_intersection_check(t, roi, 1), False)


def test_minimum_distance():
    xyz1 = np.array([[1, 0, 0], [2, 0, 0]], dtype='float32')
    xyz2 = np.array([[3, 0, 0], [4, 0, 0]], dtype='float32')
    assert_equal(pf.minimum_closest_distance(xyz1, xyz2), 1.0)


def test_most_similar_mam():
    xyz1 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]],
                    dtype='float32')
    xyz2 = np.array([[0, 1, 1], [1, 0, 1], [2, 3, -2]],
                    dtype='float32')
    xyz3 = np.array([[-1, 0, 0], [2, 0, 0], [2, 3, 0], [3, 0, 0]],
                    dtype='float32')
    tracks = [xyz1, xyz2, xyz3]
    for metric in ('avg', 'min', 'max'):
        # pf should be much faster and the results equivalent
        pf.most_similar_track_mam(tracks, metric=metric)


def test_cut_plane():
    dt = np.dtype(np.float32)
    refx = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]], dtype=dt)
    bundlex = [np.array([[0.5, 1, 0], [1.5, 2, 0], [2.5, 3, 0]], dtype=dt),
               np.array([[0.5, 2, 0], [1.5, 3, 0], [2.5, 4, 0]], dtype=dt),
               np.array([[0.5, 1, 1], [1.5, 2, 2], [2.5, 3, 3]], dtype=dt),
               np.array([[-0.5, 2, -1], [-1.5, 3, -2], [-2.5, 4, -3]],
                        dtype=dt)]
    expected_hit0 = [[1., 1.5, 0., 0.70710683, 0.],
                     [1., 2.5, 0., 0.70710677, 1.],
                     [1., 1.5, 1.5, 0.81649661, 2.]]
    expected_hit1 = [[2., 2.5, 0., 0.70710677, 0.],
                     [2., 3.5, 0., 0.70710677, 1.],
                     [2., 2.5, 2.5, 0.81649655, 2.]]
    hitx = pf.cut_plane(bundlex, refx)
    assert_array_almost_equal(hitx[0], expected_hit0)
    assert_array_almost_equal(hitx[1], expected_hit1)
    # check that algorithm allows types other than float32
    bundlex[0] = np.asarray(bundlex[0], dtype=np.float64)
    hitx = pf.cut_plane(bundlex, refx)
    assert_array_almost_equal(hitx[0], expected_hit0)
    assert_array_almost_equal(hitx[1], expected_hit1)
    refx = np.asarray(refx, dtype=np.float64)
    hitx = pf.cut_plane(bundlex, refx)
    assert_array_almost_equal(hitx[0], expected_hit0)
    assert_array_almost_equal(hitx[1], expected_hit1)
