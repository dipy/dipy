import numpy as np
from numpy.testing import (run_module_suite,
                           assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.align.streamlinear import (transform_streamlines,
                                     matrix44,
                                     from_matrix44_rigid,
                                     BundleSumDistance,
                                     BundleMinDistance,
                                     BundleMinDistanceFast,
                                     center_streamlines)
from dipy.tracking.metrics import downsample
from dipy.data import get_data
from nibabel import trackvis as tv
from dipy.align.streamlinear import (StreamlineLinearRegistration,
                                     StreamlineDistanceMetric,
                                     compose_transformations,
                                     vectorize_streamlines,
                                     unlist_streamlines,
                                     relist_streamlines)
from dipy.align.bmd import (_bundle_minimum_distance_rigid,
                            _bundle_minimum_distance_rigid_nomat,
                            bundles_distance_matrix_mdf)
from dipy.io.pickles import load_pickle
import scipy


def simulated_bundle(no_streamlines=10, waves=False, no_pts=12):
    t = np.linspace(-10, 10, 200)
    # parallel waves or parallel lines
    bundle = []
    for i in np.linspace(-5, 5, no_streamlines):
        if waves:
            pts = np.vstack((np.cos(t), t, i * np.ones(t.shape))).T
        else:
             pts = np.vstack((np.zeros(t.shape), t, i * np.ones(t.shape))).T
        pts = downsample(pts, no_pts)
        bundle.append(pts)

    return bundle


def fornix_streamlines(no_pts=12):
    fname = get_data('fornix')
    streams, hdr = tv.read(fname)
    streamlines = [downsample(i[0], no_pts) for i in streams]
    return streamlines


def evaluate_convergence(bundle, new_bundle2):
    pts_static = np.concatenate(bundle, axis=0)
    pts_moved = np.concatenate(new_bundle2, axis=0)
    assert_array_almost_equal(pts_static, pts_moved, 3)


def test_rigid_parallel_lines():

    bundle_initial = simulated_bundle()
    bundle, shift = center_streamlines(bundle_initial)
    mat = matrix44([20, 0, 10, 0, 40, 0])
    bundle2 = transform_streamlines(bundle, mat)

    bundle_sum_distance = BundleSumDistance()
    srr = StreamlineLinearRegistration(metric=bundle_sum_distance,
                                      x0=np.zeros(6),
                                      method='L-BFGS-B',
                                      bounds=None,
                                      fast=False,
                                      options={'maxcor':100, 'ftol':1e-9,
                                               'gtol':1e-16, 'eps':1e-3})

    new_bundle2 = srr.optimize(bundle, bundle2).transform(bundle2)
    evaluate_convergence(bundle, new_bundle2)


def test_rigid_real_bundles():

    bundle_initial = fornix_streamlines()[:20]
    bundle, shift = center_streamlines(bundle_initial)
    mat = matrix44([0, 0, 20, 45, 0, 0])
    bundle2 = transform_streamlines(bundle, mat)

    bundle_sum_distance = BundleSumDistance()
    srr = StreamlineLinearRegistration(bundle_sum_distance,
                                      x0=np.zeros(6),
                                      method='Powell',
                                      fast=False)
    new_bundle2 = srr.optimize(bundle, bundle2).transform(bundle2)

    evaluate_convergence(bundle, new_bundle2)

    srr = StreamlineLinearRegistration(x0=np.zeros(6),
                                      method='Powell',
                                      fast=False)
    new_bundle2 = srr.optimize(bundle, bundle2).transform(bundle2)

    evaluate_convergence(bundle, new_bundle2)

    assert_raises(ValueError, StreamlineLinearRegistration, method='Whatever')


def test_rigid_partial_real_bundles():

    static = fornix_streamlines()[:20]
    moving = fornix_streamlines()[20:40]
    static_center, shift = center_streamlines(static)

    mat = matrix44([0, 0, 0, 0, 40, 0])
    moving = transform_streamlines(moving, mat)

    srr = StreamlineLinearRegistration()

    moving_center = srr.optimize(static_center, moving).transform(moving)

    static_center = [downsample(s, 100) for s in static_center]
    moving_center = [downsample(s, 100) for s in moving_center]

    vol = np.zeros((100, 100, 100))
    spts = np.concatenate(static_center, axis=0)
    spts = np.round(spts).astype(np.int) + np.array([50, 50, 50])

    mpts = np.concatenate(moving_center, axis=0)
    mpts = np.round(mpts).astype(np.int) + np.array([50, 50, 50])

    for index in spts:
        i, j, k = index
        vol[i, j, k] = 1

    vol2 = np.zeros((100, 100, 100))
    for index in mpts:
        i, j, k = index
        vol2[i, j, k] = 1

    overlap = np.sum(np.logical_and(vol,vol2)) / float(np.sum(vol2))
    #print(overlap)

    assert_equal(overlap * 100 > 40, True )


def test_stream_rigid():

    static = fornix_streamlines()[:20]
    moving = fornix_streamlines()[20:40]
    static_center, shift = center_streamlines(static)

    mat = matrix44([0, 0, 0, 0, 40, 0])
    moving = transform_streamlines(moving, mat)

    srr = StreamlineLinearRegistration()

    sr_params = srr.optimize(static, moving)

    moved = transform_streamlines(moving, sr_params.matrix)

    srr = StreamlineLinearRegistration(disp=True, fast=True)

    srm = srr.optimize(static, moving)

    moved2 = transform_streamlines(moving, srm.matrix)

    moved3 = srm.transform(moving)

    assert_array_equal(moved[0], moved2[0])
    assert_array_equal(moved2[0], moved3[0])


def test_min_vs_min_fast_precision():

    static = fornix_streamlines()[:20]
    moving = fornix_streamlines()[:20]

    static = [s.astype('f8') for s in static]
    moving = [m.astype('f8') for m in moving]

    bmd = BundleMinDistance()
    bmd.set_static(static)
    bmd.set_moving(moving)

    bmdf = BundleMinDistanceFast()
    bmdf.set_static(static)
    bmdf.set_moving(moving)

    x_test = [0.01, 0, 0, 0, 0, 0]

    print(bmd.distance(x_test))
    print(bmdf.distance(x_test))
    assert_equal(bmd.distance(x_test), bmdf.distance(x_test))


def test_same_number_of_points():

    A = [np.random.rand(10, 3), np.random.rand(20, 3)]
    B = [np.random.rand(21, 3), np.random.rand(30, 3)]
    C = [np.random.rand(10, 3), np.random.rand(10, 3)]
    D = [np.random.rand(20, 3), np.random.rand(20, 3)]

    slr = StreamlineLinearRegistration()
    assert_raises(ValueError, slr.optimize, A, B)
    assert_raises(ValueError, slr.optimize, C, D)
    assert_raises(ValueError, slr.optimize, C, B)


def test_compose_transformations():

    A = np.eye(4)
    A[0, -1] = 10

    B = np.eye(4)
    B[0, -1] = -20

    C = np.eye(4)
    C[0, -1] = 10

    CBA = compose_transformations(A, B, C)

    assert_array_equal(CBA, np.eye(4))


def test_unlist_relist_streamlines():

    streamlines = [np.random.rand(10, 3),
                   np.random.rand(20, 3),
                   np.random.rand(5, 3)]

    points, offsets = unlist_streamlines(streamlines)

    assert_equal(offsets.dtype, np.dtype('i8'))

    assert_equal(points.shape, (35, 3))
    assert_equal(len(offsets), len(streamlines))

    streamlines2 = relist_streamlines(points, offsets)

    assert_equal(len(streamlines), len(streamlines2))

    for i in range(len(streamlines)):
        assert_array_equal(streamlines[i], streamlines2[i])


def test_efficient_bmd():

    a = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]])

    streamlines = [a, a + 2, a + 4]

    points, offsets = unlist_streamlines(streamlines)
    points = points.astype(np.double)
    points2 = points.copy()

    D = np.zeros((len(offsets), len(offsets)), dtype='f8')

    _bundle_minimum_distance_rigid(points, points2,
                                  len(offsets), len(offsets),
                                  a.shape[0], D)

    assert_equal(np.sum(np.diag(D)), 0)

    points2 += 2

    _bundle_minimum_distance_rigid(points, points2,
                                  len(offsets), len(offsets),
                                  a.shape[0], D)

    streamlines2 = relist_streamlines(points2, offsets)
    D2 = bundles_distance_matrix_mdf(streamlines, streamlines2)

    assert_array_almost_equal(D, D2)

    cols = D2.shape[1]
    rows = D2.shape[0]

    dist = 0.25 * (np.sum(np.min(D2, axis=0)) / float(cols) +
                   np.sum(np.min(D2, axis=1)) / float(rows)) ** 2

    dist2 = _bundle_minimum_distance_rigid_nomat(points, points2,
                                                len(offsets), len(offsets),
                                                a.shape[0])
    assert_almost_equal(dist, dist2)


def test_openmp_locks():

    static = []
    moving = []
    pts = 20

    for i in range(1000):
        s = np.random.rand(pts, 3)
        static.append(s)
        moving.append(s + 2)

    moving = moving[2:]

    points, offsets = unlist_streamlines(static)
    points2, offsets2 = unlist_streamlines(moving)

    D = np.zeros((len(offsets), len(offsets2)), dtype='f8')

    _bundle_minimum_distance_rigid(points, points2,
                                  len(offsets), len(offsets2),
                                  pts, D)

    dist1 = 0.25 * (np.sum(np.min(D, axis=0)) / float(D.shape[1]) +
                   np.sum(np.min(D, axis=1)) / float(D.shape[0])) ** 2

    dist2 = _bundle_minimum_distance_rigid_nomat(points, points2,
                                  len(offsets), len(offsets2),
                                  pts)

    assert_equal(dist1, dist2)


def test_from_to_rigid():

    t = np.array([10, 2, 3, 0.1, 20., 30.])
    mat = matrix44(t)
    vec = from_matrix44_rigid(mat)

    assert_array_almost_equal(t, vec)

    t = np.array([0, 0, 0, 180, 0., 0.])

    mat = np.eye(4)
    mat[0, 0] = -1

    vec = from_matrix44_rigid(mat)

    assert_array_almost_equal(t, vec)


def test_matrix44():

    assert_raises(ValueError, matrix44, np.ones(5))
    assert_raises(ValueError, matrix44, np.ones(9))
    assert_raises(ValueError, matrix44, np.ones(16))


def test_abstract_metric_class():

    s = StreamlineDistanceMetric()
    assert_equal(s.distance(np.ones(6)), None)


def test_evolution_of_previous_iterations():

    static = fornix_streamlines()[:20]
    moving = fornix_streamlines()[:20]

    moving = [m + np.array([10., 0., 0.]) for m in moving]

    slr = StreamlineLinearRegistration(evolution=True)

    from dipy.core.optimize import SCIPY_LESS_0_11

    if not SCIPY_LESS_0_11:

        slm = slr.optimize(static, moving)

        assert_equal(len(slm.matrix_history), slm.iterations)


def test_similarity_real_bundles():

    bundle_initial = fornix_streamlines()
    bundle_initial, shift = center_streamlines(bundle_initial)
    bundle = bundle_initial[:20]
    xgold = [0, 0, 10, 0, 0, 0, 1.5]
    mat = matrix44(xgold)
    bundle2 = transform_streamlines(bundle_initial[:20], mat)

    metric = BundleMinDistance()
    x0 = np.array([0, 0, 0, 0, 0, 0, 1], 'f8')

    slr = StreamlineLinearRegistration(metric=metric,
                                      x0=x0,
                                      method='Powell',
                                      bounds=None,
                                      fast=False,
                                      disp=False)

    slm = slr.optimize(bundle, bundle2)
    new_bundle2 = slm.transform(bundle2)
    evaluate_convergence(bundle, new_bundle2)


def test_affine_real_bundles():

    bundle_initial = fornix_streamlines()
    bundle_initial, shift = center_streamlines(bundle_initial)
    bundle = bundle_initial[:20]
    xgold = [0, 4, 2, 0, 10, 10, 1.2, 1.1, 1., 0., 0.2, 0.]
    mat = matrix44(xgold)
    bundle2 = transform_streamlines(bundle_initial[:20], mat)

    x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])

    x = 25

    bounds = [(-x, x), (-x, x), (-x, x),
              (-x, x), (-x, x), (-x, x),
              (0.1, 1.5), (0.1, 1.5), (0.1, 1.5),
              (-1, 1), (-1, 1), (-1, 1)]

    options = {'maxcor':10, 'ftol':1e-7, 'gtol':1e-5, 'eps':1e-8}

    metric = BundleMinDistance()

    slr = StreamlineLinearRegistration(metric=metric,
                                      x0=x0,
                                      method='L-BFGS-B',
                                      bounds=bounds,
                                      fast=False,
                                      disp=True,
                                      options=options)
    slm = slr.optimize(bundle, bundle2)

    new_bundle2 = slm.transform(bundle2)

    slr2 = StreamlineLinearRegistration(metric=metric,
                                      x0=x0,
                                      method='Powell',
                                      bounds=None,
                                      fast=False,
                                      disp=True,
                                      options=None)

    slm2 = slr2.optimize(bundle, new_bundle2)

    new_bundle2 = slm2.transform(new_bundle2)

    evaluate_convergence(bundle, new_bundle2)


def test_vectorize_streamlines():
    fname = get_data('cb_2')

    cingulum_bundles = load_pickle(fname)

    cb_subj1 = cingulum_bundles[0]
    cb_subj1 = vectorize_streamlines(cb_subj1, 10)
    cb_subj1_pts_no = np.array([s.shape[0] for s in cb_subj1])

    assert_equal(np.all(cb_subj1_pts_no == 10), True)


if __name__ == '__main__':

    run_module_suite()

