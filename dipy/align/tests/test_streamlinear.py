import numpy as np
from numpy.testing import (assert_,
                           assert_equal,
                           assert_almost_equal,
                           assert_array_almost_equal,
                           assert_raises)
from dipy.align.streamlinear import (compose_matrix44,
                                     decompose_matrix44,
                                     BundleSumDistanceMatrixMetric,
                                     BundleMinDistanceMatrixMetric,
                                     BundleMinDistanceMetric,
                                     StreamlineLinearRegistration,
                                     StreamlineDistanceMetric,
                                     groupwise_slr,
                                     get_unique_pairs)

from dipy.tracking.streamline import (center_streamlines,
                                      unlist_streamlines,
                                      relist_streamlines,
                                      transform_streamlines,
                                      set_number_of_points,
                                      Streamlines)
from dipy.io.streamline import load_tractogram
from dipy.core.geometry import compose_matrix

from dipy.data import get_fnames, two_cingulum_bundles, read_five_af_bundles
from dipy.align.bundlemin import (_bundle_minimum_distance_matrix,
                                  _bundle_minimum_distance,
                                  distance_matrix_mdf)
from dipy.testing.decorators import set_random_number_generator


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


def fornix_streamlines(no_pts=12):
    fname = get_fnames('fornix')

    fornix = load_tractogram(fname, 'same',
                             bbox_valid_check=False).streamlines

    fornix_streamlines = Streamlines(fornix)
    streamlines = set_number_of_points(fornix_streamlines, no_pts)
    return streamlines


def evaluate_convergence(bundle, new_bundle2):
    pts_static = np.concatenate(bundle, axis=0)
    pts_moved = np.concatenate(new_bundle2, axis=0)
    assert_array_almost_equal(pts_static, pts_moved, 3)


def test_rigid_parallel_lines():

    bundle_initial = simulated_bundle()
    bundle, shift = center_streamlines(bundle_initial)
    mat = compose_matrix44([20, 0, 10, 0, 40, 0])

    bundle2 = transform_streamlines(bundle, mat)

    bundle_sum_distance = BundleSumDistanceMatrixMetric()
    options = {'maxcor': 100, 'ftol': 1e-9, 'gtol': 1e-16, 'eps': 1e-3}
    srr = StreamlineLinearRegistration(metric=bundle_sum_distance,
                                       x0=np.zeros(6),
                                       method='L-BFGS-B',
                                       bounds=None,
                                       options=options)

    new_bundle2 = srr.optimize(bundle, bundle2).transform(bundle2)
    evaluate_convergence(bundle, new_bundle2)


def test_rigid_real_bundles():

    bundle_initial = fornix_streamlines()[:20]
    bundle, shift = center_streamlines(bundle_initial)

    mat = compose_matrix44([0, 0, 20, 45., 0, 0])

    bundle2 = transform_streamlines(bundle, mat)

    bundle_sum_distance = BundleSumDistanceMatrixMetric()
    srr = StreamlineLinearRegistration(bundle_sum_distance,
                                       x0=np.zeros(6),
                                       method='Powell')
    new_bundle2 = srr.optimize(bundle, bundle2).transform(bundle2)

    evaluate_convergence(bundle, new_bundle2)

    bundle_min_distance = BundleMinDistanceMatrixMetric()
    srr = StreamlineLinearRegistration(bundle_min_distance,
                                       x0=np.zeros(6),
                                       method='Powell')
    new_bundle2 = srr.optimize(bundle, bundle2).transform(bundle2)

    evaluate_convergence(bundle, new_bundle2)

    assert_raises(ValueError, StreamlineLinearRegistration, method='Whatever')


def test_rigid_partial_real_bundles():

    static = fornix_streamlines()[:20]
    moving = fornix_streamlines()[20:40]
    static_center, shift = center_streamlines(static)
    moving_center, shift2 = center_streamlines(moving)

    print(shift2)
    mat = compose_matrix(translate=np.array([0, 0, 0.]),
                         angles=np.deg2rad([40, 0, 0.]))
    moved = transform_streamlines(moving_center, mat)

    srr = StreamlineLinearRegistration()

    srm = srr.optimize(static_center, moved)
    print(srm.fopt)
    print(srm.iterations)
    print(srm.funcs)

    moving_back = srm.transform(moved)
    print(srm.matrix)

    static_center = set_number_of_points(static_center, 100)
    moving_center = set_number_of_points(moving_back, 100)

    vol = np.zeros((100, 100, 100))
    spts = np.concatenate(static_center, axis=0)
    spts = np.round(spts).astype(int) + np.array([50, 50, 50])

    mpts = np.concatenate(moving_center, axis=0)
    mpts = np.round(mpts).astype(int) + np.array([50, 50, 50])

    for index in spts:
        i, j, k = index
        vol[i, j, k] = 1

    vol2 = np.zeros((100, 100, 100))
    for index in mpts:
        i, j, k = index
        vol2[i, j, k] = 1

    overlap = np.sum(np.logical_and(vol, vol2)) / float(np.sum(vol2))

    assert_equal(overlap * 100 > 40, True)


def test_stream_rigid():

    static = fornix_streamlines()[:20]
    moving = fornix_streamlines()[20:40]
    center_streamlines(static)

    mat = compose_matrix44([0, 0, 0, 0, 40, 0])
    moving = transform_streamlines(moving, mat)

    srr = StreamlineLinearRegistration()
    sr_params = srr.optimize(static, moving)
    moved = transform_streamlines(moving, sr_params.matrix)

    srr = StreamlineLinearRegistration(verbose=True)
    srm = srr.optimize(static, moving)
    moved2 = transform_streamlines(moving, srm.matrix)
    moved3 = srm.transform(moving)

    assert_array_almost_equal(moved[0], moved2[0], decimal=3)
    assert_array_almost_equal(moved2[0], moved3[0], decimal=3)


def test_min_vs_min_fast_precision():

    static = fornix_streamlines()[:20]
    moving = fornix_streamlines()[:20]

    static = [s.astype('f8') for s in static]
    moving = [m.astype('f8') for m in moving]

    bmd = BundleMinDistanceMatrixMetric()
    bmd.setup(static, moving)

    bmdf = BundleMinDistanceMetric()
    bmdf.setup(static, moving)

    x_test = [0.01, 0, 0, 0, 0, 0]

    print(bmd.distance(x_test))
    print(bmdf.distance(x_test))
    assert_equal(bmd.distance(x_test), bmdf.distance(x_test))


@set_random_number_generator()
def test_same_number_of_points(rng):
    A = [rng.random((10, 3)), rng.random((20, 3))]
    B = [rng.random((21, 3)), rng.random((30, 3))]
    C = [rng.random((10, 3)), rng.random((10, 3))]
    D = [rng.random((20, 3)), rng.random((20, 3))]

    slr = StreamlineLinearRegistration()
    assert_raises(ValueError, slr.optimize, A, B)
    assert_raises(ValueError, slr.optimize, C, D)
    assert_raises(ValueError, slr.optimize, C, B)


def test_efficient_bmd():

    a = np.array([[1, 1, 1],
                  [2, 2, 2],
                  [3, 3, 3]])

    streamlines = [a, a + 2, a + 4]

    points, offsets = unlist_streamlines(streamlines)
    points = points.astype(np.double)
    points2 = points.copy()

    D = np.zeros((len(offsets), len(offsets)), dtype='f8')

    _bundle_minimum_distance_matrix(points, points2,
                                    len(offsets), len(offsets),
                                    a.shape[0], D)

    assert_equal(np.sum(np.diag(D)), 0)

    points2 += 2

    _bundle_minimum_distance_matrix(points, points2,
                                    len(offsets), len(offsets),
                                    a.shape[0], D)

    streamlines2 = relist_streamlines(points2, offsets)
    D2 = distance_matrix_mdf(streamlines, streamlines2)

    assert_array_almost_equal(D, D2)

    cols = D2.shape[1]
    rows = D2.shape[0]

    dist = 0.25 * (np.sum(np.min(D2, axis=0)) / float(cols) +
                   np.sum(np.min(D2, axis=1)) / float(rows)) ** 2

    dist2 = _bundle_minimum_distance(points, points2,
                                     len(offsets), len(offsets),
                                     a.shape[0])
    assert_almost_equal(dist, dist2)


@set_random_number_generator()
def test_openmp_locks(rng):
    static = []
    moving = []
    pts = 20

    for i in range(1000):
        s = rng.random((pts, 3))
        static.append(s)
        moving.append(s + 2)

    moving = moving[2:]

    points, offsets = unlist_streamlines(static)
    points2, offsets2 = unlist_streamlines(moving)

    D = np.zeros((len(offsets), len(offsets2)), dtype='f8')

    _bundle_minimum_distance_matrix(points, points2,
                                    len(offsets), len(offsets2),
                                    pts, D)

    dist1 = 0.25 * (np.sum(np.min(D, axis=0)) / float(D.shape[1]) +
                    np.sum(np.min(D, axis=1)) / float(D.shape[0])) ** 2

    dist2 = _bundle_minimum_distance(points, points2,
                                     len(offsets), len(offsets2),
                                     pts)

    assert_almost_equal(dist1, dist2, 6)


def test_from_to_rigid():

    t = np.array([10, 2, 3, 0.1, 20., 30.])
    mat = compose_matrix44(t)
    vec = decompose_matrix44(mat, 6)

    assert_array_almost_equal(t, vec)

    t = np.array([0, 0, 0, 180, 0., 0.])

    mat = np.eye(4)
    mat[0, 0] = -1

    vec = decompose_matrix44(mat, 6)

    assert_array_almost_equal(-t, vec)


def test_matrix44():

    assert_raises(ValueError, compose_matrix44, np.ones(5))
    assert_raises(ValueError, compose_matrix44, np.ones(13))
    assert_raises(ValueError, compose_matrix44, np.ones(16))


def test_abstract_metric_class():

    class DummyStreamlineMetric(StreamlineDistanceMetric):
        def test(self):
            pass
    assert_raises(TypeError, DummyStreamlineMetric)


def test_evolution_of_previous_iterations():

    static = fornix_streamlines()[:20]
    moving = fornix_streamlines()[:20]

    moving = [m + np.array([10., 0., 0.]) for m in moving]

    slr = StreamlineLinearRegistration(evolution=True)

    slm = slr.optimize(static, moving)

    assert_equal(len(slm.matrix_history), slm.iterations)


def test_similarity_real_bundles():

    bundle_initial = fornix_streamlines()
    bundle_initial, shift = center_streamlines(bundle_initial)
    bundle = bundle_initial[:20]
    xgold = [0, 0, 10, 0, 0, 0, 1.5]
    mat = compose_matrix44(xgold)
    bundle2 = transform_streamlines(bundle_initial[:20], mat)

    metric = BundleMinDistanceMatrixMetric()
    x0 = np.array([0, 0, 0, 0, 0, 0, 1], 'f8')

    slr = StreamlineLinearRegistration(metric=metric,
                                       x0=x0,
                                       method='Powell',
                                       bounds=None,
                                       verbose=False)

    slm = slr.optimize(bundle, bundle2)
    new_bundle2 = slm.transform(bundle2)
    evaluate_convergence(bundle, new_bundle2)


def test_affine_real_bundles():

    bundle_initial = fornix_streamlines()
    bundle_initial, shift = center_streamlines(bundle_initial)
    bundle = bundle_initial[:20]
    xgold = [0, 4, 2, 0, 10, 10, 1.2, 1.1, 1., 0., 0.2, 0.]
    mat = compose_matrix44(xgold)
    bundle2 = transform_streamlines(bundle_initial[:20], mat)

    x0 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])

    x = 25

    bounds = [(-x, x), (-x, x), (-x, x),
              (-x, x), (-x, x), (-x, x),
              (0.1, 1.5), (0.1, 1.5), (0.1, 1.5),
              (-1, 1), (-1, 1), (-1, 1)]

    options = {'maxcor': 10, 'ftol': 1e-7, 'gtol': 1e-5, 'eps': 1e-8}

    metric = BundleMinDistanceMatrixMetric()

    slr = StreamlineLinearRegistration(metric=metric,
                                       x0=x0,
                                       method='L-BFGS-B',
                                       bounds=bounds,
                                       verbose=True,
                                       options=options)
    slm = slr.optimize(bundle, bundle2)

    new_bundle2 = slm.transform(bundle2)

    slr2 = StreamlineLinearRegistration(metric=metric,
                                        x0=x0,
                                        method='Powell',
                                        bounds=None,
                                        verbose=True,
                                        options=None)

    slm2 = slr2.optimize(bundle, new_bundle2)

    new_bundle2 = slm2.transform(new_bundle2)

    evaluate_convergence(bundle, new_bundle2)


def test_vectorize_streamlines():

    cingulum_bundles = two_cingulum_bundles()

    cb_subj1 = cingulum_bundles[0]
    cb_subj1 = set_number_of_points(cb_subj1, 10)
    cb_subj1_pts_no = np.array([s.shape[0] for s in cb_subj1])

    assert_equal(np.all(cb_subj1_pts_no == 10), True)


@set_random_number_generator()
def test_x0_input(rng):
    for x0 in [6, 7, 12, "Rigid", 'rigid', "similarity", "Affine"]:
        StreamlineLinearRegistration(x0=x0)

    for x0 in [rng.random(6), rng.random(7), rng.random(12)]:
        StreamlineLinearRegistration(x0=x0)

    for x0 in [8, 20, "Whatever", rng.random(20), rng.random((20, 3))]:
        assert_raises(ValueError, StreamlineLinearRegistration, x0=x0)

    x0 = rng.random((4, 3))
    assert_raises(ValueError, StreamlineLinearRegistration, x0=x0)

    x0_6 = np.zeros(6)
    x0_7 = np.array([0, 0, 0, 0, 0, 0, 1.])
    x0_12 = np.array([0, 0, 0, 0, 0, 0, 1., 1., 1., 0, 0, 0])

    x0_s = [x0_6, x0_7, x0_12, x0_6, x0_7, x0_12]

    for i, x0 in enumerate([6, 7, 12, "Rigid", "similarity", "Affine"]):
        slr = StreamlineLinearRegistration(x0=x0)
        assert_equal(slr.x0, x0_s[i])


@set_random_number_generator()
def test_compose_decompose_matrix44(rng):
    for i in range(20):
        x0 = rng.random(12)
        mat = compose_matrix44(x0[:6])
        assert_array_almost_equal(x0[:6], decompose_matrix44(mat, size=6))
        mat = compose_matrix44(x0[:7])
        assert_array_almost_equal(x0[:7], decompose_matrix44(mat, size=7))
        mat = compose_matrix44(x0[:12])
        assert_array_almost_equal(x0[:12], decompose_matrix44(mat, size=12))

    assert_raises(ValueError, decompose_matrix44, mat, 20)


def test_cascade_of_optimizations_and_threading():

    cingulum_bundles = two_cingulum_bundles()

    cb1 = cingulum_bundles[0]
    cb1 = set_number_of_points(cb1, 20)

    test_x0 = np.array([10, 4, 3, 0, 20, 10, 1.5, 1.5, 1.5, 0., 0.2, 0])

    cb2 = transform_streamlines(cingulum_bundles[0],
                                compose_matrix44(test_x0))
    cb2 = set_number_of_points(cb2, 20)

    print('first rigid')
    slr = StreamlineLinearRegistration(x0=6, num_threads=1)
    slm = slr.optimize(cb1, cb2)

    print('then similarity')
    slr2 = StreamlineLinearRegistration(x0=7, num_threads=2)
    slm2 = slr2.optimize(cb1, cb2, slm.matrix)

    print('then affine')
    slr3 = StreamlineLinearRegistration(x0=12, options={'maxiter': 50},
                                        num_threads=None)
    slm3 = slr3.optimize(cb1, cb2, slm2.matrix)

    assert_(slm2.fopt < slm.fopt)
    assert_(slm3.fopt < slm2.fopt)


@set_random_number_generator()
def test_wrong_num_threads(rng):
    A = [rng.random((10, 3)), rng.random((10, 3))]
    B = [rng.random((10, 3)), rng.random((10, 3))]

    slr = StreamlineLinearRegistration(num_threads=0)
    assert_raises(ValueError, slr.optimize, A, B)


def test_get_unique_pairs():

    # Regular case
    pairs, exclude = get_unique_pairs(6)
    assert_equal(len(np.unique(pairs)), 6)
    assert_equal(exclude, None)

    # Odd case
    pairs, exclude = get_unique_pairs(5)
    assert_equal(len(np.unique(pairs)), 4)
    assert_equal(isinstance(exclude, (int, np.int64, np.int32)), True)

    # Iterative case
    new_pairs, new_exclude = get_unique_pairs(5, pairs)
    assert_equal(len(np.unique(pairs)), 4)
    assert_equal(exclude != new_exclude, True)

    # Check errors
    assert_raises(TypeError, get_unique_pairs, 2.7)
    assert_raises(ValueError, get_unique_pairs, 1)


def test_groupwise_slr():

    bundles = read_five_af_bundles()

    # Test regular use case with convergence
    new_bundles, T, d = groupwise_slr(bundles, verbose=True)

    assert_equal(len(new_bundles), len(bundles))
    assert_equal(type(new_bundles), list)
    assert_equal(len(T), len(bundles))
    assert_equal(type(T), list)

    # Test regular use case without convergence (few iterations)
    new_bundles, T, d = groupwise_slr(bundles, max_iter=3, tol=-10,
                                      verbose=True)
