import numpy as np
from numpy.testing import (run_module_suite,
                           assert_equal,
                           assert_array_equal,
                           assert_array_almost_equal)
from dipy.align.streamwarp import (transform_streamlines,
                                   matrix44,
                                   BundleSumDistance,                                   
                                   center_streamlines)
from dipy.tracking.metrics import downsample
from dipy.data import get_data
from nibabel import trackvis as tv
from dipy.align.streamwarp import (StreamlineRigidRegistration,
                                   compose_transformations,
                                   vectorize_streamlines,
                                   unlist_streamlines,
                                   relist_streamlines)
from dipy.align.bmd import _bundle_minimum_distance_rigid
from dipy.tracking.distances import bundles_distances_mdf


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
    srr = StreamlineRigidRegistration(similarity=bundle_sum_distance, 
                                      algorithm='L_BFGS_B', 
                                      bounds=None, 
                                      fast=False)
                                      
    new_bundle2 = srr.optimize(bundle, bundle2).transform(bundle2)
    evaluate_convergence(bundle, new_bundle2)


def test_rigid_real_bundles():

    bundle_initial = fornix_streamlines()[:20]
    bundle, shift = center_streamlines(bundle_initial)
    mat = matrix44([0, 0, 20, 45, 0, 0])
    bundle2 = transform_streamlines(bundle, mat)

    bundle_sum_distance = BundleSumDistance()
    srr = StreamlineRigidRegistration(bundle_sum_distance, 
                                      algorithm='Powell',
                                      fast=False)
    new_bundle2 = srr.optimize(bundle, bundle2).transform(bundle2)

    evaluate_convergence(bundle, new_bundle2)


def test_rigid_partial_real_bundles():

    static = fornix_streamlines()[:20]
    moving = fornix_streamlines()[20:40]
    static_center, shift = center_streamlines(static)

    mat = matrix44([0, 0, 0, 0, 40, 0])
    moving = transform_streamlines(moving, mat)

    srr = StreamlineRigidRegistration()

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

    srr = StreamlineRigidRegistration()

    sr_params = srr.optimize(static, moving)

    moved = transform_streamlines(moving, sr_params.matrix)

    srr = StreamlineRigidRegistration(disp=True)

    srm = srr.optimize(static, moving)

    moved2 = transform_streamlines(moving, srm.matrix)

    moved3 = srm.transform(moving)

    assert_array_equal(moved[0], moved2[0])
    assert_array_equal(moved2[0], moved3[0])


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
    points = points.astype(np.float32)
    points2 = points.copy()

    cache = 2 + np.zeros((len(offsets), len(offsets)), dtype=np.float32)

    D = np.zeros((len(offsets), len(offsets)), dtype='f4')

    _bundle_minimum_distance_rigid(points, points2,
                                  len(offsets), len(offsets),
                                  3, D)

    assert_equal(np.sum(np.diag(D)), 0)

    points2 = points2 + 2

    _bundle_minimum_distance_rigid(points, points2,
                                  len(offsets), len(offsets),
                                  3, D)

    streamlines2 = relist_streamlines(points2, offsets)
    D2 = bundles_distances_mdf(streamlines, streamlines2)

    assert_array_almost_equal(D, D2)


if __name__ == '__main__':

    run_module_suite()
