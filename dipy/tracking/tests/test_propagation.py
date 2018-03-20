import os
import numpy as np
import numpy.testing

from dipy.data import get_data, get_sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dti import TensorModel, quantize_evecs
from dipy.tracking import utils
from dipy.tracking.eudx import EuDX
from dipy.tracking.propspeed import ndarray_offset, eudx_both_directions
from dipy.tracking.metrics import length
from dipy.tracking.propspeed import map_coordinates_trilinear_iso

import nibabel as ni

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises, assert_almost_equal

from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           run_module_suite)


def stepped_1d(arr_1d):
    # Make a version of `arr_1d` which is not contiguous
    return np.vstack((arr_1d, arr_1d)).ravel(order='F')[::2]


def test_offset():
    # Test ndarray_offset function
    for dt in (np.int32, np.float64):
        index = np.array([1, 1], dtype=np.intp)
        A = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 3]], dtype=dt)
        strides = np.array(A.strides, np.intp)
        i_size = A.dtype.itemsize
        assert_equal(ndarray_offset(index, strides, 2, i_size), 4)
        assert_equal(A.ravel()[4], A[1, 1])
        # Index and strides arrays must be C-continuous. Test this is enforced
        # by using non-contiguous versions of the input arrays.
        assert_raises(ValueError, ndarray_offset,
                      stepped_1d(index), strides, 2, i_size)
        assert_raises(ValueError, ndarray_offset,
                      index, stepped_1d(strides), 2, i_size)


def test_trilinear_interp_cubic_voxels():
    A = np.ones((17, 17, 17))
    B = np.zeros(3)
    strides = np.array(A.strides, np.intp)
    A[7, 7, 7] = 2
    points = np.array([[0, 0, 0], [7., 7.5, 7.], [3.5, 3.5, 3.5]])
    map_coordinates_trilinear_iso(A, points, strides, 3, B)
    assert_array_almost_equal(B, np.array([1., 1.5, 1.]))
    # All of the input array, points array, strides array and output array must
    # be C-contiguous.  Check by passing in versions that aren't C contiguous
    assert_raises(ValueError, map_coordinates_trilinear_iso,
                  A.copy(order='F'), points, strides, 3, B)
    assert_raises(ValueError, map_coordinates_trilinear_iso,
                  A, points.copy(order='F'), strides, 3, B)
    assert_raises(ValueError, map_coordinates_trilinear_iso,
                  A, points, stepped_1d(strides), 3, B)
    assert_raises(ValueError, map_coordinates_trilinear_iso,
                  A, points, strides, 3, stepped_1d(B))


def test_eudx_further():
    """ Cause we love testin.. ;-)
    """

    fimg, fbvals, fbvecs = get_data('small_101D')

    img = ni.load(fimg)
    data = img.get_data()
    gtab = gradient_table(fbvals, fbvecs)
    tensor_model = TensorModel(gtab)
    ten = tensor_model.fit(data)
    x, y, z = data.shape[:3]
    seeds = np.zeros((10**4, 3))
    for i in range(10**4):
        rx = (x-1)*np.random.rand()
        ry = (y-1)*np.random.rand()
        rz = (z-1)*np.random.rand()
        seeds[i] = np.ascontiguousarray(np.array([rx, ry, rz]),
                                        dtype=np.float64)

    sphere = get_sphere('symmetric724')

    ind = quantize_evecs(ten.evecs)
    eu = EuDX(a=ten.fa, ind=ind, seeds=seeds,
              odf_vertices=sphere.vertices, a_low=.2)
    T = [e for e in eu]

    # check that there are no negative elements
    for t in T:
        assert_equal(np.sum(t.ravel() < 0), 0)

    # Test eudx with affine
    def random_affine(seeds):
        affine = np.eye(4)
        affine[:3, :] = np.random.random((3, 4))
        seeds = np.dot(seeds, affine[:3, :3].T)
        seeds += affine[:3, 3]
        return affine, seeds

    # Make two random affines and move seeds
    affine1, seeds1 = random_affine(seeds)
    affine2, seeds2 = random_affine(seeds)

    # Make tracks using different affines
    eu1 = EuDX(a=ten.fa, ind=ind, odf_vertices=sphere.vertices,
               seeds=seeds1, a_low=.2, affine=affine1)
    eu2 = EuDX(a=ten.fa, ind=ind, odf_vertices=sphere.vertices,
               seeds=seeds2, a_low=.2, affine=affine2)

    # Move from eu2 affine2 to affine1
    eu2_to_eu1 = utils.move_streamlines(eu2, output_space=affine1,
                                        input_space=affine2)
    # Check that the tracks are the same
    for sl1, sl2 in zip(eu1, eu2_to_eu1):
        assert_array_almost_equal(sl1, sl2)


def test_eudx_bad_seed():
    """Test passing a bad seed to eudx"""
    fimg, fbvals, fbvecs = get_data('small_101D')

    img = ni.load(fimg)
    data = img.get_data()
    gtab = gradient_table(fbvals, fbvecs)
    tensor_model = TensorModel(gtab)
    ten = tensor_model.fit(data)
    ind = quantize_evecs(ten.evecs)

    sphere = get_sphere('symmetric724')
    seed = [1000000., 1000000., 1000000.]
    eu = EuDX(a=ten.fa, ind=ind, seeds=[seed],
              odf_vertices=sphere.vertices, a_low=.2)
    assert_raises(ValueError, list, eu)

    print(data.shape)
    seed = [1., 5., 8.]
    eu = EuDX(a=ten.fa, ind=ind, seeds=[seed],
              odf_vertices=sphere.vertices, a_low=.2)

    seed = [-1., 1000000., 1000000.]
    eu = EuDX(a=ten.fa, ind=ind, seeds=[seed],
              odf_vertices=sphere.vertices, a_low=.2)
    assert_raises(ValueError, list, eu)


def test_eudx_boundaries():
    """
    This test checks that the tracking will exclude seeds in both directions.
    Here we create a volume of shape (50, 60, 40) and we will add 2 seeds
    exactly at the volume's boundaries (49, 0, 0) and (0, 0, 0). Those should
    not generate any streamlines as EuDX does not interpolate on the boundary
    voxels. We also add 3 seeds not in the boundaries which should generate
    streamlines without a problem.
    """

    fa = np.ones((50, 60, 40))
    ind = np.zeros(fa.shape)
    sphere = get_sphere('repulsion724')

    seed = [49., 0, 0]
    seed2 = [0., 0, 0]
    seed3 = [48., 0, 0]
    seed4 = [1., 0, 0]
    seed5 = [5., 5, 5]

    eu = EuDX(a=fa, ind=ind, seeds=[seed, seed2, seed3, seed4, seed5],
              odf_vertices=sphere.vertices, a_low=.2,
              total_weight=0.)
    track = list(eu)

    assert_equal(len(track), 3)


def test_eudx_both_directions_errors():
    # Test error conditions for both directions function
    sphere = get_sphere('symmetric724')
    seed = np.zeros(3, np.float64)
    qa = np.zeros((4, 5, 6, 7), np.float64)
    ind = qa.copy()
    # All of seed, qa, ind, odf_vertices must be C-contiguous.  Check by
    # passing in versions that aren't C contiguous
    assert_raises(ValueError, eudx_both_directions,
                  stepped_1d(seed), 0, qa, ind, sphere.vertices, 0.5, 0.1,
                  1., 1., 2)
    assert_raises(ValueError, eudx_both_directions,
                  seed, 0, qa[..., ::2], ind, sphere.vertices, 0.5, 0.1,
                  1., 1., 2)
    assert_raises(ValueError, eudx_both_directions,
                  seed, 0, qa, ind[..., ::2], sphere.vertices, 0.5, 0.1,
                  1., 1., 2)
    assert_raises(ValueError, eudx_both_directions,
                  seed, 0, qa, ind, sphere.vertices[::2], 0.5, 0.1,
                  1., 1., 2)


if __name__ == '__main__':
    run_module_suite()
