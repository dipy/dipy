import os
import numpy as np
import numpy.testing

from dipy.data import get_data, get_sphere
from dipy.core.gradients import gradient_table
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.reconst.dti import TensorModel, quantize_evecs
from dipy.tracking import utils
from dipy.tracking.eudx import EuDX
from dipy.tracking.propspeed import ndarray_offset
from dipy.tracking.metrics import length
from dipy.tracking.propspeed import map_coordinates_trilinear_iso

import nibabel as ni

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises, assert_almost_equal

from numpy.testing import (assert_array_equal,
                           assert_array_almost_equal,
                           run_module_suite)


def test_trilinear_interp_cubic_voxels():
    A=np.ones((17,17,17))
    B=np.zeros(3)
    strides=np.array(A.strides, np.intp)
    A[7,7,7]=2
    points=np.array([[0,0,0],[7.,7.5,7.],[3.5,3.5,3.5]])
    map_coordinates_trilinear_iso(A,points,strides,3,B)
    assert_array_almost_equal(B,np.array([ 1. ,  1.5,  1. ]))


def test_eudx_further():
    """ Cause we love testin.. ;-)
    """

    fimg,fbvals,fbvecs=get_data('small_101D')

    img=ni.load(fimg)
    affine=img.get_affine()
    data=img.get_data()
    gtab = gradient_table(fbvals, fbvecs)
    tensor_model = TensorModel(gtab)
    ten = tensor_model.fit(data)
    x,y,z=data.shape[:3]
    seeds=np.zeros((10**4,3))
    for i in range(10**4):
        rx=(x-1)*np.random.rand()
        ry=(y-1)*np.random.rand()
        rz=(z-1)*np.random.rand()
        seeds[i]=np.ascontiguousarray(np.array([rx,ry,rz]),dtype=np.float64)

    sphere = get_sphere('symmetric724')

    ind = quantize_evecs(ten.evecs)
    eu=EuDX(a=ten.fa, ind=ind, seeds=seeds,
            odf_vertices=sphere.vertices, a_low=.2)
    T=[e for e in eu]

    #check that there are no negative elements
    for t in T:
        assert_equal(np.sum(t.ravel()<0),0)

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
    affine = img.get_affine()
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
    track = list(eu)

    seed = [-1., 1000000., 1000000.]
    eu = EuDX(a=ten.fa, ind=ind, seeds=[seed],
              odf_vertices=sphere.vertices, a_low=.2)
    assert_raises(ValueError, list, eu)


if __name__ == '__main__':
    run_module_suite()
