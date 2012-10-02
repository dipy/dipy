""" Testing DTI

"""

import numpy as np
from nose.tools import assert_true, assert_false, \
     assert_equal, assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
from dipy.testing import parametric
import os

import dipy.reconst.dti as dti
from dipy.reconst.dti import lower_triangular, from_lower_triangular
from dipy.reconst.maskedview import MaskedView
import nibabel as nib
from dipy.io.bvectxt import read_bvec_file
from dipy.data import get_data

def test_tensor_scalar_attributes():
    """
    Tests that the tensor class scalar attributes (FA, ADC, etc...) are
    calculating properly.

    """
    ### DEFINING ANALYTICAL VALUES ###
    evals = np.array([2., 1., 0.])
    a = 1. / np.sqrt(2)
    #evec[:,j] is pair with eval[j]
    evecs = np.array([[a, 0, -a], [a, 0, a], [0, 1., 0]])
    D = np.array([[1., 1., 0], [1., 1., 0], [0, 0, 1.]])
    FA = np.sqrt(1./2*(1+4+1)/(1+4+0)) # 0.7745966692414834
    MD = 1.

    ### CALCULATE ESTIMATE VALUES ###
    dummy_data = np.ones((1,10)) #single voxel
    dummy_gtab = np.zeros((10,3))
    dummy_bval = np.zeros((10,))
    tensor = dti.Tensor(dummy_data,dummy_bval,dummy_gtab)
    tensor.model_params = np.r_['-1,2', evals, evecs.ravel()]

    ### TESTS ###
    assert_almost_equal(np.abs(np.dot(evecs[:, 2],
                tensor[0].evecs[:, 2].T)), 1.,
                msg = "Calculation of third eigenvector is not right")
    assert_array_almost_equal(D, tensor[0].D, err_msg = "Recovery of self diffusion tensor from eig not adaquate")
    assert_almost_equal(FA, tensor.fa(), msg = "Calculation of FA of self diffusion tensor is not adequate")
    assert_almost_equal(MD, tensor.md(), msg = "Calculation of MD of self diffusion tensor is not adequate")
    assert_equal(True, tensor.mask.all())

    #assert_equal(m_list.shape, n_list.shape)
    #assert_equal(m_list.ndim, 2)
    #assert_equal(m_list.shape, (45,1))
    #assert_true(np.all(np.abs(m_list) <= n_list))
    #assert_array_equal(n_list % 2, 0)
    #assert_raises(ValueError, qball.sph_harm_ind_list, 1)

def test_fa_of_zero():
    dummy_gtab = np.zeros((10,3))
    dummy_bval = np.zeros((10,))
    ten = dti.Tensor(np.zeros((0,56)), dummy_bval, dummy_gtab)
    ten.model_params = np.zeros(12)
    assert_equal(ten.fa(), 0)
    assert_true(np.isnan(ten.fa(nonans=False)))

def test_WLS_and_LS_fit():
    """
    Tests the WLS and LS fitting functions to see if they returns the correct
    eigenvalues and eigenvectors.

    Uses data/55dir_grad.bvec as the gradient table and 3by3by56.nii
    as the data.

    """

    ### Defining Test Voxel (avoid nibabel dependency) ###

    #Recall: D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,log(S_0)] and D ~ 10^-4 mm^2 /s
    b0 = 1000.
    gtab, bval = read_bvec_file(get_data('55dir_grad.bvec'))
    B = bval[1]
    #Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
    evals = np.array([2., 1., 0.]) / B
    md = evals.mean()
    tensor = from_lower_triangular(D)
    #Design Matrix
    X = dti.design_matrix(gtab, bval)
    #Signals
    Y = np.exp(np.dot(X,D))
    assert_almost_equal(Y[0], b0)
    Y.shape = (-1,) + Y.shape

    ### Testing WLS Fit on Single Voxel ###
    #Estimate tensor from test signals
    tensor_est = dti.Tensor(Y,bval,gtab.T,min_signal=1e-8)
    assert_equal(tensor_est.shape, Y.shape[:-1])
    assert_array_almost_equal(tensor_est.evals[0], evals)
    assert_array_almost_equal(tensor_est.D[0], tensor,err_msg= "Calculation of tensor from Y does not compare to analytical solution")
    assert_almost_equal(tensor_est.md()[0], md)

    #test 0d tensor
    y = Y[0]
    tensor_est = dti.Tensor(y, bval, gtab.T, min_signal=1e-8)
    assert_equal(tensor_est.shape, tuple())
    assert_array_almost_equal(tensor_est.evals, evals)
    assert_array_almost_equal(tensor_est.D, tensor)
    assert_almost_equal(tensor_est.md(), md)
    assert_array_almost_equal(tensor_est.lower_triangular(b0), D)

    tensor_est = dti.Tensor(y, bval, gtab.T, min_signal=1e-8, fit_method='LS')
    assert_equal(tensor_est.shape, tuple())
    assert_array_almost_equal(tensor_est.evals, evals)
    assert_array_almost_equal(tensor_est.D, tensor)
    assert_almost_equal(tensor_est.md(), md)
    assert_array_almost_equal(tensor_est.lower_triangular(b0), D)

def test_masked_array_with_Tensor():
    data = np.ones((2,4,56))
    mask = np.array([[True, False, False, True],
                     [True, False, True, False]])

    gtab, bval = read_bvec_file(get_data('55dir_grad.bvec'))

    tensor = dti.Tensor(data, bval, gtab.T, mask=mask, min_signal=1e-9)
    assert_equal(tensor.shape, (2,4))
    assert_equal(tensor.fa().shape, (2,4))
    assert_equal(tensor.evals.shape, (2,4,3))
    assert_equal(tensor.evecs.shape, (2,4,3,3))
    assert_equal(type(tensor.model_params), MaskedView)
    assert_array_equal(tensor.mask, mask)

    tensor = tensor[0]
    assert_equal(tensor.shape, (4,))
    assert_equal(tensor.fa().shape, (4,))
    assert_equal(tensor.evals.shape, (4,3))
    assert_equal(tensor.evecs.shape, (4,3,3))
    assert_equal(type(tensor.model_params), MaskedView)
    assert_array_equal(tensor.mask, mask[0])

    tensor = tensor[0]
    assert_equal(tensor.shape, tuple())
    assert_equal(tensor.fa().shape, tuple())
    assert_equal(tensor.evals.shape, (3,))
    assert_equal(tensor.evecs.shape, (3,3))
    assert_equal(type(tensor.model_params), np.ndarray)

def test_passing_maskedview():
    data = np.ones((2,4,56))
    mask = np.array([[True, False, False, True],
                     [True, False, True, False]])

    gtab, bval = read_bvec_file(get_data('55dir_grad.bvec'))

    data = data[mask]
    mv = MaskedView(mask, data)

    tensor = dti.Tensor(mv, bval, gtab.T, min_signal=1e-9)
    assert_equal(tensor.shape, (2,4))
    assert_equal(tensor.fa().shape, (2,4))
    assert_equal(tensor.evals.shape, (2,4,3))
    assert_equal(tensor.evecs.shape, (2,4,3,3))
    assert_equal(type(tensor.model_params), MaskedView)
    assert_array_equal(tensor.mask, mask)

    tensor = tensor[0]
    assert_equal(tensor.shape, (4,))
    assert_equal(tensor.fa().shape, (4,))
    assert_equal(tensor.evals.shape, (4,3))
    assert_equal(tensor.evecs.shape, (4,3,3))
    assert_equal(type(tensor.model_params), MaskedView)
    assert_array_equal(tensor.mask, mask[0])

    tensor = tensor[0]
    assert_equal(tensor.shape, tuple())
    assert_equal(tensor.fa().shape, tuple())
    assert_equal(tensor.evals.shape, (3,))
    assert_equal(tensor.evecs.shape, (3,3))
    assert_equal(type(tensor.model_params), np.ndarray)

def test_init():
    data = np.ones((2,4,56))
    mask = np.ones((2,4),'bool')

    gtab, bval = read_bvec_file(get_data('55dir_grad.bvec'))
    tensor = dti.Tensor(data, bval, gtab.T, mask, thresh=0)
    mask[:] = False
    assert_raises(ValueError, dti.Tensor, data, bval, gtab.T, mask)
    assert_raises(ValueError, dti.Tensor, data, bval, gtab.T,
                        min_signal=-1)
    assert_raises(ValueError, dti.Tensor, data, bval, gtab.T, thresh=1)
    assert_raises(ValueError, dti.Tensor, data, bval, gtab.T,
                        fit_method='s')
    assert_raises(ValueError, dti.Tensor, data, bval, gtab.T,
                        fit_method=0)

def test_lower_triangular():
    tensor = np.arange(9).reshape((3,3))
    D = lower_triangular(tensor)
    assert_array_equal(D, [0, 3, 4, 6, 7, 8])
    D = lower_triangular(tensor, 1)
    assert_array_equal(D, [0, 3, 4, 6, 7, 8, 0])
    assert_raises(ValueError, lower_triangular, np.zeros((2, 3)))
    shape = (4,5,6)
    many_tensors = np.empty(shape + (3,3))
    many_tensors[:] = tensor
    result = np.empty(shape + (6,))
    result[:] = [0, 3, 4, 6, 7, 8]
    D = lower_triangular(many_tensors)
    assert_array_equal(D, result)
    D = lower_triangular(many_tensors, 1)
    result = np.empty(shape + (7,))
    result[:] = [0, 3, 4, 6, 7, 8, 0]
    assert_array_equal(D, result)

def test_from_lower_triangular():
    result = np.array([[0, 1, 3],
                       [1, 2, 4],
                       [3, 4, 5]])
    D = np.arange(7)
    tensor = from_lower_triangular(D)
    assert_array_equal(tensor, result)
    result = result * np.ones((5, 4, 1, 1))
    D = D * np.ones((5, 4, 1))
    tensor = from_lower_triangular(D)
    assert_array_equal(tensor, result)
     
