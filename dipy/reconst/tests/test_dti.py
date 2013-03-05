""" Testing DTI

"""

import numpy as np
from nose.tools import (assert_true, assert_equal,
                        assert_almost_equal, assert_raises)
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_
import dipy.reconst.dti as dti
from dipy.reconst.dti import (lower_triangular,
                              from_lower_triangular,
                              color_fa,
                              fractional_anisotropy,
                              trace, mean_diffusivity,
                              radial_diffusivity, axial_diffusivity)
from dipy.reconst.maskedview import MaskedView
from dipy.io.bvectxt import read_bvec_file
from dipy.data import get_data, dsi_voxels, get_sphere
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.reconst.odf import gfa
import dipy.core.gradients as grad
from dipy.sims.voxel import single_tensor
from dipy.core.gradients import gradient_table


def test_TensorModel():
    data, gtab = dsi_voxels()
    dm = dti.TensorModel(gtab, 'LS')
    dtifit = dm.fit(data[0, 0, 0])
    assert_equal(dtifit.fa < 0.5, True)
    dm = dti.TensorModel(gtab, 'WLS')
    dtifit = dm.fit(data[0, 0, 0])
    assert_equal(dtifit.fa < 0.5, True)
    sphere = create_unit_sphere(4)
    assert_equal(len(dtifit.odf(sphere)), len(sphere.vertices))
    assert_almost_equal(dtifit.fa, gfa(dtifit.odf(sphere)), 1)

    # Check that the multivoxel case works:
    dtifit = dm.fit(data)
    # And smoke-test that all these operations return sensibly-shaped arrays:
    assert_equal(dtifit.fa.shape, data.shape[:3])
    assert_equal(dtifit.ad.shape, data.shape[:3])
    assert_equal(dtifit.md.shape, data.shape[:3])
    assert_equal(dtifit.rd.shape, data.shape[:3])
    assert_equal(dtifit.trace.shape, data.shape[:3])
    
    # Make some synthetic data
    b0 = 1000.
    bvecs, bvals = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table_from_bvals_bvecs(bvals, bvecs.T)
    # The first b value is 0., so we take the second one:
    B = bvals[1]
    #Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
    evals = np.array([2., 1., 0.]) / B
    md = evals.mean()
    tensor = from_lower_triangular(D)
    evecs = np.linalg.eigh(tensor)[1]
    #Design Matrix
    X = dti.design_matrix(bvecs, bvals)
    #Signals
    Y = np.exp(np.dot(X,D))
    assert_almost_equal(Y[0], b0)
    Y.shape = (-1,) + Y.shape

    # Test fitting with different methods: #XXX Add NNLS methods!
    for fit_method in ['OLS', 'WLS']:
        tensor_model = dti.TensorModel(gtab,
                                       fit_method=fit_method)

        tensor_fit = tensor_model.fit(Y)
        assert_true(tensor_fit.model is tensor_model)
        assert_equal(tensor_fit.shape, Y.shape[:-1])
        assert_array_almost_equal(tensor_fit.evals[0], evals)

        assert_array_almost_equal(tensor_fit.quadratic_form[0], tensor,
                                  err_msg =\
        "Calculation of tensor from Y does not compare to analytical solution")

        assert_almost_equal(tensor_fit.md[0], md)
        assert_equal(tensor_fit.directions.shape[-2], 1)
        assert_equal(tensor_fit.directions.shape[-1], 3)

    # Test error-handling:
    assert_raises(ValueError,
                  dti.TensorModel,
                  gtab,
                  fit_method='crazy_method')


def test_indexing_on_TensorFit():
    params = np.zeros([2, 3, 4, 12])
    fit = dti.TensorFit(None, params)

    # Should return a TensorFit of appropriate shape
    assert_equal(fit.shape, (2, 3, 4))
    fit1 = fit[0]
    assert_equal(fit1.shape, (3, 4))
    assert_equal(type(fit1), dti.TensorFit)
    fit1 = fit[0, 0, 0]
    assert_equal(fit1.shape, ())
    assert_equal(type(fit1), dti.TensorFit)
    fit1 = fit[[0], slice(None)]
    assert_equal(fit1.shape, (1, 3, 4))
    assert_equal(type(fit1), dti.TensorFit)

    # Should raise an index error if too many indices are passed
    assert_raises(IndexError, fit.__getitem__, (0, 0, 0, 0))


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
    RD = 0.5
    AD = 2.0
    trace = 3

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
    assert_almost_equal(AD, tensor.ad, msg = "Calculation of AD of self diffusion tensor is not adequate")
    assert_almost_equal(RD, tensor.rd, msg = "Calculation of RD of self diffusion tensor is not adequate")
    assert_almost_equal(trace, tensor.trace, msg = "Calculation of trace of self diffusion tensor is not adequate")

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

def test_diffusivities():
    psphere = get_sphere('symmetric362')
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003] ))
    mevecs = [ np.array( [ [1,0,0], [0,1,0], [0,0,1] ] ),
               np.array( [ [0,0,1], [0,1,0], [1,0,0] ] ) ]
    S = single_tensor( gtab, 100, mevals[0], mevecs[0], snr=None )

    dm = dti.TensorModel(gtab, 'LS')
    dmfit = dm.fit(S)
    
    md = mean_diffusivity(dmfit.evals)
    Trace = trace(dmfit.evals)
    rd = radial_diffusivity(dmfit.evals)
    ad = axial_diffusivity(dmfit.evals)
    
    assert_almost_equal(md, (0.0015 + 0.0003 + 0.0001) / 3)
    assert_almost_equal(Trace, (0.0015 + 0.0003 + 0.0001))
    assert_almost_equal(ad, 0.0015)
    assert_almost_equal(rd, (0.0003 + 0.0001) / 2)
    
    
    
def test_color_fa():
    data, gtab = dsi_voxels()
    dm = dti.TensorModel(gtab, 'LS')
    dmfit = dm.fit(data)
    fa = fractional_anisotropy(dmfit.evals)
    cfa = color_fa(fa, dmfit.evecs)

    # evecs should be of shape (fa, 3, 3)
    fa = np.ones((3, 3, 3))
    evecs = np.zeros(fa.shape + (3, 3))
    evecs[..., :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    assert_equal(fa.shape, evecs[..., 0, 0].shape)
    assert_equal((3, 3), evecs.shape[-2:])


    # 3D test case
    fa = np.ones((3, 3, 3))
    evecs = np.zeros(fa.shape + (3, 3))
    evecs[..., :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cfa = color_fa(fa, evecs)
    cfa_truth = np.array([1, 0, 0])
    true_cfa = np.reshape(np.tile(cfa_truth, 27), [3, 3, 3, 3])

    assert_array_equal(cfa, true_cfa)


    # 2D test case
    fa = np.ones((3, 3))
    evecs = np.zeros(fa.shape + (3, 3))
    evecs[..., :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cfa = color_fa(fa, evecs)
    cfa_truth = np.array([1, 0, 0])
    true_cfa = np.reshape(np.tile(cfa_truth, 9), [3, 3, 3])

    assert_array_equal(cfa, true_cfa)


    # 1D test case
    fa = np.ones((3))
    evecs = np.zeros(fa.shape + (3, 3))
    evecs[..., :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cfa = color_fa(fa, evecs)
    cfa_truth = np.array([1, 0, 0])
    true_cfa = np.reshape(np.tile(cfa_truth, 3), [3, 3])

    assert_array_equal(cfa, true_cfa)



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


def test_all_constant():
    """

    """
    bvecs, bvals = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table_from_bvals_bvecs(bvals, bvecs.T)
    fit_methods = ['LS', 'OLS', 'NNLS']
    for fit_method in fit_methods:
        dm = dti.TensorModel(gtab, )
        assert_almost_equal(dm.fit(np.zeros(bvals.shape[0])).fa, 0)
        assert_almost_equal(dm.fit(100 * np.ones(bvals.shape[0])).fa, 0)


def test_mask():
    data, gtab = dsi_voxels()
    dm = dti.TensorModel(gtab, 'LS')
    mask = np.zeros(data.shape[:-1], dtype=bool)
    mask[0, 0, 0] = True
    dtifit = dm.fit(data)
    dtifit_w_mask = dm.fit(data, mask=mask)
    # Without a mask it has some value
    assert_(not np.isnan(dtifit.fa[0, 0, 0]))
    # Where mask is False, evals, evecs and fa should all be 0
    assert_array_equal(dtifit_w_mask.evals[~mask], 0)
    assert_array_equal(dtifit_w_mask.evecs[~mask], 0)
    assert_array_equal(dtifit_w_mask.fa[~mask], 0)
    # Except for the one voxel that was selected by the mask:
    assert_almost_equal(dtifit_w_mask.fa[0, 0, 0], dtifit.fa[0, 0, 0])
