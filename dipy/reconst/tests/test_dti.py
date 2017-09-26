""" Testing DTI

"""
from __future__ import division, print_function, absolute_import

import numpy as np
from nose.tools import (assert_true, assert_equal,
                        assert_almost_equal, assert_raises)
import numpy.testing as npt
from numpy.testing import (assert_array_equal, assert_array_almost_equal,
                           assert_)
import nibabel as nib

import scipy.optimize as opt

import dipy.reconst.dti as dti
from dipy.reconst.dti import (axial_diffusivity, color_fa,
                              fractional_anisotropy, from_lower_triangular,
                              geodesic_anisotropy, lower_triangular,
                              mean_diffusivity, radial_diffusivity,
                              TensorModel, trace, linearity, planarity,
                              sphericity, decompose_tensor,
                              _decompose_tensor_nan)

from dipy.io.bvectxt import read_bvec_file
from dipy.data import get_data, dsi_voxels, get_sphere

from dipy.core.subdivide_octahedron import create_unit_sphere
import dipy.core.gradients as grad
import dipy.core.sphere as dps

from dipy.sims.voxel import single_tensor


def test_roll_evals():
    """

    """
    # Just making sure this never passes through
    weird_evals = np.array([1, 0.5])
    npt.assert_raises(ValueError, dti._roll_evals, weird_evals)


def test_tensor_algebra():
    """
    Test that the computation of tensor determinant and norm is correct
    """
    test_arr = np.random.rand(10, 3, 3)
    t_det = dti.determinant(test_arr)
    t_norm = dti.norm(test_arr)
    for i, x in enumerate(test_arr):
        assert_almost_equal(np.linalg.det(x), t_det[i])
        assert_almost_equal(np.linalg.norm(x), t_norm[i])


def test_odf_with_zeros():
    fdata, fbval, fbvec = get_data('small_25')
    gtab = grad.gradient_table(fbval, fbvec)
    data = nib.load(fdata).get_data()
    dm = dti.TensorModel(gtab)
    df = dm.fit(data)
    df.evals[0, 0, 0] = np.array([0, 0, 0])
    sphere = create_unit_sphere(4)
    odf = df.odf(sphere)
    npt.assert_equal(odf[0, 0, 0], np.zeros(sphere.vertices.shape[0]))


def test_tensor_model():
    fdata, fbval, fbvec = get_data('small_25')
    data1 = nib.load(fdata).get_data()
    gtab1 = grad.gradient_table(fbval, fbvec)
    data2, gtab2 = dsi_voxels()
    for data, gtab in zip([data1, data2], [gtab1, gtab2]):
        dm = dti.TensorModel(gtab, 'LS')
        dtifit = dm.fit(data[0, 0, 0])
        assert_equal(dtifit.fa < 0.9, True)
        dm = dti.TensorModel(gtab, 'WLS')
        dtifit = dm.fit(data[0, 0, 0])
        assert_equal(dtifit.fa < 0.9, True)
        assert_equal(dtifit.fa > 0, True)
        sphere = create_unit_sphere(4)
        assert_equal(len(dtifit.odf(sphere)), len(sphere.vertices))
        # Check that the multivoxel case works:
        dtifit = dm.fit(data)

        # Check that it works on signal that has already been normalized to S0:
        dm_to_relative = dti.TensorModel(gtab)
        if np.any(gtab.b0s_mask):
            relative_data = (data[0, 0, 0]/np.mean(data[0, 0, 0,
                                                        gtab.b0s_mask]))

            dtifit_to_relative = dm_to_relative.fit(relative_data)
            npt.assert_almost_equal(dtifit.fa[0, 0, 0], dtifit_to_relative.fa,
                                    decimal=3)

    # And smoke-test that all these operations return sensibly-shaped arrays:
    assert_equal(dtifit.fa.shape, data.shape[:3])
    assert_equal(dtifit.ad.shape, data.shape[:3])
    assert_equal(dtifit.md.shape, data.shape[:3])
    assert_equal(dtifit.rd.shape, data.shape[:3])
    assert_equal(dtifit.trace.shape, data.shape[:3])
    assert_equal(dtifit.mode.shape, data.shape[:3])
    assert_equal(dtifit.linearity.shape, data.shape[:3])
    assert_equal(dtifit.planarity.shape, data.shape[:3])
    assert_equal(dtifit.sphericity.shape, data.shape[:3])

    # Test for the shape of the mask
    assert_raises(ValueError, dm.fit, np.ones((10, 10, 3)), np.ones((3, 3)))

    # Make some synthetic data
    b0 = 1000.
    bvecs, bvals = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table_from_bvals_bvecs(bvals, bvecs.T)
    # The first b value is 0., so we take the second one:
    B = bvals[1]
    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
    evals = np.array([2., 1., 0.]) / B
    md = evals.mean()
    tensor = from_lower_triangular(D)
    A_squiggle = tensor - (1 / 3.0) * np.trace(tensor) * np.eye(3)
    mode = (3 * np.sqrt(6) * np.linalg.det(A_squiggle /
            np.linalg.norm(A_squiggle)))
    evals_eigh, evecs_eigh = np.linalg.eigh(tensor)
    # Sort according to eigen-value from large to small:
    evecs = evecs_eigh[:, np.argsort(evals_eigh)[::-1]]
    # Check that eigenvalues and eigenvectors are properly sorted through
    # that previous operation:
    for i in range(3):
        assert_array_almost_equal(np.dot(tensor, evecs[:, i]),
                                  evals[i] * evecs[:, i])
    # Design Matrix
    X = dti.design_matrix(gtab)
    # Signals
    Y = np.exp(np.dot(X, D))
    assert_almost_equal(Y[0], b0)
    Y.shape = (-1,) + Y.shape

    # Test fitting with different methods:
    for fit_method in ['OLS', 'WLS', 'NLLS']:
        tensor_model = dti.TensorModel(gtab,
                                       fit_method=fit_method,
                                       return_S0_hat=True)

        tensor_fit = tensor_model.fit(Y)
        assert_true(tensor_fit.model is tensor_model)
        assert_equal(tensor_fit.shape, Y.shape[:-1])
        assert_array_almost_equal(tensor_fit.evals[0], evals)
        assert_array_almost_equal(tensor_fit.S0_hat, b0, decimal=3)
        # Test that the eigenvectors are correct, one-by-one:
        for i in range(3):
            # Eigenvectors have intrinsic sign ambiguity
            # (see
            # http://prod.sandia.gov/techlib/access-control.cgi/2007/076422.pdf)
            # so we need to allow for sign flips. One of the following should
            # always be true:
            assert_(
                    np.all(np.abs(tensor_fit.evecs[0][:, i] -
                                  evecs[:, i]) < 10e-6) or
                    np.all(np.abs(-tensor_fit.evecs[0][:, i] -
                                  evecs[:, i]) < 10e-6))
            # We set a fixed tolerance of 10e-6, similar to array_almost_equal

        err_msg = "Calculation of tensor from Y does not compare to "
        err_msg += "analytical solution"
        assert_array_almost_equal(tensor_fit.quadratic_form[0], tensor,
                                  err_msg=err_msg)

        assert_almost_equal(tensor_fit.md[0], md)
        assert_array_almost_equal(tensor_fit.mode, mode, decimal=5)
        assert_equal(tensor_fit.directions.shape[-2], 1)
        assert_equal(tensor_fit.directions.shape[-1], 3)

    # Test error-handling:
    assert_raises(ValueError,
                  dti.TensorModel,
                  gtab,
                  fit_method='crazy_method')

    # Test custom fit tensor method
    try:
        model = dti.TensorModel(gtab, fit_method=lambda *args, **kwargs: 42)
        fit = model.fit_method()
    except Exception as exc:
        assert False, "TensorModel should accept custom fit methods: %s" % exc
    assert fit == 42, "Custom fit method for TensorModel returned %s." % fit

    # Test multi-voxel data
    data = np.zeros((3, Y.shape[1]))
    # Normal voxel
    data[0] = Y
    # High diffusion voxel, all diffusing weighted signal equal to zero
    data[1, gtab.b0s_mask] = b0
    data[1, ~gtab.b0s_mask] = 0
    # Masked voxel, all data set to zero
    data[2] = 0.

    tensor_model = dti.TensorModel(gtab)
    fit = tensor_model.fit(data)
    assert_array_almost_equal(fit[0].evals, evals)

    # Return S0_test
    tensor_model = dti.TensorModel(gtab, return_S0_hat=True)
    fit = tensor_model.fit(data)
    assert_array_almost_equal(fit[0].evals, evals)
    assert_array_almost_equal(fit[0].S0_hat, b0)

    # Evals should be high for high diffusion voxel
    assert_(all(fit[1].evals > evals[0] * .9))

    # Evals should be zero where data is masked
    assert_array_almost_equal(fit[2].evals, 0.)


def test_indexing_on_tensor_fit():
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


def test_fa_of_zero():
    evals = np.zeros((4, 3))
    fa = fractional_anisotropy(evals)
    assert_array_equal(fa, 0)


def test_ga_of_zero():
    evals = np.zeros((4, 3))
    ga = geodesic_anisotropy(evals)
    assert_array_equal(ga, 0)


def test_diffusivities():
    psphere = get_sphere('symmetric362')
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = grad.gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
    S = single_tensor(gtab, 100, mevals[0], mevecs[0], snr=None)

    dm = dti.TensorModel(gtab, 'LS')
    dmfit = dm.fit(S)

    md = mean_diffusivity(dmfit.evals)
    Trace = trace(dmfit.evals)
    rd = radial_diffusivity(dmfit.evals)
    ad = axial_diffusivity(dmfit.evals)
    lin = linearity(dmfit.evals)
    plan = planarity(dmfit.evals)
    spher = sphericity(dmfit.evals)

    assert_almost_equal(md, (0.0015 + 0.0003 + 0.0001) / 3)
    assert_almost_equal(Trace, (0.0015 + 0.0003 + 0.0001))
    assert_almost_equal(ad, 0.0015)
    assert_almost_equal(rd, (0.0003 + 0.0001) / 2)
    assert_almost_equal(lin, (0.0015 - 0.0003)/Trace)
    assert_almost_equal(plan, 2 * (0.0003 - 0.0001)/Trace)
    assert_almost_equal(spher, (3 * 0.0001)/Trace)


def test_color_fa():
    data, gtab = dsi_voxels()
    dm = dti.TensorModel(gtab, 'LS')
    dmfit = dm.fit(data)
    fa = fractional_anisotropy(dmfit.evals)
    cfa = color_fa(fa, dmfit.evecs)

    fa = np.ones((3, 3, 3))
    # evecs should be of shape (fa, 3, 3)
    evecs = np.zeros(fa.shape + (3, 2))
    npt.assert_raises(ValueError, color_fa, fa, evecs)

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


def test_wls_and_ls_fit():
    """
    Tests the WLS and LS fitting functions to see if they returns the correct
    eigenvalues and eigenvectors.

    Uses data/55dir_grad.bvec as the gradient table and 3by3by56.nii
    as the data.

    """

    # Defining Test Voxel (avoid nibabel dependency) ###

    # Recall: D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,log(S_0)] and D ~ 10^-4 mm^2 /s
    b0 = 1000.
    bvec, bval = read_bvec_file(get_data('55dir_grad.bvec'))
    B = bval[1]
    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
    evals = np.array([2., 1., 0.]) / B
    md = evals.mean()
    tensor = from_lower_triangular(D)
    # Design Matrix
    gtab = grad.gradient_table(bval, bvec)
    X = dti.design_matrix(gtab)
    # Signals
    Y = np.exp(np.dot(X, D))
    assert_almost_equal(Y[0], b0)
    Y.shape = (-1,) + Y.shape

    # Testing WLS Fit on Single Voxel
    # If you do something wonky (passing min_signal<0), you should get an
    # error:
    npt.assert_raises(ValueError, TensorModel, gtab, fit_method='WLS',
                      min_signal=-1)

    # Estimate tensor from test signals
    model = TensorModel(gtab, fit_method='WLS', return_S0_hat=True)
    tensor_est = model.fit(Y)
    assert_equal(tensor_est.shape, Y.shape[:-1])
    assert_array_almost_equal(tensor_est.evals[0], evals)
    assert_array_almost_equal(tensor_est.quadratic_form[0], tensor,
                              err_msg="Calculation of tensor from Y does not "
                                      "compare to analytical solution")
    assert_almost_equal(tensor_est.md[0], md)
    assert_array_almost_equal(tensor_est.S0_hat[0], b0, decimal=3)

    # Test that we can fit a single voxel's worth of data (a 1d array)
    y = Y[0]
    tensor_est = model.fit(y)
    assert_equal(tensor_est.shape, tuple())
    assert_array_almost_equal(tensor_est.evals, evals)
    assert_array_almost_equal(tensor_est.quadratic_form, tensor)
    assert_almost_equal(tensor_est.md, md)
    assert_array_almost_equal(tensor_est.lower_triangular(b0), D)

    # Test using fit_method='LS'
    model = TensorModel(gtab, fit_method='LS')
    tensor_est = model.fit(y)
    assert_equal(tensor_est.shape, tuple())
    assert_array_almost_equal(tensor_est.evals, evals)
    assert_array_almost_equal(tensor_est.quadratic_form, tensor)
    assert_almost_equal(tensor_est.md, md)
    assert_array_almost_equal(tensor_est.lower_triangular(b0), D)
    assert_array_almost_equal(tensor_est.linearity, linearity(evals))
    assert_array_almost_equal(tensor_est.planarity, planarity(evals))
    assert_array_almost_equal(tensor_est.sphericity, sphericity(evals))


def test_masked_array_with_tensor():
    data = np.ones((2, 4, 56))
    mask = np.array([[True, False, False, True],
                     [True, False, True, False]])

    bvec, bval = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table_from_bvals_bvecs(bval, bvec.T)

    tensor_model = TensorModel(gtab)
    tensor = tensor_model.fit(data, mask=mask)
    assert_equal(tensor.shape, (2, 4))
    assert_equal(tensor.fa.shape, (2, 4))
    assert_equal(tensor.evals.shape, (2, 4, 3))
    assert_equal(tensor.evecs.shape, (2, 4, 3, 3))

    tensor = tensor[0]
    assert_equal(tensor.shape, (4,))
    assert_equal(tensor.fa.shape, (4,))
    assert_equal(tensor.evals.shape, (4, 3))
    assert_equal(tensor.evecs.shape, (4, 3, 3))

    tensor = tensor[0]
    assert_equal(tensor.shape, tuple())
    assert_equal(tensor.fa.shape, tuple())
    assert_equal(tensor.evals.shape, (3,))
    assert_equal(tensor.evecs.shape, (3, 3))
    assert_equal(type(tensor.model_params), np.ndarray)


def test_fit_method_error():
    bvec, bval = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table_from_bvals_bvecs(bval, bvec.T)

    # This should work (smoke-testing!):
    TensorModel(gtab, fit_method='WLS')

    # This should raise an error because there is no such fit_method
    assert_raises(ValueError, TensorModel, gtab, min_signal=1e-9,
                  fit_method='s')


def test_lower_triangular():
    tensor = np.arange(9).reshape((3, 3))
    D = lower_triangular(tensor)
    assert_array_equal(D, [0, 3, 4, 6, 7, 8])
    D = lower_triangular(tensor, 1)
    assert_array_equal(D, [0, 3, 4, 6, 7, 8, 0])
    assert_raises(ValueError, lower_triangular, np.zeros((2, 3)))
    shape = (4, 5, 6)
    many_tensors = np.empty(shape + (3, 3))
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
    bvecs, bvals = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table_from_bvals_bvecs(bvals, bvecs.T)
    fit_methods = ['LS', 'OLS', 'NNLS', 'RESTORE']
    for fit_method in fit_methods:
        dm = dti.TensorModel(gtab)
        assert_almost_equal(dm.fit(100 * np.ones(bvals.shape[0])).fa, 0)
        # Doesn't matter if the signal is smaller than 1:
        assert_almost_equal(dm.fit(0.4 * np.ones(bvals.shape[0])).fa, 0)


def test_all_zeros():
    bvecs, bvals = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table_from_bvals_bvecs(bvals, bvecs.T)
    fit_methods = ['LS', 'OLS', 'NNLS', 'RESTORE']
    for fit_method in fit_methods:
        dm = dti.TensorModel(gtab)
        assert_array_almost_equal(dm.fit(np.zeros(bvals.shape[0])).evals, 0)


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

    # Test with returning S0_hat
    dm = dti.TensorModel(gtab, 'LS', return_S0_hat=True)
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
    assert_array_equal(dtifit_w_mask.S0_hat[~mask], 0)
    # Except for the one voxel that was selected by the mask:
    assert_almost_equal(dtifit_w_mask.fa[0, 0, 0], dtifit.fa[0, 0, 0])
    assert_almost_equal(dtifit_w_mask.S0_hat[0, 0, 0], dtifit.S0_hat[0, 0, 0])


def test_nnls_jacobian_fucn():
    b0 = 1000.
    bvecs, bval = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table(bval, bvecs)
    B = bval[1]

    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B

    # Design Matrix
    X = dti.design_matrix(gtab)

    # Signals
    Y = np.exp(np.dot(X, D))

    # Test Jacobian at D
    args = [X, Y]
    analytical = dti._nlls_jacobian_func(D, *args)
    for i in range(len(X)):
        args = [X[i], Y[i]]
        approx = opt.approx_fprime(D, dti._nlls_err_func, 1e-8, *args)
        assert_true(np.allclose(approx, analytical[i]))

    # Test Jacobian at zero
    D = np.zeros_like(D)
    args = [X, Y]
    analytical = dti._nlls_jacobian_func(D, *args)
    for i in range(len(X)):
        args = [X[i], Y[i]]
        approx = opt.approx_fprime(D, dti._nlls_err_func, 1e-8, *args)
        assert_true(np.allclose(approx, analytical[i]))


def test_nlls_fit_tensor():
    """
    Test the implementation of NLLS and RESTORE
    """

    b0 = 1000.
    bvecs, bval = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table(bval, bvecs)
    B = bval[1]

    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
    evals = np.array([2., 1., 0.]) / B
    md = evals.mean()
    tensor = from_lower_triangular(D)

    # Design Matrix
    X = dti.design_matrix(gtab)

    # Signals
    Y = np.exp(np.dot(X, D))
    Y.shape = (-1,) + Y.shape

    # Estimate tensor from test signals and compare against expected result
    # using non-linear least squares:
    tensor_model = dti.TensorModel(gtab, fit_method='NLLS')
    tensor_est = tensor_model.fit(Y)
    assert_equal(tensor_est.shape, Y.shape[:-1])
    assert_array_almost_equal(tensor_est.evals[0], evals)
    assert_array_almost_equal(tensor_est.quadratic_form[0], tensor)
    assert_almost_equal(tensor_est.md[0], md)

    # You can also do this without the Jacobian (though it's slower):
    tensor_model = dti.TensorModel(gtab, fit_method='NLLS', jac=False)
    tensor_est = tensor_model.fit(Y)
    assert_equal(tensor_est.shape, Y.shape[:-1])
    assert_array_almost_equal(tensor_est.evals[0], evals)
    assert_array_almost_equal(tensor_est.quadratic_form[0], tensor)
    assert_almost_equal(tensor_est.md[0], md)

    # Using the gmm weighting scheme:
    tensor_model = dti.TensorModel(gtab, fit_method='NLLS', weighting='gmm')
    tensor_est = tensor_model.fit(Y)
    assert_equal(tensor_est.shape, Y.shape[:-1])
    assert_array_almost_equal(tensor_est.evals[0], evals)
    assert_array_almost_equal(tensor_est.quadratic_form[0], tensor)
    assert_almost_equal(tensor_est.md[0], md)

    # If you use sigma weighting, you'd better provide a sigma:
    tensor_model = dti.TensorModel(gtab, fit_method='NLLS', weighting='sigma')
    npt.assert_raises(ValueError, tensor_model.fit, Y)

    # Use NLLS with some actual 4D data:
    data, bvals, bvecs = get_data('small_25')
    gtab = grad.gradient_table(bvals, bvecs)
    tm1 = dti.TensorModel(gtab, fit_method='NLLS')
    dd = nib.load(data).get_data()
    tf1 = tm1.fit(dd)
    tm2 = dti.TensorModel(gtab)
    tf2 = tm2.fit(dd)

    assert_array_almost_equal(tf1.fa, tf2.fa, decimal=1)


def test_restore():
    """
    Test the implementation of the RESTORE algorithm
    """
    b0 = 1000.
    bvecs, bval = read_bvec_file(get_data('55dir_grad.bvec'))
    gtab = grad.gradient_table(bval, bvecs)
    B = bval[1]

    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
    evals = np.array([2., 1., 0.]) / B
    tensor = from_lower_triangular(D)

    # Design Matrix
    X = dti.design_matrix(gtab)

    # Signals
    Y = np.exp(np.dot(X, D))
    Y.shape = (-1,) + Y.shape
    for drop_this in range(1, Y.shape[-1]):
        for jac in [True, False]:
            # RESTORE estimates should be robust to dropping
            this_y = Y.copy()
            this_y[:, drop_this] = 1.0
            for sigma in [67.0, np.ones(this_y.shape[-1]) * 67.0]:
                tensor_model = dti.TensorModel(gtab, fit_method='restore',
                                               jac=jac,
                                               sigma=67.0)

                tensor_est = tensor_model.fit(this_y)
                assert_array_almost_equal(tensor_est.evals[0], evals,
                                          decimal=3)
                assert_array_almost_equal(tensor_est.quadratic_form[0], tensor,
                                          decimal=3)

    # If sigma is very small, it still needs to work:
    tensor_model = dti.TensorModel(gtab, fit_method='restore', sigma=0.0001)
    tensor_model.fit(Y.copy())

    # Test return_S0_hat
    tensor_model = dti.TensorModel(gtab, fit_method='restore', sigma=0.0001,
                                   return_S0_hat=True)
    tmf = tensor_model.fit(Y.copy())
    assert_almost_equal(tmf[0].S0_hat, b0)


def test_adc():
    """
    Test the implementation of the calculation of apparent diffusion
    coefficient
    """
    data, gtab = dsi_voxels()
    dm = dti.TensorModel(gtab, 'LS')
    mask = np.zeros(data.shape[:-1], dtype=bool)
    mask[0, 0, 0] = True
    dtifit = dm.fit(data)
    # The ADC in the principal diffusion direction should be equal to the AD in
    # each voxel:

    pdd0 = dtifit.evecs[0, 0, 0, 0]
    sphere_pdd0 = dps.Sphere(x=pdd0[0], y=pdd0[1], z=pdd0[2])
    assert_array_almost_equal(dtifit.adc(sphere_pdd0)[0, 0, 0],
                              dtifit.ad[0, 0, 0], decimal=5)

    # Test that it works for cases in which the data is 1D
    dtifit = dm.fit(data[0, 0, 0])
    sphere_pdd0 = dps.Sphere(x=pdd0[0], y=pdd0[1], z=pdd0[2])
    assert_array_almost_equal(dtifit.adc(sphere_pdd0),
                              dtifit.ad, decimal=5)


def test_predict():
    """
    Test model prediction API
    """
    psphere = get_sphere('symmetric362')
    bvecs = np.concatenate(([[1, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = grad.gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
    S = single_tensor(gtab, 100, mevals[0], mevecs[0], snr=None)

    dm = dti.TensorModel(gtab, 'LS', return_S0_hat=True)
    dmfit = dm.fit(S)
    assert_array_almost_equal(dmfit.predict(gtab, S0=100), S)
    assert_array_almost_equal(dmfit.predict(gtab), S)
    assert_array_almost_equal(dm.predict(dmfit.model_params, S0=100), S)

    fdata, fbvals, fbvecs = get_data()
    data = nib.load(fdata).get_data()
    # Make the data cube a bit larger:
    data = np.tile(data.T, 2).T
    gtab = grad.gradient_table(fbvals, fbvecs)
    dtim = dti.TensorModel(gtab)
    dtif = dtim.fit(data)
    S0 = np.mean(data[..., gtab.b0s_mask], -1)
    p = dtif.predict(gtab, S0)
    assert_equal(p.shape, data.shape)
    # Predict using S0_hat:
    dtim = dti.TensorModel(gtab, return_S0_hat=True)
    dtif = dtim.fit(data)
    p = dtif.predict(gtab)
    assert_equal(p.shape, data.shape)
    p = dtif.predict(gtab, S0)
    assert_equal(p.shape, data.shape)

    # Test iter_fit_tensor with S0_hat
    dtim = dti.TensorModel(gtab, step=2, return_S0_hat=True)
    dtif = dtim.fit(data)
    S0 = np.mean(data[..., gtab.b0s_mask], -1)
    p = dtif.predict(gtab, S0)
    assert_equal(p.shape, data.shape)

    # Use a smaller step in predicting:

    dtim = dti.TensorModel(gtab, step=2)
    dtif = dtim.fit(data)
    S0 = np.mean(data[..., gtab.b0s_mask], -1)
    p = dtif.predict(gtab, S0)
    assert_equal(p.shape, data.shape)
    # And with a scalar S0:
    S0 = 1
    p = dtif.predict(gtab, S0)
    assert_equal(p.shape, data.shape)
    # Assign the step through kwarg:
    p = dtif.predict(gtab, S0, step=1)
    assert_equal(p.shape, data.shape)
    # And without S0:
    p = dtif.predict(gtab, step=1)
    assert_equal(p.shape, data.shape)

def test_eig_from_lo_tri():
    psphere = get_sphere('symmetric362')
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = grad.gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
    S = np.array([[single_tensor(gtab, 100, mevals[0], mevecs[0], snr=None),
                   single_tensor(gtab, 100, mevals[0], mevecs[0], snr=None)]])

    dm = dti.TensorModel(gtab, 'LS')
    dmfit = dm.fit(S)

    lo_tri = lower_triangular(dmfit.quadratic_form)
    assert_array_almost_equal(dti.eig_from_lo_tri(lo_tri), dmfit.model_params)

def test_min_signal_alone():
    fdata, fbvals, fbvecs = get_data()
    data = nib.load(fdata).get_data()
    gtab = grad.gradient_table(fbvals, fbvecs)

    idx = tuple(np.array(np.where(data == np.min(data)))[:-1, 0])
    ten_model = dti.TensorModel(gtab)
    fit_alone = ten_model.fit(data[idx])
    fit_together = ten_model.fit(data)
    npt.assert_array_almost_equal(fit_together.model_params[idx], fit_alone.model_params, decimal=12)

def test_decompose_tensor_nan():
    D_fine = np.array([1.7e-3, 0.0, 0.3e-3, 0.0, 0.0, 0.2e-3])
    D_alter = np.array([1.6e-3, 0.0, 0.4e-3, 0.0, 0.0, 0.3e-3])
    D_nan = np.nan * np.ones(6)

    lref, vref = decompose_tensor(from_lower_triangular(D_fine))
    lfine, vfine = _decompose_tensor_nan(from_lower_triangular(D_fine),
                                         from_lower_triangular(D_alter))
    assert_array_almost_equal(lfine, np.array([1.7e-3, 0.3e-3, 0.2e-3]))
    assert_array_almost_equal(vfine, vref)
    
    lref, vref = decompose_tensor(from_lower_triangular(D_alter))
    lalter, valter = _decompose_tensor_nan(from_lower_triangular(D_nan),
                                           from_lower_triangular(D_alter))
    assert_array_almost_equal(lalter, np.array([1.6e-3, 0.4e-3, 0.3e-3]))
    assert_array_almost_equal(valter, vref)
