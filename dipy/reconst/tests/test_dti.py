"""Testing DTI."""

import warnings

import numpy as np
import numpy.testing as npt
import scipy.optimize as opt

import dipy.core.gradients as grad
import dipy.core.sphere as dps
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.data import dsi_voxels, get_fnames, get_sphere
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data
import dipy.reconst.dti as dti
from dipy.reconst.dti import (
    MIN_POSITIVE_SIGNAL,
    TensorModel,
    _decompose_tensor_nan,
    axial_diffusivity,
    color_fa,
    decompose_tensor,
    fractional_anisotropy,
    from_lower_triangular,
    geodesic_anisotropy,
    linearity,
    lower_triangular,
    mean_diffusivity,
    mode,
    ols_fit_tensor,
    ols_resort_msg,
    planarity,
    radial_diffusivity,
    sphericity,
    trace,
    wls_fit_tensor,
)
from dipy.reconst.weights_method import (
    weights_method_nlls_m_est,
    weights_method_wls_m_est,
)
from dipy.sims.voxel import single_tensor
from dipy.testing.decorators import set_random_number_generator


def test_roll_evals():
    # Just making sure this never passes through
    weird_evals = np.array([1, 0.5])
    npt.assert_raises(ValueError, dti._roll_evals, weird_evals)


@set_random_number_generator()
def test_tensor_algebra(rng):
    # Test that the computation of tensor determinant and norm is correct
    test_arr = rng.random((10, 3, 3))
    t_det = dti.determinant(test_arr)
    t_norm = dti.norm(test_arr)
    for i, x in enumerate(test_arr):
        npt.assert_almost_equal(np.linalg.det(x), t_det[i])
        npt.assert_almost_equal(np.linalg.norm(x), t_norm[i])


def test_odf_with_zeros():
    fdata, fbval, fbvec = get_fnames(name="small_25")
    gtab = grad.gradient_table(fbval, bvecs=fbvec)
    data = load_nifti_data(fdata)
    dm = dti.TensorModel(gtab)
    df = dm.fit(data)
    df.evals[0, 0, 0] = np.array([0, 0, 0])
    sphere = create_unit_sphere(recursion_level=4)
    odf = df.odf(sphere)
    npt.assert_equal(odf[0, 0, 0], np.zeros(sphere.vertices.shape[0]))


def test_mode_with_isotropic():
    # mode involves a division by norm, so may be problematic for isotropic
    # voxels. In the above test, 4 voxels are produced with isotropic tensors
    # in indexes [0, 0] and [0, 1] in which norm should give mode 0. Voxels
    # with indexes [1, 0] and [1, 1] should give mode of 1 and -1 respectively.
    q_form = np.zeros((2, 2, 3, 3))
    q_form[0, 1, 0, 0] = 1
    q_form[0, 1, 1, 1] = 1
    q_form[0, 1, 2, 2] = 1
    q_form[1, 0, 0, 0] = 1
    q_form[1, 0, 1, 1] = 1
    q_form[1, 0, 2, 2] = 2
    q_form[1, 1, 0, 0] = 1
    q_form[1, 1, 1, 1] = 2
    q_form[1, 1, 2, 2] = 2
    npt.assert_array_almost_equal(mode(q_form), np.array([[0, 0], [1, -1]]))


def test_tensor_model():
    fdata, fbval, fbvec = get_fnames(name="small_25")
    data1 = load_nifti_data(fdata)
    gtab1 = grad.gradient_table(fbval, bvecs=fbvec)
    data2, gtab2 = dsi_voxels()
    for data, gtab in zip([data1, data2], [gtab1, gtab2]):
        dm = dti.TensorModel(gtab, fit_method="LS")
        dtifit = dm.fit(data[0, 0, 0])
        npt.assert_equal(dtifit.fa < 0.9, True)
        dm = dti.TensorModel(gtab, fit_method="WLS")
        dtifit = dm.fit(data[0, 0, 0])
        npt.assert_equal(dtifit.fa < 0.9, True)
        npt.assert_equal(dtifit.fa > 0, True)
        sphere = create_unit_sphere(recursion_level=4)
        npt.assert_equal(len(dtifit.odf(sphere)), len(sphere.vertices))
        # Check that the multivoxel case works:
        dtifit = dm.fit(data)

        # Check that it works on signal that has already been normalized to S0:
        dm_to_relative = dti.TensorModel(gtab)
        if np.any(gtab.b0s_mask):
            relative_data = data[0, 0, 0] / np.mean(data[0, 0, 0, gtab.b0s_mask])

            dtifit_to_relative = dm_to_relative.fit(relative_data)
            npt.assert_almost_equal(
                dtifit.fa[0, 0, 0], dtifit_to_relative.fa, decimal=3
            )

    # And smoke-test that all these operations return sensibly-shaped arrays:
    npt.assert_equal(dtifit.fa.shape, data.shape[:3])
    npt.assert_equal(dtifit.ad.shape, data.shape[:3])
    npt.assert_equal(dtifit.md.shape, data.shape[:3])
    npt.assert_equal(dtifit.rd.shape, data.shape[:3])
    npt.assert_equal(dtifit.trace.shape, data.shape[:3])
    npt.assert_equal(dtifit.mode.shape, data.shape[:3])
    npt.assert_equal(dtifit.linearity.shape, data.shape[:3])
    npt.assert_equal(dtifit.planarity.shape, data.shape[:3])
    npt.assert_equal(dtifit.sphericity.shape, data.shape[:3])

    # Test for the shape of the mask
    npt.assert_raises(ValueError, dm.fit, np.ones((10, 10, 3)), mask=np.ones((3, 3)))

    # Make some synthetic data
    b0 = 1000.0
    bvals, bvecs = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table_from_bvals_bvecs(bvals, bvecs)
    # The first b value is 0., so we take the second one:
    B = bvals[1]
    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -np.log(b0) * B]) / B
    evals = np.array([2.0, 1.0, 0.0]) / B
    md = evals.mean()
    tensor = from_lower_triangular(D)
    A_squiggle = tensor - (1 / 3.0) * np.trace(tensor) * np.eye(3)
    mode = 3 * np.sqrt(6) * np.linalg.det(A_squiggle / np.linalg.norm(A_squiggle))
    evals_eigh, evecs_eigh = np.linalg.eigh(tensor)
    # Sort according to eigen-value from large to small:
    evecs = evecs_eigh[:, np.argsort(evals_eigh)[::-1]]
    # Check that eigenvalues and eigenvectors are properly sorted through
    # that previous operation:
    for i in range(3):
        npt.assert_array_almost_equal(
            np.dot(tensor, evecs[:, i]), evals[i] * evecs[:, i]
        )
    # Design Matrix
    X = dti.design_matrix(gtab)
    # Signals
    Y = np.exp(np.dot(X, D))
    npt.assert_almost_equal(Y[0], b0)
    Y.shape = (-1,) + Y.shape

    # Test fitting with different methods:
    for fit_method in ["OLS", "WLS", "NLLS"]:
        tensor_model = dti.TensorModel(gtab, fit_method=fit_method, return_S0_hat=True)

        tensor_fit = tensor_model.fit(Y)
        assert tensor_fit.model is tensor_model
        npt.assert_equal(tensor_fit.shape, Y.shape[:-1])
        npt.assert_array_almost_equal(tensor_fit.evals[0], evals)
        npt.assert_array_almost_equal(tensor_fit.S0_hat, b0, decimal=3)
        # Test that the eigenvectors are correct, one-by-one:
        for i in range(3):
            # Eigenvectors have intrinsic sign ambiguity
            # (see
            # http://prod.sandia.gov/techlib/access-control.cgi/2007/076422.pdf)
            # so we need to allow for sign flips. One of the following should
            # always be true:
            npt.assert_(
                np.all(np.abs(tensor_fit.evecs[0][:, i] - evecs[:, i]) < 10e-6)
                or np.all(np.abs(-tensor_fit.evecs[0][:, i] - evecs[:, i]) < 10e-6)
            )
            # We set a fixed tolerance of 10e-6, similar to array_almost_equal

        err_msg = "Calculation of tensor from Y does not compare to "
        err_msg += "analytical solution"
        npt.assert_array_almost_equal(
            tensor_fit.quadratic_form[0], tensor, err_msg=err_msg
        )

        npt.assert_almost_equal(tensor_fit.md[0], md)
        npt.assert_array_almost_equal(tensor_fit.mode, mode, decimal=5)
        npt.assert_equal(tensor_fit.directions.shape[-2], 1)
        npt.assert_equal(tensor_fit.directions.shape[-1], 3)

    # Test error-handling:
    npt.assert_raises(ValueError, dti.TensorModel, gtab, fit_method="crazy_method")

    # Test custom fit tensor method
    try:
        model = dti.TensorModel(gtab, fit_method=lambda *args, **kwargs: 42)
        fit = model.fit_method()
    except Exception as exc:
        raise AssertionError(
            f"TensorModel should accept custom fit methods: {exc}"
        ) from exc
    assert fit == 42, f"Custom fit method for TensorModel returned {fit}."

    # Test multi-voxel data
    data = np.zeros((3, Y.shape[1]))
    # Normal voxel
    data[0] = Y
    # High diffusion voxel, all diffusing weighted signal equal to zero
    data[1, gtab.b0s_mask] = b0
    data[1, ~gtab.b0s_mask] = 0
    # Masked voxel, all data set to zero
    data[2] = 0.0

    tensor_model = dti.TensorModel(gtab)
    fit = tensor_model.fit(data)
    npt.assert_array_almost_equal(fit[0].evals, evals)

    # Return S0_test
    tensor_model = dti.TensorModel(gtab, return_S0_hat=True)
    fit = tensor_model.fit(data)
    npt.assert_array_almost_equal(fit[0].evals, evals)
    npt.assert_array_almost_equal(fit[0].S0_hat, b0)

    # Evals should be high for high diffusion voxel
    assert all(fit[1].evals > evals[0] * 0.9)

    # Evals should be zero where data is masked
    npt.assert_array_almost_equal(fit[2].evals, 0.0)


def test_indexing_on_tensor_fit():
    params = np.zeros([2, 3, 4, 12])
    fit = dti.TensorFit(None, params)

    # Should return a TensorFit of appropriate shape
    npt.assert_equal(fit.shape, (2, 3, 4))
    fit1 = fit[0]
    npt.assert_equal(fit1.shape, (3, 4))
    npt.assert_equal(type(fit1), dti.TensorFit)
    fit1 = fit[0, 0, 0]
    npt.assert_equal(fit1.shape, ())
    npt.assert_equal(type(fit1), dti.TensorFit)
    fit1 = fit[[0], slice(None)]
    npt.assert_equal(fit1.shape, (1, 3, 4))
    npt.assert_equal(type(fit1), dti.TensorFit)

    # Should raise an index error if too many indices are passed
    npt.assert_raises(IndexError, fit.__getitem__, (0, 0, 0, 0))


def test_fa_of_zero():
    evals = np.zeros((4, 3))
    fa = fractional_anisotropy(evals)
    npt.assert_array_equal(fa, 0)


def test_ga_of_zero():
    evals = np.zeros((4, 3))
    ga = geodesic_anisotropy(evals)
    npt.assert_array_equal(ga, 0)


def test_diffusivities():
    psphere = get_sphere(name="symmetric362")
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = grad.gradient_table(bvals, bvecs=bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    ]
    S = single_tensor(gtab, 100, evals=mevals[0], evecs=mevecs[0], snr=None)

    dm = dti.TensorModel(gtab, fit_method="LS")
    dmfit = dm.fit(S)

    md = mean_diffusivity(dmfit.evals)
    Trace = trace(dmfit.evals)
    rd = radial_diffusivity(dmfit.evals)
    ad = axial_diffusivity(dmfit.evals)
    lin = linearity(dmfit.evals)
    plan = planarity(dmfit.evals)
    spher = sphericity(dmfit.evals)

    npt.assert_almost_equal(md, (0.0015 + 0.0003 + 0.0001) / 3)
    npt.assert_almost_equal(Trace, (0.0015 + 0.0003 + 0.0001))
    npt.assert_almost_equal(ad, 0.0015)
    npt.assert_almost_equal(rd, (0.0003 + 0.0001) / 2)
    npt.assert_almost_equal(lin, (0.0015 - 0.0003) / Trace)
    npt.assert_almost_equal(plan, 2 * (0.0003 - 0.0001) / Trace)
    npt.assert_almost_equal(spher, (3 * 0.0001) / Trace)


def test_color_fa():
    data, gtab = dsi_voxels()
    dm = dti.TensorModel(gtab, fit_method="LS")
    dmfit = dm.fit(data)
    fa = fractional_anisotropy(dmfit.evals)

    fa = np.ones((3, 3, 3))
    # evecs should be of shape (fa, 3, 3)
    evecs = np.zeros(fa.shape + (3, 2))
    npt.assert_raises(ValueError, color_fa, fa, evecs)

    evecs = np.zeros(fa.shape + (3, 3))
    evecs[..., :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

    npt.assert_equal(fa.shape, evecs[..., 0, 0].shape)
    npt.assert_equal((3, 3), evecs.shape[-2:])

    # 3D test case
    fa = np.ones((3, 3, 3))
    evecs = np.zeros(fa.shape + (3, 3))
    evecs[..., :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cfa = color_fa(fa, evecs)
    cfa_truth = np.array([1, 0, 0])
    true_cfa = np.reshape(np.tile(cfa_truth, 27), [3, 3, 3, 3])

    npt.assert_array_equal(cfa, true_cfa)

    # 2D test case
    fa = np.ones((3, 3))
    evecs = np.zeros(fa.shape + (3, 3))
    evecs[..., :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cfa = color_fa(fa, evecs)
    cfa_truth = np.array([1, 0, 0])
    true_cfa = np.reshape(np.tile(cfa_truth, 9), [3, 3, 3])

    npt.assert_array_equal(cfa, true_cfa)

    # 1D test case
    fa = np.ones(3)
    evecs = np.zeros(fa.shape + (3, 3))
    evecs[..., :, :] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    cfa = color_fa(fa, evecs)
    cfa_truth = np.array([1, 0, 0])
    true_cfa = np.reshape(np.tile(cfa_truth, 3), [3, 3])

    npt.assert_array_equal(cfa, true_cfa)


def test_wls_and_ls_fit():
    """
    Tests the WLS and LS fitting functions to see if they returns the correct
    eigenvalues and eigenvectors.

    Uses data/55dir_grad as the gradient table and 3by3by56.nii
    as the data.

    """

    # Defining Test Voxel (avoid nibabel dependency) ###

    # Recall: D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,log(S_0)] and D ~ 10^-4 mm^2 /s
    b0 = 1000.0
    bval, bvec = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    B = bval[1]
    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -np.log(b0) * B]) / B
    evals = np.array([2.0, 1.0, 0.0]) / B
    md = evals.mean()
    tensor = from_lower_triangular(D)
    # Design Matrix
    gtab = grad.gradient_table(bval, bvecs=bvec)
    X = dti.design_matrix(gtab)
    # Signals
    Y = np.exp(np.dot(X, D))
    npt.assert_almost_equal(Y[0], b0)
    Y.shape = (-1,) + Y.shape

    # Testing WLS Fit on single voxel
    # If you do something wonky (passing min_signal<0), you should get an
    # error:
    npt.assert_raises(ValueError, TensorModel, gtab, fit_method="WLS", min_signal=-1)

    # Estimate tensor from test signals
    model = TensorModel(gtab, fit_method="WLS", return_S0_hat=True)
    tensor_est = model.fit(Y)
    npt.assert_equal(tensor_est.shape, Y.shape[:-1])
    npt.assert_array_almost_equal(tensor_est.evals[0], evals)
    npt.assert_array_almost_equal(
        tensor_est.quadratic_form[0],
        tensor,
        err_msg="Calculation of tensor from Y does "
        "not compare to analytical solution",
    )
    npt.assert_almost_equal(tensor_est.md[0], md)
    npt.assert_array_almost_equal(tensor_est.S0_hat[0], b0, decimal=3)

    # Test that we can fit a single voxel's worth of data (a 1d array)
    y = Y[0]
    tensor_est = model.fit(y)
    npt.assert_equal(tensor_est.shape, ())
    npt.assert_array_almost_equal(tensor_est.evals, evals)
    npt.assert_array_almost_equal(tensor_est.quadratic_form, tensor)
    npt.assert_almost_equal(tensor_est.md, md)
    npt.assert_array_almost_equal(tensor_est.lower_triangular(b0=b0), D)

    # Test using fit_method='LS'
    model = TensorModel(gtab, fit_method="LS")
    tensor_est = model.fit(y)
    npt.assert_equal(tensor_est.shape, ())
    npt.assert_array_almost_equal(tensor_est.evals, evals)
    npt.assert_array_almost_equal(tensor_est.quadratic_form, tensor)
    npt.assert_almost_equal(tensor_est.md, md)
    npt.assert_array_almost_equal(tensor_est.lower_triangular(b0=b0), D)
    npt.assert_array_almost_equal(tensor_est.linearity, linearity(evals))
    npt.assert_array_almost_equal(tensor_est.planarity, planarity(evals))
    npt.assert_array_almost_equal(tensor_est.sphericity, sphericity(evals))

    # testing that leverages are returned on request
    for fit_method in ['LS', 'WLS']:
        # Estimate tensor from test signals, not returning leverages
        model = TensorModel(gtab, fit_method=fit_method, return_S0_hat=True,
                            return_leverages=False)
        tensor_est = model.fit(Y)
        npt.assert_equal(tensor_est.model.extra, {})

        # Estimate tensor from test signals, returning leverages
        model = TensorModel(gtab, fit_method=fit_method, return_S0_hat=True,
                            return_leverages=True)
        tensor_est = model.fit(Y)
        npt.assert_equal(tensor_est.model.extra["leverages"].shape, Y.shape)

        # test value of leverage is 7 (in this case, for DTI)
        leverages = tensor_est.model.extra["leverages"]
        npt.assert_almost_equal(leverages.sum(axis=1), np.array([7.0]))

    # Test wls given S^2 weights argument, matches default wls
    design_matrix = dti.design_matrix(gtab)
    YN = Y + 10 * np.random.normal(size=Y.shape)  # error or weights irrelevant
    YN[YN < MIN_POSITIVE_SIGNAL] = MIN_POSITIVE_SIGNAL
    # wls calculation
    D_w, _ = wls_fit_tensor(design_matrix, YN, return_lower_triangular=True)
    # wls calculation, by calculating S^2 from OLS fit, then passing weights
    D_o, _ = ols_fit_tensor(design_matrix, YN, return_lower_triangular=True)
    pred_s = np.exp(np.dot(design_matrix, D_o.T)).T
    D_W, _ = wls_fit_tensor(design_matrix, YN, return_lower_triangular=True,
                            weights=pred_s**2)  # weights match WLS default
    npt.assert_almost_equal(D_w, D_W)
    # wls calculation, but passing incorrect weights
    D_W, _ = wls_fit_tensor(design_matrix, YN, return_lower_triangular=True,
                            weights=pred_s**1)
    npt.assert_raises(AssertionError, npt.assert_array_equal, D_w, D_W)
    # Test that wls implementation is correct, by comparison with result here
    W = np.diag(pred_s.squeeze()**2)
    AT_W = np.dot(design_matrix.T, W)
    inv_AT_W_A = np.linalg.pinv(np.dot(AT_W, design_matrix))
    AT_W_LS = np.dot(AT_W, np.log(YN).squeeze())
    result = np.dot(inv_AT_W_A, AT_W_LS)
    npt.assert_almost_equal(D_w.squeeze(), result)


def test_rwls_rnlls_irls_fit():

    # Recall: D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,log(S_0)] and D ~ 10^-4 mm^2 /s
    b0 = 1000.
    bval, bvec = read_bvals_bvecs(*get_fnames(name='55dir_grad'))
    B = bval[1]
    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
    evals = np.array([2., 1., 0.]) / B
    md = evals.mean()
    tensor = from_lower_triangular(D)
    # Design Matrix
    gtab = grad.gradient_table(bval, bvecs=bvec)
    X = dti.design_matrix(gtab)
    # Signals
    Y = np.exp(np.dot(X, D))
    npt.assert_almost_equal(Y[0], b0)
    Y.shape = (-1,) + Y.shape

    YN = Y + 1 * np.random.normal(size=Y.shape)  # error, or weights irrelevant
    YN[0, -1] *= 10  # note 1D array!

    for a, ar in zip(["WLS", "NLLS"], ["RWLS", "RNLLS"]):

        # Estimate tensor from test signals
        model = TensorModel(gtab, fit_method=a, return_S0_hat=True)
        tensor_est = model.fit(YN)

        model = TensorModel(gtab, fit_method=ar, return_S0_hat=True,
                            num_iter=10)
        tensor_est_R = model.fit(YN)

        npt.assert_array_less(np.linalg.norm(tensor_est_R.evals[0] - evals),
                              np.linalg.norm(tensor_est.evals[0] - evals))

        npt.assert_array_less(
            np.linalg.norm(
                tensor_est_R.quadratic_form[0] - tensor
            ),
            np.linalg.norm(
                tensor_est.quadratic_form[0] - tensor
            )
        )

        npt.assert_array_less(np.linalg.norm(tensor_est_R.md[0] - md),
                              np.linalg.norm(tensor_est.md[0] - md))

        # error is often almost exactly the same, so this test sometimes fails
        # npt.assert_array_less(np.linalg.norm(tensor_est_R.S0_hat[0] - b0),
        #                       np.linalg.norm(tensor_est.S0_hat[0] - b0))

    # test RWLS/RNLLS implemented explicitly via IRLS function
    for wm, fit_type in zip([weights_method_wls_m_est,
                             weights_method_nlls_m_est],
                            ["WLS", "NLLS"]):

        # IRLS implementation
        model = TensorModel(gtab, fit_method="IRLS", return_S0_hat=True,
                            weights_method=wm, fit_type=fit_type, num_iter=10)
        tensor_est_R1 = model.fit(YN)
        npt.assert_equal(tensor_est_R1.model.extra["robust"].shape, YN.shape)
        npt.assert_equal(tensor_est_R1.model.extra["robust"][0, -1], 0)

        # 'shortcut' method RWLS/RNLLS
        model = TensorModel(gtab, fit_method="R" + fit_type,
                            return_S0_hat=False,  # NOTE increase coverage
                            num_iter=10)
        tensor_est_R2 = model.fit(YN)
        npt.assert_equal(tensor_est_R2.model.extra["robust"].shape, YN.shape)
        npt.assert_equal(tensor_est_R2.model.extra["robust"][0, -1], 0)

        npt.assert_almost_equal(tensor_est_R1.evals[0],
                                tensor_est_R2.evals[0])

        npt.assert_almost_equal(tensor_est_R1.quadratic_form[0],
                                tensor_est_R2.quadratic_form[0])

    # test that error is raised if not enough data
    model = TensorModel(gtab, fit_method="RWLS", num_iter=10)
    npt.assert_raises(ValueError, model.fit, YN[:, 0:3])

    # force use of iter_fit_tensor without making a large test
    model = TensorModel(gtab, fit_method="RWLS", return_S0_hat=True,
                        num_iter=10)
    tensor_est_R2 = model.fit(np.repeat(YN, repeats=1e4 + 1, axis=0))


def test_masked_array_with_tensor():
    data = np.ones((2, 4, 56))
    mask = np.array([[True, False, False, True], [True, False, True, False]])

    bval, bvec = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table_from_bvals_bvecs(bval, bvec)

    # test self.extra with mask by using return_leverages
    tensor_model = TensorModel(gtab, return_leverages=True)
    tensor = tensor_model.fit(data, mask=mask)
    npt.assert_equal(tensor.shape, (2, 4))
    npt.assert_equal(tensor.fa.shape, (2, 4))
    npt.assert_equal(tensor.evals.shape, (2, 4, 3))
    npt.assert_equal(tensor.evecs.shape, (2, 4, 3, 3))

    tensor = tensor[0]
    npt.assert_equal(tensor.shape, (4,))
    npt.assert_equal(tensor.fa.shape, (4,))
    npt.assert_equal(tensor.evals.shape, (4, 3))
    npt.assert_equal(tensor.evecs.shape, (4, 3, 3))

    tensor = tensor[0]
    npt.assert_equal(tensor.shape, ())
    npt.assert_equal(tensor.fa.shape, ())
    npt.assert_equal(tensor.evals.shape, (3,))
    npt.assert_equal(tensor.evecs.shape, (3, 3))
    npt.assert_equal(type(tensor.model_params), np.ndarray)


def test_fit_method_error():
    bval, bvec = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table_from_bvals_bvecs(bval, bvec)

    # This should work (smoke-testing!):
    TensorModel(gtab, fit_method="WLS")

    # This should raise an error because there is no such fit_method
    npt.assert_raises(ValueError, TensorModel, gtab, min_signal=1e-9, fit_method="s")


def test_lower_triangular():
    tensor = np.arange(9).reshape((3, 3))
    D = lower_triangular(tensor)
    npt.assert_array_equal(D, [0, 3, 4, 6, 7, 8])
    D = lower_triangular(tensor, b0=1)
    npt.assert_array_equal(D, [0, 3, 4, 6, 7, 8, 0])
    npt.assert_raises(ValueError, lower_triangular, np.zeros((2, 3)))
    shape = (4, 5, 6)
    many_tensors = np.empty(shape + (3, 3))
    many_tensors[:] = tensor
    result = np.empty(shape + (6,))
    result[:] = [0, 3, 4, 6, 7, 8]
    D = lower_triangular(many_tensors)
    npt.assert_array_equal(D, result)
    D = lower_triangular(many_tensors, b0=1)
    result = np.empty(shape + (7,))
    result[:] = [0, 3, 4, 6, 7, 8, 0]
    npt.assert_array_equal(D, result)


def test_from_lower_triangular():
    result = np.array([[0, 1, 3], [1, 2, 4], [3, 4, 5]])
    D = np.arange(7)
    tensor = from_lower_triangular(D)
    npt.assert_array_equal(tensor, result)
    result = result * np.ones((5, 4, 1, 1))
    D = D * np.ones((5, 4, 1))
    tensor = from_lower_triangular(D)
    npt.assert_array_equal(tensor, result)


def test_all_constant():
    bvals, bvecs = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table_from_bvals_bvecs(bvals, bvecs)
    fit_methods = ["LS", "OLS", "NNLS", "RESTORE"]
    for _ in fit_methods:
        dm = dti.TensorModel(gtab)
        npt.assert_almost_equal(dm.fit(100 * np.ones(bvals.shape[0])).fa, 0)
        # Doesn't matter if the signal is smaller than 1:
        npt.assert_almost_equal(dm.fit(0.4 * np.ones(bvals.shape[0])).fa, 0)


def test_all_zeros():
    bvals, bvecs = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table_from_bvals_bvecs(bvals, bvecs)
    fit_methods = ["LS", "OLS", "NNLS", "RESTORE"]
    for _ in fit_methods:
        dm = dti.TensorModel(gtab)
        evals = dm.fit(np.zeros(bvals.shape[0])).evals
        npt.assert_array_almost_equal(evals, 0)


def test_mask():
    data, gtab = dsi_voxels()
    for fit_type in ["LS", "NLLS"]:
        dm = dti.TensorModel(gtab, fit_method=fit_type)
        mask = np.zeros(data.shape[:-1], dtype=bool)
        mask[0, 0, 0] = True
        dtifit = dm.fit(data)
        dtifit_w_mask = dm.fit(data, mask=mask)
        # Without a mask it has some value
        assert not np.isnan(dtifit.fa[0, 0, 0])
        # Where mask is False, evals, evecs and fa should all be 0
        npt.assert_array_equal(dtifit_w_mask.evals[~mask], 0)
        npt.assert_array_equal(dtifit_w_mask.evecs[~mask], 0)
        npt.assert_array_equal(dtifit_w_mask.fa[~mask], 0)
        # Except for the one voxel that was selected by the mask:
        npt.assert_almost_equal(dtifit_w_mask.fa[0, 0, 0], dtifit.fa[0, 0, 0])

        # Test with returning S0_hat
        dm = dti.TensorModel(gtab, fit_method=fit_type, return_S0_hat=True)
        mask = np.zeros(data.shape[:-1], dtype=bool)
        mask[0, 0, 0] = True
        for mask_more in [True, False]:
            if mask_more:
                mask[0, 0, 1] = True
            dtifit = dm.fit(data)
            dtifit_w_mask = dm.fit(data, mask=mask)
            # Without a mask it has some value
            assert not np.isnan(dtifit.fa[0, 0, 0])
            # Where mask is False, evals, evecs and fa should all be 0
            npt.assert_array_equal(dtifit_w_mask.evals[~mask], 0)
            npt.assert_array_equal(dtifit_w_mask.evecs[~mask], 0)
            npt.assert_array_equal(dtifit_w_mask.fa[~mask], 0)
            npt.assert_array_equal(dtifit_w_mask.S0_hat[~mask], 0)
            # Except for the one voxel that was selected by the mask:
            npt.assert_almost_equal(dtifit_w_mask.fa[0, 0, 0],
                                    dtifit.fa[0, 0, 0])
            npt.assert_almost_equal(dtifit_w_mask.S0_hat[0, 0, 0],
                                    dtifit.S0_hat[0, 0, 0])


@set_random_number_generator()
def test_nnls_jacobian_func(rng):
    b0 = 1000.0
    bval, bvecs = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table(bval, bvecs=bvecs)
    B = bval[1]

    # Scale the eigenvalues and tensor by the B value so the units match
    D_orig = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -np.log(b0) * B]) / B

    # Design Matrix
    X = dti.design_matrix(gtab)

    # Signals
    Y = np.exp(np.dot(X, D_orig))
    scale = 10
    error = rng.normal(scale=scale, size=Y.shape)
    Y = Y + error

    nlls = dti._NllsHelper()
    sigma_scalar = 1.4826 * np.median(np.abs(error - np.median(error)))
    sigma_array = np.full_like(Y, sigma_scalar)
    for sigma in [sigma_scalar, sigma_array]:
        weights = 1 / sigma**2
        for D in [D_orig, np.zeros_like(D_orig)]:

            # Test Jacobian at D
            args = [D, X, Y, weights]
            # 1. call 'err_func', to set internal stuff in the class
            nlls.err_func(*args)
            # 2. call 'jabobian_func', corresponds to last err_func call
            analytical = nlls.jacobian_func(*args)

            # test analytical gradient (needs to be performed per data-point)
            for i in range(len(X)):

                args = [X[i], Y[i], weights]

                # FIXME: this is sometimes failing in tests on Github
                # approx = opt.approx_fprime(D, nlls.err_func, 1e-8, *args)
                #
                #        approx_fprime wants nlls.err_func to return a scalar
                #        value, which it ought to do if called with a single
                #        data point (otherwise, it returns an array, consistent
                #        with scipy.opt.leastsq) but something seems broken in
                #        some tests, so let's make a function that ensures a
                #        scalar is returned. Issue for this *test*, not for
                #        nlls.err_func, which works correctly
                def ef(x):
                    tmp = nlls.err_func(x, *args)
                    return tmp if np.isscalar(tmp) else tmp[0]
                approx = opt.approx_fprime(D, ef, 1e-8)

                # NOTE: approx_fprime not accurate enough to pass this test
                #       even though it will pass if using autograd code
                #       to ensure a truly accurate derivative of nlls.err_func
                # assert np.allclose(approx, analytical[i])
                assert True


def test_nlls_fit_tensor():
    """
    Test the implementation of NLLS and RESTORE
    """

    b0 = 1000.0
    bvals, bvecs = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table(bvals, bvecs=bvecs)
    B = bvals[1]

    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -np.log(b0) * B]) / B
    evals = np.array([2.0, 1.0, 0.0]) / B
    md = evals.mean()
    tensor = from_lower_triangular(D)

    # Design Matrix
    X = dti.design_matrix(gtab)

    # Signals
    Y = np.exp(np.dot(X, D))
    Y.shape = (-1,) + Y.shape

    # Estimate tensor from test signals and compare against expected result
    # using non-linear least squares:
    tensor_model = dti.TensorModel(gtab, fit_method="NLLS")
    tensor_est = tensor_model.fit(Y)
    npt.assert_equal(tensor_est.shape, Y.shape[:-1])
    npt.assert_array_almost_equal(tensor_est.evals[0], evals)
    npt.assert_array_almost_equal(tensor_est.quadratic_form[0], tensor)
    npt.assert_almost_equal(tensor_est.md[0], md)

    # You can also do this without the Jacobian (though it's slower):
    tensor_model = dti.TensorModel(gtab, fit_method="NLLS", jac=False)
    tensor_est = tensor_model.fit(Y)
    npt.assert_equal(tensor_est.shape, Y.shape[:-1])
    npt.assert_array_almost_equal(tensor_est.evals[0], evals)
    npt.assert_array_almost_equal(tensor_est.quadratic_form[0], tensor)
    npt.assert_almost_equal(tensor_est.md[0], md)

    # Using weights:
    weights = 2 * np.ones_like(Y, dtype=np.float32)
    tensor_model = dti.TensorModel(gtab, fit_method='NLLS', weights=weights)
    tensor_est = tensor_model.fit(Y)
    npt.assert_equal(tensor_est.shape, Y.shape[:-1])
    npt.assert_array_almost_equal(tensor_est.evals[0], evals)
    npt.assert_array_almost_equal(tensor_est.quadratic_form[0], tensor)
    npt.assert_almost_equal(tensor_est.md[0], md)

    # Use NLLS with some actual 4D data:
    data, bvals, bvecs = get_fnames(name="small_25")
    gtab = grad.gradient_table(bvals, bvecs=bvecs)
    tm1 = dti.TensorModel(gtab, fit_method="NLLS")
    dd = load_nifti_data(data)
    tf1 = tm1.fit(dd)
    tm2 = dti.TensorModel(gtab)
    tf2 = tm2.fit(dd)

    npt.assert_array_almost_equal(tf1.fa, tf2.fa, decimal=1)

    # Reduce amount of data, to cause NLLS to fail
    gtab_less = grad.gradient_table(gtab.bvals[0:3], bvecs=gtab.bvecs[0:3, :])
    Y_less = Y[..., 0:3].copy()

    # Test warning for failure of NLLS method, resort to OLS result
    # (reason for failure: too few data points for NLLS)
    tensor_model = dti.TensorModel(gtab_less, fit_method='NLLS',
                                   return_S0_hat=True)
    tmf = npt.assert_warns(UserWarning, tensor_model.fit, Y_less)

    # Test fail_is_nan=True, failed NLLS method gives NaN
    tensor_model = dti.TensorModel(gtab_less, fit_method='NLLS',
                                   return_S0_hat=True, fail_is_nan=True)
    tmf = npt.assert_warns(UserWarning, tensor_model.fit, Y_less)
    npt.assert_equal(tmf[0].S0_hat, np.nan)


def test_restore():
    """
    Test the implementation of the RESTORE algorithm
    """
    b0 = 1000.0
    bval, bvecs = read_bvals_bvecs(*get_fnames(name="55dir_grad"))
    gtab = grad.gradient_table(bval, bvecs=bvecs)
    B = bval[1]

    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 1.0, -np.log(b0) * B]) / B
    evals = np.array([2.0, 1.0, 0.0]) / B
    tensor = from_lower_triangular(D)

    # Design Matrix
    X = dti.design_matrix(gtab)

    # Signals
    Y = np.exp(np.dot(X, D))
    Y = np.vstack([Y[None, :], Y[None, :]])  # two voxels
    for drop_this in range(1, Y.shape[-1]):
        for jac in [True, False]:
            # RESTORE estimates should be robust to dropping
            this_y = Y.copy()
            this_y[:, drop_this] = 1.0
            for sigma in [
                67.0,
                np.array([67.0]),
                np.ones(this_y.shape[-1]) * 67.0,
                np.array([66.0, 67.0]).reshape((-1, 1)),
            ]:
                tensor_model = dti.TensorModel(
                    gtab, fit_method="restore", jac=jac, sigma=sigma
                )

                tensor_est = tensor_model.fit(this_y)
                npt.assert_array_almost_equal(tensor_est.evals[0], evals, decimal=3)
                npt.assert_array_almost_equal(
                    tensor_est.quadratic_form[0], tensor, decimal=3
                )

                # test recording of robust signals
                npt.assert_equal(tensor_est.model.extra["robust"].shape,
                                 Y.shape)

    # If sigma is very small, it still needs to work:
    tensor_model = dti.TensorModel(gtab, fit_method="restore", sigma=0.0001)
    tensor_model.fit(Y.copy())

    # If sigma is very small, it still needs to work (it is estimated):
    tensor_model = dti.TensorModel(gtab, fit_method='restore', sigma=None)
    tensor_model.fit(Y.copy() + np.random.normal(size=Y.shape))

    # Test return_S0_hat
    tensor_model = dti.TensorModel(
        gtab, fit_method="restore", sigma=0.0001, return_S0_hat=True
    )
    tmf = tensor_model.fit(Y.copy())
    npt.assert_almost_equal(tmf[0].S0_hat, b0)

    # Test warning for failure of NLLS method, resort to OLS result
    # (reason for failure: too few data points for NLLS, due to negative sigma)
    tensor_model = dti.TensorModel(
        gtab, fit_method="restore", sigma=-1.0, return_S0_hat=True
    )
    tmf = npt.assert_warns(UserWarning, tensor_model.fit, Y.copy())

    # Test fail_is_nan=True, failed NLLS method gives NaN
    tensor_model = dti.TensorModel(
        gtab, fit_method="restore", sigma=-1.0, return_S0_hat=True, fail_is_nan=True
    )
    tmf = npt.assert_warns(UserWarning, tensor_model.fit, Y.copy())
    npt.assert_equal(tmf[0].S0_hat, np.nan)


def test_adc():
    """
    Test the implementation of the calculation of apparent diffusion
    coefficient
    """
    data, gtab = dsi_voxels()
    dm = dti.TensorModel(gtab, fit_method="LS")
    mask = np.zeros(data.shape[:-1], dtype=bool)
    mask[0, 0, 0] = True
    dtifit = dm.fit(data)
    # The ADC in the principal diffusion direction should be equal to the AD in
    # each voxel:

    pdd0 = dtifit.evecs[0, 0, 0, 0]
    sphere_pdd0 = dps.Sphere(x=pdd0[0], y=pdd0[1], z=pdd0[2])
    npt.assert_array_almost_equal(
        dtifit.adc(sphere_pdd0)[0, 0, 0], dtifit.ad[0, 0, 0], decimal=4
    )

    # Test that it works for cases in which the data is 1D
    dtifit = dm.fit(data[0, 0, 0])
    sphere_pdd0 = dps.Sphere(x=pdd0[0], y=pdd0[1], z=pdd0[2])
    npt.assert_array_almost_equal(dtifit.adc(sphere_pdd0), dtifit.ad, decimal=4)


def test_predict():
    """
    Test model prediction API
    """
    psphere = get_sphere(name="symmetric362")
    bvecs = np.concatenate(([[1, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = grad.gradient_table(bvals, bvecs=bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    ]
    S = single_tensor(gtab, 100, evals=mevals[0], evecs=mevecs[0], snr=None)

    dm = dti.TensorModel(gtab, fit_method="LS", return_S0_hat=True)
    dmfit = dm.fit(S)
    npt.assert_array_almost_equal(dmfit.predict(gtab, S0=100), S)
    npt.assert_array_almost_equal(dmfit.predict(gtab), S)
    npt.assert_array_almost_equal(dm.predict(dmfit.model_params, S0=100), S)

    fdata, fbvals, fbvecs = get_fnames()
    data = load_nifti_data(fdata)
    # Make the data cube a bit larger:
    data = np.tile(data.T, 2).T
    gtab = grad.gradient_table(fbvals, bvecs=fbvecs)
    dtim = dti.TensorModel(gtab)
    dtif = dtim.fit(data)
    S0 = np.mean(data[..., gtab.b0s_mask], -1)
    p = dtif.predict(gtab, S0=S0)
    npt.assert_equal(p.shape, data.shape)
    # Predict using S0_hat:
    dtim = dti.TensorModel(gtab, return_S0_hat=True)
    dtif = dtim.fit(data)
    p = dtif.predict(gtab)
    npt.assert_equal(p.shape, data.shape)
    p = dtif.predict(gtab, S0=S0)
    npt.assert_equal(p.shape, data.shape)

    # Test iter_fit_tensor with S0_hat
    dtim = dti.TensorModel(gtab, step=2, return_S0_hat=True)
    dtif = dtim.fit(data)
    S0 = np.mean(data[..., gtab.b0s_mask], -1)
    p = dtif.predict(gtab, S0=S0)
    npt.assert_equal(p.shape, data.shape)

    # Use a smaller step in predicting:

    dtim = dti.TensorModel(gtab, step=2)
    dtif = dtim.fit(data)
    S0 = np.mean(data[..., gtab.b0s_mask], -1)
    p = dtif.predict(gtab, S0=S0)
    npt.assert_equal(p.shape, data.shape)
    # And with a scalar S0:
    S0 = 1
    p = dtif.predict(gtab, S0=S0)
    npt.assert_equal(p.shape, data.shape)
    # Assign the step through kwarg:
    p = dtif.predict(gtab, S0=S0, step=1)
    npt.assert_equal(p.shape, data.shape)
    # And without S0:
    p = dtif.predict(gtab, step=1)
    npt.assert_equal(p.shape, data.shape)


def test_eig_from_lo_tri():
    psphere = get_sphere(name="symmetric362")
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = grad.gradient_table(bvals, bvecs=bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
        np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]]),
    ]
    S = np.array(
        [
            [
                single_tensor(gtab, 100, evals=mevals[0], evecs=mevecs[0], snr=None),
                single_tensor(gtab, 100, evals=mevals[0], evecs=mevecs[0], snr=None),
            ]
        ]
    )

    dm = dti.TensorModel(gtab, fit_method="LS")
    dmfit = dm.fit(S)

    lo_tri = lower_triangular(dmfit.quadratic_form)
    npt.assert_array_almost_equal(dti.eig_from_lo_tri(lo_tri), dmfit.model_params)


def test_min_signal_alone():
    fdata, fbvals, fbvecs = get_fnames()
    data = load_nifti_data(fdata)
    gtab = grad.gradient_table(fbvals, bvecs=fbvecs)

    idx = tuple(np.array(np.where(data == np.min(data)))[:-1, 0])
    ten_model = dti.TensorModel(gtab)
    fit_alone = ten_model.fit(data[idx])
    fit_together = ten_model.fit(data)
    npt.assert_array_almost_equal(
        fit_together.model_params[idx], fit_alone.model_params, decimal=12
    )


def test_decompose_tensor_nan():
    D_fine = np.array([1.7e-3, 0.0, 0.3e-3, 0.0, 0.0, 0.2e-3])
    D_alter = np.array([1.6e-3, 0.0, 0.4e-3, 0.0, 0.0, 0.3e-3])
    D_nan = np.nan * np.ones(6)

    lref, vref = decompose_tensor(from_lower_triangular(D_fine))
    lfine, vfine = _decompose_tensor_nan(
        from_lower_triangular(D_fine), from_lower_triangular(D_alter)
    )
    npt.assert_array_almost_equal(lfine, np.array([1.7e-3, 0.3e-3, 0.2e-3]))
    npt.assert_array_almost_equal(vfine, vref)

    lref, vref = decompose_tensor(from_lower_triangular(D_alter))
    lalter, valter = _decompose_tensor_nan(
        from_lower_triangular(D_nan), from_lower_triangular(D_alter)
    )
    npt.assert_array_almost_equal(lalter, np.array([1.6e-3, 0.4e-3, 0.3e-3]))
    npt.assert_array_almost_equal(valter, vref)


def test_design_matrix_lte():
    _, fbval, fbvec = get_fnames(name="small_25")
    gtab_btens_none = grad.gradient_table(fbval, bvecs=fbvec)
    gtab_btens_lte = grad.gradient_table(fbval, bvecs=fbvec, btens="LTE")

    B_btens_none = dti.design_matrix(gtab_btens_none)
    B_btens_lte = dti.design_matrix(gtab_btens_lte)
    npt.assert_array_almost_equal(B_btens_none, B_btens_lte, decimal=1)


def test_extra_return():
    """
    Test if returns of dictionary 'extra' from fitting functions are working
    properly.

    Uses data/55dir_grad as the gradient table and 3by3by56.nii
    as the data.

    """

    b0 = 1000.
    bval, bvecs = read_bvals_bvecs(*get_fnames(name='55dir_grad'))
    gtab = grad.gradient_table(bval, bvecs=bvecs)
    B = bval[1]

    # Scale the eigenvalues and tensor by the B value so the units match
    D = np.array([1., 1., 1., 0., 0., 1., -np.log(b0) * B]) / B
    evals = np.array([2., 1., 0.]) / B
    tensor = from_lower_triangular(D)

    # Design Matrix
    X = dti.design_matrix(gtab)

    # Signals
    Y = np.exp(np.dot(X, D))
    Y = np.vstack([Y[None, :], Y[None, :]])  # two voxels
    for drop_this in range(1, 3):  # Y.shape[-1]):
        # test specific extra from specific methods
        for method in ["restore"]:
            this_y = Y.copy()
            this_y[:, drop_this] = 1.0

            sigma = 0.0001

            if method == "restore":
                tensor_model = dti.TensorModel(gtab, fit_method=method,
                                               sigma=sigma)

            tensor_est = tensor_model.fit(this_y)
            npt.assert_equal(tensor_est.model.extra["robust"].shape, Y.shape)
