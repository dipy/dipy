import warnings
import numpy as np
import numpy.testing as npt
import pytest
import dipy.reconst.sfm as sfm
import dipy.data as dpd
import dipy.core.gradients as grad
import dipy.sims.voxel as sims
import dipy.core.optimize as opt
import dipy.reconst.cross_validation as xval
from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data
from dipy.utils.optpkg import optional_package

warnings.filterwarnings('ignore')

sklearn, has_sklearn, _ = optional_package('sklearn')
needs_sklearn = pytest.mark.skipif(not has_sklearn, reason="Requires sklearn")


def test_design_matrix():
    data, gtab = dpd.dsi_voxels()
    sphere = dpd.get_sphere()
    # Make it with NNLS, so that it gets tested regardless of sklearn
    sparse_fascicle_model = sfm.SparseFascicleModel(gtab, sphere,
                                                    solver='NNLS')
    npt.assert_equal(sparse_fascicle_model.design_matrix.shape,
                     (np.sum(~gtab.b0s_mask), sphere.vertices.shape[0]))


@needs_sklearn
def test_sfm():
    fdata, fbvals, fbvecs = dpd.get_fnames()
    data = load_nifti_data(fdata)
    gtab = grad.gradient_table(fbvals, fbvecs)
    for n_procs in [1, 2]:
        for iso in [sfm.ExponentialIsotropicModel, None]:
            sfmodel = sfm.SparseFascicleModel(gtab, isotropic=iso)
            sffit1 = sfmodel.fit(data[0, 0, 0], num_processes=n_procs)
            sphere = dpd.get_sphere()
            odf1 = sffit1.odf(sphere)
            pred1 = sffit1.predict(gtab)
            mask = np.ones(data.shape[:-1])
            sffit2 = sfmodel.fit(data, mask, num_processes=n_procs)
            pred2 = sffit2.predict(gtab)
            odf2 = sffit2.odf(sphere)
            sffit3 = sfmodel.fit(data, num_processes=n_procs)
            pred3 = sffit3.predict(gtab)
            odf3 = sffit3.odf(sphere)
            npt.assert_almost_equal(pred3, pred2, decimal=2)
            npt.assert_almost_equal(pred3[0, 0, 0], pred1, decimal=2)
            npt.assert_almost_equal(odf3[0, 0, 0], odf1, decimal=2)
            npt.assert_almost_equal(odf3[0, 0, 0], odf2[0, 0, 0], decimal=2)
            # Fit zeros and you will get back zeros
            npt.assert_almost_equal(
                sfmodel.fit(np.zeros(data[0, 0, 0].shape),
                            num_processes=n_procs).beta,
                np.zeros(sfmodel.design_matrix[0].shape[-1]))


@needs_sklearn
def test_predict():
    SNR = 1000
    S0 = 100
    _, fbvals, fbvecs = dpd.get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = grad.gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    angles = [(0, 0), (60, 0)]
    S, sticks = sims.multi_tensor(gtab, mevals, S0, angles=angles,
                                  fractions=[10, 90], snr=SNR)

    sfmodel = sfm.SparseFascicleModel(gtab, response=[0.0015, 0.0003, 0.0003])
    sffit = sfmodel.fit(S)
    pred = sffit.predict()
    npt.assert_(xval.coeff_of_determination(pred, S) > 97)

    # Should be possible to predict using a different gtab:
    new_gtab = grad.gradient_table(bvals[::2], bvecs[::2])
    new_pred = sffit.predict(new_gtab)
    npt.assert_(xval.coeff_of_determination(new_pred, S[::2]) > 97)

    # Should be possible to predict for a single direction:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        new_gtab = grad.gradient_table(bvals[1][None], bvecs[1][None, :])
    new_pred = sffit.predict(new_gtab)

    # Fitting and predicting with a volume of data:
    fdata, fbval, fbvec = dpd.get_fnames('small_25')
    gtab = grad.gradient_table(fbval, fbvec)
    data = load_nifti_data(fdata)
    sfmodel = sfm.SparseFascicleModel(gtab, response=[0.0015, 0.0003, 0.0003])
    sffit = sfmodel.fit(data)
    pred = sffit.predict()

    # Should be possible to predict using a different gtab:
    new_gtab = grad.gradient_table(bvals[::2], bvecs[::2])
    new_pred = sffit.predict(new_gtab)
    npt.assert_equal(new_pred.shape, data.shape[:-1] + bvals[::2].shape, )

    # Should be possible to predict for a single direction:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        new_gtab = grad.gradient_table(bvals[1][None], bvecs[1][None, :])
    new_pred = sffit.predict(new_gtab)
    npt.assert_equal(new_pred.shape, data.shape[:-1])

    # Fitting and predicting with masked data:
    mask = np.zeros(data.shape[:3])
    mask[2:5, 2:5, :] = 1
    sffit = sfmodel.fit(data, mask=mask)
    pred = sffit.predict()
    npt.assert_equal(pred.shape, data.shape)

    # Should be possible to predict using a different gtab:
    new_gtab = grad.gradient_table(bvals[::2], bvecs[::2])
    new_pred = sffit.predict(new_gtab)
    npt.assert_equal(new_pred.shape, data.shape[:-1] + bvals[::2].shape,)
    npt.assert_equal(new_pred[0, 0, 0], 0)

    # Should be possible to predict for a single direction:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        new_gtab = grad.gradient_table(bvals[1][None], bvecs[1][None, :])
    new_pred = sffit.predict(new_gtab)
    npt.assert_equal(new_pred.shape, data.shape[:-1])
    npt.assert_equal(new_pred[0, 0, 0], 0)


def test_sfm_background():
    fdata, fbvals, fbvecs = dpd.get_fnames()
    data = load_nifti_data(fdata)
    gtab = grad.gradient_table(fbvals, fbvecs)
    to_fit = data[0, 0, 0]
    to_fit[gtab.b0s_mask] = 0
    sfmodel = sfm.SparseFascicleModel(gtab, solver='NNLS')
    sffit = sfmodel.fit(to_fit)
    npt.assert_equal(sffit.beta, np.zeros_like(sffit.beta))


def test_sfm_stick():
    fdata, fbvals, fbvecs = dpd.get_fnames()
    data = load_nifti_data(fdata)
    gtab = grad.gradient_table(fbvals, fbvecs)
    sfmodel = sfm.SparseFascicleModel(gtab, solver='NNLS',
                                      response=[0.001, 0, 0])
    sffit1 = sfmodel.fit(data[0, 0, 0])
    sphere = dpd.get_sphere()
    sffit1.odf(sphere)
    sffit1.predict(gtab)

    SNR = 1000
    S0 = 100
    mevals = np.array(([0.001, 0, 0],
                       [0.001, 0, 0]))
    angles = [(0, 0), (60, 0)]
    S, sticks = sims.multi_tensor(gtab, mevals, S0, angles=angles,
                                  fractions=[50, 50], snr=SNR)

    sfmodel = sfm.SparseFascicleModel(gtab, solver='NNLS',
                                      response=[0.001, 0, 0])
    sffit = sfmodel.fit(S)
    pred = sffit.predict()
    npt.assert_(xval.coeff_of_determination(pred, S) > 96)


@needs_sklearn
def test_sfm_sklearnlinearsolver():
    class SillySolver(opt.SKLearnLinearSolver):
        def fit(self, X, y):
            self.coef_ = np.ones(X.shape[-1])

    class EvenSillierSolver:
        def fit(self, X, y):
            self.coef_ = np.ones(X.shape[-1])

    fdata, fbvals, fbvecs = dpd.get_fnames()
    gtab = grad.gradient_table(fbvals, fbvecs)
    sfmodel = sfm.SparseFascicleModel(gtab, solver=SillySolver())

    npt.assert_(isinstance(sfmodel.solver, SillySolver))
    npt.assert_raises(ValueError,
                      sfm.SparseFascicleModel,
                      gtab,
                      solver=EvenSillierSolver())


@needs_sklearn
def test_exponential_iso():
    fdata, fbvals, fbvecs = dpd.get_fnames()
    data_dti = load_nifti_data(fdata)
    gtab_dti = grad.gradient_table(fbvals, fbvecs)
    data_multi, gtab_multi = dpd.dsi_deconv_voxels()

    for data, gtab in zip([data_dti, data_multi], [gtab_dti, gtab_multi]):
        sfmodel = sfm.SparseFascicleModel(
                  gtab, isotropic=sfm.ExponentialIsotropicModel)

        sffit1 = sfmodel.fit(data[0, 0, 0])
        sphere = dpd.get_sphere()
        odf = sffit1.odf(sphere)
        pred = sffit1.predict(gtab)
        npt.assert_equal(pred.shape, data[0, 0, 0].shape)
        npt.assert_equal(odf.shape,
                         data[0, 0, 0].shape[:-1] + (sphere.x.shape[0],))

        sffit2 = sfmodel.fit(data)
        sphere = dpd.get_sphere()
        odf = sffit2.odf(sphere)
        pred = sffit2.predict(gtab)
        npt.assert_equal(pred.shape, data.shape)
        npt.assert_equal(odf.shape,
                         data.shape[:-1] + (sphere.x.shape[0],))

        mask = np.zeros(data.shape[:3])
        mask[2:5, 2:5, :] = 1
        sffit3 = sfmodel.fit(data, mask=mask)
        sphere = dpd.get_sphere()
        odf = sffit3.odf(sphere)
        pred = sffit3.predict(gtab)
        npt.assert_equal(pred.shape, data.shape)
        npt.assert_equal(odf.shape, data.shape[:-1] + (sphere.x.shape[0],))

        SNR = 1000
        S0 = 100
        mevals = np.array(([0.0015, 0.0005, 0.0005],
                           [0.0015, 0.0005, 0.0005]))
        angles = [(0, 0), (60, 0)]
        S, sticks = sims.multi_tensor(gtab, mevals, S0, angles=angles,
                                      fractions=[50, 50], snr=SNR)
        sffit = sfmodel.fit(S)
        pred = sffit.predict()
        npt.assert_(xval.coeff_of_determination(pred, S) > 96)
