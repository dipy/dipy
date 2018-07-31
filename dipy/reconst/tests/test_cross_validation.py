"""

Testing cross-validation analysis

"""
from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.testing as npt
import nibabel as nib
import dipy.reconst.cross_validation as xval
import dipy.data as dpd
import dipy.reconst.dti as dti
import dipy.core.gradients as gt
import dipy.sims.voxel as sims
import dipy.reconst.csdeconv as csd
import dipy.reconst.base as base


# We'll set these globally:
fdata, fbval, fbvec = dpd.get_data('small_64D')


def test_coeff_of_determination():
    """
    Test the calculation of the coefficient of determination
    """

    model = np.random.randn(10, 10, 10, 150)
    data = np.copy(model)
    # If the model predicts the data perfectly, the COD is all 100s:
    cod = xval.coeff_of_determination(data, model)
    npt.assert_array_equal(100, cod)


def test_dti_xval():
    """
    Test k-fold cross-validation
    """
    data = nib.load(fdata).get_data()
    gtab = gt.gradient_table(fbval, fbvec)
    dm = dti.TensorModel(gtab, 'LS')
    # The data has 102 directions, so will not divide neatly into 10 bits
    npt.assert_raises(ValueError, xval.kfold_xval, dm, data, 10)

    # In simulation with no noise, COD should be perfect:
    psphere = dpd.get_sphere('symmetric362')
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = gt.gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
    S = sims.single_tensor(gtab, 100, mevals[0], mevecs[0], snr=None)

    dm = dti.TensorModel(gtab, 'LS')
    kf_xval = xval.kfold_xval(dm, S, 2)
    cod = xval.coeff_of_determination(S, kf_xval)
    npt.assert_array_almost_equal(cod, np.ones(kf_xval.shape[:-1]) * 100)

    # Test with 2D data for use of a mask
    S = np.array([[S, S], [S, S]])
    mask = np.ones(S.shape[:-1], dtype=bool)
    mask[1, 1] = 0
    kf_xval = xval.kfold_xval(dm, S, 2, mask=mask)
    cod2d = xval.coeff_of_determination(S, kf_xval)
    npt.assert_array_almost_equal(np.round(cod2d[0, 0]), cod)


def test_csd_xval():
    # First, let's see that it works with some data:
    data = nib.load(fdata).get_data()[1:3, 1:3, 1:3]  # Make it *small*
    gtab = gt.gradient_table(fbval, fbvec)
    S0 = np.mean(data[..., gtab.b0s_mask])
    response = ([0.0015, 0.0003, 0.0001], S0)
    csdm = csd.ConstrainedSphericalDeconvModel(gtab, response)

    # In simulation, it should work rather well (high COD):
    psphere = dpd.get_sphere('symmetric362')
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = gt.gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
    S0 = 100
    S = sims.single_tensor(gtab, S0, mevals[0], mevecs[0], snr=None)
    sm = csd.ConstrainedSphericalDeconvModel(gtab, response)
    np.random.seed(12345)
    response = ([0.0015, 0.0003, 0.0001], S0)
    kf_xval = xval.kfold_xval(sm, S, 2, response, sh_order=2)
    # Because of the regularization, COD is not going to be perfect here:
    cod = xval.coeff_of_determination(S, kf_xval)
    # We'll just test for regressions:
    csd_cod = 97  # pre-computed by hand for this random seed

    # We're going to be really lenient here:
    npt.assert_array_almost_equal(np.round(cod), csd_cod)
    # Test for sD data with more than one voxel for use of a mask:
    S = np.array([[S, S], [S, S]])
    mask = np.ones(S.shape[:-1], dtype=bool)
    mask[1, 1] = 0
    kf_xval = xval.kfold_xval(sm, S, 2, response, sh_order=2,
                              mask=mask)

    cod = xval.coeff_of_determination(S, kf_xval)
    npt.assert_array_almost_equal(np.round(cod[0]), csd_cod)


def test_no_predict():
    """
    Test that if you try to do this with a model that doesn't have a `predict`
    method, you get something reasonable.
    """
    class NoPredictModel(base.ReconstModel):
        def __init__(self, gtab):
            base.ReconstModel.__init__(self, gtab)

        def fit(self, data, mask=None):
            return NoPredictFit(self, data, mask=mask)

    class NoPredictFit(base.ReconstFit):
        def __init__(self, model, data, mask=None):
            base.ReconstFit.__init__(self, model, data)

    gtab = gt.gradient_table(fbval, fbvec)
    my_model = NoPredictModel(gtab)
    data = nib.load(fdata).get_data()[1:3, 1:3, 1:3]  # Whatever

    npt.assert_raises(ValueError,  xval.kfold_xval, my_model, data, 2)
