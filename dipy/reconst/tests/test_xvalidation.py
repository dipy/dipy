"""

Testing cross-validation analysis


"""
from __future__ import division, print_function, absolute_import
import numpy as np
import numpy.testing as npt
import nibabel as nib
import dipy.reconst.xvalidation as xval
import dipy.data as dpd
import dipy.reconst.dti as dti
import dipy.core.gradients as gt
import dipy.sims.voxel as sims
import dipy.reconst.csdeconv as csd

np.random.seed(12345)

def test_coeff_of_determination():
    """
    Test the calculation of the coefficient of determination
    """

    model = np.random.randn(10,10,10,150)
    data = np.copy(model)
    # If the model predicts the data perfectly, the COD is all 100s:
    cod = xval.coeff_of_determination(data, model)
    npt.assert_array_equal(100 * np.ones(data.shape[:3]), cod)


def test_kfold_xval():
    """
    Test k-fold cross-validation
    """
    fdata, fbval, fbvec  = dpd.get_data('small_64D')
    data = nib.load(fdata).get_data()
    gtab = gt.gradient_table(fbval, fbvec)
    dm = dti.TensorModel(gtab, 'LS')
    # The data has 102 directions, so will not divide neatly into 10 bits
    npt.assert_raises(ValueError, xval.kfold_xval, dm, data, 10)

    psphere = dpd.get_sphere('symmetric362')
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = gt.gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [ np.array( [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ] ),
               np.array( [ [0, 0, 1], [0, 1, 0], [1, 0, 0] ] ) ]
    S = sims.single_tensor( gtab, 100, mevals[0], mevecs[0], snr=None )

    dm = dti.TensorModel(gtab, 'LS')
    dmfit = dm.fit(S)

    kf_xval = xval.kfold_xval(dm, S, 2)
    cod = xval.coeff_of_determination(S, kf_xval)
    npt.assert_array_almost_equal(cod, np.ones(kf_xval.shape[:-1])*100)


def test_csd_xval():
    psphere = dpd.get_sphere('symmetric362')
    bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    bvals = np.zeros(len(bvecs)) + 1000
    bvals[0] = 0
    gtab = gt.gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0001], [0.0015, 0.0003, 0.0003]))
    mevecs = [ np.array( [ [1, 0, 0], [0, 1, 0], [0, 0, 1] ] ),
               np.array( [ [0, 0, 1], [0, 1, 0], [1, 0, 0] ] ) ]
    S = sims.single_tensor( gtab, 100, mevals[0], mevecs[0], snr=None )
    response = ([0.0015, 0.0003, 0.0001], 1)
    sm = csd.ConstrainedSphericalDeconvModel(gtab, response)
    smfit = sm.fit(S)
    kf_xval = xval.kfold_xval(sm, S, 2, response, sh_order=2)
    # Because of the regularization, COD is not going to be perfect here:
    cod = xval.coeff_of_determination(S, kf_xval)
    # We'll just test for regressions:
    my_cod = 91.09995062835976 # pre-computed by hand for this random seed
    npt.assert_array_almost_equal(cod,
                                  np.ones(kf_xval.shape[:-1]) * my_cod)
