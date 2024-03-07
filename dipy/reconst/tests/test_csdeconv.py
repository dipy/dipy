import warnings
import numpy as np
import numpy.testing as npt
from numpy.testing import (assert_, assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_array_equal)
from dipy.testing import assert_greater, assert_greater_equal
from dipy.data import get_sphere, get_fnames, default_sphere, small_sphere
from dipy.sims.voxel import (multi_tensor,
                             single_tensor,
                             multi_tensor_odf,
                             all_tensor_evecs, single_tensor_odf)
from dipy.core.gradients import gradient_table
from dipy.core.sphere import Sphere, HemiSphere
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel,
                                   forward_sdeconv_mat,
                                   odf_deconv,
                                   odf_sh_to_sharp,
                                   mask_for_response_ssst,
                                   response_from_mask_ssst,
                                   response_from_mask,
                                   auto_response_ssst,
                                   auto_response,
                                   recursive_response)
from dipy.direction.peaks import peak_directions
from dipy.reconst.dti import TensorModel, fractional_anisotropy
from dipy.reconst.shm import (
    descoteaux07_legacy_msg,
    QballModel,
    sf_to_sh,
    sh_to_sf,
    real_sh_descoteaux,
    sph_harm_ind_list
)
from dipy.reconst.shm import lazy_index
from dipy.utils.deprecator import ExpiredDeprecationError
from dipy.io.gradients import read_bvals_bvecs
from dipy.testing.decorators import set_random_number_generator


def get_test_data():
    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    evals_list = [np.array([1.7E-3, 0.4E-3, 0.4E-3]),
                  np.array([4.0E-4, 4.0E-4, 4.0E-4]),
                  np.array([3.0E-3, 3.0E-3, 3.0E-3])]
    s0 = [0.8, 1, 4]
    signals = [single_tensor(gtab, x[0], x[1]) for x in zip(s0, evals_list)]
    tissues = [0, 0, 2, 0, 1, 0, 0, 1, 2]
    data = [signals[tissue] for tissue in tissues]
    data = np.asarray(data).reshape((3, 3, 1, len(signals[0])))
    evals = [evals_list[tissue] for tissue in tissues]
    evals = np.asarray(evals).reshape((3, 3, 1, 3))
    tissues = np.asarray(tissues).reshape((3, 3, 1))
    mask = np.where(tissues == 0, 1, 0)
    response = (evals_list[0], s0[0])
    fa = fractional_anisotropy(evals)
    return gtab, data, mask, response, fa


def test_auto_response_deprecated():
    gtab, data, _, _, _ = get_test_data()
    npt.assert_raises(ExpiredDeprecationError, auto_response,
                      gtab, data, roi_center=None, roi_radius=1, fa_thr=0.7)


def test_response_from_mask_deprecated():
    gtab, data, mask, _, _ = get_test_data()
    npt.assert_raises(ExpiredDeprecationError, response_from_mask,
                      gtab, data, mask)


def test_recursive_response_calibration():
    """
    Test the recursive response calibration method.
    """
    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_fnames('small_64D')

    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    sphere = default_sphere

    gtab = gradient_table(bvals, bvecs)
    evals = np.array([0.0015, 0.0003, 0.0003])
    evecs = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]]).T
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    angles = [(0, 0), (90, 0)]

    where_dwi = lazy_index(~gtab.b0s_mask)

    S_cross, _ = multi_tensor(gtab, mevals, S0, angles=angles,
                              fractions=[50, 50], snr=SNR)

    S_single = single_tensor(gtab, S0, evals, evecs, snr=SNR)

    data = np.concatenate((np.tile(S_cross, (8, 1)),
                           np.tile(S_single, (2, 1))),
                          axis=0)

    odf_gt_cross = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])

    odf_gt_single = single_tensor_odf(sphere.vertices, evals, evecs)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        response = recursive_response(gtab, data, mask=None, sh_order_max=8,
                                      peak_thr=0.01, init_fa=0.05,
                                      init_trace=0.0021, iter=8,
                                      convergence=0.001, parallel=False)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd = ConstrainedSphericalDeconvModel(gtab, response)

    csd_fit = csd.fit(data)

    assert_equal(np.all(csd_fit.shm_coeff[:, 0] >= 0), True)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        fodf = csd_fit.odf(sphere)

    directions_gt_single, _, _ = peak_directions(odf_gt_single, sphere)
    directions_gt_cross, _, _ = peak_directions(odf_gt_cross, sphere)
    directions_single, _, _ = peak_directions(fodf[8, :], sphere)
    directions_cross, _, _ = peak_directions(fodf[0, :], sphere)

    ang_sim = angular_similarity(directions_cross, directions_gt_cross)
    assert_equal(ang_sim > 1.9, True)
    assert_equal(directions_cross.shape[0], 2)
    assert_equal(directions_gt_cross.shape[0], 2)

    ang_sim = angular_similarity(directions_single, directions_gt_single)
    assert_equal(ang_sim > 0.9, True)
    assert_equal(directions_single.shape[0], 1)
    assert_equal(directions_gt_single.shape[0], 1)

    with warnings.catch_warnings(record=True) as w:
        sphere = Sphere(xyz=gtab.gradients[where_dwi])
        npt.assert_equal(len(w), 1)
        npt.assert_(issubclass(w[0].category, UserWarning))
        npt.assert_("Vertices are not on the unit sphere" in str(w[0].message))
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        sf = response.on_sphere(sphere)
    S = np.concatenate(([response.S0], sf))

    tenmodel = TensorModel(gtab, min_signal=0.001)

    tenfit = tenmodel.fit(S)
    FA = fractional_anisotropy(tenfit.evals)
    FA_gt = fractional_anisotropy(evals)
    assert_almost_equal(FA, FA_gt, 1)


def test_mask_for_response_ssst():
    gtab, data, mask_gt, _, _ = get_test_data()

    mask = mask_for_response_ssst(gtab, data,
                                  roi_center=None,
                                  roi_radii=(1, 1, 0),
                                  fa_thr=0.7)

    # Verifies that mask is not empty:
    assert_equal(int(np.sum(mask)) != 0, True)

    assert_array_almost_equal(mask_gt, mask)


def test_mask_for_response_ssst_nvoxels():
    gtab, data, _, _, _ = get_test_data()

    mask = mask_for_response_ssst(gtab, data,
                                  roi_center=None,
                                  roi_radii=(1, 1, 0),
                                  fa_thr=0.7)

    nvoxels = np.sum(mask)
    assert_equal(nvoxels, 5)

    with warnings.catch_warnings(record=True) as w:
        mask = mask_for_response_ssst(gtab, data,
                                      roi_center=None,
                                      roi_radii=(1, 1, 0),
                                      fa_thr=1)
        npt.assert_equal(len(w), 1)
        npt.assert_(issubclass(w[0].category, UserWarning))
        npt.assert_("No voxel with a FA higher than 1 were found" in
                    str(w[0].message))

    nvoxels = np.sum(mask)
    assert_equal(nvoxels, 0)


def test_response_from_mask_ssst():
    gtab, data, mask_gt, response_gt, _ = get_test_data()

    response, _ = response_from_mask_ssst(gtab, data, mask_gt)

    assert_array_almost_equal(response[0], response_gt[0])
    assert_equal(response[1], response_gt[1])


def test_auto_response_ssst():
    gtab, data, _, _, _ = get_test_data()

    response_auto, ratio_auto = auto_response_ssst(gtab,
                                                   data,
                                                   roi_center=None,
                                                   roi_radii=(1, 1, 0),
                                                   fa_thr=0.7)

    mask = mask_for_response_ssst(gtab, data,
                                  roi_center=None,
                                  roi_radii=(1, 1, 0),
                                  fa_thr=0.7)

    response_from_mask, ratio_from_mask = response_from_mask_ssst(gtab,
                                                                  data,
                                                                  mask)

    assert_array_equal(response_auto[0], response_from_mask[0])
    assert_equal(response_auto[1], response_from_mask[1])
    assert_array_equal(ratio_auto, ratio_from_mask)


def test_csdeconv():
    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_fnames('small_64D')

    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs, b0_threshold=0)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (60, 0)]

    S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric362')
    odf_gt = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])
    response = (np.array([0.0015, 0.0003, 0.0003]), S0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd = ConstrainedSphericalDeconvModel(gtab, response)
    csd_fit = csd.fit(S)
    assert_equal(csd_fit.shm_coeff[0] > 0, True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        fodf = csd_fit.odf(sphere)

    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions, directions2)

    assert_equal(ang_sim > 1.9, True)
    assert_equal(directions.shape[0], 2)
    assert_equal(directions2.shape[0], 2)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        _ = ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=10)
        assert_greater(len([lw for lw in w if issubclass(lw.category,
                                                         UserWarning)]), 0)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        ConstrainedSphericalDeconvModel(gtab, response, sh_order_max=8)
        assert_equal(len([lw for lw in w if issubclass(lw.category,
                                                       UserWarning)]), 0)

    mevecs = []
    for s in sticks:
        mevecs += [all_tensor_evecs(s).T]

    S2 = single_tensor(gtab, 100, mevals[0], mevecs[0], snr=None)
    big_S = np.zeros((10, 10, 10, len(S2)))
    big_S[:] = S2

    aresponse, aratio = auto_response_ssst(gtab, big_S, roi_center=(5, 5, 4),
                                           roi_radii=3, fa_thr=0.5)
    assert_array_almost_equal(aresponse[0], response[0])
    assert_almost_equal(aresponse[1], 100)
    assert_almost_equal(aratio, response[0][1] / response[0][0])

    auto_response_ssst(gtab, big_S, roi_radii=3, fa_thr=0.5)
    assert_array_almost_equal(aresponse[0], response[0])


def test_odfdeconv():
    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (90, 0)]
    S, _ = multi_tensor(gtab, mevals, S0, angles=angles,
                        fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric362')

    odf_gt = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])

    e1 = 15.0
    e2 = 3.0
    ratio = e2 / e1

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd = ConstrainedSDTModel(gtab, ratio, None)

    csd_fit = csd.fit(S)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        fodf = csd_fit.odf(sphere)

    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions, directions2)

    assert_equal(ang_sim > 1.9, True)

    assert_equal(directions.shape[0], 2)
    assert_equal(directions2.shape[0], 2)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=PendingDeprecationWarning)

        ConstrainedSDTModel(gtab, ratio, sh_order_max=10)
        w_count = len(w)
        # A warning is expected from the ConstrainedSDTModel constructor
        # and additional warnings should be raised where legacy SH bases
        # are used
        assert_equal(w_count > 1, True)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=PendingDeprecationWarning)

        ConstrainedSDTModel(gtab, ratio, sh_order_max=8)
        # Test that the warning from ConstrainedSDTModel
        # constructor is no more raised
        assert_equal(len(w) == w_count - 1, True)

    csd_fit = csd.fit(np.zeros_like(S))
    fodf = csd_fit.odf(sphere)
    assert_array_equal(fodf, np.zeros_like(fodf))

    odf_sh = np.zeros_like(fodf)
    odf_sh[1] = np.nan

    fodf, _ = odf_deconv(odf_sh, csd.R, csd.B_reg)
    assert_array_equal(fodf, np.zeros_like(fodf))


def test_odf_sh_to_sharp():
    SNR = None
    S0 = 1
    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, _ = multi_tensor(gtab, mevals, S0, angles=[(10, 0), (100, 0)],
                        fractions=[50, 50], snr=SNR)

    sphere = default_sphere

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        qb = QballModel(gtab, sh_order_max=8, assume_normed=True)

    qbfit = qb.fit(S)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        odf_gt = qbfit.odf(sphere)

    Z = np.linalg.norm(odf_gt)

    odfs_gt = np.zeros((3, 1, 1, odf_gt.shape[0]))
    odfs_gt[:, :, :] = odf_gt[:]

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        odfs_sh = sf_to_sh(odfs_gt, sphere, sh_order_max=8, basis_type=None)

    odfs_sh /= Z

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        fodf_sh = odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15.,
                                  sh_order_max=8, lambda_=1., tau=0.1)

        fodf = sh_to_sf(fodf_sh, sphere, sh_order_max=8, basis_type=None)

    directions2, _, _ = peak_directions(fodf[0, 0, 0], sphere)

    assert_equal(directions2.shape[0], 2)


def test_forward_sdeconv_mat():
    _, l_values = sph_harm_ind_list(4)
    mat = forward_sdeconv_mat(np.array([0, 2, 4]), l_values)
    expected = np.diag([0, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    npt.assert_array_equal(mat, expected)

    sh_order_max = 8
    expected_size = (sh_order_max + 1) * (sh_order_max + 2) / 2
    r_rh = np.arange(0, sh_order_max + 1, 2)
    m_values, l_values = sph_harm_ind_list(sh_order_max)
    mat = forward_sdeconv_mat(r_rh, l_values)
    npt.assert_equal(mat.shape, (expected_size, expected_size))
    npt.assert_array_equal(mat.diagonal(), l_values)

    # Odd spherical harmonic degrees should raise a ValueError
    l_values[2] = 3
    npt.assert_raises(ValueError, forward_sdeconv_mat, r_rh, l_values)


def test_r2_term_odf_sharp():
    SNR = None
    S0 = 1
    angle = 45  # 45 degrees is a very tight angle to disentangle

    _, fbvals, fbvecs = get_fnames('small_64D')  # get_fnames('small_64D')

    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

    sphere = default_sphere
    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (angle, 0)]

    S, _ = multi_tensor(gtab, mevals, S0, angles=angles,
                        fractions=[50, 50], snr=SNR)

    odf_gt = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        odfs_sh = sf_to_sh(odf_gt, sphere, sh_order_max=8, basis_type=None)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        fodf_sh = odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15.,
                                  sh_order_max=8, lambda_=1., tau=0.1,
                                  r2_term=True)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        fodf = sh_to_sf(fodf_sh, sphere, sh_order_max=8, basis_type=None)

    directions_gt, _, _ = peak_directions(odf_gt, sphere)
    directions, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions_gt, directions)
    assert_equal(ang_sim > 1.9, True)
    assert_equal(directions.shape[0], 2)

    # This should pass as well
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        sdt_model = ConstrainedSDTModel(gtab, ratio=3/15., sh_order_max=8)
    sdt_fit = sdt_model.fit(S)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        fodf = sdt_fit.odf(sphere)

    directions_gt, _, _ = peak_directions(odf_gt, sphere)
    directions, _, _ = peak_directions(fodf, sphere)
    ang_sim = angular_similarity(directions_gt, directions)
    assert_equal(ang_sim > 1.9, True)
    assert_equal(directions.shape[0], 2)


@set_random_number_generator()
def test_csd_predict(rng):
    """
    Test prediction API
    """
    SNR = 100
    S0 = 1
    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))
    angles = [(0, 0), (60, 0)]
    S, _ = multi_tensor(gtab, mevals, S0, angles=angles,
                        fractions=[50, 50], snr=SNR)
    sphere = small_sphere
    multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])
    response = (np.array([0.0015, 0.0003, 0.0003]), S0)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd = ConstrainedSphericalDeconvModel(gtab, response)
    csd_fit = csd.fit(S)

    # Predicting from a fit should give the same result as predicting from a
    # model, S0 is 1 by default
    prediction1 = csd_fit.predict()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        prediction2 = csd.predict(csd_fit.shm_coeff)
    npt.assert_array_equal(prediction1, prediction2)
    npt.assert_array_equal(prediction1[..., gtab.b0s_mask], 1.)

    # Same with a different S0
    prediction1 = csd_fit.predict(S0=123.)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        prediction2 = csd.predict(csd_fit.shm_coeff, S0=123.)
    npt.assert_array_equal(prediction1, prediction2)
    npt.assert_array_equal(prediction1[..., gtab.b0s_mask], 123.)

    # For "well behaved" coefficients, the model should be able to find the
    # coefficients from the predicted signal.
    coeff = rng.random(csd_fit.shm_coeff.shape) - .5
    coeff[..., 0] = 10.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        S = csd.predict(coeff)
    csd_fit = csd.fit(S)
    npt.assert_array_almost_equal(coeff, csd_fit.shm_coeff)

    # Test predict on nd-data set
    S_nd = np.zeros((2, 3, 4, S.size))
    S_nd[:] = S
    fit = csd.fit(S_nd)
    predict1 = fit.predict()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        predict2 = csd.predict(fit.shm_coeff)
    npt.assert_array_almost_equal(predict1, predict2)


@set_random_number_generator()
def test_csd_predict_multi(rng):
    """
    Check that we can predict reasonably from multi-voxel fits:

    """
    S0 = 123.
    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    response = (np.array([0.0015, 0.0003, 0.0003]), S0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd = ConstrainedSphericalDeconvModel(gtab, response)
    coeff = rng.random(45) - .5
    coeff[..., 0] = 10.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        S = csd.predict(coeff, S0=123.)
    multi_S = np.array([[S, S], [S, S]])
    csd_fit_multi = csd.fit(multi_S)
    S0_multi = np.mean(multi_S[..., gtab.b0s_mask], -1)
    pred_multi = csd_fit_multi.predict(S0=S0_multi)
    npt.assert_array_almost_equal(pred_multi, multi_S)


def test_sphere_scaling_csdmodel():
    """Check that mirroring regularization sphere does not change the result of
    the model"""
    _, fbvals, fbvecs = get_fnames('small_64D')

    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (60, 0)]

    S, _ = multi_tensor(gtab, mevals, 100., angles=angles,
                        fractions=[50, 50], snr=None)

    hemi = small_sphere
    sphere = hemi.mirror()

    response = (np.array([0.0015, 0.0003, 0.0003]), 100)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        model_full = ConstrainedSphericalDeconvModel(gtab, response,
                                                     reg_sphere=sphere)
        model_hemi = ConstrainedSphericalDeconvModel(gtab, response,
                                                     reg_sphere=hemi)
    csd_fit_full = model_full.fit(S)
    csd_fit_hemi = model_hemi.fit(S)

    assert_array_almost_equal(csd_fit_full.shm_coeff, csd_fit_hemi.shm_coeff)


def test_default_lambda_csdmodel():
    """We check that the default value of lambda is the expected value with
    the symmetric362 sphere. This value has empirically been found to work well
    and changes to this default value should be discussed with the dipy team.
    """
    expected_lambda = {4: 27.5230088, 8: 82.5713865, 16: 216.0843135}
    expected_csdmodel_warnings = {4: 0, 8: 0, 16: 1}
    expected_sh_basis_deprecation_warnings = 3
    sphere = default_sphere

    # Create gradient table
    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    # Some response function
    response = (np.array([0.0015, 0.0003, 0.0003]), 100)

    for sh_order_max, expected, e_warn in zip(expected_lambda.keys(),
                                          expected_lambda.values(),
                                          expected_csdmodel_warnings.values()):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", category=PendingDeprecationWarning)
            s_o_m = sh_order_max
            model_full = ConstrainedSphericalDeconvModel(gtab, response,
                                                         sh_order_max=s_o_m,
                                                         reg_sphere=sphere)
            npt.assert_equal(len(w) - expected_sh_basis_deprecation_warnings,
                             e_warn)
            if e_warn:
                npt.assert_(issubclass(w[0].category, UserWarning))
                npt.assert_("Number of parameters required " in str(w[0].
                                                                    message))

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=descoteaux07_legacy_msg,
                category=PendingDeprecationWarning)
            B_reg, _, _ = real_sh_descoteaux(
                sh_order_max, sphere.theta, sphere.phi)
        npt.assert_array_almost_equal(model_full.B_reg, expected * B_reg)


def test_csd_superres():
    """ Check the quality of csdfit with high SH order. """
    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    # img, gtab = read_stanford_hardi()
    evals = np.array([[1.5, .3, .3]]) * [[1.], [1.]] / 1000.
    S, sticks = multi_tensor(gtab, evals, snr=None, fractions=[55., 45.])

    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings(action="always",
                                message="Number of parameters required.*",
                                category=UserWarning)
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        model16 = ConstrainedSphericalDeconvModel(gtab, (evals[0], 3.),
                                                  sh_order_max=16)
        assert_greater_equal(len(w), 1)
        npt.assert_(issubclass(w[0].category, UserWarning))

    fit16 = model16.fit(S)

    sphere = HemiSphere.from_sphere(get_sphere('symmetric724'))
    # print local_maxima(fit16.odf(default_sphere), default_sphere.edges)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        d, v, ind = peak_directions(fit16.odf(sphere), sphere,
                                    relative_peak_threshold=.2,
                                    min_separation_angle=0)

    # Check that there are two peaks
    assert_equal(len(d), 2)

    # Check that peaks line up with sticks
    cos_sim = abs((d * sticks).sum(1)) ** .5
    assert_(all(cos_sim > .99))


def test_csd_convergence():
    """ Check existence of `convergence` keyword in CSD model """
    _, fbvals, fbvecs = get_fnames('small_64D')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)

    evals = np.array([[1.5, .3, .3]]) * [[1.], [1.]] / 1000.
    S, sticks = multi_tensor(gtab, evals, snr=None, fractions=[55., 45.])

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        model_w_conv = ConstrainedSphericalDeconvModel(
            gtab,
            (evals[0], 3.),
            sh_order_max=8,
            convergence=50,
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        model_wo_conv = ConstrainedSphericalDeconvModel(gtab, (evals[0], 3.),
                                                        sh_order_max=8)

    assert_equal(model_w_conv.fit(S).shm_coeff, model_wo_conv.fit(S).shm_coeff)
