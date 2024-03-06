import warnings

from dipy.reconst.mcsd import (mask_for_response_msmt,
                               response_from_mask_msmt,
                               auto_response_msmt)
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
import numpy as np
import numpy.testing as npt
import pytest

from dipy.sims.voxel import single_tensor, multi_tensor, add_noise
from dipy.reconst import shm
from dipy.data import default_sphere, get_3shell_gtab
from dipy.core.gradients import GradientTable
from dipy.testing.decorators import set_random_number_generator

from dipy.utils.optpkg import optional_package

cvx, have_cvxpy, _ = optional_package("cvxpy", min_version="1.4.1")
needs_cvxpy = pytest.mark.skipif(not have_cvxpy, reason="Requires CVXPY")


wm_response = np.array([[1.7E-3, 0.4E-3, 0.4E-3, 25.],
                        [1.7E-3, 0.4E-3, 0.4E-3, 25.],
                        [1.7E-3, 0.4E-3, 0.4E-3, 25.]])
csf_response = np.array([[3.0E-3, 3.0E-3, 3.0E-3, 100.],
                         [3.0E-3, 3.0E-3, 3.0E-3, 100.],
                         [3.0E-3, 3.0E-3, 3.0E-3, 100.]])
gm_response = np.array([[4.0E-4, 4.0E-4, 4.0E-4, 40.],
                        [4.0E-4, 4.0E-4, 4.0E-4, 40.],
                        [4.0E-4, 4.0E-4, 4.0E-4, 40.]])


def get_test_data(rng):
    gtab = get_3shell_gtab()
    evals_list = [np.array([1.7E-3, 0.4E-3, 0.4E-3]),
                  np.array([6.0E-4, 4.0E-4, 4.0E-4]),
                  np.array([3.0E-3, 3.0E-3, 3.0E-3])]
    s0 = [0.8, 1, 4]
    signals = [single_tensor(gtab, x[0], x[1]) for x in zip(s0, evals_list)]
    tissues = [0, 0, 2, 0, 1, 0, 0, 1, 2]  # wm=0, gm=1, csf=2
    data = [add_noise(signals[tissue], 80, s0[0], rng=rng) \
            for tissue in tissues]
    data = np.asarray(data).reshape((3, 3, 1, len(signals[0])))
    tissues = np.asarray(tissues).reshape((3, 3, 1))
    masks = [np.where(tissues == x, 1, 0) for x in range(3)]
    responses = [np.concatenate((x[0], [x[1]])) for x in zip(evals_list, s0)]
    return gtab, data, masks, responses


def _expand(m, iso, coeff):
    params = np.zeros(len(m))
    params[m == 0] = coeff[iso:]
    params = np.concatenate([coeff[:iso], params])
    return params


@needs_cvxpy
def test_mcsd_model_delta():
    sh_order_max = 8
    gtab = get_3shell_gtab()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        response = multi_shell_fiber_response(sh_order_max,
                                              [0, 1000, 2000, 3500],
                                              wm_response,
                                              gm_response,
                                              csf_response)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        model = MultiShellDeconvModel(gtab, response)
    iso = response.iso

    theta, phi = default_sphere.theta, default_sphere.phi
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        B = shm.real_sh_descoteaux_from_index(
            response.m_values, response.l_values, theta[:, None], phi[:, None])

    wm_delta = model.delta.copy()
    # set isotropic components to zero
    wm_delta[:iso] = 0.
    wm_delta = _expand(model.m_values, iso, wm_delta)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        for i, s in enumerate([0, 1000, 2000, 3500]):
            g = GradientTable(default_sphere.vertices * s)
            signal = model.predict(wm_delta, g)
            expected = np.dot(response.response[i, iso:], B.T)
            npt.assert_array_almost_equal(signal, expected)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        signal = model.predict(wm_delta, gtab)
    fit = model.fit(signal)
    m = model.m_values
    npt.assert_array_almost_equal(fit.shm_coeff[m != 0], 0., 2)


@needs_cvxpy
def test_MultiShellDeconvModel_response():
    gtab = get_3shell_gtab()

    sh_order_max = 8
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        response = multi_shell_fiber_response(sh_order_max,
                                              [0, 1000, 2000, 3500],
                                              wm_response,
                                              gm_response,
                                              csf_response)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        model_1 = MultiShellDeconvModel(gtab, response,
                                        sh_order_max=sh_order_max)
    responses = np.array([wm_response, gm_response, csf_response])
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        model_2 = MultiShellDeconvModel(gtab, responses,
                                        sh_order_max=sh_order_max)
    response_1 = model_1.response.response
    response_2 = model_2.response.response
    npt.assert_array_almost_equal(response_1, response_2, 0)

    npt.assert_raises(ValueError, MultiShellDeconvModel,
                      gtab, np.ones((4, 3, 4)))
    npt.assert_raises(ValueError, MultiShellDeconvModel,
                      gtab, np.ones((3, 3, 4)), iso=3)


@needs_cvxpy
def test_MultiShellDeconvModel():
    gtab = get_3shell_gtab()

    mevals = np.array([wm_response[0, :3], wm_response[0, :3]])
    angles = [(0, 0), (60, 0)]

    S_wm, sticks = multi_tensor(gtab, mevals, wm_response[0, 3], angles=angles,
                                fractions=[30., 70.], snr=None)
    S_gm = gm_response[0, 3] * np.exp(-gtab.bvals * gm_response[0, 0])
    S_csf = csf_response[0, 3] * np.exp(-gtab.bvals * csf_response[0, 0])

    sh_order_max = 8
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        response = multi_shell_fiber_response(sh_order_max,
                                              [0, 1000, 2000, 3500],
                                              wm_response,
                                              gm_response,
                                              csf_response)
        model = MultiShellDeconvModel(gtab, response)
    vf = [0.325, 0.2, 0.475]
    signal = sum(i * j for i, j in zip(vf, [S_csf, S_gm, S_wm]))
    fit = model.fit(signal)

    # Testing both ways to predict
    S_pred_fit = fit.predict()
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        S_pred_model = model.predict(fit.all_shm_coeff)

    npt.assert_array_almost_equal(S_pred_fit, S_pred_model, 0)
    npt.assert_array_almost_equal(S_pred_fit, signal, 0)


@needs_cvxpy
def test_MSDeconvFit():
    gtab = get_3shell_gtab()

    mevals = np.array([wm_response[0, :3], wm_response[0, :3]])
    angles = [(0, 0), (60, 0)]

    S_wm, sticks = multi_tensor(gtab, mevals, wm_response[0, 3], angles=angles,
                                fractions=[30., 70.], snr=None)
    S_gm = gm_response[0, 3] * np.exp(-gtab.bvals * gm_response[0, 0])
    S_csf = csf_response[0, 3] * np.exp(-gtab.bvals * csf_response[0, 0])

    sh_order_max = 8
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        response = multi_shell_fiber_response(sh_order_max,
                                              [0, 1000, 2000, 3500],
                                              wm_response,
                                              gm_response,
                                              csf_response)
        model = MultiShellDeconvModel(gtab, response)
    vf = [0.325, 0.2, 0.475]
    signal = sum(i * j for i, j in zip(vf, [S_csf, S_gm, S_wm]))
    fit = model.fit(signal)

    # Testing volume fractions
    npt.assert_array_almost_equal(fit.volume_fractions, vf, 1)


def test_multi_shell_fiber_response():

    sh_order_max = 8
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        response = multi_shell_fiber_response(sh_order_max,
                                              [0, 1000, 2000, 3500],
                                              wm_response,
                                              gm_response,
                                              csf_response)

    npt.assert_equal(response.response.shape, (4, 7))

    btens = ["LTE", "PTE", "STE", "CTE"]
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        response = multi_shell_fiber_response(sh_order_max,
                                              [0, 1000, 2000, 3500],
                                              wm_response,
                                              gm_response,
                                              csf_response,
                                              btens=btens)

    npt.assert_equal(response.response.shape, (4, 7))

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=PendingDeprecationWarning)
        response = multi_shell_fiber_response(sh_order_max, [1000, 2000, 3500],
                                              wm_response,
                                              gm_response,
                                              csf_response)
        # Test that the number of warnings raised is greater than 1, with
        # deprecation warnings being raised from using legacy SH bases as well
        # as a warning from multi_shell_fiber_response
        npt.assert_(len(w) > 1)
        # The last warning in list is the one from multi_shell_fiber_response
        npt.assert_(issubclass(w[-1].category, UserWarning))
        npt.assert_("""No b0 given. Proceeding either way.""" in
                    str(w[-1].message))
        npt.assert_equal(response.response.shape, (3, 7))


@set_random_number_generator()
def test_mask_for_response_msmt(rng):
    gtab, data, masks_gt, _ = get_test_data(rng)

    with warnings.catch_warnings(record=True) as w:
        wm_mask, gm_mask, csf_mask = mask_for_response_msmt(gtab, data,
                                                            roi_center=None,
                                                            roi_radii=(1, 1, 0),
                                                            wm_fa_thr=0.7,
                                                            gm_fa_thr=0.3,
                                                            csf_fa_thr=0.15,
                                                            gm_md_thr=0.001,
                                                            csf_md_thr=0.0032)

    npt.assert_equal(len(w), 1)
    npt.assert_(issubclass(w[0].category, UserWarning))
    npt.assert_("""Some b-values are higher than 1200.""" in
                str(w[0].message))

    # Verifies that masks are not empty:
    masks_sum = int(np.sum(wm_mask) + np.sum(gm_mask) + np.sum(csf_mask))
    npt.assert_equal(masks_sum != 0, True)

    npt.assert_array_almost_equal(masks_gt[0], wm_mask)
    npt.assert_array_almost_equal(masks_gt[1], gm_mask)
    npt.assert_array_almost_equal(masks_gt[2], csf_mask)


@set_random_number_generator()
def test_mask_for_response_msmt_nvoxels(rng):
    gtab, data, _, _ = get_test_data(rng)

    with warnings.catch_warnings(record=True) as w:
        wm_mask, gm_mask, csf_mask = mask_for_response_msmt(gtab, data,
                                                            roi_center=None,
                                                            roi_radii=(1, 1, 0),
                                                            wm_fa_thr=0.7,
                                                            gm_fa_thr=0.3,
                                                            csf_fa_thr=0.15,
                                                            gm_md_thr=0.001,
                                                            csf_md_thr=0.0032)

    npt.assert_equal(len(w), 1)
    npt.assert_(issubclass(w[0].category, UserWarning))
    npt.assert_("""Some b-values are higher than 1200.""" in
                str(w[0].message))

    wm_nvoxels = np.sum(wm_mask)
    gm_nvoxels = np.sum(gm_mask)
    csf_nvoxels = np.sum(csf_mask)
    npt.assert_equal(wm_nvoxels, 5)
    npt.assert_equal(gm_nvoxels, 2)
    npt.assert_equal(csf_nvoxels, 2)

    with warnings.catch_warnings(record=True) as w:
        wm_mask, gm_mask, csf_mask = mask_for_response_msmt(gtab, data,
                                                            roi_center=None,
                                                            roi_radii=(1, 1, 0),
                                                            wm_fa_thr=1,
                                                            gm_fa_thr=0,
                                                            csf_fa_thr=0,
                                                            gm_md_thr=0,
                                                            csf_md_thr=0)
        npt.assert_equal(len(w), 6)
        npt.assert_(issubclass(w[0].category, UserWarning))
        npt.assert_("""Some b-values are higher than 1200.""" in
                    str(w[0].message))
        npt.assert_("No voxel with a FA higher than 1 were found" in
                    str(w[1].message))
        npt.assert_("No voxel with a FA lower than 0 were found" in
                    str(w[2].message))
        npt.assert_("No voxel with a MD lower than 0 were found" in
                    str(w[3].message))
        npt.assert_("No voxel with a FA lower than 0 were found" in
                    str(w[4].message))
        npt.assert_("No voxel with a MD lower than 0 were found" in
                    str(w[5].message))

    wm_nvoxels = np.sum(wm_mask)
    gm_nvoxels = np.sum(gm_mask)
    csf_nvoxels = np.sum(csf_mask)
    npt.assert_equal(wm_nvoxels, 0)
    npt.assert_equal(gm_nvoxels, 0)
    npt.assert_equal(csf_nvoxels, 0)


@set_random_number_generator()
def test_response_from_mask_msmt(rng):
    gtab, data, masks_gt, responses_gt = get_test_data(rng)

    response_wm, response_gm, response_csf \
        = response_from_mask_msmt(gtab, data, masks_gt[0],
                                  masks_gt[1], masks_gt[2], tol=20)

    # Verifying that csf's response is greater than gm's
    npt.assert_equal(np.sum(response_csf[:, :3]) > np.sum(response_gm[:, :3]),
                     True)
    # Verifying that csf and gm are described by spheres
    npt.assert_almost_equal(response_csf[:, 1], response_csf[:, 2])
    npt.assert_allclose(response_csf[:, 0], response_csf[:, 1], rtol=1, atol=0)
    npt.assert_almost_equal(response_gm[:, 1], response_gm[:, 2])
    npt.assert_allclose(response_gm[:, 0], response_gm[:, 1], rtol=1, atol=0)
    # Verifying that wm is anisotropic in one direction
    npt.assert_almost_equal(response_wm[:, 1], response_wm[:, 2])
    npt.assert_equal(response_wm[:, 0] > 2.5 * response_wm[:, 1], True)

    # Verifying with ground truth for the first bvalue
    npt.assert_array_almost_equal(response_wm[0], responses_gt[0], 1)
    npt.assert_array_almost_equal(response_gm[0], responses_gt[1], 1)
    npt.assert_array_almost_equal(response_csf[0], responses_gt[2], 1)


@set_random_number_generator()
def test_auto_response_msmt(rng):
    gtab, data, _, _ = get_test_data(rng)

    with warnings.catch_warnings(record=True) as w:
        response_auto_wm, response_auto_gm, response_auto_csf = \
            auto_response_msmt(gtab, data, tol=20,
                               roi_center=None, roi_radii=(1, 1, 0),
                               wm_fa_thr=0.7, gm_fa_thr=0.3, csf_fa_thr=0.15,
                               gm_md_thr=0.001, csf_md_thr=0.0032)

        npt.assert_(issubclass(w[0].category, UserWarning))
        npt.assert_("""Some b-values are higher than 1200.
        The DTI fit might be affected. It is advised to use
        mask_for_response_msmt with bvalues lower than 1200, followed by
        response_from_mask_msmt with all bvalues to overcome this."""
                    in str(w[0].message))

        mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data,
                                                            roi_center=None,
                                                            roi_radii=(1, 1, 0),
                                                            wm_fa_thr=0.7,
                                                            gm_fa_thr=0.3,
                                                            csf_fa_thr=0.15,
                                                            gm_md_thr=0.001,
                                                            csf_md_thr=0.0032)

        response_from_mask_wm, response_from_mask_gm, response_from_mask_csf = \
            response_from_mask_msmt(gtab, data,
                                    mask_wm, mask_gm, mask_csf,
                                    tol=20)

        npt.assert_array_equal(response_auto_wm, response_from_mask_wm)
        npt.assert_array_equal(response_auto_gm, response_from_mask_gm)
        npt.assert_array_equal(response_auto_csf, response_from_mask_csf)
