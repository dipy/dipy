import warnings

from dipy.reconst.mcsd import (mask_for_response_msmt,
                               response_from_mask_msmt,
                               auto_response_msmt)
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from dipy.reconst import mcsd
import numpy as np
import numpy.testing as npt
import pytest

from dipy.sims.voxel import single_tensor, multi_tensor, add_noise
from dipy.reconst import shm
from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity
from dipy.data import default_sphere, get_3shell_gtab, get_fnames
from dipy.core.gradients import GradientTable, gradient_table

from dipy.io.gradients import read_bvals_bvecs

from dipy.utils.optpkg import optional_package
cvx, have_cvxpy, _ = optional_package("cvxpy")

needs_cvxpy = pytest.mark.skipif(not have_cvxpy)


wm_response = np.array([1.7E-3, 0.4E-3, 0.4E-3, 25.])
csf_response = np.array([3.0E-3, 3.0E-3, 3.0E-3, 100.])
gm_response = np.array([4.0E-4, 4.0E-4, 4.0E-4, 40.])


def get_test_data():
    _, fbvals, fbvecs, _ = get_fnames('cfin_multib')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs)
    evals_list = [np.array([1.7E-3, 0.4E-3, 0.4E-3]),
                  np.array([4.0E-4, 4.0E-4, 4.0E-4]),
                  np.array([3.0E-3, 3.0E-3, 3.0E-3])]
    s0 = [25., 100., 40.]
    signals = [single_tensor(gtab, x[0], x[1]) for x in zip(s0, evals_list)]
    tissues = [0, 0, 2, 0, 1, 0, 0, 1, 2]
    data = [add_noise(signals[tissue], None, s0[0]) for tissue in tissues]
    data = np.asarray(data).reshape((3, 3, 1, len(signals[0])))
    evals = [evals_list[tissue] for tissue in tissues]
    evals = np.asarray(evals).reshape((3, 3, 1, 3))
    tissues = np.asarray(tissues).reshape((3, 3, 1))
    masks = [np.where(tissues == x, 1, 0) for x in range(3)]
    responses = [np.concatenate((x[0], [x[1]])) for x in zip(evals_list, s0)]
    fa = fractional_anisotropy(evals)
    md = mean_diffusivity(evals)
    return (gtab, data, masks, responses, fa, md)


def _expand(m, iso, coeff):
    params = np.zeros(len(m))
    params[m == 0] = coeff[iso:]
    params = np.concatenate([coeff[:iso], params])
    return params


@pytest.mark.skipif(not mcsd.have_cvxpy, reason="Requires CVXPY")
def test_mcsd_model_delta():
    sh_order = 8
    gtab = get_3shell_gtab()
    shells = np.unique(gtab.bvals // 100.) * 100.
    response = multi_shell_fiber_response(sh_order, shells,
                                          wm_response,
                                          gm_response,
                                          csf_response)
    model = MultiShellDeconvModel(gtab, response)
    iso = response.iso

    theta, phi = default_sphere.theta, default_sphere.phi
    B = shm.real_sph_harm(response.m, response.n, theta[:, None], phi[:, None])

    wm_delta = model.delta.copy()
    # set isotropic components to zero
    wm_delta[:iso] = 0.
    wm_delta = _expand(model.m, iso, wm_delta)

    for i, s in enumerate(shells):
        g = GradientTable(default_sphere.vertices * s)
        signal = model.predict(wm_delta, g)
        expected = np.dot(response.response[i, iso:], B.T)
        npt.assert_array_almost_equal(signal, expected)

    signal = model.predict(wm_delta, gtab)
    fit = model.fit(signal)
    m = model.m
    npt.assert_array_almost_equal(fit.shm_coeff[m != 0], 0., 2)


@pytest.mark.skipif(not mcsd.have_cvxpy, reason="Requires CVXPY")
def test_compartments():
    # test for failure if no. of compartments less than 2
    gtab = get_3shell_gtab()
    sh_order = 8
    response = multi_shell_fiber_response(sh_order, [0, 1000, 2000, 3500],
                                          wm_response,
                                          gm_response,
                                          csf_response)
    npt.assert_raises(ValueError, MultiShellDeconvModel, gtab, response, iso=1)


@pytest.mark.skipif(not mcsd.have_cvxpy, reason="Requires CVXPY")
def test_MultiShellDeconvModel():
    gtab = get_3shell_gtab()

    mevals = np.array([wm_response[:3], wm_response[:3]])
    angles = [(0, 0), (60, 0)]

    S_wm, sticks = multi_tensor(gtab, mevals, wm_response[3], angles=angles,
                                fractions=[30., 70.], snr=None)
    S_gm = gm_response[3] * np.exp(-gtab.bvals * gm_response[0])
    S_csf = csf_response[3] * np.exp(-gtab.bvals * csf_response[0])

    sh_order = 8
    response = multi_shell_fiber_response(sh_order, [0, 1000, 2000, 3500],
                                          wm_response,
                                          gm_response,
                                          csf_response)
    model = MultiShellDeconvModel(gtab, response)
    vf = [0.325, 0.2, 0.475]
    signal = sum(i * j for i, j in zip(vf, [S_csf, S_gm, S_wm]))
    fit = model.fit(signal)

    npt.assert_array_almost_equal(fit.volume_fractions, vf, 1)

    S_pred = fit.predict()
    npt.assert_array_almost_equal(S_pred, signal, 0)


# def test_mask_for_response_msmt():
#     gtab, data, masks_gt, _, fa, md = get_test_data()

#     wm_mask, gm_mask, csf_mask = mask_for_response_msmt(gtab, data,
#                                     roi_center=None, roi_radii=(1, 1, 0),
#                                     fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
#                                     csf_fa_thr=0.15, md_data=None,
#                                     gm_md_thr=0.001, csf_md_thr=0.0032)

#     # Verifies that masks are not empty:
#     masks_sum = int(np.sum(wm_mask) + np.sum(gm_mask) + np.sum(csf_mask))
#     npt.assert_equal(masks_sum != 0, True)

#     wm_mask_fa_md, gm_mask_fa_md, csf_mask_fa_md = mask_for_response_msmt(
#                                 gtab, data,
#                                 roi_center=None, roi_radii=(1, 1, 0),
#                                 fa_data=fa, wm_fa_thr=0.7, gm_fa_thr=0.3,
#                                 csf_fa_thr=0.15, md_data=md,
#                                 gm_md_thr=0.001, csf_md_thr=0.0032)

#     npt.assert_array_almost_equal(wm_mask_fa_md, wm_mask)
#     npt.assert_array_almost_equal(gm_mask_fa_md, gm_mask)
#     npt.assert_array_almost_equal(csf_mask_fa_md, csf_mask)
#     npt.assert_array_almost_equal(masks_gt[0], wm_mask_fa_md)
#     npt.assert_array_almost_equal(masks_gt[1], gm_mask_fa_md)
#     npt.assert_array_almost_equal(masks_gt[2], csf_mask_fa_md)


# def test_mask_for_response_msmt_nvoxels():
#     gtab, data, _, _, _, _ = get_test_data()

#     wm_mask, gm_mask, csf_mask = mask_for_response_msmt(gtab, data,
#                                     roi_center=None, roi_radii=(1, 1, 0),
#                                     fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
#                                     csf_fa_thr=0.15, md_data=None,
#                                     gm_md_thr=0.001, csf_md_thr=0.0032)

#     wm_nvoxels = np.sum(wm_mask)
#     gm_nvoxels = np.sum(gm_mask)
#     csf_nvoxels = np.sum(csf_mask)
#     npt.assert_equal(wm_nvoxels, 5)
#     npt.assert_equal(gm_nvoxels, 2)
#     npt.assert_equal(csf_nvoxels, 2)

#     with warnings.catch_warnings(record=True) as w:
#         wm_mask, gm_mask, csf_mask = mask_for_response_msmt(gtab, data,
#                                         roi_center=None, roi_radii=(1, 1, 0),
#                                         fa_data=None, wm_fa_thr=1, gm_fa_thr=0,
#                                         csf_fa_thr=0, md_data=None,
#                                         gm_md_thr=0, csf_md_thr=0)
#         npt.assert_equal(len(w), 5)
#         npt.assert_(issubclass(w[0].category, UserWarning))
#         npt.assert_("No voxel with a FA higher than 1 were found" in
#                     str(w[0].message))
#         npt.assert_("No voxel with a FA lower than 0 were found" in
#                     str(w[1].message))
#         npt.assert_("No voxel with a MD lower than 0 were found" in
#                     str(w[2].message))
#         npt.assert_("No voxel with a FA lower than 0 were found" in
#                     str(w[3].message))
#         npt.assert_("No voxel with a MD lower than 0 were found" in
#                     str(w[4].message))

#     wm_nvoxels = np.sum(wm_mask)
#     gm_nvoxels = np.sum(gm_mask)
#     csf_nvoxels = np.sum(csf_mask)
#     npt.assert_equal(wm_nvoxels, 0)
#     npt.assert_equal(gm_nvoxels, 0)
#     npt.assert_equal(csf_nvoxels, 0)


# def test_response_from_mask_msmt():
#     gtab, data, masks_gt, responses_gt, _, _ = get_test_data()

#     response_wm, response_gm, response_csf = response_from_mask_msmt(gtab,
#                                                 data, masks_gt[0],
#                                                 masks_gt[1], masks_gt[2],
#                                                 tol=20)

#     # Verifying that csf's response is greater than gm's
#     npt.assert_equal(np.sum(response_csf[:3]) > np.sum(response_gm[:3]), True)
#     # Verifying that csf and gm are described by spheres
#     npt.assert_almost_equal(response_csf[1], response_csf[2])
#     npt.assert_almost_equal(response_csf[0], response_csf[1])
#     npt.assert_almost_equal(response_gm[1], response_gm[2])
#     npt.assert_allclose(response_gm[0], response_gm[1], rtol=1, atol=0)
#     # Verifying that wm is anisotropic in one direction
#     npt.assert_almost_equal(response_wm[1], response_wm[2])
#     npt.assert_equal(response_wm[0] > 2.5 * response_wm[1], True)

#     # Verifying with ground truth
#     npt.assert_array_almost_equal(response_wm, responses_gt[0])
#     npt.assert_array_almost_equal(response_gm, responses_gt[1])
#     npt.assert_array_almost_equal(response_csf, responses_gt[2])


# def test_auto_response_msmt():
#     gtab, data, _, _, _, _ = get_test_data()

#     response_auto_wm, response_auto_gm, response_auto_csf = \
#         auto_response_msmt(gtab, data, tol=20,
#                            roi_center=None, roi_radii=(1, 1, 0),
#                            fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
#                            csf_fa_thr=0.15, md_data=None,
#                            gm_md_thr=0.001, csf_md_thr=0.0032)

#     mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab, data,
#                                     roi_center=None, roi_radii=(1, 1, 0),
#                                     fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
#                                     csf_fa_thr=0.15, md_data=None,
#                                     gm_md_thr=0.001, csf_md_thr=0.0032)

#     response_from_mask_wm, response_from_mask_gm, response_from_mask_csf = \
#         response_from_mask_msmt(gtab, data,
#                                 mask_wm, mask_gm, mask_csf,
#                                 tol=20)

#     npt.assert_array_equal(response_auto_wm, response_from_mask_wm)
#     npt.assert_array_equal(response_auto_gm, response_from_mask_gm)
#     npt.assert_array_equal(response_auto_csf, response_from_mask_csf)


if __name__ == "__main__":
    npt.run_module_suite()
