import warnings

from dipy.reconst.mcsd import (mask_for_response_msmt,
                               response_from_mask_msmt,
                               auto_response_msmt)
from dipy.reconst.mcsd import MultiShellDeconvModel, multi_shell_fiber_response
from dipy.reconst import mcsd
import numpy as np
import numpy.testing as npt
import pytest

from dipy.sims.voxel import single_tensor, multi_tensor
from dipy.reconst import shm
from dipy.reconst.dti import fractional_anisotropy, mean_diffusivity
from dipy.data import default_sphere, get_3shell_gtab, get_fnames
from dipy.core.gradients import GradientTable, gradient_table

from dipy.io.gradients import read_bvals_bvecs
from dipy.io.image import load_nifti_data

from dipy.utils.optpkg import optional_package
cvx, have_cvxpy, _ = optional_package("cvxpy")

needs_cvxpy = pytest.mark.skipif(not have_cvxpy)


csf_md = 3e-3
gm_md = .76e-3
evals_d = np.array([.992, .254, .254]) * 1e-3


_, fbvals, fbvecs = get_fnames('small_64D')  # Need multi-shell data!!!
bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
gtab_test = gradient_table(bvals, bvecs)
evals_wm = np.array([1.7E-3, 0.4E-3, 0.4E-3])
evals_gm = np.array([4.0E-4, 4.0E-4, 4.0E-4])
evals_csf = np.array([3.0E-3, 3.0E-3, 3.0E-3])
S0_wm = 0.8
S0_gm = 1
S0_csf = 4
signal_wm = single_tensor(gtab_test, S0_wm, evals_wm)
signal_gm = single_tensor(gtab_test, S0_gm, evals_gm)
signal_csf = single_tensor(gtab_test, S0_csf, evals_csf)
signals = [signal_wm, signal_gm, signal_csf]
tissues = [0, 0, 2, 0, 1, 0, 0, 1, 2]
data_test = [signals[tissue] for tissue in tissues]
evals = np.ndarray((9, 3))
for i, tissue in enumerate(tissues):
    if tissue == 0:
        evals[i] = evals_wm
    elif tissues == 1:
        evals[i] = evals_gm
    else:
        evals[i] = evals_csf
evals = evals.reshape((3, 3, 1, 3))

tissues = np.asarray(tissues).reshape((3, 3, 1))
data_test = np.asarray(data_test).reshape((3, 3, 1, len(signal_wm)))
mask_wm_test = np.where(tissues == 0, 1, 0)
mask_gm_test = np.where(tissues == 1, 1, 0)
mask_csf_test = np.where(tissues == 2, 1, 0)
response_wm_test = np.concatenate((evals_wm, [S0_wm]))
response_gm_test = np.concatenate((evals_gm, [S0_gm]))
response_csf_test = np.concatenate((evals_csf, [S0_csf]))
fa_test = fractional_anisotropy(evals)
md_test = mean_diffusivity(evals)


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
    response = multi_shell_fiber_response(sh_order, shells, evals_d, csf_md,
                                          gm_md)
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
                                          evals_d, csf_md, gm_md)
    npt.assert_raises(ValueError, MultiShellDeconvModel, gtab, response, iso=1)


@pytest.mark.skipif(not mcsd.have_cvxpy, reason="Requires CVXPY")
def test_MultiShellDeconvModel():

    gtab = get_3shell_gtab()

    S0 = 1.
    evals = np.array([.992, .254, .254]) * 1e-3
    mevals = np.array([evals, evals])
    angles = [(0, 0), (60, 0)]

    S_wm, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                                fractions=[30., 70.], snr=None)
    S_gm = np.exp(-gtab.bvals * gm_md)
    S_csf = np.exp(-gtab.bvals * csf_md)

    sh_order = 8
    response = multi_shell_fiber_response(sh_order, [0, 1000, 2000, 3500],
                                          evals_d, csf_md, gm_md)
    model = MultiShellDeconvModel(gtab, response)
    vf = [1.3, .8, 1.9]
    signal = sum(i * j for i, j in zip(vf, [S_csf, S_gm, S_wm]))
    fit = model.fit(signal)

    npt.assert_array_almost_equal(fit.volume_fractions, vf, 0)

    S_pred = fit.predict()
    npt.assert_array_almost_equal(S_pred, signal, 0)


def test_mask_for_response_msmt():
    # fdata, fbvals, fbvecs, ffa, fmd, fmask_wm, fmask_gm, fmask_csf = \
    # get_fnames('???')
    # bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    # data = load_nifti_data(fdata)
    # fa = load_nifti_data(ffa)
    # md = load_nifti_data(fmd)
    # wm_mask_gt = load_nifti_data(fmask_wm)
    # gm_mask_gt = load_nifti_data(fmask_gm)
    # csf_mask_gt = load_nifti_data(fmask_csf)

    # gtab = gradient_table(bvals, bvecs)

    wm_mask, gm_mask, csf_mask = mask_for_response_msmt(gtab_test, data_test,
                                    roi_center=None, roi_radii=3,
                                    fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
                                    csf_fa_thr=0.15, md_data=None,
                                    gm_md_thr=0.001, csf_md_thr=0.003)

    # Verifies that masks are not empty:
    masks_sum = int(np.sum(wm_mask) + np.sum(gm_mask) + np.sum(csf_mask))
    npt.assert_equal(masks_sum == 0, True)

    wm_mask_fa_md, gm_mask_fa_md, csf_mask_fa_md = mask_for_response_msmt(
                                gtab_test, data_test,
                                roi_center=None, roi_radii=3,
                                fa_data=fa_test, wm_fa_thr=0.7, gm_fa_thr=0.3,
                                csf_fa_thr=0.15, md_data=md_test,
                                gm_md_thr=0.001, csf_md_thr=0.003)

    npt.assert_array_almost_equal(wm_mask_fa_md, wm_mask)
    npt.assert_array_almost_equal(gm_mask_fa_md, gm_mask)
    npt.assert_array_almost_equal(csf_mask_fa_md, csf_mask)
    npt.assert_array_almost_equal(mask_wm_test, wm_mask_fa_md)
    npt.assert_array_almost_equal(mask_gm_test, gm_mask_fa_md)
    npt.assert_array_almost_equal(mask_csf_test, csf_mask_fa_md)


def test_mask_for_response_msmt_nvoxels():
    # fdata, fbvals, fbvecs = get_fnames('???')
    # bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    # data = load_nifti_data(fdata)

    # gtab = gradient_table(bvals, bvecs)

    wm_mask, gm_mask, csf_mask = mask_for_response_msmt(gtab_test, data_test,
                                    roi_center=None, roi_radii=3,
                                    fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
                                    csf_fa_thr=0.15, md_data=None,
                                    gm_md_thr=0.001, csf_md_thr=0.003)

    wm_nvoxels = np.sum(wm_mask)
    gm_nvoxels = np.sum(gm_mask)
    csf_nvoxels = np.sum(csf_mask)
    npt.assert_equal(wm_nvoxels, 5)
    npt.assert_equal(gm_nvoxels, 2)
    npt.assert_equal(csf_nvoxels, 2)

    with warnings.catch_warnings(record=True) as w:
        wm_mask, gm_mask, csf_mask = mask_for_response_msmt(gtab_test,
                                        data_test,
                                        roi_center=None, roi_radii=3,
                                        fa_data=None, wm_fa_thr=1, gm_fa_thr=0,
                                        csf_fa_thr=0, md_data=None,
                                        gm_md_thr=1, csf_md_thr=1)
        npt.assert_equal(len(w), 5)
        npt.assert_(issubclass(w[0].category, UserWarning))
        npt.assert_("No voxel with a FA higher than 1 were found" in
                    str(w[0].message))
        npt.assert_("No voxel with a FA lower than 0 were found" in
                    str(w[0].message))
        npt.assert_("No voxel with a MD higher than 1 were found" in
                    str(w[0].message))
        npt.assert_("No voxel with a FA lower than 0 were found" in
                    str(w[0].message))
        npt.assert_("No voxel with a MD higher than 1 were found" in
                    str(w[0].message))

    wm_nvoxels = np.sum(wm_mask)
    gm_nvoxels = np.sum(gm_mask)
    csf_nvoxels = np.sum(csf_mask)
    npt.assert_equal(wm_nvoxels, 0)
    npt.assert_equal(gm_nvoxels, 0)
    npt.assert_equal(csf_nvoxels, 0)


def test_response_from_mask_msmt():
    # fdata, fbvals, fbvecs, fmask_wm, fmask_gm, fmask_csf, \
    # fresponse_wm, fresponse_gm, fresponse_csf = get_fnames('???')
    # bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    # data = load_nifti_data(fdata)
    # mask_wm = load_nifti_data(fmask_wm)
    # mask_gm = load_nifti_data(fmask_gm)
    # mask_csf = load_nifti_data(fmask_csf)
    # response_wm_gt = np.loadtxt(fresponse_wm).T
    # response_gm_gt = np.loadtxt(fresponse_gm).T
    # response_csf_gt = np.loadtxt(fresponse_csf).T

    # gtab = gradient_table(bvals, bvecs)

    response_wm, response_gm, response_csf = response_from_mask_msmt(gtab_test,
                                                data_test, mask_wm_test,
                                                mask_gm_test, mask_csf_test,
                                                tol=20)

    # Verifying that csf's response is greater than gm's
    npt.assert_equal(np.sum(response_csf[:3]) > np.sum(response_gm[:3]), True)
    # Verifying that csf and gm are described by spheres
    npt.assert_almost_equal(response_csf[1], response_csf[2])
    npt.assert_almost_equal(response_csf[0], response_csf[1])
    npt.assert_almost_equal(response_gm[1], response_gm[2])
    npt.assert_almost_equal(response_gm[0], response_gm[1])
    # Verifying that wm is anisotropic in one direction
    npt.assert_almost_equal(response_wm[1], response_wm[2])
    npt.assert_equal(response_wm[0] > response_wm[1], True)  # > by how much??

    # Way to test response[3] ??? b0

    # Verifying with ground truth
    npt.assert_array_almost_equal(response_wm, response_wm_test)
    npt.assert_array_almost_equal(response_gm, response_gm_test)
    npt.assert_array_almost_equal(response_csf, response_csf_test)


def test_auto_response_msmt():
    # fdata, fbvals, fbvecs = get_fnames('small_64D')
    # bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    # data = load_nifti_data(fdata)

    # gtab = gradient_table(bvals, bvecs)

    response_auto_wm, response_auto_gm, response_auto_csf = \
        auto_response_msmt(gtab_test, data_test, tol=20,
                           roi_center=None, roi_radii=10,
                           fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
                           csf_fa_thr=0.15, md_data=None,
                           gm_md_thr=0.001, csf_md_thr=0.003)

    mask_wm, mask_gm, mask_csf = mask_for_response_msmt(gtab_test, data_test,
                                    roi_center=None, roi_radii=3,
                                    fa_data=None, wm_fa_thr=0.7, gm_fa_thr=0.3,
                                    csf_fa_thr=0.15, md_data=None,
                                    gm_md_thr=0.001, csf_md_thr=0.003)

    response_from_mask_wm, response_from_mask_gm, response_from_mask_csf = \
        response_from_mask_msmt(gtab_test, data_test,
                                mask_wm, mask_gm, mask_csf,
                                tol=20)

    npt.assert_array_equal(response_auto_wm, response_from_mask_wm)
    npt.assert_array_equal(response_auto_gm, response_from_mask_gm)
    npt.assert_array_equal(response_auto_csf, response_from_mask_csf)


if __name__ == "__main__":
    npt.run_module_suite()
