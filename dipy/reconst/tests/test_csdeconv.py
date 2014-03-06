import warnings
import numpy as np
import numpy.testing as npt
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_almost_equal,
                           run_module_suite)
from dipy.data import get_sphere, get_data
from dipy.sims.voxel import (multi_tensor,
                             single_tensor,
                             multi_tensor_odf,
                             all_tensor_evecs)
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel,
                                   forward_sdeconv_mat,
                                   odf_sh_to_sharp,
                                   auto_response)
from dipy.reconst.peaks import peak_directions
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.shm import (sf_to_sh, sh_to_sf, QballModel, 
                              CsaOdfModel, sph_harm_ind_list)


def test_csdeconv():
    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (60, 0)]

    S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric362')

    odf_gt = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])

    response = (np.array([0.0015, 0.0003, 0.0003]), S0)

    csd = ConstrainedSphericalDeconvModel(gtab, response)

    csd_fit = csd.fit(S)

    assert_equal(csd_fit.shm_coeff[0] > 0, True)

    fodf = csd_fit.odf(sphere)

    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions, directions2)

    assert_equal(ang_sim > 1.9, True)
    assert_equal(directions.shape[0], 2)
    assert_equal(directions2.shape[0], 2)

    with warnings.catch_warnings(record=True) as w:

        ConstrainedSphericalDeconvModel(gtab, response, sh_order=10)
        assert_equal(len(w) > 0, True)

    with warnings.catch_warnings(record=True) as w:

        ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
        assert_equal(len(w) > 0, False)

    mevecs = []
    for s in sticks:
        mevecs += [all_tensor_evecs(s).T]

    S2 = single_tensor(gtab, 100, mevals[0], mevecs[0], snr=None)
    big_S = np.zeros((10, 10, 10, len(S2)))
    big_S[:] = S2

    aresponse, aratio = auto_response(gtab, big_S, roi_center=(5, 5, 4), roi_radius=3, fa_thr=0.5)
    assert_array_almost_equal(aresponse[0], response[0])
    assert_almost_equal(aresponse[1], 100)
    assert_almost_equal(aratio, response[0][1]/response[0][0])

    aresponse2, aratio2 = auto_response(gtab, big_S, roi_radius=3, fa_thr=0.5)
    assert_array_almost_equal(aresponse[0], response[0])


def test_odfdeconv():
    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (90, 0)]
    S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric362')

    odf_gt = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])

    e1 = 15.0
    e2 = 3.0
    ratio = e2 / e1

    csd = ConstrainedSDTModel(gtab, ratio, None)

    csd_fit = csd.fit(S)
    fodf = csd_fit.odf(sphere)

    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions, directions2)

    assert_equal(ang_sim > 1.9, True)

    assert_equal(directions.shape[0], 2)
    assert_equal(directions2.shape[0], 2)

    with warnings.catch_warnings(record=True) as w:

        ConstrainedSDTModel(gtab, ratio, sh_order=10)
        assert_equal(len(w) > 0, True)

    with warnings.catch_warnings(record=True) as w:

        ConstrainedSDTModel(gtab, ratio, sh_order=8)
        assert_equal(len(w) > 0, False)


def test_odf_sh_to_sharp():

    SNR = None
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(10, 0), (100, 0)],
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric724')

    qb = QballModel(gtab, sh_order=8, assume_normed=True)

    qbfit = qb.fit(S)
    odf_gt = qbfit.odf(sphere)

    Z = np.linalg.norm(odf_gt)

    odfs_gt = np.zeros((3, 1, 1, odf_gt.shape[0]))
    odfs_gt[:,:,:] = odf_gt[:]

    odfs_sh = sf_to_sh(odfs_gt, sphere, sh_order=8, basis_type=None)

    odfs_sh /= Z

    fodf_sh = odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15.,
                              sh_order=8, lambda_=1., tau=0.1)

    fodf = sh_to_sf(fodf_sh, sphere, sh_order=8, basis_type=None)

    directions2, _, _ = peak_directions(fodf[0, 0, 0], sphere)

    assert_equal(directions2.shape[0], 2)


def test_forward_sdeconv_mat():
    m, n = sph_harm_ind_list(4)
    mat = forward_sdeconv_mat(np.array([0, 2, 4]), n)
    expected = np.diag([0, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4])
    npt.assert_array_equal(mat, expected)

    sh_order = 8
    expected_size = (sh_order + 1) * (sh_order + 2) / 2
    r_rh = np.arange(0, sh_order + 1, 2)
    m, n = sph_harm_ind_list(sh_order)
    mat = forward_sdeconv_mat(r_rh, n)
    npt.assert_equal(mat.shape, (expected_size, expected_size))
    npt.assert_array_equal(mat.diagonal(), n)

    # Odd spherical harmonic degrees should raise a ValueError
    n[2] = 3
    npt.assert_raises(ValueError, forward_sdeconv_mat, r_rh, n)


def test_r2_term_odf_sharp():
    SNR = None
    S0 = 1
    angle = 75

    _, fbvals, fbvecs = get_data('small_64D')  #get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    sphere = get_sphere('symmetric724')
    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0), (angle, 0)]

    S, sticks = multi_tensor(gtab, mevals, S0, angles=angles,
                             fractions=[50, 50], snr=SNR)    

    odf_gt = multi_tensor_odf(sphere.vertices, mevals, angles, [50, 50])
    odfs_sh = sf_to_sh(odf_gt, sphere, sh_order=8, basis_type=None)
    fodf_sh = odf_sh_to_sharp(odfs_sh, sphere, basis=None, ratio=3 / 15.,
                              sh_order=8, lambda_=1., tau=0.1, r2_term=True)
    fodf = sh_to_sf(fodf_sh, sphere, sh_order=8, basis_type=None)

    directions_gt, _, _ = peak_directions(odf_gt, sphere)
    directions, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions_gt, directions)
    assert_equal(ang_sim > 1.9, True)
    assert_equal(directions.shape[0], 2)



if __name__ == '__main__':
    run_module_suite()

