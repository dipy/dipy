import warnings

import numpy as np

from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_allclose,
                           assert_raises,)
from numpy.testing import assert_

from dipy.reconst.rumba import RumbaSDModel, generate_kernel
from dipy.reconst.csdeconv import AxSymShResponse
from dipy.data import get_fnames, dsi_voxels, default_sphere, get_sphere
from dipy.core.gradients import gradient_table, unique_bvals_tolerance
from dipy.core.geometry import cart2sphere
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies
from dipy.sims.voxel import sticks_and_ball, multi_tensor, single_tensor
from dipy.direction.peaks import peak_directions
from dipy.reconst.shm import descoteaux07_legacy_msg


def test_rumba():

    # Test fODF results from ideal examples.
    sphere = default_sphere  # repulsion 724
    sphere2 = get_sphere('symmetric362')

    btable = np.loadtxt(get_fnames('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)
    data, golden_directions = sticks_and_ball(gtab, d=0.0015, S0=100,
                                              angles=[(0, 0), (90, 0)],
                                              fractions=[50, 50], snr=None)

    # Testing input validation
    msg = "b0_threshold .*"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=msg,
                                category=UserWarning)
        gtab_broken = gradient_table(bvals[~gtab.b0s_mask],
                                     bvecs[~gtab.b0s_mask])

    assert_raises(ValueError, RumbaSDModel, gtab_broken)

    with warnings.catch_warnings(record=True) as w:
        _ = RumbaSDModel(gtab, verbose=True)
        assert_equal(len(w), 1)
        assert_(w[0].category, UserWarning)

    assert_raises(ValueError, RumbaSDModel, gtab, use_tv=True)
    assert_raises(ValueError, RumbaSDModel, gtab, n_iter=0)
    rumba_broken = RumbaSDModel(gtab, recon_type='test')
    assert_raises(ValueError, rumba_broken.fit, data)

    # Models to validate
    rumba_smf = RumbaSDModel(gtab, n_iter=20, recon_type='smf', n_coils=1,
                             sphere=sphere)
    rumba_sos = RumbaSDModel(gtab, n_iter=20, recon_type='sos', n_coils=32,
                             sphere=sphere)
    model_list = [rumba_smf, rumba_sos]

    # Test on repulsion724 sphere
    for model in model_list:
        model_fit = model.fit(data)

        # Verify only works on original sphere
        assert_raises(ValueError, model_fit.odf, sphere2)
        odf = model_fit.odf(sphere)

        directions, _, _ = peak_directions(odf, sphere, .35, 25)
        assert_equal(len(directions), 2)
        assert_almost_equal(angular_similarity(directions, golden_directions),
                            2, 1)

    # Test on data with 1, 2, 3, or no peaks
    sb_dummies = sticks_and_ball_dummies(gtab)
    for model in model_list:
        for sbd in sb_dummies:
            data, golden_directions = sb_dummies[sbd]
            model_fit = model.fit(data)
            odf = model_fit.odf(sphere)
            directions, _, _ = peak_directions(
                odf, sphere, .35, 25)
            if len(directions) <= 3:
                # Verify small isotropic fraction in anisotropic case
                assert_equal(model_fit.f_iso < 0.1, True)
                assert_equal(len(directions), len(golden_directions))
            if len(directions) > 3:
                # Verify large isotropic fraction in isotropic case
                assert_equal(model_fit.f_iso > 0.8, True)


def test_predict():

    # Test signal reconstruction on ideal example

    sphere = default_sphere

    btable = np.loadtxt(get_fnames('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)

    rumba = RumbaSDModel(gtab, n_iter=600, sphere=sphere)

    # Simulated data
    data = single_tensor(gtab, S0=1, evals=rumba.wm_response)
    rumba_fit = rumba.fit(data)
    data_pred = rumba_fit.predict()

    assert_allclose(data_pred, data, atol=0.01, rtol=0.05)


def test_recursive_rumba():

    # Test with recursive data-driven response

    sphere = default_sphere  # repulsion 724

    btable = np.loadtxt(get_fnames('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)
    data, golden_directions = sticks_and_ball(gtab, d=0.0015, S0=100,
                                              angles=[(0, 0), (90, 0)],
                                              fractions=[50, 50], snr=None)

    wm_response = AxSymShResponse(480, np.array([570.35065982,
                                                 -262.81741086,
                                                 80.23104069,
                                                 -16.93940972,
                                                 2.57628738]))
    model = RumbaSDModel(gtab, wm_response, n_iter=20, sphere=sphere)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        model_fit = model.fit(data)

    # Test peaks
    odf = model_fit.odf(sphere)
    directions, _, _ = peak_directions(odf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions),
                        2, 1)


def test_multishell_rumba():

    # Test with multi-shell response

    sphere = default_sphere  # repulsion 724

    btable = np.loadtxt(get_fnames('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)
    data, golden_directions = sticks_and_ball(gtab, d=0.0015, S0=100,
                                              angles=[(0, 0), (90, 0)],
                                              fractions=[50, 50], snr=None)

    ms_eigenval_count = len(unique_bvals_tolerance(gtab.bvals)) - 1
    wm_response = np.tile(
        np.array([1.7E-3, 0.2E-3, 0.2E-3]), (ms_eigenval_count, 1))
    model = RumbaSDModel(gtab, wm_response, n_iter=20, sphere=sphere)
    model_fit = model.fit(data)

    # Test peaks
    odf = model_fit.odf(sphere)
    directions, _, _ = peak_directions(odf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions),
                        2, 1)


def test_mvoxel_rumba():

    # Verify form of results in multi-voxel situation.

    data, gtab = dsi_voxels()  # multi-voxel data
    sphere = default_sphere  # repulsion 724

    # Models to validate
    rumba_smf = RumbaSDModel(gtab, n_iter=5, recon_type='smf', n_coils=1,
                             sphere=sphere)
    rumba_sos = RumbaSDModel(gtab, n_iter=5, recon_type='sos', n_coils=32,
                             sphere=sphere)
    model_list = [rumba_smf, rumba_sos]

    msg = "There is overlap in clustering of b-value.*"
    for model in model_list:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=msg,
                                    category=UserWarning)
            model_fit = model.fit(data)

        odf = model_fit.odf(sphere)
        f_iso = model_fit.f_iso
        f_wm = model_fit.f_wm
        f_gm = model_fit.f_gm
        f_csf = model_fit.f_csf
        combined = model_fit.combined_odf_iso

        # Verify prediction properties

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=msg,
                                    category=UserWarning)
            pred_sig_1 = model_fit.predict()
            pred_sig_2 = model_fit.predict(S0=1)
            pred_sig_3 = model_fit.predict(S0=np.ones(odf.shape[:-1]))
            pred_sig_4 = model_fit.predict(gtab=gtab)

        assert_equal(pred_sig_1, pred_sig_2)
        assert_equal(pred_sig_3, pred_sig_4)
        assert_equal(pred_sig_1, pred_sig_3)
        assert_equal(data.shape, pred_sig_1.shape)
        assert_equal(np.all(np.isreal(pred_sig_1)), True)
        assert_equal(np.all(pred_sig_1 > 0), True)

        # Verify shape, positivity, realness of results
        assert_equal(data.shape[:-1] + (len(sphere.vertices),), odf.shape)
        assert_equal(np.all(np.isreal(odf)), True)
        assert_equal(np.all(odf > 0), True)

        assert_equal(data.shape[:-1], f_iso.shape)
        assert_equal(np.all(np.isreal(f_iso)), True)
        assert_equal(np.all(f_iso > 0), True)

        # Verify properties of fODF and volume fractions
        assert_equal(f_iso, f_gm + f_csf)
        assert_equal(combined, odf + f_iso[..., None] / len(sphere.vertices))
        assert_almost_equal(f_iso + f_wm, np.ones(f_iso.shape))
        assert_almost_equal(np.sum(combined, axis=3), np.ones(f_iso.shape))
        assert_equal(np.sum(odf, axis=3), f_wm)


def test_global_fit():

    # Test fODF results on ideal examples in global fitting paradigm.

    sphere = default_sphere  # repulsion 724

    btable = np.loadtxt(get_fnames('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)
    data, golden_directions = sticks_and_ball(gtab, d=0.0015, S0=100,
                                              angles=[(0, 0), (90, 0)],
                                              fractions=[50, 50], snr=None)

    # global_fit requires 4D argument
    data = data[None, None, None, :]
    # TV requires non-singleton size in all volume dimensions
    data_mvoxel = np.tile(data, (2, 2, 2, 1))

    # Model to validate
    rumba = RumbaSDModel(gtab, n_iter=20, recon_type='smf', n_coils=1, R=2,
                         voxelwise=False, sphere=sphere)
    rumba_tv = RumbaSDModel(gtab, n_iter=20, recon_type='smf', n_coils=1, R=2,
                            voxelwise=False, use_tv=True, sphere=sphere)

    # Testing input validation
    assert_raises(ValueError, rumba.fit, data[:, :, :, 0])  # Must be 4D
    # TV can't work with singleton dimensions in data volume
    assert_raises(ValueError, rumba_tv.fit, data)
    # Mask must match first 3 dimensions of data
    assert_raises(ValueError, rumba.fit, data, mask=np.ones(data.shape))
    # Recon type validation
    rumba_broken = RumbaSDModel(gtab, recon_type='test', voxelwise=False)
    assert_raises(ValueError, rumba_broken.fit, data)

    # Test on repulsion 724 sphere, with/without TV regularization
    for ix, model in enumerate([rumba, rumba_tv]):

        if ix:
            model_fit = model.fit(data_mvoxel)
        else:
            model_fit = model.fit(data)

        odf = model_fit.odf(sphere)

        directions, _, _ = peak_directions(
            odf[0, 0, 0], sphere, .35, 25)
        assert_equal(len(directions), 2)
        assert_almost_equal(angular_similarity(directions, golden_directions),
                            2, 1)

    # Test on data with 1, 2, 3, or no peaks
    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        data = data[None, None, None, :]  # make 4D

        rumba_fit = rumba.fit(data)
        odf = rumba_fit.odf(sphere)
        f_iso = rumba_fit.f_iso

        directions, _, _ = peak_directions(
            odf[0, 0, 0], sphere, .35, 25)
        if len(directions) <= 3:
            # Verify small isotropic fraction in anisotropic case
            assert_equal(f_iso[0, 0, 0] < 0.1, True)
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            # Verify large isotropic fraction in isotropic case
            assert_equal(f_iso[0, 0, 0] > 0.8, True)


def test_mvoxel_global_fit():

    # Verify form of results in global fitting paradigm.

    data, gtab = dsi_voxels()  # multi-voxel data
    sphere = default_sphere  # repulsion 724

    # Models to validate
    rumba_sos = RumbaSDModel(gtab, recon_type='sos', n_iter=5, n_coils=32, R=1,
                             voxelwise=False, verbose=True, sphere=sphere)
    rumba_sos_tv = RumbaSDModel(gtab, recon_type='sos', n_iter=5, n_coils=32,
                                R=1, voxelwise=False, use_tv=True,
                                sphere=sphere)
    rumba_r = RumbaSDModel(gtab, recon_type='smf', n_iter=5, n_coils=1, R=2,
                           voxelwise=False, sphere=sphere)
    rumba_r_tv = RumbaSDModel(gtab, recon_type='smf', n_iter=5, n_coils=1, R=2,
                              voxelwise=False, use_tv=True, sphere=sphere)
    model_list = [rumba_sos, rumba_sos_tv, rumba_r, rumba_r_tv]

    # Test each model with/without TV regularization
    for model in model_list:
        msg = "There is overlap in clustering of b-value.*"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=msg,
                                    category=UserWarning)
            model_fit = model.fit(data)
        odf = model_fit.odf(sphere)
        f_iso = model_fit.f_iso
        f_wm = model_fit.f_wm
        f_gm = model_fit.f_gm
        f_csf = model_fit.f_csf
        combined = model_fit.combined_odf_iso

        # Verify shape, positivity, realness of results
        assert_equal(data.shape[:-1] + (len(sphere.vertices),), odf.shape)
        assert_equal(np.all(np.isreal(odf)), True)
        assert_equal(np.all(odf > 0), True)

        assert_equal(data.shape[:-1], f_iso.shape)
        assert_equal(np.all(np.isreal(f_iso)), True)
        assert_equal(np.all(f_iso > 0), True)

        # Verify normalization
        assert_equal(f_iso, f_gm + f_csf)
        assert_equal(combined, odf +
                     f_iso[..., None] / len(sphere.vertices))
        assert_almost_equal(f_iso + f_wm, np.ones(f_iso.shape))
        assert_almost_equal(np.sum(combined, axis=3), np.ones(f_iso.shape))
        assert_equal(np.sum(odf, axis=3), f_wm)


def test_generate_kernel():

    # Test form and content of kernel generation result.

    # load repulsion 724 sphere
    sphere = default_sphere

    btable = np.loadtxt(get_fnames('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)

    # Kernel parameters
    wm_response = np.array([1.7e-3, 0.2e-3, 0.2e-3])
    gm_response = 0.2e-4
    csf_response = 3.0e-3

    # Test kernel shape
    kernel = generate_kernel(
        gtab, sphere, wm_response, gm_response, csf_response)
    assert_equal(kernel.shape, (len(gtab.bvals), len(sphere.vertices) + 2))

    # Verify first column of kernel
    _, theta, phi = cart2sphere(
        sphere.x,
        sphere.y,
        sphere.z
    )
    S0 = 1  # S0 assumed to be 1
    fi = 100  # volume fraction assumed to be 100%

    S, _ = multi_tensor(gtab, np.array([wm_response]),
                        S0, [[theta[0] * 180 / np.pi, phi[0] * 180 / np.pi]],
                        [fi],
                        None)
    assert_almost_equal(kernel[:, 0], S)

    # Multi-shell version
    ms_eigenval_count = len(unique_bvals_tolerance(gtab.bvals)) - 1
    wm_response_multi = np.tile(wm_response, (ms_eigenval_count, 1))
    kernel_multi = generate_kernel(
        gtab, sphere, wm_response_multi, gm_response, csf_response)
    assert_equal(kernel.shape, (len(gtab.bvals), len(sphere.vertices) + 2))
    assert_almost_equal(kernel, kernel_multi)

    # Test optional isotropic compartment; should cause last column of zeroes
    kernel = generate_kernel(
        gtab, sphere, wm_response, gm_response=None, csf_response=None)
    assert_array_equal(kernel[:, -2], np.zeros(len(gtab.bvals)))
    assert_array_equal(kernel[:, -1], np.zeros(len(gtab.bvals)))
