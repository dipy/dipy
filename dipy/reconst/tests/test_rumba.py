import warnings

import numpy as np

from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_raises,
                           run_module_suite)

from dipy.reconst.rumba import RumbaSD, global_fit, generate_kernel
from dipy.data import get_fnames, dsi_voxels, default_sphere
from dipy.core.gradients import gradient_table
from dipy.core.geometry import cart2sphere
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies
from dipy.sims.voxel import sticks_and_ball, multi_tensor
from dipy.direction.peaks import peak_directions


def test_rumba():
    '''
    Test fODF results from ideal examples.
    '''

    sphere = default_sphere  # repulsion 724

    btable = np.loadtxt(get_fnames('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)
    data, golden_directions = sticks_and_ball(gtab, d=0.0015, S0=100,
                                              angles=[(0, 0), (90, 0)],
                                              fractions=[50, 50], snr=None)

    # Testing input validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gtab_broken = gradient_table(
            bvals[~gtab.b0s_mask], bvecs[~gtab.b0s_mask])
        assert_raises(ValueError, RumbaSD, gtab_broken)

    assert_raises(ValueError, RumbaSD, gtab_broken, lambda1=-1.0)
    assert_raises(ValueError, RumbaSD, gtab, lambda_iso=-1.0)
    assert_raises(ValueError, RumbaSD, gtab, n_iter=0)

    rumba_broken = RumbaSD(gtab, recon_type='test')  # recon_type validation
    broken_fit = rumba_broken.fit(data)
    assert_raises(ValueError, broken_fit.odf, sphere)

    # Models to validate
    rumba_smf = RumbaSD(gtab, n_iter=20, recon_type='smf', n_coils=1)
    rumba_sos = RumbaSD(gtab, n_iter=20, recon_type='sos', n_coils=32)
    model_list = [rumba_smf, rumba_sos]

    # Test on repulsion724 sphere
    for model in model_list:
        odf = model.fit(data).odf(sphere)
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
                assert_equal(model_fit.f_iso(sphere) < 0.1, True)
                assert_equal(len(directions), len(golden_directions))
            if len(directions) > 3:
                # Verify large isotropic fraction in isotropic case
                assert_equal(model_fit.f_iso(sphere) > 0.8, True)


def test_mvoxel_rumba():
    '''
    Verify form of results in multi-voxel situation.
    '''
    data, gtab = dsi_voxels()  # multi-voxel data
    sphere = default_sphere  # repulsion 724

    # Models to validate
    rumba_smf = RumbaSD(gtab, n_iter=5, recon_type='smf', n_coils=1)
    rumba_sos = RumbaSD(gtab, n_iter=5, recon_type='sos', n_coils=32)
    model_list = [rumba_smf, rumba_sos]

    for model in model_list:
        model_fit = model.fit(data)

        odf = model_fit.odf(sphere)
        f_iso = model_fit.f_iso(sphere)
        f_wm = model_fit.f_wm(sphere)
        combined = model_fit.combined_odf_iso(sphere)

        # Verify shape, positivity, realness of results
        assert_equal(data.shape[:-1] + (len(sphere.vertices),), odf.shape)
        assert_equal(np.alltrue(np.isreal(odf)), True)
        assert_equal(np.alltrue(odf > 0), True)

        assert_equal(data.shape[:-1], f_iso.shape)
        assert_equal(np.alltrue(np.isreal(f_iso)), True)
        assert_equal(np.alltrue(f_iso > 0), True)

        # Verify properties of fODF and volume fractions
        assert_equal(combined, odf + f_iso[..., None] / len(sphere.vertices))
        assert_almost_equal(f_iso + f_wm, np.ones(f_iso.shape))
        assert_almost_equal(np.sum(combined, axis=3), np.ones(f_iso.shape))
        assert_equal(np.sum(odf, axis=3), f_wm)


def test_global_fit():
    '''
    Test fODF results on ideal examples in global fitting paradigm.
    '''

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
    rumba = RumbaSD(gtab, n_iter=20, recon_type='smf', n_coils=1, R=2)

    # Testing input validation
    assert_raises(ValueError, global_fit, rumba,
                  data[:, :, :, 0], sphere, use_tv=False)  # Must be 4D
    # TV can't work with singleton dimensions in data volume
    assert_raises(ValueError, global_fit, rumba,
                  data, sphere, use_tv=True)
    # Mask must match first 3 dimensions of data
    assert_raises(ValueError, global_fit, rumba, data, sphere, mask=np.ones(
        data.shape), use_tv=False)
    # Recon type validation
    rumba_broken = RumbaSD(gtab, recon_type='test')
    assert_raises(ValueError, global_fit, rumba_broken, data, sphere)

    # Test on repulsion 724 sphere
    for use_tv in [True, False]:  # test with/without TV regularization
        if use_tv:
            odf, f_iso, _, _ = global_fit(
                rumba, data_mvoxel, sphere, use_tv=True)
        else:
            odf, f_iso, _, _ = global_fit(
                rumba, data, sphere, use_tv=False)

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
        odf, f_iso, _, _ = global_fit(rumba, data, sphere, use_tv=False)
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
    '''
    Verify form of results in global fitting paradigm.
    '''
    data, gtab = dsi_voxels()  # multi-voxel data
    sphere = default_sphere  # repulsion 724

    # Models to validate
    rumba_sos = RumbaSD(gtab, recon_type='sos', n_iter=5, n_coils=32, R=1)
    rumba_r = RumbaSD(gtab, recon_type='smf', n_iter=5, n_coils=1, R=2)
    model_list = [rumba_sos, rumba_r]

    # Test each model with/without TV regularization
    for model in model_list:
        for use_tv in [True, False]:
            odf, f_iso, f_wm, combined = global_fit(
                model, data, sphere, verbose=True, use_tv=use_tv)

            # Verify shape, positivity, realness of results
            assert_equal(data.shape[:-1] + (len(sphere.vertices),), odf.shape)
            assert_equal(np.alltrue(np.isreal(odf)), True)
            assert_equal(np.alltrue(odf > 0), True)

            assert_equal(data.shape[:-1], f_iso.shape)
            assert_equal(np.alltrue(np.isreal(f_iso)), True)
            assert_equal(np.alltrue(f_iso > 0), True)

            # Verify normalization
            assert_equal(combined, odf +
                         f_iso[..., None] / len(sphere.vertices))
            assert_almost_equal(f_iso + f_wm, np.ones(f_iso.shape))
            assert_almost_equal(np.sum(combined, axis=3), np.ones(f_iso.shape))
            assert_equal(np.sum(odf, axis=3), f_wm)


def test_generate_kernel():
    '''
    Test form and content of kernel generation result.
    '''

    # load repulsion 724 sphere
    sphere = default_sphere

    btable = np.loadtxt(get_fnames('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)

    # Kernel parameters
    lambda1 = 1.7e-3
    lambda2 = 0.2e-3
    lambda_iso = 3e-3

    # Test kernel shape
    kernel = generate_kernel(
        gtab, sphere, lambda1=lambda1, lambda2=lambda2, lambda_iso=lambda_iso)
    assert_equal(kernel.shape, (len(gtab.bvals), len(sphere.vertices)+1))

    # Verify first column of kernel
    _, theta, phi = cart2sphere(
        sphere.x,
        sphere.y,
        sphere.z
    )
    S0 = 1  # S0 assumed to be 1
    fi = 100  # volume fraction assumed to be 100%

    S, _ = multi_tensor(gtab, np.array([[lambda1, lambda2, lambda2]]),
                        S0, [[theta[0]*180/np.pi, phi[0]*180/np.pi]], [fi],
                        None)
    assert_array_equal(kernel[:, 0], S)

    # Test optional isotropic compartment; should cause last column of zeroes
    kernel = generate_kernel(
        gtab, sphere, lambda1=lambda1, lambda2=lambda2, lambda_iso=None)
    assert_array_equal(kernel[:, -1], np.zeros(len(gtab.bvals)))


if __name__ == '__main__':
    run_module_suite()
