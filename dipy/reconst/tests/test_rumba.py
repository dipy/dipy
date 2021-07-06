import warnings

import numpy as np

from dipy.reconst.rumba import RumbaSD, global_fit, generate_kernel, combine_odf_csf
from dipy.data import get_fnames, dsi_voxels, get_sphere, default_sphere
from dipy.core.gradients import gradient_table
from dipy.core.geometry import cart2sphere
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies
from dipy.sims.voxel import sticks_and_ball, multi_tensor
from dipy.direction.peaks import peak_directions

from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           assert_raises,
                           run_module_suite)
from numpy.random import random


def get_f_csf(model_fit):
    '''
    Used for testing access of f_csf property on RumbaFit objects
    '''
    f = model_fit.f_csf


def test_rumba():
    '''
    Test fODF results from ideal examples
    '''

    sphere = default_sphere  # repulsion 724
    sphere2 = create_unit_sphere(5)  # icosahedron sphere

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
    assert_raises(ValueError, RumbaSD, gtab, lambda_csf=-1.0)
    assert_raises(ValueError, RumbaSD, gtab, n_iter=0)
    assert_raises(ValueError, RumbaSD, gtab, recon_type='test')

    # Models to validate
    rumba_smf = RumbaSD(gtab, n_iter=20, recon_type='smf', n_coils=1)
    rumba_sos = RumbaSD(gtab, n_iter=20, recon_type='sos', n_coils=32)
    model_list = [rumba_smf, rumba_sos]

    # Test on repulsion724 sphere
    for model in model_list:
        odf = model.fit(data).odf(sphere)
        directions, _, _ = peak_directions(odf, sphere, .35, 25)
        assert_equal(len(directions), 2)
        assert_almost_equal(angular_similarity(directions, golden_directions), 2,
                            1)

    # Test on icosahedron sphere
    for model in model_list:
        odf = model.fit(data).odf(sphere2)
        directions, values, indices = peak_directions(odf, sphere2, .35, 25)
        assert_equal(len(directions), 2)
        assert_almost_equal(angular_similarity(directions, golden_directions), 2,
                            1)

    # Test on data with 1, 2, 3, or no peaks
    sb_dummies = sticks_and_ball_dummies(gtab)
    for model in model_list:
        for sbd in sb_dummies:
            data, golden_directions = sb_dummies[sbd]
            model_fit = model.fit(data)
            odf = model_fit.odf(sphere2)
            directions, values, indices = peak_directions(
                odf, sphere2, .35, 25)
            if len(directions) <= 3:
                # verify small CSF fraction in anisotropic case
                assert_equal(model_fit.f_csf < 0.1, True)
                assert_equal(len(directions), len(golden_directions))
            if len(directions) > 3:
                # verify large CSF fraction in isotropic case
                assert_equal(model_fit.f_csf > 0.8, True)


def test_mvoxel_rumba():
    '''
    Verify form of results in multi-voxel situation
    '''
    data, gtab = dsi_voxels()  # multi-voxel data
    sphere = get_sphere('symmetric724')

    # Models to validate
    rumba_smf = RumbaSD(gtab, n_iter=10, recon_type='smf', n_coils=1)
    rumba_sos = RumbaSD(gtab, n_iter=10, recon_type='sos', n_coils=32)
    model_list = [rumba_smf, rumba_sos]

    for model in model_list:
        model_fit = model.fit(data)

        # can't access before odf call
        assert_raises(RuntimeError, get_f_csf, model_fit)

        odf = model_fit.odf(sphere)
        f_csf = model_fit.f_csf

        # Verify shape, positivity, realness of results
        assert_equal(data.shape[:-1] + (len(sphere.vertices),), odf.shape)
        assert_equal(np.alltrue(np.isreal(odf)), True)
        assert_equal(np.alltrue(odf > 0), True)

        assert_equal(data.shape[:-1], f_csf.shape)
        assert_equal(np.alltrue(np.isreal(f_csf)), True)
        assert_equal(np.alltrue(f_csf > 0), True)


def test_mvoxel_global_fit():
    '''
    Verify form of results in global fitting paradigm 
    '''
    data, gtab = dsi_voxels()  # multi-voxel data
    sphere = get_sphere('symmetric724')

    # Models to validate
    rumba_smf = RumbaSD(gtab, recon_type='smf', n_iter=10, n_coils=1)
    rumba_sos = RumbaSD(gtab, recon_type='sos', n_iter=10, n_coils=32)
    rumba_R = RumbaSD(gtab, recon_type='smf', n_iter=10, n_coils=1, R=2)
    model_list = [rumba_smf, rumba_sos, rumba_R]

    # Test each model with/without TV regularization
    for model in model_list:
        for use_tv in [True, False]:
            odf, f_csf = global_fit(model, data, sphere, use_tv=use_tv)

            # Verify shape, positivity, realness of results
            assert_equal(data.shape[:-1] + (len(sphere.vertices),), odf.shape)
            assert_equal(np.alltrue(np.isreal(odf)), True)
            assert_equal(np.alltrue(odf > 0), True)

            assert_equal(data.shape[:-1], f_csf.shape)
            assert_equal(np.alltrue(np.isreal(f_csf)), True)
            assert_equal(np.alltrue(f_csf > 0), True)


def test_global_fit():
    '''
    Test fODF results on ideal examples in global fitting paradigm
    '''

    sphere = default_sphere  # repulsion 724

    btable = np.loadtxt(get_fnames('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)
    data, golden_directions = sticks_and_ball(gtab, d=0.0015, S0=100,
                                              angles=[(0, 0), (90, 0)],
                                              fractions=[50, 50], snr=None)

    # global_fit requires 4D argument with multiple voxels
    data = data[None, None, None, :]
    data_2voxel = np.tile(data, (2, 1, 1, 1))  # duplicate data
    # TV requires non-singleton size in all volume dimensions
    data_mvoxel = np.tile(data, (2, 2, 2, 1))

    # Models to validate
    rumba_smf = RumbaSD(gtab, n_iter=20, recon_type='smf', n_coils=1)
    rumba_sos = RumbaSD(gtab, n_iter=20, recon_type='sos', n_coils=32)
    rumba_r = RumbaSD(gtab, n_iter=20, recon_type='smf', n_coils=1, R=2)
    model_list = [rumba_smf, rumba_sos, rumba_r]

    # Testing input validation
    assert_raises(ValueError, global_fit, rumba_smf, data,
                  sphere, use_tv=False)  # Must have multiple voxels
    assert_raises(ValueError, global_fit, rumba_smf,
                  data_2voxel[:, :, :, 0], sphere, use_tv=False)  # Must be 4D
    # TV can't work with singleton dimensions in data volume
    assert_raises(ValueError, global_fit, rumba_smf,
                  data_2voxel, sphere, use_tv=True)
    assert_raises(ValueError, global_fit, rumba_smf, data_2voxel, sphere, mask=np.ones(
        data.shape), use_tv=False)  # Mask must match first 3 dimensions of data

    # Test on repulsion 724 sphere
    for model in model_list:
        for use_tv in [True, False]:  # test with/without TV regularization
            if use_tv:
                odf, f_csf = global_fit(
                    model, data_mvoxel, sphere, use_tv=True)
            else:
                odf, f_csf = global_fit(
                    model, data_2voxel, sphere, use_tv=False)

            directions, _, _ = peak_directions(
                odf[0, 0, 0], sphere, .35, 25)
            assert_equal(len(directions), 2)
            assert_almost_equal(angular_similarity(directions, golden_directions), 2,
                                1)

    # Test on data with 1, 2, 3, or no peaks
    sb_dummies = sticks_and_ball_dummies(gtab)
    for model in model_list:
        for sbd in sb_dummies:
            data, golden_directions = sb_dummies[sbd]
            data_2voxel = np.tile(data, (2, 1, 1, 1))
            odf, f_csf = global_fit(model, data_2voxel, sphere, use_tv=False)
            directions, values, indices = peak_directions(
                odf[0, 0, 0], sphere, .35, 25)
            if len(directions) <= 3:
                # verify small CSF fraction in anisotropic case
                assert_equal(f_csf[0, 0, 0] < 0.1, True)
                assert_equal(len(directions), len(golden_directions))
            if len(directions) > 3:
                # verify large CSF fraction in isotropic case
                assert_equal(f_csf[0, 0, 0] > 0.8, True)


def test_generate_kernel():
    '''
    Test form and content of kernel generation result
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
    lambda_csf = 3e-3

    # Test kernel shape
    kernel = generate_kernel(
        gtab, sphere, lambda1=lambda1, lambda2=lambda2, lambda_csf=lambda_csf)
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
                        S0, [[theta[0]*180/np.pi, phi[0]*180/np.pi]], [fi], None)
    assert_array_equal(kernel[:, 0], S)

    # Test optional isotropic compartment; should cause last column of zeroes
    kernel = generate_kernel(
        gtab, sphere, lambda1=lambda1, lambda2=lambda2, lambda_csf=None)
    assert_array_equal(kernel[:, -1], np.zeros(len(gtab.bvals)))


def test_combine():
    '''
    Test fODF and CSF volume fraction combination function
    '''

    vol_shape = (5, 10, 10)
    n = 12

    # test that zeros result in unchanged odf
    odf = random((*vol_shape, n))
    f_csf = np.zeros((*vol_shape,))
    combined = combine_odf_csf(odf, f_csf)
    assert_array_equal(odf, combined)

    # test even distribution
    odf = np.zeros((*vol_shape, n))
    f_csf = np.ones((*vol_shape,))
    combined = combine_odf_csf(odf, f_csf)
    combined_check = 1/12 * np.ones((*vol_shape, n))
    assert_array_equal(combined, combined_check)


if __name__ == '__main__':
    run_module_suite()
