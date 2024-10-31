import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_less,
)

from dipy.data import get_sphere
from dipy.reconst.bingham import (
    _bingham_fit_peak,
    _convert_bingham_pars,
    _single_bingham_to_sf,
    _single_sf_to_bingham,
    bingham_fiber_density,
    bingham_fiber_spread,
    k2odi,
    odi2k,
    sf_to_bingham,
    sh_to_bingham,
)
from dipy.reconst.shm import sf_to_sh


def setup_module():
    global sphere
    sphere = get_sphere(name="repulsion724")
    sphere = sphere.subdivide(n=2)


def teardown_module():
    global sphere
    sphere = None


def test_bingham_fit():
    """Tests for bingham function and single Bingham fit"""
    peak_dir = np.array([1, 0, 0])
    ma_axis = np.array([0, 1, 0])
    mi_axis = np.array([0, 0, 1])
    k1 = 2
    k2 = 6
    f0 = 3

    # Test if maximum amplitude is in the expected Bingham main direction
    # which should be perpendicular to both ma_axis and mi_axis
    odf_test = _single_bingham_to_sf(f0, k1, k2, ma_axis, mi_axis, peak_dir)
    assert_almost_equal(odf_test, f0)

    # Test Bingham fit on full sampled GT Bingham function
    odf_gt = _single_bingham_to_sf(f0, k1, k2, ma_axis, mi_axis, sphere.vertices)
    a0, c1, c2, mu0, mu1, mu2 = _bingham_fit_peak(odf_gt, peak_dir, sphere, 45)

    # check scalar parameters
    assert_almost_equal(a0, f0, decimal=3)
    assert_almost_equal(c1, k1, decimal=3)
    assert_almost_equal(c2, k2, decimal=3)

    # check if measured peak direction and dispersion axis are aligned to their
    # respective GT
    Mus = np.array([mu0, mu1, mu2])
    Mus_ref = np.array([peak_dir, ma_axis, mi_axis])
    assert_array_almost_equal(
        np.abs(np.diag(np.dot(Mus, Mus_ref))), np.ones(3), decimal=5
    )

    # check the same for bingham_fit_odf
    fits, n = _single_sf_to_bingham(odf_gt, sphere, max_search_angle=45)
    assert_almost_equal(fits[0][0], f0, decimal=3)
    assert_almost_equal(fits[0][1], k1, decimal=3)
    assert_almost_equal(fits[0][2], k2, decimal=3)
    Mus = np.array([fits[0][3], fits[0][4], fits[0][5]])
    # I had to decrease the precision in the assert below because main peak
    # direction is now calculated (before the GT direction was given)
    assert_array_almost_equal(
        np.abs(np.diag(np.dot(Mus, Mus_ref))), np.ones(3), decimal=5
    )


def test_bingham_metrics():
    axis0 = np.array([1, 0, 0])
    axis1 = np.array([0, 1, 0])
    axis2 = np.array([0, 0, 1])
    k1 = 2
    k2 = 6
    f0_lobe1 = 3
    f0_lobe2 = 1

    # define the parameters of two bingham functions with different amplitudes
    fits = [(f0_lobe1, k1, k2, axis0, axis1, axis2)]
    fits.append((f0_lobe2, k1, k2, axis0, axis1, axis2))

    # First test just to check right parameter conversion
    ref_pars = np.zeros((2, 12))
    ref_pars[0, 0] = f0_lobe1
    ref_pars[1, 0] = f0_lobe2
    ref_pars[0, 1] = ref_pars[1, 1] = k1
    ref_pars[0, 2] = ref_pars[1, 2] = k2
    ref_pars[0, 3:6] = ref_pars[1, 3:6] = axis0
    ref_pars[0, 6:9] = ref_pars[1, 6:9] = axis1
    ref_pars[0, 9:12] = ref_pars[1, 9:12] = axis2
    bpars = _convert_bingham_pars(fits, 2)
    assert_array_almost_equal(bpars, ref_pars)

    # TEST: Bingham Fiber density
    # As the amplitude of the first bingham function is 3 times higher than
    # the second, its surface integral have to be also 3 times larger.
    fd = bingham_fiber_density(bpars)

    assert_almost_equal(fd[0] / fd[1], 3)

    # Fiber density using the default sphere should be close to the fd obtained
    # using a high-resolution sphere (2621442 vertices)
    fd_hires = bingham_fiber_density(bpars, subdivide=9)

    # We must lower the precision for the test to pass, but still shows that
    # the fiber density is precise up to 4 decimals using only 0.39% of the
    # samples (10242 versus 2621442 vertices)
    assert_array_almost_equal(fd, fd_hires, decimal=4)

    # Rotating the Bingham distribution should not bias the FD estimate
    xfits = [
        (f0_lobe1, k1, k2, axis0, axis1, axis2),
        (f0_lobe1, k1, k2, axis1, axis2, axis0),
        (f0_lobe1, k1, k2, axis2, axis0, axis1),
    ]
    xpars = _convert_bingham_pars(xfits, 3)
    xfd = bingham_fiber_density(xpars)

    assert_almost_equal(xfd[0], xfd[1])
    assert_almost_equal(xfd[0], xfd[2])

    # If the Bingham function is a sphere of unit radius, the
    # fiber density should be 4*np.pi.
    sphere_fit = [(1.0, 0.0, 0.0, axis0, axis1, axis2)]
    sphere_pars = _convert_bingham_pars(sphere_fit, 1)

    fd_sphere = bingham_fiber_density(sphere_pars)

    assert_almost_equal(fd_sphere[0], 4.0 * np.pi)

    # TEST: k2odi and odi2k conversions
    assert_almost_equal(odi2k(k2odi(np.array(k1))), k1)
    assert_almost_equal(odi2k(k2odi(np.array(k2))), k2)

    # TEST: Fiber spread
    f0s = np.array([f0_lobe1, f0_lobe2])
    fs = bingham_fiber_spread(f0s, fd)

    assert_array_almost_equal(fs, fd / f0s)


def test_bingham_from_odf():
    # Reconstruct multi voxel ODFs to test bingham_from_odf
    ma_axis = np.array([0, 1, 0])
    mi_axis = np.array([0, 0, 1])
    k1 = 2
    k2 = 6
    f0 = 3
    odf = _single_bingham_to_sf(f0, k1, k2, ma_axis, mi_axis, sphere.vertices)

    # Perform Bingham fit in multi-voxel odf
    multi_odfs = np.zeros((2, 2, 1, len(sphere.vertices)))
    multi_odfs[...] = odf
    bim = sf_to_bingham(multi_odfs, sphere, npeaks=2, max_search_angle=45)

    # check model_params
    assert_almost_equal(bim.model_params[0, 0, 0, 0, 0], f0, decimal=3)
    assert_almost_equal(bim.model_params[0, 0, 0, 0, 1], k1, decimal=3)
    assert_almost_equal(bim.model_params[0, 0, 0, 0, 2], k2, decimal=3)
    # check if estimates for a second lobe are zero (note that a single peak
    # ODF is assumed here for this test GT)
    assert_array_almost_equal(bim.model_params[0, 0, 0, 1], np.zeros(12))

    # check if we have estimates in the right lobe for all voxels
    peak_v = bim.model_params[0, 0, 0, 0, 0]
    assert_array_almost_equal(bim.amplitude_lobe[..., 0], peak_v * np.ones((2, 2, 1)))
    assert_array_almost_equal(bim.amplitude_lobe[..., 1], np.zeros((2, 2, 1)))

    # check kappas
    assert_almost_equal(bim.kappa1_lobe[0, 0, 0, 0], k1, decimal=3)
    assert_almost_equal(bim.kappa2_lobe[0, 0, 0, 0], k2, decimal=3)
    assert_almost_equal(bim.kappa_total_lobe[0, 0, 0, 0], np.sqrt(k1 * k2), decimal=3)

    # check ODI
    assert_almost_equal(bim.odi1_lobe[0, 0, 0, 0], k2odi(np.array(k1)), decimal=3)
    assert_almost_equal(bim.odi2_lobe[0, 0, 0, 0], k2odi(np.array(k2)), decimal=3)
    # ODI2 < ODI total < ODI1
    assert_array_less(bim.odi2_lobe[..., 0], bim.odi1_lobe[..., 0])
    assert_array_less(bim.odi2_lobe[..., 0], bim.odi_total_lobe[..., 0])
    assert_array_less(bim.odi_total_lobe[..., 0], bim.odi1_lobe[..., 0])

    # check fiber_density estimates (larger than zero for lobe 0)
    assert_array_less(np.zeros((2, 2, 1)), bim.fd_lobe[:, :, :, 0])
    assert_almost_equal(np.zeros((2, 2, 1)), bim.fd_lobe[:, :, :, 1])

    # check global metrics: since this simulations only have one lobe, global
    # metrics have to give the same values than their counterparts for lobe 1
    assert_almost_equal(bim.odi1_voxel, bim.odi1_lobe[..., 0])
    assert_almost_equal(bim.odi2_voxel, bim.odi2_lobe[..., 0])
    assert_almost_equal(bim.odi_total_voxel, bim.odi_total_lobe[..., 0])
    assert_almost_equal(bim.fd_voxel, bim.fd_lobe[..., 0])

    # check fiber spread
    fs_v = bim.fd_lobe[0, 0, 0, 0] / peak_v
    assert_almost_equal(bim.fs_lobe[..., 0], fs_v)

    # check reconstructed odf
    reconst_odf = bim.odf(sphere)
    assert_almost_equal(reconst_odf[0, 0, 0], odf, decimal=2)


def test_bingham_from_sh():
    ma_axis = np.array([0, 1, 0])
    mi_axis = np.array([0, 0, 1])
    k1 = 2
    k2 = 6
    f0 = 3
    odf = _single_bingham_to_sf(f0, k1, k2, ma_axis, mi_axis, sphere.vertices)

    bim_odf = sf_to_bingham(odf, sphere, npeaks=2, max_search_angle=45)
    sh = sf_to_sh(odf, sphere, sh_order_max=16, legacy=False)
    bim_sh = sh_to_bingham(sh, sphere, legacy=False, npeaks=2, max_search_angle=45)
    assert_array_almost_equal(bim_sh.model_params, bim_odf.model_params, decimal=3)
