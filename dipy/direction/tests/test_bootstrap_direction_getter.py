import warnings

import numpy as np
import numpy.testing as npt


from dipy.core.geometry import cart2sphere
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, unit_icosahedron
from dipy.data import get_sphere, get_fnames
from dipy.direction.bootstrap_direction_getter import BootDirectionGetter
from dipy.io.gradients import read_bvals_bvecs
from dipy.reconst import dti, shm
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   TensorModel)
from dipy.sims.voxel import multi_tensor, single_tensor
from dipy.testing.decorators import set_random_number_generator


def test_bdg_initial_direction():
    """This tests the number of initial directions."
    """

    hsph_updated = HemiSphere.from_sphere(unit_icosahedron).subdivide(2)
    vertices = hsph_updated.vertices
    bvecs = vertices
    bvals = np.ones(len(vertices)) * 1000
    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, 0, 0)
    gtab = gradient_table(bvals, bvecs)

    # test that we get one direction when we have a single tensor
    sphere = HemiSphere.from_sphere(get_sphere('symmetric724'))
    voxel = single_tensor(gtab).reshape([1, 1, 1, -1])
    dti_model = dti.TensorModel(gtab)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_dg = BootDirectionGetter.from_data(voxel, dti_model, 30,
                                                sphere=sphere, sh_order=6)
    initial_direction = boot_dg.initial_direction(np.zeros(3))
    npt.assert_equal(len(initial_direction), 1)
    npt.assert_allclose(initial_direction[0], [1, 0, 0], atol=0.1)

    # test that we get multiple directions when we have a multi-tensor
    mevals = np.array([[1.5, 0.4, 0.4], [1.5, 0.4, 0.4]]) * 1e-3
    fracs = [60, 40]
    voxel, primary_evecs = multi_tensor(gtab, mevals, fractions=fracs,
                                        snr=None)
    voxel = voxel.reshape([1, 1, 1, -1])
    response = (np.array([0.0015, 0.0004, 0.0004]), 1)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response=response,
                                                    sh_order_max=4)
        boot_dg = BootDirectionGetter.from_data(voxel, csd_model, 30,
                                                sphere=sphere,)
        initial_direction = boot_dg.initial_direction(np.zeros(3))

    npt.assert_equal(len(initial_direction), 2)
    npt.assert_allclose(initial_direction, primary_evecs, atol=0.1)


def test_bdg_get_direction():
    """This tests the direction found by the bootstrap direction getter.
    """

    _, fbvals, fbvecs = get_fnames('small_64D')

    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs)
    gtab = gradient_table(bvals, bvecs, b0_threshold=0)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    angles = [(0, 0)]

    voxel, _ = multi_tensor(gtab, mevals, 1, angles=angles, fractions=[100],
                            snr=100)
    data = np.tile(voxel, (3, 3, 3, 1))
    sphere = get_sphere('symmetric362')
    response = (np.array([0.0015, 0.0003, 0.0003]), 1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd_model = ConstrainedSphericalDeconvModel(gtab,
                                                    response,
                                                    sh_order_max=6)

    point = np.array([0., 0., 0.])
    prev_direction = sphere.vertices[5]

    # test case in which no valid direction is found with default max attempts
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_dg = BootDirectionGetter(data, model=csd_model, max_angle=10.,
                                      sphere=sphere)
        npt.assert_equal(boot_dg.get_direction(point, prev_direction), 1)

    # test case in which no valid direction is found with new max attempts
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_dg = BootDirectionGetter(data, model=csd_model, max_angle=10,
                                      sphere=sphere, max_attempts=3)
        npt.assert_equal(boot_dg.get_direction(point, prev_direction), 1)

    # test case in which a valid direction is found
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_dg = BootDirectionGetter(data, model=csd_model, max_angle=60.,
                                      sphere=sphere, max_attempts=5)
        npt.assert_equal(boot_dg.get_direction(point, prev_direction), 0)

    # test invalid max_attempts parameters
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        npt.assert_raises(
            ValueError,
            lambda: BootDirectionGetter(data, csd_model, 60, sphere=sphere,
                                        max_attempts=0))


@set_random_number_generator()
def test_bdg_residual(rng):
    """This tests the bootstrapping residual.
    """

    hsph_updated = HemiSphere.from_sphere(unit_icosahedron).subdivide(2)
    vertices = hsph_updated.vertices
    bvecs = vertices
    bvals = np.ones(len(vertices)) * 1000
    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, 0, 0)
    gtab = gradient_table(bvals, bvecs)
    r, theta, phi = cart2sphere(*vertices.T)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        B, m, n = shm.real_sh_descoteaux(6, theta, phi)
    shm_coeff = rng.random(B.shape[1])

    # sphere_func is sampled of the spherical function for each point of
    # the sphere
    sphere_func = np.dot(shm_coeff, B.T)

    response = (np.array([1.5e3, 0.3e3, 0.3e3]), 1)
    voxel = np.concatenate((np.zeros(1), sphere_func))
    data = np.tile(voxel, (3, 3, 3, 1))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd_model = ConstrainedSphericalDeconvModel(gtab,
                                                    response,
                                                    sh_order_max=6)
        boot_dg = BootDirectionGetter.from_data(data, model=csd_model,
                                                max_angle=60,
                                                sphere=hsph_updated,
                                                sh_order=6)

        # Two boot samples should be the same
        odf1 = boot_dg.get_pmf(np.array([1.5, 1.5, 1.5]))
        odf2 = boot_dg.get_pmf(np.array([1.5, 1.5, 1.5]))
        npt.assert_array_almost_equal(odf1, odf2)

    # A boot sample with less sh coeffs should have residuals, thus the two
    # should be different
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_dg2 = BootDirectionGetter.from_data(data, model=csd_model,
                                                 max_angle=60,
                                                 sphere=hsph_updated,
                                                 sh_order=4)
    odf1 = boot_dg2.get_pmf(np.array([1.5, 1.5, 1.5]))
    odf2 = boot_dg2.get_pmf(np.array([1.5, 1.5, 1.5]))
    npt.assert_(np.any(odf1 != odf2))

    # test with a gtab with two shells and assert you get an error
    bvals[-1] = 2000
    gtab = gradient_table(bvals, bvecs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd_model = ConstrainedSphericalDeconvModel(gtab,
                                                    response,
                                                    sh_order_max=6)
    npt.assert_raises(ValueError, BootDirectionGetter, data, csd_model, 60,
                      hsph_updated, 6)


def test_boot_pmf():
    # This tests the local model used for the bootstrapping.
    hsph_updated = HemiSphere.from_sphere(unit_icosahedron)
    vertices = hsph_updated.vertices
    bvecs = vertices
    bvals = np.ones(len(vertices)) * 1000
    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, 0, 0)
    gtab = gradient_table(bvals, bvecs)
    voxel = single_tensor(gtab)
    data = np.tile(voxel, (3, 3, 3, 1))
    point = np.array([1., 1., 1.])
    tensor_model = TensorModel(gtab)
    response = (np.array([1.5e3, 0.3e3, 0.3e3]), 1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_dg = BootDirectionGetter(data, model=tensor_model, max_angle=60,
                                      sphere=hsph_updated)
    no_boot_pmf = boot_dg.get_pmf_no_boot(point)
    model_pmf = tensor_model.fit(voxel).odf(hsph_updated)

    npt.assert_equal(len(hsph_updated.vertices), no_boot_pmf.shape[0])
    npt.assert_array_almost_equal(no_boot_pmf, model_pmf)

    # test model spherical harmonic order different than bootstrap order
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        warnings.simplefilter("always", category=PendingDeprecationWarning)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response,
                                                    sh_order_max=6)
        # Tests that the first caught warning comes from the CSD model
        # constructor
        npt.assert_(issubclass(w[0].category, UserWarning))
        npt.assert_("Number of parameters required " in str(w[0].message))

        # Tests that additional warnings are raised for outdated SH basis
        npt.assert_(len(w) > 1)

        boot_dg_sh4 = BootDirectionGetter(data, model=csd_model,
                                          max_angle=60, sphere=hsph_updated,
                                          sh_order=4)
        pmf_sh4 = boot_dg_sh4.get_pmf(point)
        npt.assert_equal(len(hsph_updated.vertices), pmf_sh4.shape[0])
        npt.assert_(np.sum(pmf_sh4.shape) > 0)

        boot_dg_sh8 = BootDirectionGetter(data, model=csd_model,
                                          max_angle=60, sphere=hsph_updated,
                                          sh_order=8)
        pmf_sh8 = boot_dg_sh8.get_pmf(point)
        npt.assert_equal(len(hsph_updated.vertices), pmf_sh8.shape[0])
        npt.assert_(np.sum(pmf_sh8.shape) > 0)


    # test b_tol parameter
    bvals[-2] = 1100
    gtab = gradient_table(bvals, bvecs)
    tensor_model = TensorModel(gtab)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        npt.assert_raises(ValueError, BootDirectionGetter, data, tensor_model,
                          60, hsph_updated, 6, 20)
        npt.assert_raises(ValueError, BootDirectionGetter, data, tensor_model,
                          60, hsph_updated, 6, -1)

