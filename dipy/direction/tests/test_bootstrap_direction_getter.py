import warnings

import numpy as np
import numpy.testing as npt

from dipy.core.geometry import cart2sphere
from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, unit_icosahedron

from dipy.data import get_sphere
from dipy.direction.bootstrap_direction_getter import BootDirectionGetter
from dipy.direction.pmf import BootPmfGen, SimplePmfGen
from dipy.reconst import shm
from dipy.reconst import dti
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.sims.voxel import single_tensor, multi_tensor


DEFAULT_SH = 4
response = (np.array([1.5e3, 0.3e3, 0.3e3]), 1)


def test_bdg_initial_direction():
    """This test the number of inital direction."
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
                                                    sh_order=4)
        boot_dg = BootDirectionGetter.from_data(voxel, csd_model, 30,
                                                sphere=sphere,)
        initial_direction = boot_dg.initial_direction(np.zeros(3))

    npt.assert_equal(len(initial_direction), 2)
    npt.assert_allclose(initial_direction, primary_evecs, atol=0.1)


def test_bdg_get_direction():
    """This tests the direction found by the bootstrap direction getter.
    """

    sphere = HemiSphere.from_sphere(unit_icosahedron.subdivide())
    two_neighbors = sphere.edges[0]
    direction1 = sphere.vertices[two_neighbors[0]]
    direction2 = sphere.vertices[two_neighbors[1]]
    angle = np.rad2deg(direction1.dot(direction2))
    point = np.zeros(3)
    prev_direction = direction2.copy()

    pmf = np.zeros([1, 1, 1, len(sphere.vertices)])
    pmf[:, :, :, two_neighbors[0]] = 1
    pmf_gen = SimplePmfGen(pmf, sphere)

    # test case in which no valid direction is found with default maxdir
    boot_dg = BootDirectionGetter(pmf_gen, angle / 2., sphere=sphere)
    npt.assert_equal(boot_dg.get_direction(point, prev_direction), 1)
    npt.assert_equal(direction2, prev_direction)

    # test case in which no valid direction is found with new max attempts
    boot_dg = BootDirectionGetter(pmf_gen, angle / 2., sphere=sphere,
                                  max_attempts=3)
    npt.assert_equal(boot_dg.get_direction(point, prev_direction), 1)
    npt.assert_equal(direction2, prev_direction)

    # test case in which a valid direction is found
    boot_dg = BootDirectionGetter(pmf_gen, angle * 2., sphere=sphere,
                                  max_attempts=1)
    npt.assert_equal(boot_dg.get_direction(point, prev_direction), 0)
    npt.assert_equal(direction1, prev_direction)

    # test invalid max_attempts parameters
    npt.assert_raises(
        ValueError,
        lambda: BootDirectionGetter(pmf_gen, angle * 2., sphere=sphere,
                                    max_attempts=0))


def test_bdg_residual():
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
    shm_coeff = np.random.random(B.shape[1])

    # sphere_func is sampled of the spherical function for each point of
    # the sphere
    sphere_func = np.dot(shm_coeff, B.T)

    voxel = np.concatenate((np.zeros(1), sphere_func))
    data = np.tile(voxel, (3, 3, 3, 1))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
        boot_pmf_gen = BootPmfGen(data, model=csd_model, sphere=hsph_updated,
                                  sh_order=6)

        # Two boot samples should be the same
        odf1 = boot_pmf_gen.get_pmf(np.array([1.5, 1.5, 1.5]))
    odf2 = boot_pmf_gen.get_pmf(np.array([1.5, 1.5, 1.5]))
    npt.assert_array_almost_equal(odf1, odf2)

    # A boot sample with less sh coeffs should have residuals, thus the two
    # should be different
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_pmf_gen2 = BootPmfGen(data, model=csd_model, sphere=hsph_updated,
                                   sh_order=4)
    odf1 = boot_pmf_gen2.get_pmf(np.array([1.5, 1.5, 1.5]))
    odf2 = boot_pmf_gen2.get_pmf(np.array([1.5, 1.5, 1.5]))
    npt.assert_(np.any(odf1 != odf2))

    # test with a gtab with two shells and assert you get an error
    bvals[-1] = 2000
    gtab = gradient_table(bvals, bvecs)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=shm.descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response, sh_order=6)
    npt.assert_raises(ValueError, BootPmfGen, data, csd_model, hsph_updated, 6)
