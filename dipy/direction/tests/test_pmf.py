import warnings
import numpy as np
import numpy.testing as npt

from dipy.core.gradients import gradient_table
from dipy.core.sphere import HemiSphere, unit_octahedron
from dipy.data import default_sphere, get_sphere
from dipy.direction.pmf import SimplePmfGen, SHCoeffPmfGen, BootPmfGen
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.reconst.dti import TensorModel
from dipy.reconst.shm import descoteaux07_legacy_msg
from dipy.sims.voxel import single_tensor

response = (np.array([1.5e3, 0.3e3, 0.3e3]), 1)


def test_pmf_val():
    sphere = get_sphere('symmetric724')
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        pmfgen = SHCoeffPmfGen(np.random.random([2, 2, 2, 28]), sphere, None)
    point = np.array([1, 1, 1], dtype='float')

    for idx in [0, 5, 15, -1]:
        pmf = pmfgen.get_pmf(point)
        # Create a direction vector close to the vertex idx
        xyz = sphere.vertices[idx] + np.random.random([3]) / 100
        pmf_idx = pmfgen.get_pmf_value(point, xyz)
        # Test that the pmf sampled for the direction xyz is correct
        npt.assert_array_equal(pmf[idx], pmf_idx)


def test_pmf_from_sh():
    sphere = HemiSphere.from_sphere(unit_octahedron)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        pmfgen = SHCoeffPmfGen(np.ones([2, 2, 2, 28]), sphere, None)

    # Test that the pmf is greater than 0 for a valid point
    pmf = pmfgen.get_pmf(np.array([0, 0, 0], dtype='float'))
    npt.assert_equal(np.sum(pmf) > 0, True)

    # Test that the pmf is 0 for invalid Points
    npt.assert_array_equal(pmfgen.get_pmf(np.array([-1, 0, 0], dtype='float')),
                           np.zeros(len(sphere.vertices)))
    npt.assert_array_equal(pmfgen.get_pmf(np.array([0, 0, 10], dtype='float')),
                           np.zeros(len(sphere.vertices)))


def test_pmf_from_array():
    sphere = HemiSphere.from_sphere(unit_octahedron)
    pmfgen = SimplePmfGen(np.ones([2, 2, 2, len(sphere.vertices)]), sphere)

    # Test that the pmf is greater than 0 for a valid point
    pmf = pmfgen.get_pmf(np.array([0, 0, 0], dtype='float'))
    npt.assert_equal(np.sum(pmf) > 0, True)

    # Test that the pmf is 0 for invalid Points
    npt.assert_array_equal(pmfgen.get_pmf(np.array([-1, 0, 0], dtype='float')),
                           np.zeros(len(sphere.vertices)))
    npt.assert_array_equal(pmfgen.get_pmf(np.array([0, 0, 10], dtype='float')),
                           np.zeros(len(sphere.vertices)))

    # Test ValueError for negative pmf
    npt.assert_raises(
        ValueError,
        lambda: SimplePmfGen(np.ones([2, 2, 2, len(sphere.vertices)])*-1,
                             sphere))
    # Test ValueError for non matching pmf and sphere
    npt.assert_raises(
        ValueError,
        lambda: SimplePmfGen(np.ones([2, 2, 2, len(sphere.vertices)]),
                             default_sphere))


def test_boot_pmf():
    # This tests the local model used for the bootstrapping.
    hsph_updated = HemiSphere.from_sphere(unit_octahedron)
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

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_pmf_gen = BootPmfGen(
            data, model=tensor_model, sphere=hsph_updated)
    no_boot_pmf = boot_pmf_gen.get_pmf_no_boot(point)

    model_pmf = tensor_model.fit(voxel).odf(hsph_updated)

    npt.assert_equal(len(hsph_updated.vertices), no_boot_pmf.shape[0])
    npt.assert_array_almost_equal(no_boot_pmf, model_pmf)

    # test model spherical harmonic order different than bootstrap order
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", category=UserWarning)
        warnings.simplefilter("always", category=PendingDeprecationWarning)
        csd_model = ConstrainedSphericalDeconvModel(gtab, response,
                                                    sh_order=6)
        # Tests that the first catched warning comes from
        # the CSD model  constructor
        npt.assert_(issubclass(w[0].category, UserWarning))
        npt.assert_("Number of parameters required " in str(w[0].message))
        # Tests that additional warnings are raised for outdated SH basis
        npt.assert_(len(w) > 1)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_pmf_gen_sh4 = BootPmfGen(data, sphere=hsph_updated,
                                      model=csd_model, sh_order=4)
        pmf_sh4 = boot_pmf_gen_sh4.get_pmf(point)
    npt.assert_equal(len(hsph_updated.vertices), pmf_sh4.shape[0])
    npt.assert_(np.sum(pmf_sh4.shape) > 0)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=descoteaux07_legacy_msg,
            category=PendingDeprecationWarning)
        boot_pmf_gen_sh8 = BootPmfGen(data, model=csd_model,
                                      sphere=hsph_updated, sh_order=8)
    pmf_sh8 = boot_pmf_gen_sh8.get_pmf(point)
    npt.assert_equal(len(hsph_updated.vertices), pmf_sh8.shape[0])
    npt.assert_(np.sum(pmf_sh8.shape) > 0)
