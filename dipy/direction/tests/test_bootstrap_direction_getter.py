import numpy as np
import numpy.testing as npt
import dipy
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             single_tensor_odf,
                             all_tensor_evecs)
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
import dipy.reconst.dti as dti
import dipy.reconst.csdeconv as csd
from dipy.core.gradients import gradient_table
from dipy.sims.phantom import single_tensor
from dipy.direction import peaks_from_model
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.data import default_sphere
from dipy.tracking import utils
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel
from dipy.core.sphere import unit_icosahedron
import dipy.direction.bootstrap_direction_getter as bdg
from dipy.reconst import shm
from dipy.core.geometry import cart2sphere

# Set up the toy example


def uniform_toy_data():
    toydict = {}

    hsph_updated = HemiSphere.from_sphere(unit_icosahedron).subdivide(2)

    vertices = hsph_updated.vertices

    bvecs = vertices
    bvals = np.ones(len(vertices)) * 1000

    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, 0, 0)

    gtab = gradient_table(bvals, bvecs)

    toy_voxel = single_tensor(gtab)
    toy_data = np.tile(toy_voxel, (11, 11, 11, 1))
    toy_tissue_classifier = np.ones(toy_data.shape[:-1])
    toy_affine = np.eye(4)

    # make a slice roi that should track in-plane only
    toy_roi_long_plane = np.zeros(toy_data.shape[:-1])
    toy_roi_long_plane[:, :, 0]=1
    # make a slice roi that should track directly across the whole volume
    toy_roi_radial_plane = np.zeros(toy_data.shape[:-1])
    toy_roi_radial_plane[0, :, :] = 1
    # make an roi that contains the center voxel only
    toy_roi_center_vox = np.zeros(toy_data.shape[:-1])
    toy_roi_center_vox[5, 5, 5] = 1

    toydict['gtab'] = gtab
    toydict['toy_data'] = toy_data
    toydict['toy_affine'] = toy_affine
    toydict['toy_roi_long_plane'] = toy_roi_long_plane
    toydict['toy_roi_radial_plane'] = toy_roi_radial_plane
    toydict['toy_roi_center_vox'] = toy_roi_center_vox
    toydict['toy_tissue_classifier'] = toy_tissue_classifier

    return toydict

def test_bdg_initial_direction():
    # test that we get one direction when we have a single tensor
    toydict = {}

    hsph_updated = HemiSphere.from_sphere(unit_icosahedron).subdivide(2)

    vertices = hsph_updated.vertices

    bvecs = vertices
    bvals = np.ones(len(vertices)) * 1000

    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, 0, 0)

    gtab = gradient_table(bvals, bvecs)

    toy_voxel = single_tensor(gtab).reshape([1, 1, 1, -1])

    model = dti.TensorModel(gtab)

    direction_getter = bdg.BootDirectionGetter.from_data(toy_voxel, model, 30)

    initial_direction = direction_getter.initial_direction(np.zeros(3))

    npt.assert_equal(len(initial_direction), 1)

    npt.assert_allclose(initial_direction[0], [1, 0, 0], atol=0.1)

    # test that we get multiple directions when we have a multi-tensor

    mevals = np.array([[1.5, 0.4, 0.4], [1.5, 0.4, 0.4]]) * 1e-3
    fracs = [60, 40]
    toy_voxel, primary_evecs = multi_tensor(gtab, mevals,
                                            fractions=fracs, snr=None)
    toy_voxel = toy_voxel.reshape([1, 1, 1, -1])
    model = csd.ConstrainedSphericalDeconvModel(gtab, response=None)
    direction_getter = bdg.BootDirectionGetter.from_data(toy_voxel, model, 30)
    initial_direction = direction_getter.initial_direction(np.zeros(3))
    print(initial_direction)
    npt.assert_equal(len(initial_direction), 2)
    npt.assert_allclose(initial_direction, primary_evecs, atol=0.1)

def test_bdg_get_direction():
    ''' test if direction getter goes 5 tries if '''

    sphere = HemiSphere.from_sphere(unit_icosahedron.subdivide())

    two_neighbors = sphere.edges[0]
    direction1 = sphere.vertices[two_neighbors[0]]
    direction2 = sphere.vertices[two_neighbors[1]]
    angle = np.rad2deg(direction1.dot(direction2))

    class Fakepmf():
        count = 0
        def get_pmf(self, point):
            pmf = np.zeros(len(sphere.vertices))
            pmf[two_neighbors[0]] = 1
            self.count += 1
            return pmf

        def pmf_no_boot(self, point):
            pass


    myfakepmf = Fakepmf()
    mydirgetter = bdg.BootDirectionGetter(myfakepmf, angle/2., sphere=sphere)

    mypoint = np.zeros(3)
    myprev_direction = direction2.copy()

    # test case in which no valid direction is found with default maxdir
    no_valid_direction = mydirgetter.get_direction(mypoint, myprev_direction)

    npt.assert_equal(no_valid_direction, 1)
    npt.assert_equal(direction2, myprev_direction)
    npt.assert_equal(myfakepmf.count, 5)

    # test case in which no valid direction is found with new max attempts
    myfakepmf.count = 0
    mydirgetter = bdg.BootDirectionGetter(myfakepmf, angle/2., sphere=sphere,
                                          max_attempts=3)
    no_valid_direction = mydirgetter.get_direction(mypoint, myprev_direction)

    npt.assert_equal(no_valid_direction, 1)
    npt.assert_equal(direction2, myprev_direction)
    npt.assert_equal(myfakepmf.count, 3)

    # now test case in which valid direction is found
    myfakepmf.count = 0
    mydirgetter = bdg.BootDirectionGetter(myfakepmf, angle*2., sphere=sphere)
    no_valid_direction = mydirgetter.get_direction(mypoint, myprev_direction)

    npt.assert_equal(no_valid_direction, 0)
    npt.assert_equal(direction1, myprev_direction)
    npt.assert_equal(myfakepmf.count, 1)

def test_bdg_bog_pmfnoboot():
    # does it return the right size of the sphere we give it?
    #check the model you explicitly fit is the one that is returned

    hsph_updated = HemiSphere.from_sphere(unit_icosahedron).subdivide(2)

    vertices = hsph_updated.vertices

    bvecs = vertices
    bvals = np.ones(len(vertices)) * 1000

    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, 0, 0)

    gtab = gradient_table(bvals, bvecs)

    toy_voxel = single_tensor(gtab)
    toy_data = np.tile(toy_voxel, (3, 3, 3, 1))

    tensor_model = dti.TensorModel(gtab)

    mybog = bdg.BootOdfGen(toy_data, model=tensor_model, sphere=hsph_updated)
    odf_fit = mybog.pmf_no_boot(np.array([1.,1.,1.]))

    myfit = tensor_model.fit(toy_voxel).odf(hsph_updated)

    npt.assert_equal(len(hsph_updated.vertices), odf_fit.shape[0])
    npt.assert_array_almost_equal(myfit, odf_fit)


def test_bdg_bog_pmfboot():
    #test if residuals = 0
    #make a perfect dataset that can be fit

    hsph_updated = HemiSphere.from_sphere(unit_icosahedron).subdivide(2)
    vertices = hsph_updated.vertices
    bvecs = vertices

    bvals = np.ones(len(vertices)) * 1000

    bvecs = np.insert(bvecs, 0, np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, 0, 0)

    gtab = gradient_table(bvals, bvecs)

    r, theta, phi = cart2sphere(*vertices.T)

    B, m, n = shm.real_sym_sh_basis(6, theta, phi)

    shm_coeff = np.random.random(B.shape[1])

    # sphere_func is samples of the spherical function for each point of the
    # sphere
    sphere_func = np.dot(shm_coeff, B.T)

    toy_voxel = np.concatenate((np.zeros(1), sphere_func))
    toy_data = np.tile(toy_voxel, (3, 3, 3, 1))

    csd_model = ConstrainedSphericalDeconvModel(gtab,
                                                None, sh_order=6)

    mybog = bdg.BootOdfGen(toy_data, model=csd_model, sphere=hsph_updated,
                           sh_order=6)
    # Two boot samples should be the same if there are no residuales
    myodf1 = mybog.get_pmf(np.array([1.5, 1.5, 1.5]))
    myodf2 = mybog.get_pmf(np.array([1.5, 1.5, 1.5]))

    npt.assert_array_almost_equal(myodf1, myodf2)

    # A boot sample with less sh coeffs should have residuals, thus the two
    # should be different
    mybog2 = bdg.BootOdfGen(toy_data, model=csd_model, sphere=hsph_updated,
                           sh_order=4)
    myodf1 = mybog2.get_pmf(np.array([1.5, 1.5, 1.5]))
    myodf2 = mybog2.get_pmf(np.array([1.5, 1.5, 1.5]))

    npt.assert_(np.any(myodf1 != myodf2))

    # throw in a gtab with two shells and assert you get an error
    bvals[-1] = 2000
    gtab = gradient_table(bvals, bvecs)

    csd_model = ConstrainedSphericalDeconvModel(gtab,
                                                None, sh_order=6)
    npt.assert_raises(ValueError, bdg.BootOdfGen, toy_data, csd_model, hsph_updated,6)


def test_num_sls():

    toydict = uniform_toy_data()
    csd_model = ConstrainedSphericalDeconvModel(toydict['gtab'],
                                                None, sh_order=6)
    csd_fit = csd_model.fit(toydict['toy_data'])

    sltest_list = [('toy_roi_long_plane', 121),
                   ('toy_roi_radial_plane', 121), 
                   ('toy_roi_center_vox', 1)]

    classifier = ThresholdTissueClassifier(toydict['toy_tissue_classifier'], .1)
    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(
        csd_fit.shm_coeff,max_angle=30.,sphere=default_sphere)

    expected_sl_length = 11
    for roi, num_sl in sltest_list:
        seed = utils.seeds_from_mask(toydict[roi])
        streamlines = LocalTracking(detmax_dg, classifier, seed,
                                    toydict['toy_affine'], step_size=1)
        streamlines = list(streamlines)
        npt.assert_equal(len(streamlines), num_sl)
        for sl in streamlines:
            npt.assert_equal(len(sl), expected_sl_length)

if __name__ == '__main__':
    npt.run_module_suite()