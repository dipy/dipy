import numpy as np
import numpy.testing as npt
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             single_tensor_odf,
                             all_tensor_evecs)
from dipy.data import get_sphere
from dipy.core.sphere import disperse_charges, Sphere, HemiSphere
import dipy.reconst.dti as dti
from dipy.core.gradients import gradient_table
from dipy.sims.phantom import SingleTensor
from dipy.direction import peaks_from_model
from dipy.direction import DeterministicMaximumDirectionGetter
from dipy.tracking.local import LocalTracking, ThresholdTissueClassifier
from dipy.data import default_sphere
from dipy.tracking import utils
from dipy.reconst.csdeconv import ConstrainedSphericalDeconvModel


def uniform_toy_data():
    toydict = {}
    n_pts = 64
    theta = np.pi * np.random.rand(n_pts)
    phi = 2 * np.pi * np.random.rand(n_pts)
    hsph_initial = HemiSphere(theta=theta, phi=phi)
    hsph_updated, potential = disperse_charges(hsph_initial, 5000)

    vertices = hsph_updated.vertices
    values = np.ones(vertices.shape[0])

    bvecs = np.vstack((vertices))
    bvals = np.hstack((1000 * values))

    bvecs = np.insert(bvecs, (0, bvecs.shape[0]), np.array([0, 0, 0]), axis=0)
    bvals = np.insert(bvals, (0, bvals.shape[0]), 0)

    gtab = gradient_table(bvals, bvecs)

    toy_voxel = SingleTensor(gtab)
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


def test_num_sls():

    toydict = uniform_toy_data()
    csd_model = ConstrainedSphericalDeconvModel(toydict['gtab'], None, sh_order=6)
    csd_fit = csd_model.fit(toydict['toy_data'])    
    sltest_list = [('toy_roi_long_plane', 121), 
                   ('toy_roi_radial_plane', 121), 
                   ('toy_roi_center_vox', 1)]

    classifier = ThresholdTissueClassifier(toydict['toy_tissue_classifier'], .1)
    detmax_dg = DeterministicMaximumDirectionGetter.from_shcoeff(csd_fit.shm_coeff,
                                                                 max_angle=30.,
                                                                 sphere=default_sphere)

    expected_sl_length = 11
    for roi, num_sl in sltest_list:
        seed = utils.seeds_from_mask(toydict[roi])
        streamlines = LocalTracking(detmax_dg, classifier, seed, toydict['toy_affine'], step_size=1)
        streamlines = list(streamlines)
        npt.assert_equal(len(streamlines), num_sl)
        for sl in streamlines:
            npt.assert_equal(len(sl), expected_sl_length)
