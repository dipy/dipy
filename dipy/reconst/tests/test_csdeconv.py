import warnings
import numpy as np
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           assert_array_equal,
                           run_module_suite)

from dipy.data import get_sphere, get_data
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             all_tensor_evecs)
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel,
                                   odf_sh_to_sharp)
from dipy.reconst.odf import peak_directions
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.shm import sf_to_sh, sh_to_sf


def test_csdeconv():
    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (60, 0)],
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric724')
    sphere = sphere.subdivide(1)

    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]

    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)

    response = (np.array([0.0015, 0.0003, 0.0003]), S0)

    csd = ConstrainedSphericalDeconvModel(gtab, response)

    csd_fit = csd.fit(S)

    fodf = csd_fit.odf(sphere)

    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions, directions2)

    assert_equal(ang_sim > 1.98, True)
    assert_array_equal(directions.shape[0], 2)
    assert_array_equal(directions2.shape[0], 2)

    with warnings.catch_warnings(True) as w:

        csd = ConstrainedSphericalDeconvModel(gtab, response, sh_order=16)
        assert_equal(len(w) > 0, True)

    with warnings.catch_warnings(True) as w:

        csd = ConstrainedSphericalDeconvModel(gtab, response, sh_order=8)
        assert_equal(len(w) > 0, False)


def test_odfdeconv():
    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0017, 0.0003, 0.0003],
                       [0.0017, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (65, 0)],
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric724')
    sphere = sphere.subdivide(1)

    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]

    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)

    e1 = 17.0
    e2 = 3.0
    ratio = e2 / e1

    # print 'Deconvolution eigen value ratio is %f'%ratio
    csd = ConstrainedSDTModel(gtab, ratio, None)

    csd_fit = csd.fit(S)
    fodf = csd_fit.odf(sphere)

    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions, directions2)

    assert_equal(ang_sim > 1.98, True)

    assert_array_equal(directions.shape[0], 2)
    assert_array_equal(directions2.shape[0], 2)


def test_odf_sh_to_sharp():

    SNR = 100
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (45, 0)],
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric724')
    sphere = sphere.subdivide(1)

    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]

    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)

    odfs_gt = np.zeros((3, 1, 1, odf_gt.shape[0]))
    odfs_gt[:,:,:] = odf_gt[:]

    odfs_sh = sf_to_sh(odfs_gt, sphere, sh_order=8, basis_type='mrtrix')

    fodf_sh = odf_sh_to_sharp(odfs_sh, sphere, basis='mrtrix', ratio=3 / 15.,
                              sh_order=8, Lambda=.1, tau=.02)

    fodf = sh_to_sf(fodf_sh, sphere, sh_order=8, basis_type='mrtrix')

    directions, _, _ = peak_directions(odf_gt, sphere, min_separation_angle=25.)
    directions2, _, _ = peak_directions(fodf[0, 0, 0], sphere,
                                        min_separation_angle=25.)

    def two_fibers_angle(directions):

        return np.arccos(np.abs(np.dot(directions[0], directions[1]))) * 180. / np.pi

    angle = two_fibers_angle(directions)
    angle2 = two_fibers_angle(directions2)

    #print angle, angle2

    #print directions.shape, directions2.shape

    # print np.sum(gfa(fodf)) / 3**3
    # print np.sum(gfa(odfs_gt)) / 3**3

    # assert_equal(np.sum(gfa(fodf))/3**3 < np.sum(gfa(odfs_gt))/3**3, True)

    """
    from dipy.viz import fvtk
    r = fvtk.ren()
    fvtk.add(r, fvtk.sphere_funcs(odf_gt, sphere))
    fvtk.show(r)

    fvtk.clear(r)
    fvtk.add(r, fvtk.sphere_funcs(fodf[0, 0, 0], sphere))
    fvtk.show(r)
    """


if __name__ == '__main__':
    # run_module_suite()
    test_odf_sh_to_sharp()
