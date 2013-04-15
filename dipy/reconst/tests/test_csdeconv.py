import numpy as np
from numpy.testing import (assert_almost_equal,
                           assert_array_equal,
                           run_module_suite)

from dipy.data import get_sphere, get_data
from dipy.sims.voxel import (multi_tensor,
                             multi_tensor_odf,
                             all_tensor_evecs)
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import (ConstrainedSphericalDeconvModel,
                                   ConstrainedSDTModel)
from dipy.reconst.odf import peak_directions
from dipy.core.sphere_stats import angular_similarity


def test_csdeconv():
    SNR = 30
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

    assert_almost_equal(ang_sim > 1.95, True)
    assert_array_equal(directions.shape[0], 2)
    assert_array_equal(directions2.shape[0], 2)


def test_odfdeconv():
    SNR = 30  # 20, 10
    S0 = 1

    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003],
                       [0.0015, 0.0003, 0.0003]))

    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (60, 0)],
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric362')

    mevecs = [all_tensor_evecs(sticks[0]).T,
              all_tensor_evecs(sticks[1]).T]

    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)

    e1 = 15.0  # 13.9
    e2 = 3.0  # 3.55
    ratio = e2/e1

    # print 'Deconvolution eigen value ratio is %f'%ratio
    csd = ConstrainedSDTModel(gtab, ratio, None)

    csd_fit = csd.fit(S)
    fodf = csd_fit.odf(sphere)

    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)

    ang_sim = angular_similarity(directions, directions2)

    assert_almost_equal(ang_sim > 1.95, True)
    assert_array_equal(directions.shape[0], 2)
    assert_array_equal(directions2.shape[0], 2)


if __name__ == '__main__':
    run_module_suite()
