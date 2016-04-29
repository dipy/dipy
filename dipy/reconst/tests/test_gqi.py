import numpy as np
from dipy.data import get_data, dsi_voxels
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.sims.voxel import SticksAndBall
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.data import get_sphere
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite)
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.odf import gfa
from dipy.direction.peaks import peak_directions


def test_gqi():
    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')
    # load icosahedron sphere
    sphere2 = create_unit_sphere(5)
    btable = np.loadtxt(get_data('dsi515btable'))
    bvals = btable[:, 0]
    bvecs = btable[:, 1:]
    gtab = gradient_table(bvals, bvecs)
    data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                            S0=100, angles=[(0, 0), (90, 0)],
                                            fractions=[50, 50], snr=None)
    gq = GeneralizedQSamplingModel(gtab, method='gqi2', sampling_length=1.4)

    # symmetric724
    gqfit = gq.fit(data)
    odf = gqfit.odf(sphere)
    directions, values, indices = peak_directions(odf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2,
                        1)

    # 5 subdivisions
    gqfit = gq.fit(data)
    odf2 = gqfit.odf(sphere2)
    directions, values, indices = peak_directions(odf2, sphere2, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2,
                        1)

    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        odf = gq.fit(data).odf(sphere2)
        directions, values, indices = peak_directions(odf, sphere2, .35, 25)
        if len(directions) <= 3:
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(odf) < 0.1, True)


def test_mvoxel_gqi():
    data, gtab = dsi_voxels()
    sphere = get_sphere('symmetric724')

    gq = GeneralizedQSamplingModel(gtab, 'standard')
    gqfit = gq.fit(data)
    all_odfs = gqfit.odf(sphere)

    # Check that the first and last voxels each have 2 peaks
    odf = all_odfs[0, 0, 0]
    directions, values, indices = peak_directions(odf, sphere, .35, 25)
    assert_equal(directions.shape[0], 2)
    odf = all_odfs[-1, -1, -1]
    directions, values, indices = peak_directions(odf, sphere, .35, 25)
    assert_equal(directions.shape[0], 2)


if __name__ == "__main__":
    run_module_suite()
