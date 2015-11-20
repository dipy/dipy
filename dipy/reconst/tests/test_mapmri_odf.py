import numpy as np
from dipy.data import get_sphere, get_3shell_gtab, get_isbi2013_2shell_gtab
from dipy.reconst.mapmri import MapmriModel
from dipy.direction.peaks import gfa, peak_directions
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite,
                           assert_array_equal,
                           assert_raises)
from dipy.sims.voxel import SticksAndBall
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies
from dipy.sims.voxel import MultiTensor

def test_mapmri_odf():

    gtab = get_3shell_gtab()
    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')

    # load icosahedron sphere
    sphere2 = create_unit_sphere(5)
    evals = np.array(([0.0017, 0.0003, 0.0003],
                      [0.0017, 0.0003, 0.0003]))
    data, golden_directions = MultiTensor(
        gtab, evals, S0=1.0, angles=[(0, 0), (90, 0)], fractions=[50, 50], snr=None)
    map_model = MapmriModel(gtab, radial_order=4)
    # symmetric724
    mapfit = map_model.fit(data)
    odf = mapfit.odf(sphere)
    directions, _ , _ = peak_directions(odf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, golden_directions), 2, 1)

    # 5 subdivisions
    odf = mapfit.odf(sphere2)
    directions, _ , _ = peak_directions(odf, sphere2, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, golden_directions), 2, 1)

    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        mapfit = map_model.fit(data)
        odf = mapfit.odf(sphere2)
        directions, _ , _ = peak_directions(odf, sphere2, .35, 25)
        if len(directions) <= 3:
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(odf) < 0.1, True)


def test_multivox_mapmri():
    gtab = get_3shell_gtab()

    data = np.random.random([20, 30, 1, gtab.gradients.shape[0]])
    radial_order = 4
    map_model = MapmriModel(gtab, radial_order=radial_order)
    mapfit = map_model.fit(data)
    c_map = mapfit.mapmri_coeff
    assert_equal(c_map.shape[0:3], data.shape[0:3])
    assert_equal(np.alltrue(np.isreal(c_map)), True)


if __name__ == '__main__':
    run_module_suite()
