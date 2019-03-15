import numpy as np
from dipy.data import get_sphere, get_3shell_gtab, get_isbi2013_2shell_gtab
from dipy.reconst.shore import ShoreModel
from dipy.reconst.shm import sh_to_sf
from dipy.direction.peaks import peak_directions
from dipy.reconst.odf import gfa
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite)
from dipy.sims.voxel import SticksAndBall
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies


def test_shore_odf():
    gtab = get_isbi2013_2shell_gtab()

    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')

    # load icosahedron sphere
    sphere2 = create_unit_sphere(5)
    data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                            S0=100, angles=[(0, 0), (90, 0)],
                                            fractions=[50, 50], snr=None)
    asm = ShoreModel(gtab, radial_order=6,
                     zeta=700, lambdaN=1e-8, lambdaL=1e-8)
    # symmetric724
    asmfit = asm.fit(data)
    odf = asmfit.odf(sphere)
    odf_sh = asmfit.odf_sh()
    odf_from_sh = sh_to_sf(odf_sh, sphere, 6, basis_type=None)
    assert_almost_equal(odf, odf_from_sh, 10)

    directions, _, _ = peak_directions(odf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, golden_directions), 2, 1)

    # 5 subdivisions
    odf = asmfit.odf(sphere2)
    directions, _, _ = peak_directions(odf, sphere2, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(
        angular_similarity(directions, golden_directions), 2, 1)

    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        asmfit = asm.fit(data)
        odf = asmfit.odf(sphere2)
        directions, _ , _ = peak_directions(odf, sphere2, .35, 25)
        if len(directions) <= 3:
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(odf) < 0.1, True)


def test_multivox_shore():
    gtab = get_3shell_gtab()

    data = np.random.random([20, 30, 1, gtab.gradients.shape[0]])
    radial_order = 4
    zeta = 700
    asm = ShoreModel(gtab, radial_order=radial_order,
                     zeta=zeta, lambdaN=1e-8, lambdaL=1e-8)
    asmfit = asm.fit(data)
    c_shore = asmfit.shore_coeff
    assert_equal(c_shore.shape[0:3], data.shape[0:3])
    assert_equal(np.alltrue(np.isreal(c_shore)), True)


if __name__ == '__main__':
    run_module_suite()
