import numpy as np
from dipy.data import get_data, two_shells_voxels, three_shells_voxels, get_sphere
from dipy.data.fetcher import (fetch_isbi2013_2shell, read_isbi2013_2shell,
                               fetch_sherbrooke_3shell, read_sherbrooke_3shell)
from dipy.reconst.shore import ShoreModel
from dipy.reconst.shm import QballModel
from dipy.reconst.odf import gfa, peak_directions
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
from numpy.testing import (assert_equal,
                           assert_almost_equal,
                           run_module_suite,
                           assert_array_equal,
                           assert_raises)
from dipy.sims.voxel import SticksAndBall, multi_tensor
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity
from dipy.reconst.tests.test_dsi import sticks_and_ball_dummies
import nibabel as nib


def test_shore():
    fetch_isbi2013_2shell()
    img, gtab=read_isbi2013_2shell()

    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')

    # load icosahedron sphere
    sphere2 = create_unit_sphere(5)    
    data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                            S0=100, angles=[(0, 0), (90, 0)],
                                            fractions=[50, 50], snr=None)
    asm = ShoreModel(gtab,radialOrder=6, zeta=700, lambdaN=1e-8, lambdaL=1e-8)
    # symmetric724
    asmfit = asm.fit(data)
    odf = asmfit.odf(sphere)

    directions, _, _ = peak_directions(odf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2, 1)

    # 5 subdivisions
    odf = asmfit.odf(sphere2)
    directions, _, _ = peak_directions(odf, sphere2, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2, 1)

    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        asmfit = asm.fit(data)
        odf = asmfit.odf(sphere2)
        directions, _, _ = peak_directions(odf, sphere2, .35, 25)
        if len(directions) <= 3:
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(odf) < 0.1, True)


def test_multivox_shore():    
    fetch_sherbrooke_3shell()
    img, gtab=read_sherbrooke_3shell()

    test = img.get_data()
    data = test[45:65, 35:65, 33:34]
    radialOrder = 4
    zeta = 700
    asm = ShoreModel(gtab, radialOrder=radialOrder, zeta=zeta, lambdaN=1e-8, lambdaL=1e-8)
    asmfit = asm.fit(data)
    c_shore=asmfit.shore_coeff
    assert_equal(c_shore.shape[0:3], data.shape[0:3])
    assert_equal(np.alltrue(np.isreal(c_shore)), True)


if __name__ == '__main__':
    run_module_suite()

    
