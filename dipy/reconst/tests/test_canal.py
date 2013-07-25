import numpy as np
import pylab as py
from dipy.data import get_data, two_shells_voxels, three_shells_voxels, get_sphere
from dipy.reconst.canal import ShoreModel
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
from dipy.reconst.shm import sh_to_sf
from dipy.viz import fvtk
import nibabel as nib


def test_canal():
    # load symmetric 724 sphere
    sphere = get_sphere('symmetric724')
    # load icosahedron sphere
    sphere2 = create_unit_sphere(5)
    fimg, fbvals, fbvecs = get_data('ISBI_testing_2shells_table')
    bvals, bvecs = read_bvals_bvecs(fbvals, fbvecs) 
    gtab = gradient_table(bvals[1:], bvecs[1:,:])
    data, golden_directions = SticksAndBall(gtab, d=0.0015,
                                            S0=100, angles=[(0, 0), (90, 0)],
                                            fractions=[50, 50], snr=None)

    asm = ShoreModel(gtab)

    # symmetric724
    asmfit = asm.fit(data)
    Cshore = asmfit.l2estimation(radialOrder=6, zeta=700, lambdaN=1e-8, lambdaL=1e-8)
    Csh = asmfit.odf()
    odf = sh_to_sf(Csh, sphere, 6, basis_type="fibernav")
    directions, _, _ = peak_directions(odf, sphere, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2, 1)

    # 5 subdivisions
    asmfit = asm.fit(data)
    Cshore = asmfit.l2estimation(radialOrder=6, zeta=700, lambdaN=1e-8, lambdaL=1e-8)
    Csh = asmfit.odf()
    odf = sh_to_sf(Csh, sphere2, 6, basis_type="fibernav")
    directions, _, _ = peak_directions(odf, sphere2, .35, 25)
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2, 1)

    sb_dummies = sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        asmfit = asm.fit(data)
        Cshore = asmfit.l2estimation(radialOrder=6, zeta=700, lambdaN=1e-8, lambdaL=1e-8)
        Csh = asmfit.odf()
        odf = sh_to_sf(Csh, sphere2, 6, basis_type="fibernav")
        directions, _, _ = peak_directions(odf, sphere2, .35, 25)
        if len(directions) <= 3:
            assert_equal(len(directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(odf) < 0.1, True)


def test_multivox_canal():

  
    # data, affine, gtab = two_shells_voxels(10, 40, 10, 40, 25, 26)
    data, affine, gtab = three_shells_voxels(45, 65, 35, 65, 33, 34)

    asm = ShoreModel(gtab)
    asmfit = asm.fit(data)
    radialOrder = 4
    zeta = 700
    Cshore = asmfit.l2estimation(radialOrder=radialOrder, zeta=zeta, lambdaN=1e-8, lambdaL=1e-8)

    assert_equal(Cshore.shape[0:3], data.shape[0:3])
    assert_equal(np.alltrue(np.isreal(Cshore)), True)


if __name__ == '__main__':
    run_module_suite()
    #test_multivox_canal()
    #test_canal()
