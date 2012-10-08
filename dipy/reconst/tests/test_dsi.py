import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, run_module_suite)
from dipy.data import get_data
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.reconst.odf import gfa
from dipy.sims.voxel import SticksAndBall
from dipy.core.sphere import Sphere
from dipy.core.gradients import gradient_table
from dipy.data import get_sphere
from numpy.testing import assert_equal
from dipy.core.subdivide_octahedron import create_unit_sphere
from dipy.core.sphere_stats import angular_similarity


def test_dsi():
    #load symmetric 724 sphere
    sphere = get_sphere('symmetric724')
    #load icosahedron sphere
    sphere2 = create_unit_sphere(5)
    btable = np.loadtxt(get_data('dsi515btable'))    
    bvals = btable[:,0]
    bvecs = btable[:,1:]        
    data, golden_directions = SticksAndBall(bvals, bvecs, d=0.0015, 
                               S0=100, angles=[(0, 0), (90, 0)], 
                               fractions=[50, 50], snr=None) 
    gtab = gradient_table(bvals, bvecs) 
    ds = DiffusionSpectrumModel(gtab)
    #symmetric724
    ds.direction_finder.config(sphere=sphere, min_separation_angle=25,
                               relative_peak_threshold=.35)
    dsfit = ds.fit(data)
    odf = dsfit.odf(sphere)
    directions = dsfit.directions
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 
                            2, 1)
    #5 subdivisions
    ds.direction_finder.config(sphere=sphere2, min_separation_angle=25,
                              relative_peak_threshold=.35)
    dsfit = ds.fit(data)
    odf2 = dsfit.odf(sphere2)
    directions = dsfit.directions
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 
                            2, 1)
    #from dipy.viz._show_odfs import show_odfs
    #show_odfs(odf[None,None,None,:], (sphere.vertices, sphere.faces))
    #show_odfs(odf2[None,None,None,:], (sphere2.vertices, sphere2.faces))
    assert_equal(dsfit.pdf.shape, 3 * (ds.qgrid_size, ))
    sb_dummies=sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        odf = ds.fit(data).odf(sphere2)
        directions = ds.fit(data).directions
        #show_odfs(odf[None, None, None, :], (sphere2.vertices, sphere2.faces))
        if len(directions) <= 3:
            assert_equal(len(ds.fit(data).directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(ds.fit(data).odf(sphere2)) < 0.1, True)


def sticks_and_ball_dummies(gtab):
    bvals=gtab.bvals
    bvecs=gtab.bvecs
    sb_dummies={}
    S, sticks = SticksAndBall(bvals, bvecs, d=0.0015, S0=100, 
                              angles=[(0, 0)], 
                              fractions=[100], snr=None)   
    sb_dummies['1fiber'] = (S, sticks)
    S, sticks = SticksAndBall(bvals, bvecs, d=0.0015, S0=100, 
                             angles=[(0, 0), (90, 0)], 
                             fractions=[50, 50], snr=None)   
    sb_dummies['2fiber'] = (S, sticks)
    S, sticks = SticksAndBall(bvals, bvecs, d=0.0015, S0=100, 
                           angles=[(0, 0), (90, 0), (90, 90)], 
                           fractions=[33, 33, 33], snr=None)   
    sb_dummies['3fiber'] = (S, sticks)
    S, sticks = SticksAndBall(bvals, bvecs, d=0.0015, S0=100, 
                             angles=[(0, 0), (90, 0), (90, 90)], 
                             fractions=[0, 0, 0], snr=None)
    sb_dummies['isotropic'] = (S, sticks)
    return sb_dummies


if __name__ == '__main__':
    run_module_suite()
    #test_dsi_rf()
