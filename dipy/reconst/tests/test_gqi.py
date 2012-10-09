import numpy as np
from dipy.data import get_data
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


def test_gqi():
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
    gq = GeneralizedQSamplingModel(gtab, method='gqi2', sampling_length=1.4)
    #symmetric724
    gq.direction_finder.config(sphere=sphere, min_separation_angle=25,
                               relative_peak_threshold=.35)
    gqfit = gq.fit(data)
    odf = gqfit.odf(sphere)
    #from dipy.viz._show_odfs import show_odfs
    #show_odfs(odf[None,None,None,:], (sphere.vertices, sphere.faces))
    directions = gqfit.directions
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2, 1)
    #5 subdivisions
    gq.direction_finder.config(sphere=sphere2, min_separation_angle=25,
                              relative_peak_threshold=.35)
    gqfit = gq.fit(data)
    odf2 = gqfit.odf(sphere2)
    directions = gqfit.directions
    assert_equal(len(directions), 2)
    assert_almost_equal(angular_similarity(directions, golden_directions), 2, 1)
    #show_odfs(odf[None,None,None,:], (sphere.vertices, sphere.faces))
    sb_dummies=sticks_and_ball_dummies(gtab)
    for sbd in sb_dummies:
        data, golden_directions = sb_dummies[sbd]
        odf = gq.fit(data).odf(sphere2)
        directions = gq.fit(data).directions
        #show_odfs(odf[None, None, None, :], (sphere2.vertices, sphere2.faces))
        if len(directions) <= 3:
            assert_equal(len(gq.fit(data).directions), len(golden_directions))
        if len(directions) > 3:
            assert_equal(gfa(gq.fit(data).odf(sphere2)) < 0.1, True)


if __name__ == "__main__":
    run_module_suite()









