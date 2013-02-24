import numpy as np
from numpy.testing import (assert_equal,
						   assert_almost_equal,
						   assert_array_equal,
						   assert_array_almost_equal)

from dipy.core.sphere import unit_icosahedron, unit_octahedron
from dipy.reconst.shm import sf_to_sh
from dipy.data import get_sphere
from dipy.sims.voxel import (single_tensor, single_tensor_odf,
                             multi_tensor, multi_tensor_odf)
from dipy.core.gradients import gradient_table
from dipy.reconst.csdeconv import sh_to_rh
from dipy.reconst.shm import sf_to_sh, sh_to_sf
from dipy.reconst.csdeconv import (forward_sdeconv_mat, 
                                   csdeconv,
                                   ConstrainedSphericalDeconvModel)


def test_csdeconv():

    SNR = 30
    bvalue = 1000
    S0 = 1
    sh_order = 8
    visu = False

    from dipy.data import get_data
    _, fbvals, fbvecs = get_data('small_64D')

    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)

    gtab = gradient_table(bvals, bvecs)
    mevals = np.array(([0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]))
    mevecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
              np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
    
    S, sticks = multi_tensor(gtab, mevals, S0, angles=[(0, 0), (90, 0)],
                             fractions=[50, 50], snr=SNR)

    sphere = get_sphere('symmetric362')
    
    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], mevals, mevecs)
    
    csd = ConstrainedSphericalDeconvModel(gtab)
    from time import time
    t1 = time()
    fodf_sh, fodf = csd.fit(S)
    print time()-t1
    
    if visu:
        from dipy.viz import fvtk
        r = fvtk.ren()
        fvtk.add(r, fvtk.sphere_funcs(np.vstack((odf_gt, fodf)), sphere))
        fvtk.show(r)

    from dipy.reconst.odf import peak_directions
    directions, _, _ = peak_directions(odf_gt, sphere)
    directions2, _, _ = peak_directions(fodf, sphere)
    
    from dipy.core.sphere_stats import angular_similarity
    ang_sim = angular_similarity(directions, directions2)
    
    assert_almost_equal(ang_sim > 1.95, True)
    assert_array_equal(directions.shape[0], 2) 
    assert_array_equal(directions2.shape[0], 2)


if __name__ == '__main__':
    test_csdeconv()