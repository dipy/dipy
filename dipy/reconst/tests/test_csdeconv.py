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

    SNR = 100  # None 10, 20, 30
    bvalue = 1000
    S0 = 1
    sh_order = 8
    visu = True

    # signal gtab
    #sphere = get_sphere('symmetric362')
    sphere = unit_octahedron
    sphere = sphere.subdivide(3)



    # sphere = sphere.subdivide(2)
    s_bvecs = np.concatenate(([[0, 0, 0]], sphere.vertices))
    s_bvals = np.zeros(len(s_bvecs)) + bvalue
    s_bvals[0] = 0
    s_gtab = gradient_table(s_bvals, s_bvecs)
    s_mevals = np.array(([0.0015, 0.0003, 0.0003], [0.0015, 0.0003, 0.0003]))
    s_mevecs = [np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
                np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])]
    S, sticks = multi_tensor(s_gtab, s_mevals, S0, angles=[(0, 0), (90, 0)],
                             fractions=[50, 50], snr=SNR)
    odf_gt = multi_tensor_odf(sphere.vertices, [0.5, 0.5], s_mevals, s_mevecs)

    if visu:
        from dipy.viz import fvtk
        r = fvtk.ren()
        fvtk.add(r, fvtk.sphere_funcs(np.vstack((S[1:], odf_gt)), sphere))
        fvtk.show(r)

    # single fiber response function gtab on a sphere of 362 points
    psphere = get_sphere('symmetric362')
    r_bvecs = np.concatenate(([[0, 0, 0]], psphere.vertices))
    r_bvals = np.zeros(len(r_bvecs)) + bvalue
    r_bvals[0] = 0
    r_gtab = gradient_table(r_bvals, r_bvecs)
    R = single_tensor(r_gtab, S0, s_mevals[1], s_mevecs[1], snr=None)
    r_odf = single_tensor_odf(psphere.vertices, s_mevals[1], s_mevecs[1])

    if visu:
        from dipy.viz import fvtk
        r = fvtk.ren()
        fvtk.add(r, fvtk.sphere_funcs(np.vstack((R[1:], r_odf)), psphere))
        fvtk.show(r)

    s_sh, B_dwi = sf_to_sh(S[1:], sphere, sh_order, ret_bmatrix=True)
    r_sh, B_regul = sf_to_sh(R[1:], psphere, sh_order, ret_bmatrix=True)
    r_rh = sh_to_rh(r_sh, sh_order)
    
    R = forward_sdeconv_mat(r_rh, sh_order)

    # checking standard csdeconv function

    Lambda = 1.0
    Lambda = Lambda * R.shape[0] * r_rh[0] / B_regul.shape[0]  

    fodf_sh, num_it = csdeconv(s_sh, sh_order, R, B_regul, Lambda, 0.1)
    fodf = sh_to_sf(fodf_sh, psphere, sh_order)
    print 'converged after %d iterations' % num_it
    
    if visu:
        odf_gt = multi_tensor_odf(psphere.vertices, [0.5, 0.5], s_mevals, s_mevecs)
        from dipy.viz import fvtk
        r = fvtk.ren()
        fvtk.add(r, fvtk.sphere_funcs(np.vstack((odf_gt, fodf)), psphere))
        fvtk.show(r)

    # checking model class

    csd = ConstrainedSphericalDeconvModel(s_gtab)
    from time import time
    t1 = time()    
    fodf_sh2, fodf2 = csd.fit(S)
    print time()-t1
    assert_array_almost_equal(fodf_sh2, fodf_sh)
    assert_array_almost_equal(fodf2, fodf)

    if visu:
        from dipy.viz import fvtk
        r = fvtk.ren()
        fvtk.add(r, fvtk.sphere_funcs(np.vstack((odf_gt, fodf2)), psphere))
        fvtk.show(r)


if __name__ == '__main__':
    test_csdeconv()