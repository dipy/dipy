""" Testing vizualization with fvtk
"""
import numpy as np

from dipy.viz import fvtk

from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_
import numpy.testing as npt


@npt.dec.skipif(not fvtk.have_vtk)
def test_fvtk_functions():

    # Create a renderer
    r = fvtk.ren()

    # Create 2 lines with 2 different colors
    lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    colors = np.random.rand(2, 3)
    c = fvtk.line(lines, colors)
    fvtk.add(r, c)

    # Create a volume and return a volumetric actor using volumetric rendering
    vol = 100 * np.random.rand(100, 100, 100)
    vol = vol.astype('uint8')
    r = fvtk.ren()
    v = fvtk.volume(vol)
    fvtk.add(r, v)

    # Remove all objects
    fvtk.rm_all(r)

    # Put some text
    l = fvtk.label(r, text='Yes Men')
    fvtk.add(r, l)

    # Show everything
    # fvtk.show(r)


@npt.dec.skipif(not fvtk.have_vtk)
def test_fvtk_ellipsoid():

    from dipy.data import get_data
    _, fbvals, fbvecs = get_data('small_64D')
    bvals = np.load(fbvals)
    bvecs = np.load(fbvecs)
    from dipy.core.gradients import gradient_table
    gtab = gradient_table(bvals, bvecs)

    evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    evecs = np.eye(3)

    from dipy.sims.voxel import SingleTensor
    S = SingleTensor(gtab, 100, evals, evecs, snr=None)
    assert_array_almost_equal(S[gtab.b0s_mask], 100)
    assert_(np.mean(S[~gtab.b0s_mask]) < 100)

    from dipy.reconst.dti import TensorModel

    tm = TensorModel(gtab)
    tf = tm.fit(S)

    assert_array_almost_equal(tf.fa, 0.707, decimal=3)

    ea = np.diag([4, 1, 1.])
    ev = tf.evecs

    from dipy.data import get_sphere

    sphere = get_sphere('symmetric724')

    proj = np.dot(ea, sphere.vertices.T)

    # els = np.abs(np.sqrt(proj[0, :]**2 + proj[1, :]**2 + proj[2, :]**2) - 1)

    els = np.sqrt(proj[0,:]**2 + proj[1,:]**2 + proj[2,:]**2)
    els = els / els.min()

    ren = fvtk.ren()

    #fvtk.add(ren, fvtk.point((sphere.vertices.T * els).T / np.abs(els).max(), fvtk.red))

    fvtk.add(ren, fvtk.point((sphere.vertices * els[:, None]), fvtk.red,theta=10, phi=10 ))

    fvtk.show(ren)

    #fvtk.clear(ren)

    fvtk.add(ren, fvtk.point(proj.T, fvtk.green, theta=10, phi=10))

    fvtk.show(ren)

    fvtk.add(ren, fvtk.point(sphere.vertices, fvtk.yellow, theta=10, phi=10))

    fvtk.show(ren)

    pds = proj.T / sphere.vertices

    fvtk.add(ren, fvtk.point(sphere.vertices * pds, fvtk.blue, theta=10, phi=10))

    fvtk.show(ren)



    1/0
    # fvtk.clear(ren)

    # fvtk.add(ren, fvtk.sphere_funcs(els-els.min(), sphere, colormap=None,
    #    norm=True, radial_scale=True))

    # fvtk.show(ren)


test_fvtk_ellipsoid()
