""" Testing vizualization with fvtk
"""
import numpy as np

from dipy.viz import fvtk

from numpy.testing import assert_equal, assert_
import numpy.testing as npt

import tempfile
import os
import shutil


@npt.dec.skipif(not fvtk.have_vtk)
@npt.dec.skipif(not fvtk.have_vtk_colors)
def test_fvtk_functions():

    # Create a renderer
    r = fvtk.renderer()

    # Create 2 lines with 2 different colors
    lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    colors = np.random.rand(2, 3)
    c = fvtk.line(lines, colors)
    fvtk.add(r, c)

    # create streamtubes of the same lines and shift them a bit
    c2 = fvtk.streamtube(lines, colors)
    c2.SetPosition(2, 0, 0)
    fvtk.add(r, c2)

    # Create a volume and return a volumetric actor using volumetric rendering
    vol = 100 * np.random.rand(100, 100, 100)
    vol = vol.astype('uint8')
    r = fvtk.renderer()
    v = fvtk.volume(vol)
    fvtk.add(r, v)

    # Remove all objects
    fvtk.rm_all(r)

    # Put some text
    l = fvtk.label(r, text='Yes Men')
    fvtk.add(r, l)

    # Slice the volume
    fvtk.add(r, fvtk.slicer(vol, plane_i=[50]))

    # Change the position of the active camera
    fvtk.camera(r, pos=(0.6, 0, 0), verbose=False)

    # Show everything
    # fvtk.show(r)


@npt.dec.skipif(not fvtk.have_vtk)
@npt.dec.skipif(not fvtk.have_vtk_colors)
def test_fvtk_record():
    d = tempfile.mkdtemp(prefix='dipy_fvtk')
    try:
        ren = fvtk.renderer()
        fvtk.record(ren, os.path.join(d, 'snapshot'), size=(10, 10))
        assert_(os.path.exists(os.path.join(d, 'snapshot.png')))

        fvtk.record(ren, os.path.join(d, 'snapshot'), size=(10, 10),
                    path_numbering=True)
        assert_(os.path.exists(os.path.join(d, 'snapshot0000000.png')))

        fvtk.record(ren, os.path.join(d, 'snapshot'), size=(10, 10),
                    path_numbering=True, n_frames=3)
        for i in range(3):
            assert_(os.path.exists(os.path.join(d, 'snapshot000000%d.png' % i)))

        fvtk.record(ren, os.path.join(d, 'snapshot'), size=(10, 10),
                    magnification=2)


    finally:
        shutil.rmtree(d)


@npt.dec.skipif(not fvtk.have_vtk)
@npt.dec.skipif(not fvtk.have_vtk_colors)
def test_fvtk_object_API():
    ren = fvtk.renderer()
    axes = fvtk.axes()
    ren.add(axes)
    ren.rm(axes)
    ren.add(axes)
    ren.clear()


@npt.dec.skipif(not fvtk.have_vtk)
@npt.dec.skipif(not fvtk.have_vtk_colors)
def test_fvtk_ellipsoid():
    evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    evecs = np.eye(3)

    mevals = np.zeros((3, 2, 4, 3))
    mevecs = np.zeros((3, 2, 4, 3, 3))

    mevals[..., :] = evals
    mevecs[..., :, :] = evecs

    from dipy.data import get_sphere

    sphere = get_sphere('symmetric724')

    ren = fvtk.renderer()

    fvtk.add(ren, fvtk.tensor(mevals, mevecs, sphere=sphere))

    fvtk.add(ren, fvtk.tensor(mevals, mevecs, np.ones(mevals.shape), sphere=sphere))

    assert_equal(ren.GetActors().GetNumberOfItems(), 2)
