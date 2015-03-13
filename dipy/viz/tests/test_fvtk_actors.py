# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy as sp

from dipy.viz import actor, window, utils

import numpy.testing as npt
from nibabel.tmpdirs import TemporaryDirectory


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_butcher():

    renderer = window.renderer()

    data = (255 * np.random.rand(50, 50, 50))
    affine = np.eye(4)
    slicer = actor.butcher(data, affine)
    window.add(renderer, slicer)

    # window.show(renderer)

    # copy pixels in numpy array directly
    arr = window.snapshot(renderer, None)
    npt.assert_(window.analyze_snapshot(renderer, arr))

    # The slicer can cut directly a smaller part of the image
    slicer.SetDisplayExtent(10, 30, 10, 30, 35, 35)
    slicer.Update()
    renderer.ResetCamera()

    window.add(renderer, slicer)

    # save pixels in png file not a numpy array
    with TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'butcher.png')
        # window.show(renderer)
        window.snapshot(renderer, fname)
        # imshow(window.snapshot(renderer), origin='lower')
        npt.assert_(window.analyze_snapshot(renderer, fname))


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_streamtube_and_line_actors():

    renderer = window.renderer()

    lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    colors = np.random.rand(2, 3)
    c = actor.line(lines, colors)
    window.add(renderer, c)

    # create streamtubes of the same lines and shift them a bit
    c2 = actor.streamtube(lines, colors)
    c2.SetPosition(2, 0, 0)
    window.add(renderer, c2)

    # window.show(renderer)
    arr = window.snapshot(renderer)
    npt.assert_(window.analyze_snapshot(renderer, arr))


if __name__ == "__main__":

    npt.run_module_suite()
    # test_butcher()
