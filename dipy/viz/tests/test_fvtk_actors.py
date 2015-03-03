# -*- coding: utf-8 -*-
import os
import numpy as np
import scipy as sp

from dipy.viz import actor
from dipy.viz import window

import numpy.testing as npt
from nibabel.tmpdirs import TemporaryDirectory


def analyze_output(renderer, fname, cleanup=True):
    result = sp.misc.imread(fname)
    bg = renderer.GetBackground()
    if bg == (0, 0, 0):
        npt.assert_equal(result.sum() > 0, True)
    else:
        raise ValueError('The background of the renderer is not black')

    return True


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_butcher():

    renderer = window.renderer()

    # data = np.random.randint(0, 255, (50, 50, 50))
    data = (255 * np.random.rand(50, 50, 50))
    affine = np.eye(4)

    #from dipy.viz import fvtk
    #slicer = fvtk.slicer(data)

    slicer = actor.butcher(data, affine)

    window.add(renderer, slicer)

    window.show(renderer)


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
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

    with TemporaryDirectory() as tmpdir:
        fname = os.path.join(tmpdir, 'streamtube.png')
        window.record(renderer, out_path=fname)
        npt.assert_(analyze_output(renderer, fname))


if __name__ == "__main__":

    # npt.run_module_suite()
    test_butcher()
