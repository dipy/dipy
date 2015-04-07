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
    arr = window.snapshot(renderer)
    report = window.analyze_snapshot(renderer, arr, find_objects=True)
    npt.assert_equal(report.objects, 1)

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
        npt.assert_equal(window.analyze_snapshot(renderer, fname).objects, 1)


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
@npt.dec.skipif(not window.have_imread)
def test_streamtube_and_line_actors():

    renderer = window.renderer()

    line1 = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2.]])
    line2 = line1 + np.array([0.5, 0., 0.])

    lines = [line1, line2]
    colors = np.array([[1, 0, 0], [0, 0, 1.]])
    c = actor.line(lines, colors, linewidth=3)
    window.add(renderer, c)

    # create streamtubes of the same lines and shift them a bit
    c2 = actor.streamtube(lines, colors, linewidth=.1)
    c2.SetPosition(2, 0, 0)
    window.add(renderer, c2)

    window.show(renderer)
    arr = window.snapshot(renderer)
    # npt.assert_(window.analyze_snapshot(renderer, arr))
    report = window.analyze_snapshot(renderer, arr,
                                     colors=[(255, 0, 0), (0, 0, 255)],
                                     find_objects=True)


    imshow(arr)
    imshow(report.labels)
    print(report.objects)
    print(report.colors_found)


if __name__ == "__main__":

    # npt.run_module_suite()
    # test_butcher()
    test_streamtube_and_line_actors()
