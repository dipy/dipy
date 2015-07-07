import numpy as np
from dipy.viz import actor, window, fvtk
from dipy.data import fetch_viz_icons, read_viz_icons
import numpy.testing as npt


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_renderer():

    ren = window.Renderer()

    # background color for renderer
    bg_float = (1, 0.5, 0)

    # that will come in the image in the 0-255 uint scale
    bg_color = tuple((np.floor(255 * np.array(bg_float))).astype('uint8'))

    ren.background(bg_float)

    # window.show(ren)
    arr = window.snapshot(ren)
    print(bg_color)
    report = window.analyze_snapshot(arr,
                                     bg_color=bg_color,
                                     colors=[bg_color, (0, 127, 0)])
    npt.assert_equal(report.objects, 0)
    npt.assert_equal(report.colors_found, [True, False])

    axes = fvtk.axes()
    ren.add(axes)
    # window.show(ren)

    arr = window.snapshot(ren)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 1)

    ren.rm(axes)
    arr = window.snapshot(ren)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 0)

    window.add(ren, axes)
    arr = window.snapshot(ren)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 1)

    ren.rm_all()
    arr = window.snapshot(ren)
    report = window.analyze_snapshot(arr, bg_color)
    npt.assert_equal(report.objects, 0)

    ren2 = window.renderer(bg_float)
    ren2.background((0, 0, 0.))

    report = window.analyze_renderer(ren2)
    npt.assert_equal(report.bg_color, (0, 0, 0))

    ren2.add(axes)

    report = window.analyze_renderer(ren2)
    npt.assert_equal(report.actors, 3)

    window.rm(ren2, axes)
    report = window.analyze_renderer(ren2)
    npt.assert_equal(report.actors, 0)


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_parallel_projection():

    ren = window.Renderer()
    axes = fvtk.axes()
    ren.add(axes)

    axes2 = fvtk.axes()
    axes2.SetPosition((2, 0, 0))
    ren.add(axes2)

    # Put the camera on a angle so that the
    # camera can show the difference between perspective
    # and parallel projection
    fvtk.camera(ren, pos=(1.5, 1.5, 1.5))
    ren.GetActiveCamera().Zoom(2)

    # window.show(ren, reset_camera=True)
    ren.reset_camera()
    arr = window.snapshot(ren)

    ren.projection('parallel')
    # window.show(ren, reset_camera=False)
    arr2 = window.snapshot(ren)
    # Because of the parallel projection the two axes
    # will have the same size and therefore occupy more
    # pixels rather than in perspective projection were
    # the axes being further will be smaller.
    npt.assert_equal(np.sum(arr2 > 0) > np.sum(arr > 0), True)


if __name__ == '__main__':

    test_renderer()
    #npt.run_module_suite()
