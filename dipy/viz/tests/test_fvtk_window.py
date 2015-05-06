import numpy as np
from dipy.viz import actor, window, fvtk
from dipy.data import fetch_viz_icons, read_viz_icons
import numpy.testing as npt


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_renderer():

    ren = window.Renderer()

    bg_float = (1, 0.5, 0)

    bg_color = tuple((np.round(255 * np.array(bg_float))).astype('uint8'))

    ren.background(bg_float)

    # window.show(ren)
    arr = window.snapshot(ren)
    report = window.analyze_snapshot(arr,
                                     bg_color=bg_color,
                                     colors=[bg_color, (0, 127, 0)])
    npt.assert_equal(report.objects, 0)
    npt.assert_equal(report.colors_found, [True, False])

    axes = fvtk.axes()
    ren.add(axes)
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



if __name__ == '__main__':

    test_renderer()
    # npt.run_module_suite()
