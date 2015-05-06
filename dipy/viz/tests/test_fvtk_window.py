import numpy as np
from dipy.viz import actor, window, fvtk
from dipy.data import fetch_viz_icons, read_viz_icons
import numpy.testing as npt


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_renderer():

    ren = window.Renderer()

    ren.background((1, 0.5, 0))

    # window.show(ren)

    arr = window.snapshot(ren)

    report = window.analyze_snapshot(arr,
                                     colors=[(255, 128, 0), (0, 127, 0)])

    npt.assert_equal(report.objects, 1)
    npt.assert_equal(report.colors_found, [True, False])

    axes = fvtk.axes()

    ren.add(axes)

    window.show(ren)

    ren.rm(axes)

    window.show(ren)

    window.add(ren, axes)

    window.show(ren)

    ren.rm_all()

    window.show(ren)



if __name__ == '__main__':

    test_renderer()
    # npt.run_module_suite()
