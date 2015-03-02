import numpy as np
import scipy as sp

from dipy.viz import actor
from dipy.viz import window

import numpy.testing as npt


@npt.dec.skipif(not actor.have_vtk)
@npt.dec.skipif(not actor.have_vtk_colors)
def test_streamtube_and_line_actors():

    # Create a renderer
    renderer = window.renderer()

    # Create 2 lines with 2 different colors
    lines = [np.random.rand(10, 3), np.random.rand(20, 3)]
    colors = np.random.rand(2, 3)
    c = actor.line(lines, colors)
    window.add(renderer, c)

    # create streamtubes of the same lines and shift them a bit
    c2 = actor.streamtube(lines, colors)
    c2.SetPosition(2, 0, 0)
    window.add(renderer, c2)

    # window.show(renderer)

    window.record(renderer, out_path='streamtube.png')
    result = sp.misc.imread('streamtube.png')
    bg = renderer.GetBackground()

    if bg == (0, 0, 0):

        npt.assert_equal(result.sum() > 0, True)

    else:

        raise ValueError('Renderer background not black')


test_streamtube_and_line_actors()
