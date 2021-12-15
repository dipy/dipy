import pytest
import numpy as np
import numpy.testing as npt
from dipy.testing.decorators import use_xvfb
from dipy.utils.optpkg import optional_package
from dipy.data import DATA_DIR
from dipy.align.reslice import reslice
from nibabel.tmpdirs import TemporaryDirectory

fury, has_fury, setup_module = optional_package('fury')

if has_fury:
    from fury.io import load_image
    from fury import actor, window

skip_it = use_xvfb == 'skip'

@pytest.mark.skipif(skip_it or not has_fury,
                    reason="Needs xvfb")
def test_slicer():
    scene = window.Scene()
    data = (255 * np.random.rand(50, 50, 50))
    affine = np.diag([1, 3, 2, 1])
    data2, affine2 = reslice(data, affine, zooms=(1, 3, 2),
                                new_zooms=(1, 1, 1))

    slicer = actor.slicer(data2, affine2, interpolation='linear')
    slicer.display(None, None, 25)

    scene.add(slicer)
    scene.reset_camera()
    scene.reset_clipping_range()

    window.show(scene, reset_camera=False)
    arr = window.snapshot(scene, offscreen=True)
    report = window.analyze_snapshot(arr, find_objects=True)
    npt.assert_equal(report.objects, 1)
    npt.assert_array_equal([1, 3, 2] * np.array(data.shape),
                            np.array(slicer.shape))


test_slicer()