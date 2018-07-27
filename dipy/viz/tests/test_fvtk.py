"""Testing visualization with fvtk."""
import os
import warnings
import numpy as np
from distutils.version import LooseVersion

from dipy.viz import fvtk
from dipy import data

import dipy.utils.testing as npt
from dipy.testing.decorators import xvfb_it
from dipy.utils.optpkg import optional_package
import pytest

use_xvfb = os.environ.get('TEST_WITH_XVFB', False)
if use_xvfb == 'skip':
    skip_it = True
else:
    skip_it = False

cm, have_matplotlib, _ = optional_package('matplotlib.cm')

if have_matplotlib:
    import matplotlib
    mpl_version = LooseVersion(matplotlib.__version__)


@pytest.mark.skipif(not fvtk.have_vtk or not fvtk.have_vtk_colors or skip_it, reason=" Requires vtk or vtk_colors")
@xvfb_it
def test_fvtk_functions():
    # This tests will fail if any of the given actors changed inputs or do
    # not exist

    # Create a renderer
    r = fvtk.ren()

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
    r = fvtk.ren()
    v = fvtk.volume(vol)
    fvtk.add(r, v)

    # Remove all objects
    fvtk.rm_all(r)

    # Put some text
    l = fvtk.label(r, text='Yes Men')
    fvtk.add(r, l)

    # Slice the volume
    slicer = fvtk.slicer(vol)
    slicer.display(50, None, None)
    fvtk.add(r, slicer)

    # Change the position of the active camera
    fvtk.camera(r, pos=(0.6, 0, 0), verbose=False)

    fvtk.clear(r)

    # Peak directions
    p = fvtk.peaks(np.random.rand(3, 3, 3, 5, 3))
    fvtk.add(r, p)

    p2 = fvtk.peaks(np.random.rand(3, 3, 3, 5, 3),
                    np.random.rand(3, 3, 3, 5),
                    colors=(0, 1, 0))
    fvtk.add(r, p2)


@pytest.mark.skipif(not fvtk.have_vtk or not fvtk.have_vtk_colors or skip_it, reason=" Requires vtk or vtk_colors")
@xvfb_it
def test_fvtk_ellipsoid():

    evals = np.array([1.4, .35, .35]) * 10 ** (-3)
    evecs = np.eye(3)

    mevals = np.zeros((3, 2, 4, 3))
    mevecs = np.zeros((3, 2, 4, 3, 3))

    mevals[..., :] = evals
    mevecs[..., :, :] = evecs

    from dipy.data import get_sphere

    sphere = get_sphere('symmetric724')

    ren = fvtk.ren()

    fvtk.add(ren, fvtk.tensor(mevals, mevecs, sphere=sphere))

    fvtk.add(ren, fvtk.tensor(mevals, mevecs, np.ones(mevals.shape),
             sphere=sphere))

    npt.assert_equal(ren.GetActors().GetNumberOfItems(), 2)


def test_colormap():
    v = np.linspace(0., .5)
    map1 = fvtk.create_colormap(v, 'bone', auto=True)
    map2 = fvtk.create_colormap(v, 'bone', auto=False)
    npt.assert_(not np.allclose(map1, map2))

    npt.assert_raises(ValueError, fvtk.create_colormap, np.ones((2, 3)))
    npt.assert_raises(ValueError, fvtk.create_colormap, v, 'no such map')


@pytest.mark.skipif(not fvtk.have_matplotlib, reason="Requires matplotlib")
def test_colormaps_matplotlib():
    v = np.random.random(1000)
    # The "Accent" colormap is deprecated as of 0.12:
    with warnings.catch_warnings(record=True) as w:
        # Cause all warnings to always be triggered.
        warnings.simplefilter("always")
        data.get_cmap("Accent")
        # Test that the deprecation warning was raised:
        npt.assert_(len(w) > 0)

    names = ['jet', 'Blues', 'bone']

    if have_matplotlib and mpl_version < "2":
        names.append('Accent')

    for name in names:
        with warnings.catch_warnings(record=True) as w:
            # Matplotlib version of get_cmap
            rgba1 = fvtk.get_cmap(name)(v)
            # Dipy version of get_cmap
            rgba2 = data.get_cmap(name)(v)
            # dipy's colormaps are close to matplotlibs colormaps, but not
            # perfect:
            npt.assert_array_almost_equal(rgba1, rgba2, 1)
            npt.assert_(len(w) == (1 if name == 'Accent' else 0))



if __name__ == "__main__":
    pytest.main()
