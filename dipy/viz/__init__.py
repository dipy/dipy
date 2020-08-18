# Init file for visualization package
import warnings

from distutils.version import LooseVersion
from dipy.utils.optpkg import optional_package
# Allow import, but disable doctests if we don't have fury
fury, has_fury, _ = optional_package(
    'fury',
    "You do not have FURY installed. Some visualization functions"
    "might not work for you. For installation instructions, please visit: "
    "https://fury.gl/")


if has_fury:
    from fury import actor, window, colormap, interactor, ui, utils
    from fury.window import vtk
    from fury.data import (fetch_viz_icons, read_viz_icons,
                           DATA_DIR as FURY_DATA_DIR)

    HAVE_VTK_9_PLUS = vtk.vtkVersion.GetVTKMajorVersion() >= 9
    HAVE_FURY_0_6_LESS = LooseVersion(fury.__version__) < LooseVersion('0.6.0')
    if HAVE_VTK_9_PLUS and HAVE_FURY_0_6_LESS:
        msg = "You are using VTK 9+ and FURY {}. ".format(fury.__version__)
        msg += "Some functions might not work. To have a better experience, "
        msg += "please upgrade to FURY version 0.6.0+. "
        msg += "pip install -U fury"
        warnings.warn(msg)


else:
    warnings.warn(
        "You do not have FURY installed. "
        "Some visualization functions might not work for you. "
        "For installation instructions, please visit: https://fury.gl/")

# We make the visualization requirements optional imports:
_, has_mpl, _ = optional_package(
    'matplotlib',
    "You do not have Matplotlib installed. Some visualization functions"
    "might not work for you. For installation instructions, please visit: "
    "https://matplotlib.org/")

if has_mpl:
    from . import projections
