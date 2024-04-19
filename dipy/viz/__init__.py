# Init file for visualization package
import warnings

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.app import horizon

# Allow import, but disable doctests if we don't have fury
fury, has_fury, _ = optional_package(
    'fury',
    trip_msg="You do not have FURY installed. Some visualization functions"
    "might not work for you. For installation instructions, please visit: "
    "https://fury.gl/", min_version="0.10.0")


if has_fury:
    from fury import actor, colormap, interactor, lib, shaders, ui, utils, window
    from fury.data import DATA_DIR as FURY_DATA_DIR, fetch_viz_icons, read_viz_icons

else:
    warnings.warn(
        "You do not have FURY installed. "
        "Therefore, 3D visualization functions will not work for you. "
        "Please install or upgrade FURY using pip install -U fury"
        "For detailed installation instructions visit: https://fury.gl/")

# We make the visualization requirements optional imports:
_, has_mpl, _ = optional_package(
    'matplotlib',
    trip_msg="You do not have Matplotlib installed. Some visualization "
    "functions might not work for you. For installation instructions, "
    "please visit: https://matplotlib.org/")

if has_mpl:
    from . import projections
