"""Visualization tools.

Functions and classes for visualizing diffusion MRI data,
streamlines, and tensors (powered by FURY and Matplotlib).
"""

# Init file for visualization package
import warnings

from dipy.utils.optpkg import optional_package
from dipy.viz.horizon.app import horizon
from dipy.viz.skyline.app import skyline, skyline_from_files

fury_pckg_msg = (
    "You do not have FURY installed. Some visualization functions might not "
    "work for you. Please install or upgrade FURY using pip install -U fury --pre. "
    "For detailed installation instructions visit: https://fury.gl/"
)

# Allow import, but disable doctests if we don't have fury
fury, has_fury, _ = optional_package(
    "fury", trip_msg=fury_pckg_msg, min_version="2.0.0a6"
)
