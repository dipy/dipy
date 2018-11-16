# Init file for visualization package
from __future__ import division, print_function, absolute_import


from dipy.utils.optpkg import optional_package
# Allow import, but disable doctests if we don't have fury
fury, have_fury, _ = optional_package('fury')


if have_fury:
    from fury import actor, window, widget, colormap, interactor, ui, utils
    from fury.window import vtk
    from fury.data import (fetch_viz_icons, read_viz_icons,
                           DATA_DIR as FURY_DATA_DIR)

# We make the visualization requirements optional imports:
_, has_mpl, _ = optional_package('matplotlib',
                                 "You do not have Matplotlib installed. Some"
                                 " visualization functions might not work for"
                                 " you")

if has_mpl:
    from . import projections
