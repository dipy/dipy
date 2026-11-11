"""
==================================
Advanced interactive visualization
==================================

In DIPY_ we created a thin interface to access many of the capabilities
available in the FURY 3D visualization library :footcite:p:`Garyfallidis2021`
but tailored to the needs of structural and diffusion imaging.

Let's start by importing the necessary modules.
"""

import numpy as np

from dipy.data.fetcher import fetch_bundles_2_subjects, read_bundles_2_subjects
from dipy.tracking.streamline import Streamlines
from dipy.viz import actor, colormap, ui, window

###############################################################################
# In ``window`` we have all the objects that connect what needs to be rendered
# to the display or the disk e.g., for saving screenshots. So, there you will
# find key objects and functions like the ``Scene`` class which holds and
# provides access to all the actors and the ``show`` function which displays
# what is in the scene on a window. Also, this module provides access to
# functions for opening/saving dialogs and printing screenshots
# (see ``snapshot``).
#
# In the ``actor`` module we can find all the different primitives e.g.,
# streamtubes, lines, image slices, etc.
#
# In the ``ui`` module we have some other objects which allow to add buttons
# and sliders and these interact both with windows and actors. Because of
# this they need input from the operating system so they can process events.
#
# Let's get started. In this tutorial, we will visualize some bundles
# together with FA or T1. We will be able to change the slices using
# a ``LineSlider2D`` widget.
#
# First we need to fetch and load some datasets.

fetch_bundles_2_subjects()

###############################################################################
# The following function outputs a dictionary with the required bundles e.g.
# ``af left`` (left arcuate fasciculus) and maps, e.g. FA for a specific
# subject.

res = read_bundles_2_subjects(
    subj_id="subj_1", metrics=["t1", "fa"], bundles=["af.left", "cst.right", "cc_1"]
)

###############################################################################
# We will use 3 bundles, FA and the affine transformation that brings the voxel
# coordinates to world coordinates (RAS 1mm).

streamlines = Streamlines(res["af.left"])
streamlines.extend(res["cst.right"])
streamlines.extend(res["cc_1"])

data = res["fa"]
shape = data.shape
affine = res["affine"]

###############################################################################
# With our current design it is easy to decide in which space you want the
# streamlines and slices to appear. The default we have here is to appear in
# world coordinates (RAS 1mm).

world_coords = True

###############################################################################
# If we want to see the objects in native space we need to make sure that all
# objects which are currently in world coordinates are transformed back to
# native space using the inverse of the affine.

if not world_coords:
    from dipy.tracking.streamline import transform_streamlines

    streamlines = transform_streamlines(streamlines, np.linalg.inv(affine))

###############################################################################
# Now we create, a ``Scene`` object and add the streamlines using the ``line``
# function and an image plane using the ``slice`` function.

scene = window.Scene()
stream_actor = actor.streamlines(
    streamlines,
    colors=colormap.line_colors(streamlines),
    thickness=0.05,
    outline_thickness=0,
)

if not world_coords:
    image_actor = actor.volume_slicer(data, affine=np.eye(4))
else:
    image_actor = actor.volume_slicer(data, affine=affine)

###############################################################################
# We can also change also the opacity of the slicer.

slicer_opacity = 0.6
actor.set_group_opacity(image_actor, slicer_opacity)

###############################################################################
# The ``volume_slicer`` actor already holds the three orthogonal planes,
# centered in the volume. We keep track of their positions in world
# coordinates to move them later.

bounds = image_actor.get_bounding_box()
slice_positions = bounds.mean(axis=0)

###############################################################################
# Connect the actors with the Scene.

scene.add(stream_actor)
scene.add(image_actor)

###############################################################################
# Now we would like to change the position of each ``image_actor`` using a
# slider. The sliders are widgets which require access to different areas of
# the visualization pipeline and therefore we don't recommend using them with
# ``show``. The more appropriate way is to use them with the ``ShowManager``
# object which allows accessing the pipeline in different areas. Here is how:

show_m = window.ShowManager(title="bundles_and_3_slices", size=(1200, 900), scene=scene)

###############################################################################
# Now we can go ahead and create sliders to move the slices and change their
# opacity. The slice sliders work in world coordinates.

line_slider_z = ui.LineSlider2D(
    min_value=bounds[0, 2],
    max_value=bounds[1, 2],
    initial_value=slice_positions[2],
    text_template="{value:.0f}",
    length=140,
)

line_slider_x = ui.LineSlider2D(
    min_value=bounds[0, 0],
    max_value=bounds[1, 0],
    initial_value=slice_positions[0],
    text_template="{value:.0f}",
    length=140,
)

line_slider_y = ui.LineSlider2D(
    min_value=bounds[0, 1],
    max_value=bounds[1, 1],
    initial_value=slice_positions[1],
    text_template="{value:.0f}",
    length=140,
)

opacity_slider = ui.LineSlider2D(
    min_value=0.0, max_value=1.0, initial_value=slicer_opacity, length=140
)

###############################################################################
# Now we will write callbacks for the sliders and register them. A translucent
# slicer must not write depth, otherwise it hides the bundles behind it; a
# fully opaque one must write depth so that it does hide them.


def change_slice_z(slider):
    slice_positions[2] = slider.value
    actor.show_slices(image_actor, slice_positions)


def change_slice_x(slider):
    slice_positions[0] = slider.value
    actor.show_slices(image_actor, slice_positions)


def change_slice_y(slider):
    slice_positions[1] = slider.value
    actor.show_slices(image_actor, slice_positions)


def change_opacity(slider):
    actor.set_group_opacity(image_actor, slider.value)
    for plane in image_actor.children:
        plane.material.depth_write = bool(slider.value >= 1)


line_slider_z.on_change = change_slice_z
line_slider_x.on_change = change_slice_x
line_slider_y.on_change = change_slice_y
opacity_slider.on_change = change_opacity

###############################################################################
# We'll also create text labels to identify the sliders.


def build_label(text):
    label = ui.TextBlock2D(dynamic_bbox=True)
    label.message = text
    label.font_size = 18
    label.font_family = "Arial"
    label.justification = "left"
    label.bold = False
    label.italic = False
    label.shadow = False
    label.color = (1, 1, 1)

    return label


line_slider_label_z = build_label(text="Z Slice")
line_slider_label_x = build_label(text="X Slice")
line_slider_label_y = build_label(text="Y Slice")
opacity_slider_label = build_label(text="Opacity")

###############################################################################
# Now we will create a ``panel`` to contain the sliders and labels.

panel = ui.Panel2D(size=(300, 280), color=(1, 1, 1), opacity=0.1, align="right")

panel.add_element(line_slider_label_x, (0.1, 0.2))
panel.add_element(line_slider_x, (0.38, 0.2))
panel.add_element(line_slider_label_y, (0.1, 0.415))
panel.add_element(line_slider_y, (0.38, 0.415))
panel.add_element(line_slider_label_z, (0.1, 0.63))
panel.add_element(line_slider_z, (0.38, 0.63))
panel.add_element(opacity_slider_label, (0.1, 0.845))
panel.add_element(opacity_slider, (0.38, 0.845))

scene.add(panel)

###############################################################################
# The panel is placed in pixels from the top-left corner of the window. To keep
# it on the right side when the window is resized, we reposition it from the
# show manager's resize callback.


def place_panel(window_size):
    panel.set_position((window_size[0] - 320, 20))


place_panel(show_m.size)
show_m.resize_callback(place_panel)

###############################################################################
# Finally, start the interaction with the datasets in 3D. Use the sliders to
# move the slices and change their opacity.

show_m.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# A few bundles with interactive slicing.

###############################################################################
# References
# ----------
#
# .. footbibliography::
#
