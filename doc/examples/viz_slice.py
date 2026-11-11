"""
=====================
Simple volume slicing
=====================

Here we present an example for visualizing slices from 3D images.

Let's start by importing the relevant modules.
"""

from functools import partial
from pathlib import Path

from dipy.data import fetch_bundles_2_subjects
from dipy.io.image import load_nifti, load_nifti_data
from dipy.viz import actor, ui, window

###############################################################################
# Let's download and load a T1.

fetch_bundles_2_subjects()

fname_t1 = (
    Path("~").expanduser()
    / ".dipy"
    / "exp_bundles_and_maps"
    / "bundles_2_subjects"
    / "subj_1"
    / "t1_warped.nii.gz"
)

data, affine = load_nifti(fname_t1)

###############################################################################
# Create a Scene object which holds all the actors which we want to visualize.

scene = window.Scene(background=(0.5, 0.5, 0.5))

###############################################################################
# Render slices from T1 with a specific value range
# =================================================
#
# The T1 has usually a higher range of values than what can be visualized in an
# image. We can set the range that we would like to see.

mean, std = data[data > 0].mean(), data[data > 0].std()
value_range = (mean - 0.5 * std, mean + 1.5 * std)

###############################################################################
# The ``volume_slicer`` function places the data in world coordinates using the
# affine transformation matrix. It holds three orthogonal planes (x, y, z)
# through the middle of the volume. We start by showing only the axial (z)
# plane. ``depth_write=True`` lets the opaque planes hide each other where they
# cross.

slice_actor = actor.volume_slicer(
    data,
    affine=affine,
    value_range=value_range,
    visibility=(False, False, True),
    depth_write=True,
)

scene.add(slice_actor)

###############################################################################
# The same actor can show any combination of its planes with
# ``set_group_visibility`` (and ``show_slices`` moves them to other world
# positions). Here the sagittal plane is shown too.

actor.set_group_visibility(slice_actor, (True, False, True))

###############################################################################
# The camera looks from the left, anterior and superior side so both planes are
# visible. ``start`` opens an interactive window (documentation builds save a
# screenshot instead).

show_manager = window.ShowManager(title="slices", size=(600, 600), scene=scene)
screen = show_manager.screens[0]
screen.camera.show_object(screen.scene.main_scene, view_dir=(1, -1, -1), up=(0, 0, 1))
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Simple slice viewer.
#
#
# Render slices from FA with your colormap
# ========================================
#
# It is also possible to set the colormap of your preference. Here we are
# loading an FA image and showing it in a non-standard way using an HSV
# colormap.

fname_fa = (
    Path("~").expanduser()
    / ".dipy"
    / "exp_bundles_and_maps"
    / "bundles_2_subjects"
    / "subj_1"
    / "fa_1x1x1.nii.gz"
)

fa = load_nifti_data(fname_fa)

###############################################################################
# ``volume_slicer`` shows scalar volumes in gray levels and RGB volumes as
# given. So the HSV colormap is built with a scalar bar, and its ``map_values``
# turns the FA volume into an RGB volume. FA above 80% of the maximum gets the
# last color of the colormap. ``value_range=(0, 1)`` keeps the RGB colors
# unscaled. Nearest interpolation draws each voxel as a flat square, so what
# you see is the voxel you click on next. The default camera looks from the
# superior side with anterior at the top.

fa_bar = actor.scalar_bar(
    clim=(fa.min(), fa.max() * 0.8),
    hsv_range=((0.4, 1.0), (1.0, 1.0), (0.0, 1.0)),
    orientation="vertical",
    title="FA",
    tick_format=".2f",
    position=(910, 300),
)
fa_rgb = fa_bar.map_values(fa)

fa_actor = actor.volume_slicer(
    fa_rgb,
    affine=affine,
    value_range=(0, 1),
    interpolation="nearest",
    visibility=(False, False, True),
)

fa_scene = window.Scene(background=(0.5, 0.5, 0.5))
fa_scene.add(fa_actor)
fa_scene.ui_scene.add(fa_bar)

show_manager = window.ShowManager(title="slices_lut", size=(1200, 900), scene=fa_scene)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Simple slice viewer with an HSV colormap.
#
#
#
# Now we would like to add the ability to click on a voxel and show its value
# on a panel in the window. The panel is a UI element positioned in pixels from
# the top-left corner of the window. Clicking a voxel of the slice fills it.

label_position = ui.TextBlock2D(text="Position:", dynamic_bbox=True)
label_value = ui.TextBlock2D(text="Value:", dynamic_bbox=True)

result_position = ui.TextBlock2D(text="", dynamic_bbox=True)
result_value = ui.TextBlock2D(text="", dynamic_bbox=True)

panel_picking = ui.Panel2D(
    size=(250, 125), position=(20, 20), color=(0, 0, 0), opacity=0.75, align="left"
)

panel_picking.add_element(label_position, (0.1, 0.25))
panel_picking.add_element(label_value, (0.1, 0.55))

panel_picking.add_element(result_position, (0.45, 0.25))
panel_picking.add_element(result_value, (0.45, 0.55))

fa_scene.add(panel_picking)

###############################################################################
# Add a ``pointer_down`` callback to the slicer. The picked voxel index is in
# ``event.pick_info["index"]``.


def show_voxel_value(event):
    """Show the index and FA value of the clicked voxel in the panel.

    Parameters
    ----------
    event : fury.lib.PointerEvent
        Click on the FA slicer; ``event.pick_info["index"]`` is the voxel.
    """
    i, j, k = event.pick_info["index"]
    result_position.message = f"({i}, {j}, {k})"
    result_value.message = f"{fa[i, j, k]:.8f}"


fa_actor.add_event_handler(show_voxel_value, "pointer_down")

show_manager = window.ShowManager(
    title="slices_picking", size=(1200, 900), scene=fa_scene
)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# FA slice with a panel showing the index and value of the clicked voxel.
#
#
# Create a mosaic
# ===============
#
# Each tile of the mosaic is a ``volume_slicer`` of a one-slice slab
# ``data[:, :, k : k + 1]``, so only that slice is uploaded. Tiles are built
# without affine, so they are laid out in voxel units and shifted with
# ``local.position``. We need a callback that reports the slice of the clicked
# tile.


def show_mosaic_voxel_value(event, *, k):
    """Show the index and T1 value of the voxel clicked in a mosaic tile.

    Parameters
    ----------
    event : fury.lib.PointerEvent
        Click on a tile; ``event.pick_info["index"]`` is the voxel of the
        one-slice tile.
    k : int
        Axial slice shown by the clicked tile.
    """
    i, j, _ = event.pick_info["index"]
    result_position.message = f"({i}, {j}, {k})"
    result_value.message = f"{data[i, j, k]:.8f}"


###############################################################################
# Now we create the tiles in a grid of 15 columns. The number of rows follows
# from the number of slices. The default camera looks straight at the flat
# mosaic, so no parallel projection is needed, and ``zoom`` enlarges it. In the
# window, right-drag pans, the wheel zooms and a left click shows the voxel
# value in the panel.

result_position.message = ""
result_value.message = ""

X, Y, Z = data.shape
cols = 15
border = 10

mosaic_scene = window.Scene(background=(0.5, 0.5, 0.5))
for k in range(Z):
    row, col = divmod(k, cols)
    tile = actor.volume_slicer(
        data[:, :, k : k + 1],
        value_range=value_range,
        interpolation="nearest",
        visibility=(False, False, True),
    )
    tile.local.position = (col * (X + border), -row * (Y + border), 0)
    tile.add_event_handler(partial(show_mosaic_voxel_value, k=k), "pointer_down")
    mosaic_scene.add(tile)
mosaic_scene.add(panel_picking)

show_manager = window.ShowManager(title="mosaic", size=(900, 600), scene=mosaic_scene)
show_manager.screens[0].camera.zoom = 1.6
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# A mosaic of all the slices in the T1 volume.
