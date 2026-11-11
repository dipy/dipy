"""
======================
Fast Streamline Search
======================

This example explains how Fast Streamline Search :footcite:p:`StOnge2022`
can be used to find all similar streamlines.

First import the necessary modules.
"""

import numpy as np

from dipy.data import (
    fetch_bundle_atlas_hcp842,
    fetch_target_tractogram_hcp,
    get_target_tractogram_hcp,
    get_two_hcp842_bundles,
)
from dipy.io.streamline import load_tractogram
from dipy.segment.fss import FastStreamlineSearch, nearest_from_matrix_row
from dipy.viz import actor, colormap, window

###############################################################################
# Download and read data for this tutorial

fetch_bundle_atlas_hcp842()
fetch_target_tractogram_hcp()

hcp_file = get_target_tractogram_hcp()
streamlines = load_tractogram(hcp_file, "same", bbox_valid_check=False).streamlines

###############################################################################
# Visualize the atlas (ref) bundle and full brain tractogram

line_actor = actor.line(
    streamlines, colors=colormap.line_colors(streamlines), material="thin"
)
line_actor.local.position = (0, 0, 0)

scene = window.Scene(background=(1, 1, 1, 1))
scene.add(line_actor)

show_manager = window.ShowManager(
    title="tractograms_initial", size=(600, 600), scene=scene
)
actor.gizmo(show_manager)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Atlas bundle and source streamlines before registration.
#
#
#
# Read Arcuate Fasciculus Left and Corticospinal Tract Left bundles from
# already fetched atlas data to use them as model bundle. Let's visualize the
# Arcuate Fasciculus Left model bundle.

model_af_l_file, model_cst_l_file = get_two_hcp842_bundles()
sft_af_l = load_tractogram(model_af_l_file, "same", bbox_valid_check=False)
model_af_l = sft_af_l.streamlines

camera_view_dir = (341.94, 10.90, 47.36)
camera_up = (-0.03, 0.028, 0.89)

line_actor = actor.streamlines(
    model_af_l, colors=(0, 1, 0), thickness=0.05, outline_thickness=0
)

scene = window.Scene(background=(1, 1, 1, 1))
scene.add(line_actor)

show_manager = window.ShowManager(
    title="AF_L_model_bundle", size=(600, 600), scene=scene
)
screen = show_manager.screens[0]
screen.camera.show_object(
    screen.scene.main_scene, view_dir=camera_view_dir, up=camera_up
)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Model Arcuate Fasciculus Left bundle
#
#
#
#
# Search for all similar streamlines :footcite:p:`StOnge2022`
#
# Fast Streamline Search can do a radius search to find all streamlines that
# are similar to from one tractogram to another. It returns the distance
# matrix mapping between both tractograms. The same list of streamlines can
# be given to recover the self distance matrix.
#
# here are the ``FastStreamlinesSearch`` class need the following
# initialization arguments:
#
# - ref_streamlines : reference streamlines, that will be searched in (tree)
# - max_radius : is the maximum distance that can be used with radius search
#
# Then, the ``radius_search()`` method needs the following arguments:
#
# - radius : for each streamline search find all similar ones in the
#   "ref_streamlines" that are within the given radius
#
# Be cautious, a large radius might result in a dense distance computation,
# requiring a large amount of time and memory. Recommended range of the
# radius is from 1 to 10 mm.

radius = 7.0
fs_tree_af = FastStreamlineSearch(ref_streamlines=model_af_l, max_radius=radius)
coo_mdist_mtx = fs_tree_af.radius_search(streamlines, radius=radius)

###############################################################################
# Extract indices of streamlines with an similar ones in the reference

ids_s = np.unique(coo_mdist_mtx.row)
ids_ref = np.unique(coo_mdist_mtx.col)

recognized_af_l = streamlines[ids_s].copy()

###############################################################################
# let's visualize streamlines similar to the Arcuate Fasciculus Left bundle

line_actor = actor.streamlines(
    recognized_af_l, colors=(0, 0, 1), thickness=0.05, outline_thickness=0
)

scene = window.Scene(background=(1, 1, 1, 1))
scene.add(line_actor)

show_manager = window.ShowManager(
    title="AF_L_recognized_bundle", size=(600, 600), scene=scene
)
screen = show_manager.screens[0]
screen.camera.show_object(
    screen.scene.main_scene, view_dir=camera_view_dir, up=camera_up
)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Recognized Arcuate Fasciculus Left bundle
#
#
#
#
# Color the resulting AF by the nearest streamlines distance

nn_s, nn_ref, nn_dist = nearest_from_matrix_row(coo_mdist_mtx)

dist_bar = actor.scalar_bar(
    clim=(nn_dist.min(), nn_dist.max()),
    hsv_range=((0.8, 0.0), (1.0, 1.0), (0.8, 0.8)),
    title="distance to atlas (mm)",
    tick_format=".1f",
    color=(0, 0, 0),
)
dist_colors = dist_bar.map_values(nn_dist)

line_actor = actor.streamlines(
    recognized_af_l, colors=dist_colors, thickness=0.05, outline_thickness=0
)

scene = window.Scene(background=(1, 1, 1, 1))
scene.add(line_actor)
scene.ui_scene.add(dist_bar)

show_manager = window.ShowManager(
    title="AF_L_recognized_bundle_dist", size=(600, 600), scene=scene
)
screen = show_manager.screens[0]
screen.camera.show_object(
    screen.scene.main_scene, view_dir=camera_view_dir, up=camera_up
)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Recognized Arcuate Fasciculus Left bundle colored by distance to ref
#
#
#
# Display the streamlines reference reached in green

# Default red color
ref_color = np.zeros((len(model_af_l), 3), dtype=float)
ref_color[:] = (1.0, 0.0, 0.0)

# Reached in green color
ref_color[ids_ref] = (0.0, 1.0, 0.0)

line_actor = actor.streamlines(
    model_af_l, colors=ref_color, thickness=0.05, outline_thickness=0
)

scene = window.Scene(background=(1, 1, 1, 1))
scene.add(line_actor)

show_manager = window.ShowManager(
    title="AF_L_model_bundle_reached", size=(600, 600), scene=scene
)
screen = show_manager.screens[0]
screen.camera.show_object(
    screen.scene.main_scene, view_dir=camera_view_dir, up=camera_up
)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Arcuate Fasciculus Left model reached (green) in radius
#
#
#
#
# References
# ----------
#
# .. footbibliography::
#
