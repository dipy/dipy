"""
====================================
BUAN Bundle Assignment Maps Creation
====================================

This example explains how we can use BUAN :footcite:p:`Chandio2020a` to create
assignment maps on a bundle. Divide bundle into N smaller segments.


First import the necessary modules.
"""

import numpy as np

from dipy.data import fetch_bundle_atlas_hcp842, get_two_hcp842_bundles
from dipy.io.streamline import load_tractogram
from dipy.stats.analysis import assignment_map
from dipy.viz import actor, colormap, window

###############################################################################
# Download and read data for this tutorial

atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()

###############################################################################
# Read AF left and CST left bundles from already fetched atlas data to use them
# as model bundles

model_af_l_file, model_cst_l_file = get_two_hcp842_bundles()

sft_af_l = load_tractogram(model_af_l_file, reference="same", bbox_valid_check=False)
model_af_l = sft_af_l.streamlines

###############################################################################
# let's visualize Arcuate Fasiculus Left (AF_L) bundle before assignment maps
# from the left side, with the camera looking along camera_view_dir.

camera_view_dir = (341.94, 10.90, 47.36)
camera_up = (-0.03, 0.028, 0.89)

scene = window.Scene(background=(1, 1, 1, 1))
scene.add(
    actor.streamlines(
        model_af_l,
        colors=colormap.line_colors(model_af_l),
        outline_thickness=0,
    )
)
show_manager = window.ShowManager(
    title="af_l_before_assignment_maps", size=(600, 600), scene=scene
)
screen = show_manager.screens[0]
screen.camera.show_object(
    screen.scene.main_scene, view_dir=camera_view_dir, up=camera_up
)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# AF_L before assignment maps
#
#
#
# Creating 100 bundle assignment maps on AF_L using BUAN
# :footcite:p:`Chandio2020a`

rng = np.random.default_rng()

n = 100
_, indx = assignment_map(model_af_l, model_af_l, n)
indx = np.array(indx)

colors = rng.random((n, 3))
disks_color = colors[indx]

###############################################################################
# let's visualize Arcuate Fasiculus Left (AF_L) bundle after assignment maps

scene = window.Scene(background=(1, 1, 1, 1))
scene.add(
    actor.streamlines(
        model_af_l,
        colors=disks_color,
        outline_thickness=0,
    )
)
show_manager = window.ShowManager(
    title="af_l_after_assignment_maps", size=(600, 600), scene=scene
)
screen = show_manager.screens[0]
screen.camera.show_object(
    screen.scene.main_scene, view_dir=camera_view_dir, up=camera_up
)
show_manager.start()

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# AF_L after assignment maps
#
#
# References
# ----------
#
# .. footbibliography::
#
