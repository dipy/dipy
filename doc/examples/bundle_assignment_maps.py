"""
====================================
BUAN Bundle Assignment Maps Creation
====================================

This example explains how we can use BUAN [Chandio2020]_ to create assignment
maps on a bundle. Divide bundle into N smaller segments.


First import the necessary modules.
"""

import numpy as np

from dipy.data import get_two_hcp842_bundles
from dipy.data import fetch_bundle_atlas_hcp842
from dipy.io.streamline import load_trk
from dipy.stats.analysis import assignment_map
from dipy.viz import actor, window

###############################################################################
# Download and read data for this tutorial

atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()

###############################################################################
# Read AF left and CST left bundles from already fetched atlas data to use them
# as model bundles

model_af_l_file, model_cst_l_file = get_two_hcp842_bundles()

sft_af_l = load_trk(model_af_l_file, "same", bbox_valid_check=False)
model_af_l = sft_af_l.streamlines

###############################################################################
# let's visualize Arcuate Fasiculus Left (AF_L) bundle before assignment maps

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_af_l, fake_tube=True, linewidth=6))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='af_l_before_assignment_maps.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# AF_L before assignment maps
#
#
#
# Creating 100 bundle assignment maps on AF_L using BUAN [Chandio2020]_

rng = np.random.default_rng()

n = 100
indx = assignment_map(model_af_l, model_af_l, n)
indx = np.array(indx)

colors = [rng.random(3) for si in range(n)]

disks_color = []
for i in range(len(indx)):
    disks_color.append(tuple(colors[indx[i]]))

###############################################################################
# let's visualize Arcuate Fasiculus Left (AF_L) bundle after assignment maps

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_af_l, fake_tube=True, colors=disks_color,
                     linewidth=6))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='af_l_after_assignment_maps.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# AF_L after assignment maps
#
#
# References
# ----------
#
# .. [Chandio2020] Chandio, B.Q., Risacher, S.L., Pestilli, F.,
#         Bullock, D., Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and
#         Garyfallidis, E. Bundle analytics, a computational framework for
#         investigating the shapes and profiles of brain pathways across
#         populations. Sci Rep 10, 17149 (2020)
