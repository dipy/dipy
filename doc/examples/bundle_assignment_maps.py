"""
====================================
BUAN Bundle Assignment Maps Creation
====================================

This example explains how we can use BUAN [Chandio2020]_ to create assignment
maps on a bundle. Divide bundle into N smaller segments.


First import the necessary modules.
"""

from dipy.data import get_two_hcp842_bundles
from dipy.data import fetch_bundle_atlas_hcp842
import numpy as np
from fury import actor, window
from dipy.stats.analysis import assignment_map
from dipy.io.streamline import load_trk

"""
Download and read data for this tutorial
"""

atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()

"""
Read AF left and CST left bundles from already fetched atlas data to use them
as model bundles
"""

model_af_l_file, model_cst_l_file = get_two_hcp842_bundles()

sft_af_l = load_trk(model_af_l_file, "same", bbox_valid_check=False)
model_af_l = sft_af_l.streamlines

"""
let's visualize Arcuate Fasiculus Left (AF_L) bundle before assignment maps
"""

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(model_af_l))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -340.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='af_l_before_assignment_maps.png',
              size=(600, 600))
if interactive:
    window.show(scene)

"""
.. figure:: af_l_before_assignment_maps.png
   :align: center

   AF_L before assignment maps
"""


"""
Creating 100 bundle assignment maps on AF_L using BUAN [Chandio2020]_
"""

n = 100
indx = assignment_map(model_af_l, model_af_l, n)
indx = np.array(indx)

colors = [np.random.rand(3) for si in range(n)]

disks_color = []
for i in range(len(indx)):
    disks_color.append(tuple(colors[indx[i]]))

"""
let's visualize Arcuate Fasiculus Left (AF_L) bundle after assignment maps
"""

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(model_af_l, colors=disks_color))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -340.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='af_l_after_assignment_maps.png',
              size=(600, 600))
if interactive:
    window.show(scene)

"""
.. figure:: af_l_after_assignment_maps.png
   :align: center

   AF_L after assignment maps

"""

"""

References
----------

.. [Chandio2020] Chandio, B.Q., Risacher, S.L., Pestilli, F.,
        Bullock, D., Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and
        Garyfallidis, E. Bundle analytics, a computational framework for
        investigating the shapes and profiles of brain pathways across
        populations. Sci Rep 10, 17149 (2020)

"""
