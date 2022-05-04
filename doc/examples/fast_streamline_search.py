"""
=======================
Fast Streamlines Search
=======================

This example explains how Fast Streamlines Search [StOnge2021]_
can be used to find similar streamlines.

First import the necessary modules.
"""

import numpy as np

from dipy.data import (fetch_target_tractogram_hcp,
                       get_target_tractogram_hcp,
                       get_two_hcp842_bundles)
from dipy.io.streamline import load_trk
from dipy.io.utils import create_tractogram_header
from dipy.segment.search import FastStreamlineSearch, nearest_from_matrix_row
from dipy.viz import actor, window

"""
Download and read data for this tutorial
"""


target_file, target_folder = fetch_target_tractogram_hcp()
target_file = get_target_tractogram_hcp()

sft_target = load_trk(target_file, "same", bbox_valid_check=False)
target = sft_target.streamlines
target_header = create_tractogram_header(target_file,
                                         *sft_target.space_attributes)


"""
Visualize atlas tractogram and target tractogram
"""

interactive = True

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(target))
if interactive:
    window.show(scene)
else:
    window.record(scene, out_path='tractograms_initial.png', size=(600, 600))

"""
.. figure:: tractograms_initial.png
   :align: center

   Atlas and target before registration.

"""

"""
Read Arcuate Fasciculus Left and Corticospinal Tract Left bundles from already
fetched atlas data to use them as model bundle. Let's visualize the
Arcuate Fasciculus Left model bundle.
"""

model_af_l_file, model_cst_l_file = get_two_hcp842_bundles()
sft_af_l = load_trk(model_af_l_file, "same", bbox_valid_check=False)
model_af_l = sft_af_l.streamlines


scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_af_l, colors=(0, 1, 0)))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
if interactive:
    window.show(scene)
else:
    window.record(scene, out_path='AF_L_model_bundle.png',
                  size=(600, 600))

"""
.. figure:: AF_L_model_bundle.png
   :align: center

   Model Arcuate Fasciculus Left bundle

"""


"""
Search for all similar streamlines  [StOnge2021]_

Fast Streamlines Search, apply a radius search from one to another set of streamlines.
It return the distance matrix mapping both tractograms.
The same list of streamlines can be given to recover the self distance matrix.

    - radius : is the maximum distance between streamlines returned. 
    Be cautious, a large radius might result in a dense distance computation,
    requiring a large amount of time and memory.
    Recommended range of the radius is 1 - 10 mm.
"""

"""
Compute the sparse distance matrix using Fast Streamlines Search [StOnge2021]_
"""

l21_radius = 7.0
fss = FastStreamlineSearch(model_af_l, l21_radius, resampling=32)
coo_mdist_mtx = fss.radius_search(target, l21_radius)

"""
Extract streamlines with an similar ones in the reference
"""
ids_target = np.unique(coo_mdist_mtx.row)
ids_ref = np.unique(coo_mdist_mtx.col)

recognized_af_l = target[ids_target]
"""
let's visualize extracted Arcuate Fasciculus Left bundle
"""

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_af_l, colors=(0, 1, 0)))
scene.add(actor.line(recognized_af_l, colors=(0, 0, 1)))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
if interactive:
    window.show(scene)
else:
    window.record(scene, out_path='AF_L_recognized_bundle.png',
                  size=(600, 600))

"""
.. figure:: AF_L_recognized_bundle.png
   :align: center

   Recognized Arcuate Fasciculus Left bundle

"""

"""
Color the resulting AF by the nearest streamlines distance
"""
nn_target, nn_ref, nn_dist = nearest_from_matrix_row(coo_mdist_mtx)

scene = window.Scene()
scene.SetBackground(1, 1, 1)
cmap = actor.colormap_lookup_table(scale_range=(nn_dist.min(), nn_dist.max()))
scene.add(actor.line(recognized_af_l, colors=nn_dist, lookup_colormap=cmap))
scene.add(actor.scalar_bar(cmap, title="distance to ref (mm)"))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
if interactive:
    window.show(scene)
else:
    window.record(scene, out_path='AF_L_recognized_bundle_dist.png',
                  size=(600, 600))

"""
.. figure:: AF_L_recognized_bundle_dist.png
   :align: center

   Recognized Arcuate Fasciculus Left bundle colored by distance to ref

"""

"""
Display the streamlines reference reached
"""

ref_in_reach = np.zeros([len(model_af_l)], dtype=bool)
ref_in_reach[ids_ref] = True

scene = window.Scene()
scene.SetBackground(1, 1, 1)
if np.any(ref_in_reach):
    scene.add(actor.line(model_af_l[ref_in_reach], colors=(0, 1, 0)))
if np.any(~ref_in_reach):
    scene.add(actor.line(model_af_l[~ref_in_reach], colors=(1, 0, 0)))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))

if interactive:
    window.show(scene)
else:
    window.record(scene, out_path='AF_L_model_bundle_reached.png',
                  size=(600, 600))

"""
.. figure:: AF_L_model_bundle_reached.png
   :align: center

   Arcuate Fasciculus Left model reached (green) in radius

"""

"""
References
----------

.. [StOnge2021] St-Onge E. et al., Fast Tractography Streamline Search,
        International Workshop on Computational Diffusion MRI,
        pp. 82-95. Springer, Cham, 2021.
"""
