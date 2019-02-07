"""
==================================================
Automatic Fiber Bundle Extraction with RecoBundles
==================================================

This example explains how we can use RecokBundles [Garyfallidis17]_ to
extract bundles from tractograms.

First import the necessary modules.
"""

from dipy.segment.bundles import RecoBundles
from dipy.align.streamlinear import whole_brain_slr
from dipy.viz import window, actor
from dipy.io.streamline import load_trk, save_trk


"""
Download and read data for this tutorial
"""

from dipy.data.fetcher import (fetch_target_tractogram_hcp,
                               fetch_bundle_atlas_hcp842,
                               get_bundle_atlas_hcp842,
                               get_target_tractogram_hcp)

target_file, target_folder = fetch_target_tractogram_hcp()
atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()

atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
target_file = get_target_tractogram_hcp()

atlas, atlas_header = load_trk(atlas_file)
target, target_header = load_trk(target_file)

"""
let's visualize atlas tractogram and target tractogram before registration
"""

interactive = False

ren = window.Renderer()
ren.SetBackground(1, 1, 1)
ren.add(actor.line(atlas, colors=(1,0,1)))
ren.add(actor.line(target, colors=(1,1,0)))
window.record(ren, out_path='tractograms_initial.png', size=(600, 600))
if interactive:
    window.show(ren)

"""
.. figure:: tractograms_initial.png
   :align: center

   Atlas and target before registration.

"""

"""
We will register target tractogram to model atlas' space using streamlinear
registeration (SLR) [Garyfallidis15]_
"""

moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
            atlas, target, x0='affine', verbose=True, progressive=True)

"""
let's visualize atlas tractogram and target tractogram after registration
"""

interactive = False

ren = window.Renderer()
ren.SetBackground(1, 1, 1)
ren.add(actor.line(atlas, colors=(1,0,1)))
ren.add(actor.line(moved, colors=(1,1,0)))
window.record(ren, out_path='tractograms_after_registration.png',
              size=(600, 600))
if interactive:
    window.show(ren)

"""
.. figure:: tractograms_after_registration.png
   :align: center

   Atlas and target after registration.

"""

"""
Read AF left and CST left bundles from already fetched atlas data to use them
as model bundles
"""

from dipy.data.fetcher import get_two_hcp842_bundles
bundle1, bundle2 = get_two_hcp842_bundles()

"""
Extracting bundles using recobundles [Garyfallidis17]_
"""

model_bundle, _ = load_trk(bundle1)

rb = RecoBundles(moved, verbose=True)

recognized_bundle, rec_labels = rb.recognize(model_bundle=model_bundle,
                                             model_clust_thr=5.,
                                             reduction_thr=10,
                                             reduction_distance='mam',
                                             slr=True,
                                             slr_metric='asymmetric',
                                             pruning_distance='mam')

"""
let's visualize extracted Arcuate Fasciculus Left bundle and model bundle
together
"""

interactive = False

ren = window.Renderer()
ren.SetBackground(1, 1, 1)
ren.add(actor.line(model_bundle, colors=(.1,.7,.26)))
ren.add(actor.line(recognized_bundle, colors=(.1,.1,6)))
ren.set_camera(focal_point=(320.21296692, 21.28884506,  17.2174015),
               position=(2.11, 200.46, 250.44) , view_up=(0.1, -1.028, 0.18))
window.record(ren, out_path='AF_L_recognized_bundle.png',
              size=(600, 600))
if interactive:
    window.show(ren)

"""
.. figure:: AF_L_recognized_bundle.png
   :align: center

   Extracted Arcuate Fasciculus Left bundle and model bundle

"""

model_bundle, _ = load_trk(bundle2)

recognized_bundle, rec_labels = rb.recognize(model_bundle=model_bundle,
                                             model_clust_thr=5.,
                                             reduction_thr=10,
                                             reduction_distance='mam',
                                             slr=True,
                                             slr_metric='asymmetric',
                                             pruning_distance='mam')

"""
let's visualize extracted Corticospinal Tract (CST) Left bundle and model
bundle together
"""

interactive = False

ren = window.Renderer()
ren.SetBackground(1, 1, 1)
ren.add(actor.line(model_bundle, colors=(.1,.7,.26)))
ren.add(actor.line(recognized_bundle, colors=(.1,.1,6)))
ren.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
               position=(-360.11, -340.46, -40.44),
               view_up=(-0.03, 0.028, 0.89))
window.record(ren, out_path='CST_L_recognized_bundle.png',
              size=(600, 600))
if interactive:
    window.show(ren)


"""
.. figure:: CST_L_recognized_bundle.png
   :align: center

   Extracted Corticospinal Tract (CST) Left bundle and model bundle

"""
