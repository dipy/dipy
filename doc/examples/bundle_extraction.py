"""
==================================================
Automatic Fiber Bundle Extraction with RecoBundles
==================================================

This example explains how we can use RecoBundles [Garyfallidis17]_ to extract
bundles from tractograms.

First import the necessary modules.
"""

import numpy as np

from dipy.align.streamlinear import whole_brain_slr
from dipy.data import get_two_hcp842_bundles
from dipy.data import (fetch_target_tractogram_hcp,
                       fetch_bundle_atlas_hcp842,
                       get_bundle_atlas_hcp842,
                       get_target_tractogram_hcp)
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_trk, save_trk
from dipy.io.utils import create_tractogram_header
from dipy.segment.bundles import RecoBundles
from dipy.viz import actor, window

###############################################################################
# Download and read data for this tutorial

target_file, target_folder = fetch_target_tractogram_hcp()
atlas_file, atlas_folder = fetch_bundle_atlas_hcp842()

atlas_file, all_bundles_files = get_bundle_atlas_hcp842()
target_file = get_target_tractogram_hcp()

sft_atlas = load_trk(atlas_file, "same", bbox_valid_check=False)
atlas = sft_atlas.streamlines
atlas_header = create_tractogram_header(atlas_file,
                                        *sft_atlas.space_attributes)

sft_target = load_trk(target_file, "same", bbox_valid_check=False)
target = sft_target.streamlines
target_header = create_tractogram_header(target_file,
                                         *sft_target.space_attributes)

###############################################################################
# let's visualize atlas tractogram and target tractogram before registration

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(atlas, colors=(1, 0, 1)))
scene.add(actor.line(target, colors=(1, 1, 0)))
window.record(scene, out_path='tractograms_initial.png', size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Atlas and target before registration.
#
#
# We will register target tractogram to model atlas' space using streamlinear
# registration (SLR) [Garyfallidis15]_

moved, transform, qb_centroids1, qb_centroids2 = whole_brain_slr(
    atlas, target, x0='affine', verbose=True, progressive=True,
    rng=np.random.RandomState(1984))


###############################################################################
# We save the transform generated in this registration, so that we can use
# it in the bundle profiles example

np.save("slr_transform.npy", transform)

###############################################################################
# let's visualize atlas tractogram and target tractogram after registration

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(atlas, colors=(1, 0, 1)))
scene.add(actor.line(moved, colors=(1, 1, 0)))
window.record(scene, out_path='tractograms_after_registration.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Atlas and target after registration.
#
#
#
# Extracting bundles using RecoBundles [Garyfallidis17]_
#
# RecoBundles requires a model (reference) bundle and tries to extract similar
# looking bundle from the input tractogram. There are some key parameters that
# users can set as per requirements. Here are the key threshold parameters
# measured in millimeters and their function in Recobundles:
#
# - model_clust_thr : It will use QuickBundles to get the centroids of the
#   model bundle and work with centroids instead of all streamlines. This
#   helps to make RecoBundles faster. The larger the value of the threshold,
#   the fewer centroids will be, the and smaller the threshold value, the
#   more centroids will be. If you prefer to use all the streamlines of the
#   model bundle, you can set this threshold to 0.01 mm.
#   Recommended range of the model_clust_thr is 0.01 - 3.0 mm.
#
# - reduction_thr : This threshold will be used to reduce the search space
#   for finding the streamlines that match model bundle streamlines in shape.
#   Instead of looking at the entire tractogram, now we will be looking at
#   neighboring region of a model bundle in the tractogram. Increase the
#   threshold to increase the search space.
#   Recommended range of the reduction_thr is 15 - 30 mm.
#
# - pruning_thr : This threshold will filter the streamlines for which the
#   distance to the model bundle is greater than the pruning_thr.
#   This serves to filter the neighborhood area (search space) to get
#   streamlines that are like the model bundle.
#   Recommended range of the pruning_thr is 8 - 12 mm.
#
# - reduction_distance and pruning_distance : Distance method used
#   internally. Minimum Diferect Flip distance (mdf) or Mean Average Minimum
#   (mam). Default is set to mdf.
#
# - slr : If slr flag is set to True, local registration of model bundle
#   with neighbouring area will be performed. Default and recommended is True.
#
#
#
# Read Arcuate Fasciculus Left and Corticospinal Tract Left bundles from
# already fetched atlas data to use them as model bundle. Let's visualize the
# Arcuate Fasciculus Left model bundle.

model_af_l_file, model_cst_l_file = get_two_hcp842_bundles()
sft_af_l = load_trk(model_af_l_file, "same", bbox_valid_check=False)
model_af_l = sft_af_l.streamlines

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_af_l))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='AF_L_model_bundle.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Model Arcuate Fasciculus Left bundle

rb = RecoBundles(moved, verbose=True, rng=np.random.RandomState(2001))

recognized_af_l, af_l_labels = rb.recognize(model_bundle=model_af_l,
                                            model_clust_thr=0.1,
                                            reduction_thr=15,
                                            pruning_thr=7,
                                            reduction_distance='mdf',
                                            pruning_distance='mdf',
                                            slr=True)

###############################################################################
# let's visualize extracted Arcuate Fasciculus Left bundle

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(recognized_af_l.copy()))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='AF_L_recognized_bundle.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Extracted Arcuate Fasciculus Left bundle
#
#
#
# Save the bundle as a trk file. Let's save the recognized bundle in the
# common space (atlas space), in this case, MNI space.

reco_af_l = StatefulTractogram(recognized_af_l, atlas_header,
                               Space.RASMM)
save_trk(reco_af_l, "AF_L_rec_1.trk", bbox_valid_check=False)

###############################################################################
# Let's save the recognized bundle in the original space of the subject
# anatomy.

reco_af_l = StatefulTractogram(target[af_l_labels], target_header,
                               Space.RASMM)
save_trk(reco_af_l, "AF_L_org_1.trk", bbox_valid_check=False)

###############################################################################
# Now, let's increase the reduction_thr and pruning_thr values.

recognized_af_l, af_l_labels = rb.recognize(model_bundle=model_af_l,
                                            model_clust_thr=0.1,
                                            reduction_thr=20,
                                            pruning_thr=10,
                                            reduction_distance='mdf',
                                            pruning_distance='mdf',
                                            slr=True)

###############################################################################
# let's visualize extracted Arcuate Fasciculus Left bundle.

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(recognized_af_l.copy()))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='AF_L_recognized_bundle2.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Extracted Arcuate Fasciculus Left bundle
#
#
# Save the bundle as a trk file. Let's save the recognized bundle in the
# common space (atlas space), in this case, MNI space.

reco_af_l = StatefulTractogram(recognized_af_l, atlas_header,
                               Space.RASMM)
save_trk(reco_af_l, "AF_L_rec_2.trk", bbox_valid_check=False)

###############################################################################
# Let's save the recognized bundle in the original space of the subject
# anatomy.

reco_af_l = StatefulTractogram(target[af_l_labels], target_header,
                               Space.RASMM)
save_trk(reco_af_l, "AF_L_org_2.trk", bbox_valid_check=False)

###############################################################################
# Now, let's increase the reduction_thr and pruning_thr values further.

recognized_af_l, af_l_labels = rb.recognize(model_bundle=model_af_l,
                                            model_clust_thr=0.1,
                                            reduction_thr=25,
                                            pruning_thr=12,
                                            reduction_distance='mdf',
                                            pruning_distance='mdf',
                                            slr=True)

###############################################################################
# let's visualize extracted Arcuate Fasciculus Left bundle.

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(recognized_af_l.copy()))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='AF_L_recognized_bundle3.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Extracted Arcuate Fasciculus Left bundle
#
#
# Save the bundle as a trk file. Let's save the recognized bundle in the
# common space (atlas space), in this case, MNI space.

reco_af_l = StatefulTractogram(recognized_af_l, atlas_header,
                               Space.RASMM)
save_trk(reco_af_l, "AF_L_rec_3.trk", bbox_valid_check=False)

###############################################################################
# Let's save the recognized bundle in the original space of the subject
# anatomy.

reco_af_l = StatefulTractogram(target[af_l_labels], target_header,
                               Space.RASMM)
save_trk(reco_af_l, "AF_L_org_3.trk", bbox_valid_check=False)


###############################################################################
# Let's apply auto-calibrated RecoBundles on the output of standard
# RecoBundles. This step will filter out the outlier streamlines. This time,
# the RecoBundles' extracted bundle will serve as a model bundle. As a rule of
# thumb, provide larger threshold values in standard RecoBundles function and
# smaller values in the auto-calibrated RecoBundles (refinement) step.

r_recognized_af_l, r_af_l_labels = rb.refine(
    model_bundle=model_af_l,
    pruned_streamlines=recognized_af_l,
    model_clust_thr=0.1,
    reduction_thr=15,
    pruning_thr=6,
    reduction_distance='mdf',
    pruning_distance='mdf',
    slr=True)

###############################################################################
# let's visualize extracted refined Arcuate Fasciculus Left bundle.

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(r_recognized_af_l.copy()))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='AF_L_refine_recognized_bundle.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Extracted Arcuate Fasciculus Left bundle
#
#
#
# Save the bundle as a trk file. Let's save the recognized bundle in the
# common space (atlas space), in this case, MNI space.

reco_af_l = StatefulTractogram(r_recognized_af_l, atlas_header,
                               Space.RASMM)
save_trk(reco_af_l, "AF_L_rec_refine.trk", bbox_valid_check=False)

###############################################################################
# Let's save the recognized bundle in the original space of the subject
# anatomy.

reco_af_l = StatefulTractogram(target[r_af_l_labels], target_header,
                               Space.RASMM)
save_trk(reco_af_l, "AF_L_org_refine.trk", bbox_valid_check=False)

###############################################################################
# Let's load Corticospinal Tract Left model bundle and visualize it.

sft_cst_l = load_trk(model_cst_l_file, "same", bbox_valid_check=False)
model_cst_l = sft_cst_l.streamlines

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(model_cst_l))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='CST_L_model_bundle.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# The Corticospinal tract model bundle

recognized_cst_l, cst_l_labels = rb.recognize(model_bundle=model_cst_l,
                                              model_clust_thr=0.1,
                                              reduction_thr=15,
                                              pruning_thr=7,
                                              reduction_distance='mdf',
                                              pruning_distance='mdf',
                                              slr=True)

###############################################################################
# let's visualize extracted Corticospinal tract Left bundle

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(recognized_cst_l.copy()))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='CST_L_recognized_bundle.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Extracted Corticospinal tract Left bundle
#
#
#
# Save the bundle as a trk file. Let's save the recognized bundle in the
# common space (atlas space), in this case, MNI space.

reco_cst_l = StatefulTractogram(recognized_cst_l, atlas_header,
                                Space.RASMM)
save_trk(reco_cst_l, "CST_L_rec_1.trk", bbox_valid_check=False)

###############################################################################
# Let's save the recognized bundle in the original space of the subject
# anatomy.

reco_cst_l = StatefulTractogram(target[cst_l_labels], target_header,
                                Space.RASMM)
save_trk(reco_cst_l, "CST_L_org_1.trk", bbox_valid_check=False)

###############################################################################
# Now, let's increase the reduction_thr and pruning_thr values.

recognized_cst_l, cst_l_labels = rb.recognize(model_bundle=model_cst_l,
                                              model_clust_thr=0.1,
                                              reduction_thr=20,
                                              pruning_thr=10,
                                              reduction_distance='mdf',
                                              pruning_distance='mdf',
                                              slr=True)

###############################################################################
# let's visualize extracted Corticospinal tract Left bundle.

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(recognized_cst_l.copy()))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='CST_L_recognized_bundle2.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Extracted Corticospinal tract Left bundle
#
#
# Save the bundle as a trk file. Let's save the recognized bundle in the
# common space (atlas space), in this case, MNI space.

reco_cst_l = StatefulTractogram(recognized_cst_l, atlas_header,
                                Space.RASMM)
save_trk(reco_cst_l, "CST_L_rec_2.trk", bbox_valid_check=False)

###############################################################################
# Let's save the recognized bundle in the original space of the subject
# anatomy.

reco_cst_l = StatefulTractogram(target[cst_l_labels], target_header,
                                Space.RASMM)
save_trk(reco_cst_l, "CST_L_org_2.trk", bbox_valid_check=False)

###############################################################################
# Now, let's increase the reduction_thr and pruning_thr values further.

recognized_cst_l, cst_l_labels = rb.recognize(model_bundle=model_cst_l,
                                              model_clust_thr=0.1,
                                              reduction_thr=25,
                                              pruning_thr=12,
                                              reduction_distance='mdf',
                                              pruning_distance='mdf',
                                              slr=True)

###############################################################################
# let's visualize extracted Corticospinal tract Left bundle.

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(recognized_cst_l.copy()))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='CST_L_recognized_bundle3.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Extracted Corticospinal tract Left bundle
#
#
#
#
# Save the bundle as a trk file. Let's save the recognized bundle in the
# common space (atlas space), in this case, MNI space.

reco_cst_l = StatefulTractogram(recognized_cst_l, atlas_header,
                                Space.RASMM)
save_trk(reco_cst_l, "CST_L_rec_3.trk", bbox_valid_check=False)

###############################################################################
# Let's save the recognized bundle in the original space of the subject
# anatomy.

reco_cst_l = StatefulTractogram(target[cst_l_labels], target_header,
                                Space.RASMM)
save_trk(reco_cst_l, "CST_L_org_3.trk", bbox_valid_check=False)

###############################################################################
# Let's apply auto-calibrated RecoBundles on the output of standard
# RecoBundles. This step will filter out the outlier streamlines. This time,
# the RecoBundles' extracted bundle will serve as a model bundle. As a rule of
# thumb, provide larger threshold values in standard RecoBundles function and
# smaller values in the auto-calibrated RecoBundles (refinement) step.

r_recognized_cst_l, r_cst_l_labels = rb.refine(
                                     model_bundle=model_cst_l,
                                     pruned_streamlines=recognized_cst_l,
                                     model_clust_thr=0.1,
                                     reduction_thr=15,
                                     pruning_thr=6,
                                     reduction_distance='mdf',
                                     pruning_distance='mdf',
                                     slr=True)

###############################################################################
# let's visualize extracted refined Corticospinal tract Left bundle.

interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.line(r_recognized_cst_l.copy()))
scene.set_camera(focal_point=(-18.17281532, -19.55606842, 6.92485857),
                 position=(-360.11, -30.46, -40.44),
                 view_up=(-0.03, 0.028, 0.89))
window.record(scene, out_path='CST_L_refine_recognized_bundle.png',
              size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Extracted refined Corticospinal tract Left bundle
#
#
# Save the bundle as a trk file. Let's save the recognized bundle in the
# common space (atlas space), in this case, MNI space.

reco_cst_l = StatefulTractogram(r_recognized_cst_l, atlas_header,
                                Space.RASMM)
save_trk(reco_cst_l, "CST_L_rec_refine.trk", bbox_valid_check=False)

###############################################################################
# Let's save the recognized bundle in the original space of the subject
# anatomy.

reco_cst_l = StatefulTractogram(target[r_cst_l_labels], target_header,
                                Space.RASMM)
save_trk(reco_cst_l, "CST_L_org_refine.trk", bbox_valid_check=False)

###############################################################################
# This example shows how changing different threshold parameters can change the
# output. It is up to the user to decide what is the desired output and select
# parameters accordingly.
#
# References
# ----------
#
# .. [Garyfallidis17] Garyfallidis et al. Recognition of white matter
#         bundles using local and global streamline-based registration
#         and clustering, Neuroimage, 2017.
#
# .. [Chandio2020] Chandio, B.Q., Risacher, S.L., Pestilli, F.,
#         Bullock, D., Yeh, FC., Koudoro, S., Rokem, A., Harezlak, J., and
#         Garyfallidis, E. Bundle analytics, a computational framework for
#         investigating the shapes and profiles of brain pathways across
#         populations. Sci Rep 10, 17149 (2020)
