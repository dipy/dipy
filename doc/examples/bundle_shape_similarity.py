"""
==================================
BUAN Bundle Shape Similarity Score
==================================

This example explains how we can use BUAN [Chandio2020]_ to calculate shape
similarity between two given bundles. Where, shape similarity score of 1 means
two bundles are extremely close in shape and 0 implies no shape similarity
whatsoever.

Shape similarity score can be used to compare populations or individuals.
It can also serve as a quality assurance metric, to validate streamline
registration quality, bundle extraction quality by calculating output with a
reference bundle or other issues with pre-processing by calculating shape
dissimilarity with a reference bundle.

First import the necessary modules.
"""

import numpy as np
from dipy.viz import window, actor
from dipy.segment.bundles import bundle_shape_similarity
from dipy.segment.bundles import select_random_set_of_streamlines
from dipy.data import two_cingulum_bundles

###############################################################################
# To show the concept we will use two pre-saved cingulum bundle.
# Let's start by fetching the data.

cb_subj1, _ = two_cingulum_bundles()

###############################################################################
# Let's create two streamline sets (bundles) from same bundle cb_subj1 by
# randomly selecting 60 streamlines two times.

rng = np.random.default_rng()
bundle1 = select_random_set_of_streamlines(cb_subj1, 60, rng=None)
bundle2 = select_random_set_of_streamlines(cb_subj1, 60, rng=None)

###############################################################################
# Now, let's visualize two bundles.


def show_both_bundles(bundles, colors=None, show=True, fname=None):

    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    for (i, bundle) in enumerate(bundles):
        color = colors[i]
        streamtube_actor = actor.streamtube(bundle, color, linewidth=0.3)
        streamtube_actor.RotateX(-90)
        streamtube_actor.RotateZ(90)
        scene.add(streamtube_actor)
    if show:
        window.show(scene)
    if fname is not None:
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))


show_both_bundles([bundle1, bundle2], colors=[(1, 0, 0), (0, 1, 0)],
                  show=False, fname="two_bundles.png")

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Two Cingulum Bundles.
#
#
#
# Calculate shape similarity score between two bundles.
# 0 cluster_thr because we want to use all streamlines and not the centroids of
# clusters.

clust_thr = [0]

###############################################################################
# Threshold indicates how strictly we want two bundles to be similar in shape.

threshold = 5

ba_score = bundle_shape_similarity(bundle1, bundle2, rng, clust_thr, threshold)
print("Shape similarity score = ", ba_score)

###############################################################################
# Let's change the value of threshold to 10.

threshold = 10

ba_score = bundle_shape_similarity(bundle1, bundle2, rng, clust_thr, threshold)
print("Shape similarity score = ", ba_score)

###############################################################################
# Higher value of threshold gives us higher shape similarity score as it is
# more lenient.
#
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
