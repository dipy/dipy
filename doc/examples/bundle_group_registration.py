"""
=============================
Groupwise Bundle Registration
=============================

This example explains how to coregister a set of bundles to a common space that
is not biased by any specific bundle. This is useful when we want to align all
the bundles but do not have a target reference space defined by an atlas.

How it works
============

The bundle groupwise registration framework in DIPY relies on streamline linear
registration (SLR) [Garyfallidis15]_ and an iterative process.

In each iteration, bundles are shuffled and matched in pairs. Then, each pair
of bundles are simultaneously moved to a common space in between both.

After all pairs have been aligned, a group distance metric is computed as the
mean pairwise distance between all bundle pairs. With each iteration, bundles
get closer to each other and the group distance decreases.

When the reduction in the group distance reaches a tolerance level the process
ends.

To reduce computational time, by default both registration and distance
computation are performed on streamline centroids (obtained with Quickbundles)
[Garyfallidis12]_.

Example
=======

We start by importing and creating the necessary functions:
"""

from dipy.align.streamlinear import groupwise_slr
from dipy.data import read_five_af_bundles
from dipy.viz.streamline import show_bundles

import logging
logging.basicConfig(level=logging.INFO)

###############################################################################
# To run groupwise registration we need to have our input bundles stored in a
# list.
#
# Here we load 5 left arcuate fasciculi and store them in a list.

bundles = read_five_af_bundles()

###############################################################################
# Let's now visualize our input bundles:

colors = [[0.91, 0.26, 0.35], [0.99, 0.50, 0.38], [0.99, 0.88, 0.57],
          [0.69, 0.85, 0.64], [0.51, 0.51, 0.63]]

show_bundles(bundles, interactive=False, colors=colors,
             save_as='before_group_registration.png')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Bundles before registration.
#
#
# They are in native space and, therefore, not aligned.
#
# Now running groupwise registration is as simple as:

bundles_reg, aff, d = groupwise_slr(bundles, verbose=True)

###############################################################################
# Finally, we visualize the registered bundles to confirm that they are now in
# a common space:

show_bundles(bundles_reg, interactive=False, colors=colors,
             save_as='after_group_registration.png')

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Bundles after registration.
#
#
#
# Extended capabilities
# =====================
#
# In addition to the registered bundles, `groupwise_slr` also returns a list
# with the individual transformation matrices as well as the pairwise distances
# computed in each iteration.
#
# By changing the input arguments the user can modify the transformation (up to
# affine), the number of maximum number of streamlines per bundle, the level of
# clustering, or the tolerance of the method.
#
# References
# ----------
#
# .. [Garyfallidis15] Garyfallidis et al., "Robust and efficient linear
#                     registration of white-matter fascicles in the space of
#                     streamlines", Neuroimage, 117:124-140, 2015.
# .. [Garyfallidis12] Garyfallidis E. et al., "QuickBundles a method for
#                     tractography simplification", Frontiers in Neuroscience,
#                     vol 6, no 175, 2012.
# .. [RomeroBascones22] Romero-Bascones D. et al., "BundleAtlasing: unbiased
#                     population-specific atlasing of bundles in streamline
#                     space", ISMRM, 2022.
