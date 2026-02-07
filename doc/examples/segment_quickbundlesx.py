"""
=================================================================
Streamline Clustering with QuickBundlesX
=================================================================

This example explains how we can use QuickBundlesX for hierarchical clustering
of streamlines. QuickBundlesX extends QuickBundles
:footcite:p:`Garyfallidis2012a` by building clustering hierarchies at multiple
threshold levels.

The `qbx_and_merge` function is the recommended way to use QuickBundlesX
in DIPY and is what the DIPY Horizon visualization tool uses internally.

First import the necessary modules.
"""

import numpy as np

from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import qbx_and_merge
from dipy.viz import actor, colormap, window

###############################################################################
# For educational purposes we will try to cluster a small streamline bundle
# known from neuroanatomy as the fornix.

fname = get_fnames(name="fornix")

###############################################################################
# Load fornix streamlines.

fornix = load_tractogram(fname, "same", bbox_valid_check=False)
streamlines = fornix.streamlines

###############################################################################
# Perform QuickBundlesX clustering using a sequence of distance thresholds.
# The thresholds are in mm and represent the maximum distance for a streamline
# to be considered part of a cluster. QuickBundlesX builds a hierarchy by
# clustering at each threshold level, from coarse (40mm) to fine (10mm).
# The function automatically resamples streamlines to have the same number
# of points. We set a fixed random seed for reproducibility.

rng = np.random.default_rng(42)
thresholds = [40, 30, 25, 20, 10]
clusters = qbx_and_merge(streamlines, thresholds, rng=rng)

###############################################################################
# `clusters` is a `ClusterMap` object which contains attributes that
# provide information about the clustering result.

print("Nb. clusters:", len(clusters))
cluster_sizes = [len(c) for c in clusters]
print("Cluster sizes:", cluster_sizes)
print("Large clusters:", [c for c, size in zip(clusters, cluster_sizes) if size >= 10])

###############################################################################
# Let's visualize the clusters by assigning a color to each cluster.

# Enables/disables interactive visualization
interactive = False

# Create a color for each cluster
cmap = colormap.create_colormap(np.arange(len(clusters)))

# Assign colors to streamlines based on their cluster
colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, cmap):
    colormap_full[cluster.indices] = color

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, colors=colormap_full))
window.record(scene=scene, out_path="fornix_qbx_clusters.png", size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Showing the different QuickBundlesX clusters with random colors.
#
#
# Threshold sequences can be customized for different clustering needs.
# Larger thresholds create coarser groupings while smaller thresholds create
# finer clusters. The last threshold is particularly important as it
# determines the final cluster granularity.

# For coarse segmentation
coarse_thresholds = [40, 30, 25]
coarse_clusters = qbx_and_merge(streamlines, coarse_thresholds, rng=rng)

# For fine-grained analysis
fine_thresholds = [40, 30, 25, 20, 15, 10, 5]
fine_clusters = qbx_and_merge(streamlines, fine_thresholds, rng=rng)

print("\nCoarse clustering:", len(coarse_clusters), "clusters")
print("Fine clustering:", len(fine_clusters), "clusters")

###############################################################################
# QuickBundlesX builds a hierarchy, which often produces more meaningful
# segmentations than the flat QuickBundles approach, especially for complex
# tractography data with branching structures.
#
#
# References
# ----------
#
# .. footbibliography::
