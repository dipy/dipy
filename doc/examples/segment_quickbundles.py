"""
=========================================
Tractography Clustering with QuickBundles
=========================================

This example explains how we can use QuickBundles
:footcite:p:`Garyfallidis2012a` to simplify/cluster streamlines.

First import the necessary modules.
"""

import numpy as np

from dipy.data import get_fnames
from dipy.io.pickles import save_pickle
from dipy.io.streamline import load_tractogram
from dipy.segment.clustering import QuickBundles, qbx_and_merge
from dipy.viz import actor, colormap as cmap, window

###############################################################################
# For educational purposes we will try to cluster a small streamline bundle
# known from neuroanatomy as the fornix.

fname = get_fnames(name="fornix")

###############################################################################
# Load fornix streamlines.

fornix = load_tractogram(fname, "same", bbox_valid_check=False)
streamlines = fornix.streamlines

###############################################################################
# Perform QuickBundles clustering using the MDF metric and a 10mm distance
# threshold. Keep in mind that since the MDF metric requires streamlines to
# have the same number of points, the clustering algorithm will internally use
# a representation of streamlines that have been automatically
# downsampled/upsampled so they have only 12 points (To set manually the
# number of points, see :ref:`clustering-examples-ResampleFeature`).

qb = QuickBundles(threshold=10.0)
clusters = qb.cluster(streamlines)

###############################################################################
# `clusters` is a `ClusterMap` object which contains attributes that
# provide information about the clustering result.

print("Nb. clusters:", len(clusters))
print("Cluster sizes:", map(len, clusters))
print("Small clusters:", clusters < 10)
print("Streamlines indices of the first cluster:\n", clusters[0].indices)
print("Centroid of the last cluster:\n", clusters[-1].centroid)

###############################################################################
# `clusters` also has attributes such as `centroids` (cluster representatives),
# and methods like `add`, `remove`, and `clear` to modify the clustering
# result.
#
# Let's first show the initial dataset.

# Enables/disables interactive visualization
interactive = False

scene = window.Scene()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, colors=window.colors.white))
window.record(scene=scene, out_path="fornix_initial.png", size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Initial Fornix dataset.
#
#
# Show the centroids of the fornix after clustering (with random colors):

colormap_qb = cmap.create_colormap(np.arange(len(clusters)))

scene.clear()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, colors=window.colors.white, opacity=0.05))
scene.add(actor.streamtube(clusters.centroids, colors=colormap_qb, linewidth=0.4))
window.record(scene=scene, out_path="fornix_centroids.png", size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Showing the different QuickBundles centroids with random colors.
#
#
# Show the labeled fornix (colors from centroids).

colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap_qb):
    colormap_full[cluster.indices] = color

scene.clear()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, colors=colormap_full))
window.record(scene=scene, out_path="fornix_clusters.png", size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Showing the different clusters.
#
#
# It is also possible to save the complete `ClusterMap` object with pickling.

save_pickle("QB.pkl", clusters)

###############################################################################
# QuickBundlesX (Hierarchical Clustering)
# =======================================
#
# QuickBundlesX extends QuickBundles :footcite:p:`Garyfallidis2012a` by building
# clustering hierarchies at multiple threshold levels. The `qbx_and_merge`
# function is the recommended way to use it.
#
# Let's perform clustering using a sequence of distance thresholds (in mm).
# QuickBundlesX builds a hierarchy by clustering at each threshold level, from
# coarse (40mm) to fine (10mm).

rng = np.random.default_rng(42)
thresholds = [40, 30, 25, 20, 10]
qbx_clusters = qbx_and_merge(streamlines, thresholds, rng=rng)

colormap_qbx = cmap.create_colormap(np.arange(len(qbx_clusters)))
colormap_full_qbx = np.ones((len(streamlines), 3))
for cluster, color in zip(qbx_clusters, colormap_qbx):
    colormap_full_qbx[cluster.indices] = color

scene.clear()
scene.SetBackground(1, 1, 1)
scene.add(actor.streamtube(streamlines, colors=colormap_full_qbx))
window.record(scene=scene, out_path="fornix_qbx_clusters.png", size=(600, 600))
if interactive:
    window.show(scene)

###############################################################################
# .. rst-class:: centered small fst-italic fw-semibold
#
# Showing the different QuickBundlesX clusters with random colors.
#
#
# Comparing QuickBundles and QuickBundlesX
# ========================================
#
# QuickBundles uses a single flat distance threshold, which requires
# determining an optimal global value. QuickBundlesX solves this by exploring
# multiple scales simultaneously.
#
# Let's compare the granularity of the outputs from both methods by looking at
# the number of clusters and their sizes.

qb_cluster_sizes = sorted([len(c) for c in clusters], reverse=True)
qbx_cluster_sizes = sorted([len(c) for c in qbx_clusters], reverse=True)

print(f"QuickBundles  (threshold=10mm): {len(clusters)} clusters")
print(f"QuickBundlesX (thresholds ending at 10mm): {len(qbx_clusters)} clusters")
print("QuickBundles cluster sizes:", qb_cluster_sizes)
print("QuickBundlesX cluster sizes:", qbx_cluster_sizes)

###############################################################################
# Comparing these size distributions shows how QuickBundlesX
# :footcite:p:`Garyfallidis2016` can preserve coarse organization while
# refining the final segmentation at smaller scales. It is also generally
# faster for large datasets.
#
# Finally, here is a video of QuickBundles applied on a larger dataset.
#
# .. raw:: html
#
#     <iframe width="420" height="315" src="https://www.youtube.com/embed/kstL7KKqu94" frameborder="0" allowfullscreen></iframe>
#
#
# References
# ----------
#
# .. footbibliography::
#
