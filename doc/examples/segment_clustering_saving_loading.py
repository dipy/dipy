"""
==================================================
Saving and Loading QuickBundles Clustering Results
==================================================

This example shows how to save clustering results produced by QuickBundles
:footcite:p:`Garyfallidis2012a` into tractogram files. To ensure all relevant
information is kept together, we merge original streamlines and their cluster
centroids into a single tractogram. Both TRX and TRK formats are demonstrated,
prioritizing the highly efficient TRX format.

"""

import numpy as np

from dipy.data import get_fnames
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.clustering import QuickBundles

###############################################################################
# Generating Clusters and Centroids
# =================================
#
# We fetch a sample fornix dataset and group the streamlines into clusters
# using the QuickBundles algorithm. QuickBundles produces a `ClusterMap`
# object that stores the cluster indices as well as the computed centroids
# (the representative average streamline for each cluster).

fname = get_fnames(name="fornix")

# bbox_valid_check=False is needed because the fornix streamlines
# extend slightly outside the reference bounding box.
fornix = load_tractogram(fname, "same", bbox_valid_check=False)
streamlines = fornix.streamlines

qb = QuickBundles(threshold=10.0)
clusters = qb.cluster(streamlines)

print(f"Number of clusters: {len(clusters)}")
print(f"Cluster sizes: {list(map(len, clusters))}")

###############################################################################
# Merging Streamlines and Centroids
# =================================
#
# A common workflow is to save both the original streamlines and their
# centroids together. To do this, we concatenate the centroid streamlines
# to the end of the original streamline list.
#
# We also need to keep track of which cluster each streamline belongs to,
# and distinguish the original data from the centroids. We can store this
# metadata using ``data_per_streamline``.

labels = np.empty(len(streamlines), dtype=np.int32)
for i, cluster in enumerate(clusters):
    for idx in cluster.indices:
        labels[idx] = i

centroid_labels = np.arange(len(clusters), dtype=np.int32)
all_labels = np.concatenate((labels, centroid_labels))

is_centroid = np.concatenate((
    np.zeros(len(streamlines), dtype=np.int32),
    np.ones(len(clusters), dtype=np.int32)
))

combined_streamlines = list(streamlines) + list(clusters.centroids)

sft_combined = StatefulTractogram(
    combined_streamlines,
    reference=fornix,
    space=Space.RASMM,
    data_per_streamline={
        "cluster": all_labels,
        "is_centroid": is_centroid
    },
)

###############################################################################
# Saving to TRX
# =============
#
# The TRX format is recommended because it utilizes memory-mapped arrays,
# making it extremely memory-efficient for large tractograms. Crucially,
# any metadata stored in ``data_per_streamline`` is automatically preserved
# when saving and loading.

save_tractogram(sft_combined, "clustering_results.trx", bbox_valid_check=False)

sft_trx = load_tractogram("clustering_results.trx", "same",
                          bbox_valid_check=False)

trx_labels = sft_trx.data_per_streamline["cluster"]
trx_is_centroid = sft_trx.data_per_streamline["is_centroid"]

print(f"Loaded {len(sft_trx.streamlines)} total streamlines from TRX")
print(f"TRX labels match: {np.array_equal(trx_labels, all_labels)}")
print(f"TRX centroid flags match: "
      f"{np.array_equal(trx_is_centroid, is_centroid)}")

###############################################################################
# For backward compatibility with older software, the combined tractogram
# can also be saved as a standard TRK file using identical syntax.

save_tractogram(sft_combined, "clustering_results.trk",
                bbox_valid_check=False)

###############################################################################
# References
# ----------
#
# .. footbibliography::
#
