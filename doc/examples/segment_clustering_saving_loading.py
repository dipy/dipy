"""
==================================================
Saving and Loading QuickBundles Clustering Results
==================================================

This example shows how to save clustering results produced by QuickBundles
:footcite:p:`Garyfallidis2012a` into tractogram files using two formats.

**Part 1 — TRX** uses the native TRX structure: cluster labels are stored in
``dps/clusters_QB`` and each cluster is saved as a named entry under
``groups/``. This is the recommended approach because TRX natively supports
both concepts, uses memory-mapped arrays for efficiency, and avoids any
concatenation hack.

**Part 2 — TRK** uses a merge strategy: the centroids are appended to the
streamline array and an ``is_centroid`` flag is stored in
``data_per_streamline``. This is necessary because the TRK format has no
native groups concept.

"""

import numpy as np
import trx.trx_file_memmap as tmm

from dipy.data import get_fnames
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.clustering import QuickBundles

###############################################################################
# Generating Clusters and Centroids
# =================================
#
# We fetch a sample fornix dataset and group the streamlines into clusters
# using the QuickBundles algorithm. QuickBundles produces a ``ClusterMap``
# that stores per-cluster indices and a representative centroid streamline.

fname = get_fnames(name="fornix")

# bbox_valid_check=False is required because the fornix streamlines extend
# slightly outside the reference image bounding box.
fornix = load_tractogram(fname, "same", bbox_valid_check=False)
streamlines = fornix.streamlines

qb = QuickBundles(threshold=10.0)
clusters = qb.cluster(streamlines)

print(f"Number of clusters: {len(clusters)}")
print(f"Cluster sizes: {list(map(len, clusters))}")

###############################################################################
# Part 1 — TRX: Native Groups Approach
# =====================================
#
# TRX natively supports two complementary ways to encode clustering results:
#
# * ``dps/clusters_QB`` — a per-streamline integer label array stored under
#   ``data_per_streamline``. The key name ``clusters_QB`` is a TRX convention
#   established in the OHBM demo tractogram.
# * ``groups/cluster_N`` — named index arrays, one per cluster, stored under
#   ``groups/``. Each array contains the indices of the streamlines that
#   belong to that cluster, enabling random access without loading all data.
#
# We build a ``StatefulTractogram`` with only the original streamlines and the
# ``clusters_QB`` label array, convert it to a ``TrxFile``, then attach the
# per-cluster index arrays as named groups before saving.

labels = np.empty(len(streamlines), dtype=np.int16)
for i, cluster in enumerate(clusters):
    for idx in cluster.indices:
        labels[idx] = i

sft = StatefulTractogram(
    streamlines,
    reference=fornix,
    space=Space.RASMM,
    data_per_streamline={"clusters_QB": labels},
)

trx = tmm.TrxFile.from_sft(sft)

for i, cluster in enumerate(clusters):
    trx.groups[f"cluster_{i}"] = np.array(cluster.indices, dtype=np.uint32)

tmm.save(trx, "clustering_results.trx")
trx.close()

###############################################################################
# Reloading the TRX File and Accessing Groups
# ============================================
#
# After reloading, ``data_per_streamline["clusters_QB"]`` gives the per-
# streamline label and ``groups["cluster_N"]`` gives the index array for each
# named cluster — no concatenation or sentinel flags required.

trx_loaded = tmm.load("clustering_results.trx")

print(f"Loaded {len(trx_loaded.streamlines)} streamlines from TRX")
print(f"dps keys: {list(trx_loaded.data_per_streamline.keys())}")
print(f"group keys: {list(trx_loaded.groups.keys())}")
print(
    f"labels match: "
    f"{np.array_equal(trx_loaded.data_per_streamline['clusters_QB'], labels)}"
)

trx_loaded.close()

###############################################################################
# Part 2 — TRK: Merging Streamlines and Centroids
# ================================================
#
# The TRK format has no native groups concept, so the standard approach is
# to append the centroid streamlines to the original streamlines and record
# two integer arrays in ``data_per_streamline``:
#
# * ``cluster`` — the cluster index for every streamline (centroids carry
#   their own cluster index).
# * ``is_centroid`` — 0 for original streamlines, 1 for centroids.

centroid_labels = np.arange(len(clusters), dtype=np.int32)
all_labels = np.concatenate((labels.ravel().astype(np.int32), centroid_labels))

is_centroid = np.concatenate(
    (
        np.zeros(len(streamlines), dtype=np.int32),
        np.ones(len(clusters), dtype=np.int32),
    )
)

combined_streamlines = list(streamlines) + list(clusters.centroids)

sft_combined = StatefulTractogram(
    combined_streamlines,
    reference=fornix,
    space=Space.RASMM,
    data_per_streamline={
        "cluster": all_labels,
        "is_centroid": is_centroid,
    },
)

save_tractogram(sft_combined, "clustering_results.trk", bbox_valid_check=False)

sft_trk = load_tractogram("clustering_results.trk", "same", bbox_valid_check=False)

print(f"Loaded {len(sft_trk.streamlines)} total streamlines from TRK")
print(
    f"TRK labels match: "
    f"{np.array_equal(sft_trk.data_per_streamline['cluster'], all_labels)}"
)
print(
    f"TRK centroid flags match: "
    f"{np.array_equal(sft_trk.data_per_streamline['is_centroid'], is_centroid)}"
)

###############################################################################
# References
# ----------
#
# .. footbibliography::
#
