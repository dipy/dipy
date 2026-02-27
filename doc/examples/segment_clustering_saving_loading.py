"""
==================================================
Saving and Loading QuickBundles Clustering Results
==================================================

This example shows how to save clustering results produced by QuickBundles
:footcite:p:`Garyfallidis2012a` into tractogram files. Cluster labels are
stored via ``data_per_streamline`` and centroids are saved separately.
Both TRK and TRX formats are demonstrated.

"""

import numpy as np

from dipy.data import get_fnames
from dipy.io.stateful_tractogram import Space, StatefulTractogram
from dipy.io.streamline import load_tractogram, save_tractogram
from dipy.segment.clustering import QuickBundles

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
# Build a label array that maps each streamline to its cluster index.
# ``cluster.indices`` gives the original streamline indices belonging to
# that cluster.

labels = np.empty(len(streamlines), dtype=np.int32)
for i, cluster in enumerate(clusters):
    for idx in cluster.indices:
        labels[idx] = i

###############################################################################
# Save the streamlines with their cluster labels attached via
# ``data_per_streamline``. The TRK format preserves this metadata
# through save/load round-trips.

sft_labeled = StatefulTractogram(
    streamlines,
    reference=fornix,
    space=Space.RASMM,
    data_per_streamline={"cluster": labels},
)

save_tractogram(sft_labeled, "labeled_streamlines.trk",
                bbox_valid_check=False)

# Save the cluster centroids as a separate tractogram.
sft_centroids = StatefulTractogram(
    clusters.centroids,
    reference=fornix,
    space=Space.RASMM,
)

save_tractogram(sft_centroids, "centroids.trk",
                bbox_valid_check=False)

###############################################################################
# Reload and verify.

sft_loaded = load_tractogram("labeled_streamlines.trk", "same",
                             bbox_valid_check=False)
loaded_labels = sft_loaded.data_per_streamline["cluster"]

print(f"Loaded {len(loaded_labels)} labels from TRK")
print(f"Labels match: "
      f"{np.array_equal(loaded_labels.astype(np.int32), labels)}")

sft_centr = load_tractogram("centroids.trk", "same",
                            bbox_valid_check=False)
print(f"Loaded {len(sft_centr.streamlines)} centroids from TRK")

# The same workflow works with TRX, which uses memory-mapped arrays
# and is more efficient for large tractograms.

save_tractogram(sft_labeled, "labeled_streamlines.trx",
                bbox_valid_check=False)
save_tractogram(sft_centroids, "centroids.trx",
                bbox_valid_check=False)

sft_trx = load_tractogram("labeled_streamlines.trx", "same",
                          bbox_valid_check=False)
trx_labels = sft_trx.data_per_streamline["cluster"]

print(f"Loaded {len(trx_labels)} labels from TRX")
print(f"Labels match: "
      f"{np.array_equal(trx_labels.astype(np.int32), labels)}")

sft_centr_trx = load_tractogram("centroids.trx", "same",
                                bbox_valid_check=False)
print(f"Loaded {len(sft_centr_trx.streamlines)} centroids "
      f"from TRX")

###############################################################################
# References
# ----------
#
# .. footbibliography::
#
