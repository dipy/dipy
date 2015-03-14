"""
=========================================
Tractography Clustering with QuickBundles
=========================================

This example explains how we can use QuickBundles [Garyfallidis12]_ to
simplify/cluster streamlines.

First import the necessary modules.
"""

import numpy as np
from nibabel import trackvis as tv
from dipy.tracking.streamline import set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.data import get_data
from dipy.viz import fvtk

"""
For educational purposes we will try to cluster a small streamline bundle known
from neuroanatomy as the fornix.
"""

fname = get_data('fornix')

"""
Load fornix streamlines.
"""

streams, hdr = tv.read(fname)

streamlines = [i[0] for i in streams]

"""
Perform QuickBundles clustering using the MDF metric and a 10mm distance
threshold. Since the MDF metric requires streamlines to have the same number
of points, we first downsample them so they have only 18 points and then run
the clustering algorithm.
"""

qb = QuickBundles(threshold=10.)
streamlines = set_number_of_points(streamlines, nb_points=18)
clusters = qb.cluster(streamlines)

"""
`clusters` is a `ClusterMap` object which contains attributes that
provide information about the clustering result.
"""

print("Nb. clusters:", len(clusters))
print("Cluster sizes:", map(len, clusters))
print("Small clusters:", clusters < 10)
print("Streamlines indices of the first cluster:\n", clusters[0].indices)
print("Centroid of the last cluster:\n", clusters[-1].centroid)

"""

::

    Nb. clusters: 4

    Cluster sizes: [64, 191, 44, 1]

    Small clusters: array([False, False, False, True], dtype=bool)

    Streamlines indices of the first cluster:
    [0, 7, 8, 10, 11, 12, 13, 14, 15, 18, 26, 30, 33, 35, 41, 42, 45, 65, 66, 75,
     85, 100, 101, 105, 115, 116, 119, 122, 123, 124, 125, 126, 128, 129, 135, 139,
     142, 143, 144, 148, 151, 159, 167, 175, 180, 181, 185, 200, 208, 210, 224, 237,
     246, 249, 251, 256, 267, 270, 280, 284, 293, 296, 297, 299]

    Centroid of the last cluster:
    array([[  84.83773804,  117.92590332,   77.32278442],
           [  85.89845276,  116.67261505,   80.27609253],
           [  86.2130661 ,  114.88985443,   83.13295746],
           [  86.40007019,  112.50982666,   85.55088043],
           [  86.54071045,  109.60722351,   87.31826019],
           [  86.39044189,  106.43745422,   88.54563904],
           [  86.29808807,  103.12637329,   89.29672241],
           [  85.72164154,   99.78807068,   89.04328918],
           [  84.6943512 ,   96.6314621 ,   88.31369781],
           [  83.09349823,   93.83686066,   87.22660065],
           [  81.00836945,   91.42190552,   86.07907867],
           [  78.49610138,   89.20231628,   85.63204193],
           [  75.75254822,   87.23584747,   85.22332001],
           [  72.96138   ,   85.69472504,   84.09647369],
           [  70.16287231,   85.14102173,   82.26060486],
           [  67.67449188,   85.57660675,   79.98880005],
           [  65.69326782,   86.66771698,   77.44818115],
           [  64.02451324,   88.43942261,   75.0697403 ]], dtype=float32)

"""

"""
`clusters` has also attributes like `centroids` (cluster representatives), and
methods like `add`, `remove`, and `clear` to modify the clustering result.

Lets first show the initial dataset.
"""

ren = fvtk.ren()
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white))
fvtk.record(ren, n_frames=1, out_path='fornix_initial.png', size=(600, 600))

"""
.. figure:: fornix_initial.png
   :align: center

   **Initial Fornix dataset**.

Show the centroids of the fornix after clustering (with random colors):
"""

colormap = np.random.rand(len(clusters), 3)

fvtk.clear(ren)
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white, opacity=0.05))
fvtk.add(ren, fvtk.streamtube(clusters.centroids, colormap, linewidth=0.4))
fvtk.record(ren, n_frames=1, out_path='fornix_centroids.png', size=(600, 600))

"""
.. figure:: fornix_centroids.png
   :align: center

   **Showing the different QuickBundles centroids with random colors**.

Show the labeled fornix (colors from centroids).
"""

colormap_full = np.ones((len(streamlines), 3))
for cluster, color in zip(clusters, colormap):
    colormap_full[cluster.indices] = color

fvtk.clear(ren)
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, colormap_full))
fvtk.record(ren, n_frames=1, out_path='fornix_clusters.png', size=(600, 600))

"""
.. figure:: fornix_clusters.png
   :align: center

   **Showing the different clusters**.

It is also possible to save the complete `ClusterMap` object with pickling.
"""

save_pickle('QB.pkl', clusters)

"""
Finally, here is a video of QuickBundles applied on a larger dataset.

.. raw:: html

    <iframe width="420" height="315" src="http://www.youtube.com/embed/kstL7KKqu94" frameborder="0" allowfullscreen></iframe>

.. include:: ../links_names.inc

.. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                    tractography simplification, Frontiers in Neuroscience, vol
                    6, no 175, 2012.

"""
