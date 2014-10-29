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
from dipy.segment.metric import MDF
#from dipy.segment.quickbundles import QuickBundles
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
Perform QuickBundles clustering using the MDF metric and a 10mm distance threshold
after having downsampled the streamlines to have only 18 points.
"""
streamlines = set_number_of_points(streamlines, nb_points=18)
qb = QuickBundles(streamlines, dist_thr=10., pts=18)

"""
qb has attributes like `centroids` (cluster representatives), `total_clusters`
(total number of clusters) and methods like `partitions` (complete description
of all clusters) and `label2tracksids` (provides the indices of the streamlines
which belong in a specific cluster).

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

centroids = qb.centroids
colormap = np.random.rand(len(centroids), 3)

fvtk.clear(ren)
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, fvtk.colors.white, opacity=0.05))
fvtk.add(ren, fvtk.streamtube(centroids, colormap, linewidth=0.4))
fvtk.record(ren, n_frames=1, out_path='fornix_centroids.png', size=(600, 600))

"""
.. figure:: fornix_centroids.png
   :align: center

   **Showing the different QuickBundles centroids with random colors**.

Show the labeled fornix (colors from centroids).
"""

colormap_full = np.ones((len(streamlines), 3))
for i, centroid in enumerate(centroids):
    inds = qb.label2tracksids(i)
    colormap_full[inds] = colormap[i]

fvtk.clear(ren)
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, colormap_full))
fvtk.record(ren, n_frames=1, out_path='fornix_clust.png', size=(600, 600))

"""
.. figure:: fornix_clust.png
   :align: center

   **Showing the different clusters with random colors**.

It is also possible to save the complete QuickBundles object with pickling.
"""

save_pickle('QB.pkl', qb)

"""
Finally, here is a video of QuickBundles applied on a larger dataset.

.. raw:: html

    <iframe width="420" height="315" src="http://www.youtube.com/embed/kstL7KKqu94" frameborder="0" allowfullscreen></iframe>

.. include:: ../links_names.inc

.. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                    tractography simplification, Frontiers in Neuroscience, vol
                    6, no 175, 2012.

"""
