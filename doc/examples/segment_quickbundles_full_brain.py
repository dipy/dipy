"""
============================================================================
Tractography Clustering with QuickBundles for Immense Full Brain Datasets
============================================================================

This example explains how we can use QuickBundles [Garyfallidis12]_ to
simplify/cluster streamlines.

First import the necessary modules.
"""

import numpy as np
from nibabel import trackvis as tv
from dipy.tracking import metrics as tm
from dipy.tracking.streamline import length, set_number_of_points
from dipy.segment.clustering import QuickBundles
from dipy.segment.metric import Metric, dist
from dipy.segment.metricspeed import ArcLength as ArcLengthFast
from dipy.io.pickles import save_pickle
from dipy.data import get_data
from dipy.viz import fvtk
from time import time


class ArcLength(Metric):
    def infer_features_shape(self, streamline):
        return (1, 1)

    def extract_features(self, streamline):
        length_ = length(streamline).astype('f4')
        return np.array([[length_]])

    def dist(self, features1, features2):
        return np.abs(features1 - features2)[0, 0]


class Orientation(Metric):
    def infer_features_shape(self, streamline):
        return 3

    def extract_features(self, streamline):
        vector = np.mean(np.diff(streamline, axis=0), axis=0)
        return vector/np.linalg.norm(vector)

    def dist(self, features1, features2):
        angle = np.rad2deg(np.abs(np.arccos(np.dot(features1, features2))))

        if angle > 90:
            return 180 - angle
        else:
            return angle


class MDFpy(Metric):
    def infer_features_shape(self, streamline):
        return streamline.shape[0] * streamline.shape[1]

    def extract_features(self, streamline):
        N, D = streamline.shape

        features = np.empty(N*D, dtype=streamline.base.dtype)
        for y in range(N):
            i = y*D
            features[i+0] = streamline[y, 0]
            features[i+1] = streamline[y, 1]
            features[i+2] = streamline[y, 2]

        return features

    def dist(self, features1, features2):
        D = 3
        N = features2.shape[0] // D

        d = 0.0
        for y in range(N):
            i = y*D
            dx = features1[i+0] - features2[i+0]
            dy = features1[i+1] - features2[i+1]
            dz = features1[i+2] - features2[i+2]
            d += np.sqrt(dx*dx + dy*dy + dz*dz)

        return d / N


# To become a test

s1 = np.array([[0,0,0], [1, 1, 0.]])
s1 = set_number_of_points(s1, 12)

s2 = np.array([[0,0,0], [1, -1.2, 0.]])
s2 = set_number_of_points(s2, 12)

o = Orientation()
print(o.extract_features(s1))
print(o.extract_features(s2))
print(o.dist(o.extract_features(s1), o.extract_features(s2)))


dname = '/home/eleftherios/Data/fancy_data/2013_02_26_Patrick_Delattre/'
fname =  dname + 'streamlines_500K.trk'

"""
Load full brain streamlines.
"""

streams, hdr = tv.read(fname)

streamlines = [i[0] for i in streams]
streamlines = streamlines[:200]

for s in streamlines:
    s.setflags(write=True)

pts = 20

rstreamlines = set_number_of_points(streamlines, pts)

# from ipdb import set_trace as dbg
# dbg()


t0 = time()

#qb = QuickBundles(threshold=20., metric=ArcLength())

#cluster_map = qb.cluster(rstreamlines)

# t1 = time()
# print(t1 - t0)

# qb2 = QuickBundles(threshold=20., metric=ArcLengthFast())

# cluster_map2 = qb2.cluster(rstreamlines)

# t2 = time()
# print(t2 - t1)

qb3 = QuickBundles(threshold=20.)

cluster_map3 = qb3.cluster(rstreamlines)

# t3 = time()
# print(t3 - t2)

# qb4 = QuickBundles(threshold=20., metric=Orientation())

# cluster_map4 = qb4.cluster(rstreamlines)

# t4 = time()
# print(t4 - t3)



"""
qb has attributes like `centroids` (cluster representatives), `total_clusters`
(total number of clusters) and methods like `partitions` (complete description
of all clusters) and `label2tracksids` (provides the indices of the streamlines
which belong in a specific cluster).

Lets first show the initial dataset.
"""

ren = fvtk.ren()
# ren.SetBackground(1, 1, 1)
# fvtk.add(ren, fvtk.line(streamlines, fvtk.colors.white))

# fvtk.show(ren)
# fvtk.record(ren, n_frames=1, out_path='full_brain_initial.png', size=(600, 600))

"""
.. figure:: full_brain_initial.png
   :align: center

   **Initial Fornix dataset**.

Show the centroids of the fornix after clustering (with random colors):
"""

# cs = cluster_map.centroids
# centroids = []
# for c in cs:
#     centroids.append(c.reshape(pts, 3))

clusters = cluster_map4.clusters

colormap = np.random.rand(len(clusters), 3)


# fvtk.clear(ren)
# ren.SetBackground(1, 1, 1)
# #fvtk.add(ren, fvtk.line(streamlines, fvtk.colors.red, opacity=0.05))
# fvtk.add(ren, fvtk.line(centroids, colormap, linewidth=3.))
# fvtk.show(ren)
# fvtk.record(ren, n_frames=1, out_path='full_brain_centroids.png', size=(600, 600))

# 1/0

"""
.. figure:: full_brain_centroids.png
   :align: center

   **Showing the different QuickBundles centroids with random colors**.

Show the labeled fornix (colors from centroids).
"""

colormap_full = np.ones((len(streamlines), 3))
for i, cluster in enumerate(clusters):
    inds = cluster.indices
    for j in inds:
        colormap_full[j] = colormap[i]

fvtk.clear(ren)
ren.SetBackground(1, 1, 1)
fvtk.add(ren, fvtk.streamtube(streamlines, colormap_full))
fvtk.show(ren)
fvtk.record(ren, n_frames=1, out_path='full_brain_clust.png', size=(600, 600))

"""
.. figure:: full_brain_clust.png
   :align: center

   **Showing the different clusters with random colors**.

It is also possible to save the complete QuickBundles object with pickling.
"""

save_pickle('QB.pkl', qb)

"""
Finally, here is a video of QuickBundles applied on a larger dataset.

.. include:: ../links_names.inc

.. [MarcCote14]

.. [Garyfallidis12] Garyfallidis E. et al., QuickBundles a method for
                    tractography simplification, Frontiers in Neuroscience, vol
                    6, no 175, 2012.

"""
