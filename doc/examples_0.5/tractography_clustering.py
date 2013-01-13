""" 

=============================
Tractography Clustering
=============================

Overview
========

**This example gives a tour of clustering related features of dipy.**

First import the necessary modules
----------------------------------

``numpy`` is for numerical computation

"""

import numpy as np

import time

from nibabel import trackvis as tv

from dipy.tracking import metrics as tm
from dipy.segment.quickbundles import QuickBundles
from dipy.io.pickles import save_pickle
from dipy.viz import fvtk


#fname='/home/user/Data_Backup/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
#fname='/home/user/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
from dipy.data import get_data

fname=get_data('fornix')
print(fname)

"""
Load Trackvis file for *Fornix*:
"""

streams,hdr=tv.read(fname)

"""
Copy tracks:
"""

T=[i[0] for i in streams]

"""
Downsample tracks to 12 points:
"""

tracks=[tm.downsample(t, 12) for t in T]

"""
Delete unnecessary data:
"""

del streams,hdr

"""
Perform QuickBundles clustering with a 10mm threshold:
"""

qb=QuickBundles(tracks, dist_thr=10., pts=None)

"""
Show the initial *Fornix* dataset:
"""

r=fvtk.ren()
fvtk.add(r,fvtk.line(T, fvtk.white, opacity=1, linewidth=3))
#fvtk.show(r)
fvtk.record(r,n_frames=1,out_path='fornix_initial',size=(600,600))
fvtk.clear(r)
"""
.. figure:: fornix_initial1000000.png
   :align: center

   **Initial Fornix dataset**.
"""

"""
Show the centroids of the *Fornix* after clustering (with random colors):
"""


centroids=qb.centroids
colormap = np.ones((len(centroids), 3))
for i, centroid in enumerate(centroids):
    colormap[i] = np.random.rand(3)
    fvtk.add(r, fvtk.line(centroids, colormap, opacity=1., linewidth=5))
#hack for auto camera
fvtk.add(r,fvtk.line(T,fvtk.white,opacity=0))
#fvtk.show(r)
fvtk.record(r,n_frames=1,out_path='fornix_centroids',size=(600,600))
fvtk.clear(r)

"""
.. figure:: fornix_centroids1000000.png
   :align: center

   **Showing the different clusters with random colors**.

"""

"""
Show the labeled *Fornix* (colors from centroids):
"""

colormap_full = np.ones((len(tracks), 3))
for i, centroid in enumerate(centroids):
    inds=qb.label2tracksids(i)
    colormap_full[inds]=colormap[i]
fvtk.add(r, fvtk.line(tracks, colormap_full, opacity=1., linewidth=3))


#fvtk.show(r)
fvtk.record(r,n_frames=1,out_path='fornix_clust',size=(600,600))

"""
.. figure:: fornix_clust1000000.png
   :align: center

   **Showing the different clusters with random colors**.

"""

"""
It is also possible to save the QuickBundles object with pickling.
"""
save_pickle('QB.pkl',qb)


