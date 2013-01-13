.. AUTO-GENERATED FILE -- DO NOT EDIT!

.. _example_tractography_clustering:



=============================
Tractography Clustering
=============================

Overview
========

**This example gives a tour of clustering related features of dipy.**

First import the necessary modules
----------------------------------

``numpy`` is for numerical computation


::
  
  import numpy as np
  
  import time
  
  from nibabel import trackvis as tv
  
  from dipy.tracking import metrics as tm
  from dipy.tracking import distances as td
  from dipy.io import pickles as pkl
  from dipy.viz import fvtk
  
  
  #fname='/home/user/Data_Backup/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
  #fname='/home/user/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
  from dipy.data import get_data
  
  fname=get_data('fornix')
  print(fname)
  

Load Trackvis file for *Fornix*:

::
  
  streams,hdr=tv.read(fname)
  

Copy tracks:

::
  
  T=[i[0] for i in streams]
  
  #T=T[:1000]
  

Downsample tracks to just 3 points:

::
  
  tracks=[tm.downsample(t,3) for t in T]
  

Delete unnecessary data:

::
  
  del streams,hdr
  

Perform Local Skeleton Clustering (LSC) with a 5mm threshold:

::
  
  now=time.clock()
  C=td.local_skeleton_clustering(tracks,d_thr=5)
  print('Done in %.2f s'  % (time.clock()-now,))
  
  

Reduce the number of points for faster visualization using the ``approx_polygon_track`` algorithm which retains points depending on how much they are need to define the shape of the track:

::
  
  T=[td.approx_polygon_track(t) for t in T]
  

Show the initial *Fornix* dataset:

::
  
  r=fvtk.ren()
  fvtk.add(r,fvtk.line(T,fvtk.white,opacity=1))
  #fvtk.show(r)
  

Show the *Fornix* after clustering (with random bundle colors):

::
  
  fvtk.clear(r)
  colors=np.zeros((len(T),3))
  for c in C:
      color=np.random.rand(1,3)
      for i in C[c]['indices']:
          colors[i]=color
  fvtk.add(r,fvtk.line(T,colors,opacity=1))
  #fvtk.show(r)
  

Calculate some statistics about the clusters

::
  
  lens=[len(C[c]['indices']) for c in C]
  print('max %d min %d' %(max(lens), min(lens)))
  print('singletons %d ' % lens.count(1))
  print('doubletons %d' % lens.count(2))
  print('tripletons %d' % lens.count(3))
  

Find and display the skeleton of most representative tracks in each cluster:

::
  
  skeleton=[]
  
  fvtk.clear(r)
  
  for c in C:
      
      bundle=[T[i] for i in C[c]['indices']]
      si,s=td.most_similar_track_mam(bundle,'avg')    
      skeleton.append(bundle[si])
      fvtk.label(r,text=str(len(bundle)),pos=(bundle[si][-1]),scale=(2,2,2))
  
  fvtk.add(r,fvtk.line(skeleton,colors,opacity=1))
  #fvtk.show(r)
  

Save the skeleton information in the dictionary. Now try to play with different thresholds LSC and check the different results.
Try it with your datasets and gives us some feedback.


::
  
  for (i,c) in enumerate(C):    
      C[c]['most']=skeleton[i]
      
  for c in C:    
      print('Keys in bundle %d' % c)
      print(C[c].keys())
      print('Shape of skeletal track (%d, %d) ' % C[c]['most'].shape)
  
  pkl.save_pickle('skeleton_fornix.pkl',C)
  
  
  
  

        
.. admonition:: Example source code

   You can download :download:`the full source code of this example <./tractography_clustering.py>`.
   This same script is also included in the dipy source distribution under the
   :file:`doc/examples/` directory.

