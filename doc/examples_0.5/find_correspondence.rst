.. AUTO-GENERATED FILE -- DO NOT EDIT!

.. _example_find_correspondence:


==========================================
Find correspondence between tractographies
==========================================

First import the necessary modules

numpy is for numerical computation

::
  
  import numpy as np
  

nibabel is for data formats

::
  
  import nibabel as nib
  

dipy.tracking.distances is for tractography distances

::
  
  from dipy.tracking.distances import mam_distances
  

dipy.data is for getting some small datasets used in examples and tests.

::
  
  from dipy.data import get_skeleton
  

``get_skeleton`` provides two skeletons 'C1' and 'C3'
previously generated from Local Skeleton Clustering (LSC)

::
  
  C1=get_skeleton('C1')
  C3=get_skeleton('C3')
  

We create a diagram with the two skeletons offset [100,0,0] apart

::
  
  from dipy.viz import fvtk
  r=fvtk.ren()
  
  T1=[]
  for c in C1:
      T1.append(C1[c]['most'])
  
  fvtk.add(r,fvtk.line(T1,fvtk.gray))    
  
  T3=[]
  for c in C3:
      T3.append(C3[c]['most'])    
  
  T3s=[t+ np.array([100,0,0]) for t in T3]
  
  fvtk.add(r,fvtk.line(T3s,fvtk.gray))
  
  #fvtk.show(r)    bout 
  

For each track in T1 find the minimum average distance to all the
tracks in T3 and put information about it in ``track2track``.

::
  
  indices=range(len(T1))    
  track2track=[]
  mam_threshold=6.
  
  for i in indices:                
      rt=[mam_distances(T1[i],t,'avg') for t in T3]
      rt=np.array(rt)
      if rt.min()< mam_threshold:
          track2track.append(np.array([i,rt.argmin(),rt.min()]))        
          
  track2track=np.array(track2track)
  
  np.set_printoptions(2)
  

When a track in T3 is simultaneously the nearest track to more than one track in T1 we identify the track
in T1 that has the best correspondence and remove the other.

::
  
  good_correspondence=[]
  for i in track2track[:,1]:
      
      check= np.where(track2track[:,1]==i)[0]
      if len(check) == 1:
          good_correspondence.append(check[0])
      elif len(check)>=2:
          #print check,check[np.argmin(track2track[check][:,2])]
          good_correspondence.append(check[np.argmin(track2track[check][:,2])])
          #good_correspondence.append())
  
  #print goo_correspondenced
  good_correspondence=list(set(good_correspondence))
  
  track2track=track2track[good_correspondence,:]
  
  print 'With mam_threshold %f we find %d correspondence pairs' % (mam_threshold, np.size(track2track,0))
  
  #fvtk.clear(r)
  

Now plot the corresponding tracks in the same colours

.. figure:: ../_static/find_corr.png
   :align: center

   **Showing correspondence between these two modest tractographies**.

   The labels on the corresponding tracks are the indices of the first tractography on the left.


::
  
  for row in track2track:
  
      color=np.random.rand(3)
      T=[T1[int(row[0])],T3s[int(row[1])]]
      fvtk.add(r,fvtk.line(T,color,linewidth=10))
      pos1=T1[int(row[0])][0]
      pos3=T3s[int(row[1])][0]
      fvtk.add(r,fvtk.label(r,str(int(row[0])),tuple(pos1),(5,5,5)))
      fvtk.add(r,fvtk.label(r,str(int(row[0])),tuple(pos3),(5,5,5)))
  
  #fvtk.show(r,png_magnify=1,size=(600,600))
  
  
  
  
  
  
  
  
  
  

        
.. admonition:: Example source code

   You can download :download:`the full source code of this example <./find_correspondence.py>`.
   This same script is also included in the dipy source distribution under the
   :file:`doc/examples/` directory.

