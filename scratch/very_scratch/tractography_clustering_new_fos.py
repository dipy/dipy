import time

import numpy as np

from nibabel import trackvis as tv

from dipy.core import track_metrics as tm
from dipy.core import track_performance as pf

from fos.core.scene  import Scene
from fos.core.actors import Actor
from fos.core.plots  import Plot
from fos.core.tracks import Tracks

#fname='/home/eg01/Data_Backup/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'

fname='/home/eg01/Data_Backup/Data/PBC/pbc2009icdm/brain2/brain2_scan1_fiber_track_mni.trk'


#fname='/home/eg309/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'

opacity=0.5

print 'Loading file...'
streams,hdr=tv.read(fname)

print 'Copying tracks...'
T=[i[0] for i in streams]

T=T[:len(T)/5]

#T=T[:1000]

print 'Representing tracks using only 3 pts...'
tracks=[tm.downsample(t,3) for t in T]

print 'Deleting unnecessary data...'
del streams,hdr

print 'Local Skeleton Clustering...'
now=time.clock()
C=pf.local_skeleton_clustering(tracks,d_thr=20)
print 'Done in', time.clock()-now,'s.'

print 'Reducing the number of points...'
T=[pf.approx_polygon_track(t) for t in T]

print 'Showing initial dataset.'

#r=fos.ren()
#fos.add(r,fos.line(T,fos.white,opacity=0.1))
#fos.show(r)



data=T

colors =[np.tile(np.array([1,1,1,opacity],'f'),(len(t),1)) for t in T]

t=Tracks(data,colors,line_width=1.)  

t.position=(-100,0,0)

print 'Showing dataset after clustering.'

print 'Calculating skeletal track for every bundle.'

skeletals=[]
colors2 = len(data)*[None]
colors_sk = []#len(C.keys())*[None]

for c in C:

    color=np.random.rand(3)    
    r,g,b = color
    bundle=[]

    for i in C[c]['indices']:
        
        colors2[i]=np.tile(np.array([r,g,b,opacity],'f'),(len(data[i]),1))    
        bundle.append(data[i])

    
    bi=pf.most_similar_track_mam(bundle)[0]
    C[c]['skeletal']=bundle[bi]
    

    if len(C[c]['indices'])>100 and tm.length(bundle[bi])>30.:        
        colors_sk.append( np.tile(np.array([r,g,b,opacity],'f'),(len(bundle[bi]),1)) )
        skeletals.append(bundle[bi])

        

print 'len_data', len(data)
print 'len_skeletals', len(skeletals)
print 'len_colors2', len(colors2)
print 'len_colors_sk', len(colors_sk)
    
t2=Tracks(data,colors2,line_width=1.)
t2.position=(100,0,0)

sk=Tracks(skeletals,colors_sk,line_width=3.)
sk.position=(0,0,0)
        
slot={0:{'actor':t,'slot':(0, 800000)},
      1:{'actor':t2,'slot':(0, 800000)},
      2:{'actor':sk,'slot':(0, 800000)}}

Scene(Plot(slot)).run()

print 'Some statistics about the clusters'
lens=[len(C[c]['indices']) for c in C]
print 'max ',max(lens), 'min ',min(lens)
print 'singletons ',lens.count(1)
print 'doubletons ',lens.count(2)
print 'tripletons ',lens.count(3)

""" Next Level

12: cluster0=[T[t] for t in C[0]['indices']]
13: pf.most_similar_track_mam(cluster0)

"""
