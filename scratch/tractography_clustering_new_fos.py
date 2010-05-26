import time
import numpy as np

from dipy.core import track_metrics as tm
from dipy.io import trackvis as tv
from dipy.core import track_performance as pf

from fos.core.scene  import Scene
from fos.core.actors import Actor
from fos.core.plots  import Plot
from fos.core.tracks import Tracks

#fname='/home/eg01/Data_Backup/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'

fname='/home/eg309/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'

print 'Loading file...'
streams,hdr=tv.read(fname)

print 'Copying tracks...'
T=[i[0] for i in streams]

T=T[:1000]

print 'Representing tracks using only 3 pts...'
tracks=[tm.downsample(t,3) for t in T]

print 'Deleting unnecessary data...'
del streams,hdr

print 'Local Skeleton Clustering...'
now=time.clock()
C=pf.local_skeleton_clustering(tracks,d_thr=20)
print 'Done in', time.clock()-now,'s.'

print 'Reducing the number of points...'
T=[pf.approximate_ei_trajectory(t) for t in T]

print 'Showing initial dataset.'

#r=fos.ren()
#fos.add(r,fos.line(T,fos.white,opacity=0.1))
#fos.show(r)

data=T

colors =[np.tile(np.array([1,1,1,1],'f'),(len(t),1)) for t in T]

t=Tracks(data,colors)  

print 'Showing dataset after clustering.'

colors2 = []

for c in C:

    color=np.random.rand(3)

    #print color
    
    r,g,b = color
    
    for i in C[c]['indices']:

        #print i

        colors2.append(np.tile(np.array([r,g,b,1],'f'),(len(data[i]),1)))
        
        
t2=Tracks(data,colors2)   


slot={0:{'actor':t,'slot':(0, 10000)},
      1:{'actor':t2,'slot':(10000, 100000)}}

Scene(Plot(slot)).run()


print 'Some statistics about the clusters'
lens=[len(C[c]['indices']) for c in C]
print 'max ',max(lens), 'min ',min(lens)
print 'singletons ',lens.count(1)
print 'doubletons ',lens.count(2)
print 'tripletons ',lens.count(3)


