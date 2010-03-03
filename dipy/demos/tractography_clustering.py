from dipy.core import track_metrics as tm
from dipy.viz import fos
from dipy.io import trackvis as tv
from dipy.io import pickle as pkl
from dipy.core import track_performance as pf
import time
import numpy as np


#fname='/home/eg01/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
fname='/home/eg309/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'

print 'Loading file...'
streams,hdr=tv.read(fname)

print 'Copying tracks...'
T=[i[0] for i in streams]

print 'Representing tracks using only 3 pts...'
tracks=[tm.downsample(t,3) for t in T]

print 'Deleting unnecessary data...'
del streams,hdr

print 'Hidden Structure Clustering...'
now=time.clock()
C=pf.local_skeleton_clustering(tracks,d_thr=20)
print 'Done in', time.clock()-now,'s.'

print 'Reducing the number of points...'
T=[pf.approximate_ei_trajectory(t) for t in T]

print 'Showing initial dataset.'
r=fos.ren()
fos.add(r,fos.line(T,fos.white,opacity=0.1))
fos.show(r)

print 'Showing dataset after clustering.'
fos.clear(r)
colors=np.zeros((len(T),3))
for c in C:
    color=np.random.rand(1,3)
    for i in C[c]['indices']:
        colors[i]=color
fos.add(r,fos.line(T,colors,opacity=1))
fos.show(r)

print 'Some statistics about the clusters'
lens=[len(C[c]['indices']) for c in C]
print 'max ',max(lens), 'min ',min(lens)
print 'singletons ',lens.count(1)
print 'doubletons ',lens.count(2)
print 'tripletons ',lens.count(3)



