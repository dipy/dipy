from dipy.core import track_metrics as tm
from dipy.viz import fos
from dipy.io import trackvis as tv
from dipy.io import pickle as pkl
from dipy.core import track_performance as pf
import time
import numpy as np


fname='/home/eg01/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
#fname='/home/eg309/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
tree_fname='/home/eg01/Data/tmp/larch_tree.pkl'

print 'Loading trackvis file...'
streams,hdr=tv.read(fname)

print 'Copying tracks...'
T=[i[0] for i in streams]

print 'Deleting unnecessary data...'
del streams,hdr

print '3track representation ...'
tracks3=[tm.downsample(t,3) for t in T]


print 'LARCH in process...'
tim=time.clock()
#C=pf.larch(tracks3,split_thrs=[40**2,15**2,5.**2],info=True)
C=pf.larch(tracks3,split_thrs=[50**2,20**2,10.**2],info=True)
print 'Done in ',time.clock()-tim,'seconds.'

print 'Saving result...'
pkl.save_pickle(tree_fname,C)

print 'Reducing the number of necessary points on a track...'
T=[pf.approximate_ei_trajectory(t) for t in T]

#print 'Loading result...'
#C=pkl.load_pickle(tree_fname)

skel=[]
for c in C:

    if C[c]['N']> 100:
        skel_tracks=[T[i] for i in  C[c]['indices']]
        skel.append(skel_tracks[pf.most_similar_track_zhang(skel_tracks)[0]])

print 'Showing dataset after clustering.'
r=fos.ren()
fos.clear(r)
colors=np.zeros((len(skel),3))
for (i,s) in enumerate(skel):

    color=np.random.rand(1,3)
    colors[i]=color

fos.add(r,fos.line(skel,colors,opacity=1))
fos.show(r)
