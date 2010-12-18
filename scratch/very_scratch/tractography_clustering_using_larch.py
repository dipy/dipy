import time
import os

import numpy as np

from nibabel import trackvis as tv

from dipy.viz import fos
from dipy.io import pickles as pkl
from dipy.core import track_learning as tl
from dipy.core import track_performance as pf
from dipy.core import track_metrics as tm

fname='/home/eg01/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
C_fname='/tmp/larch_tree.pkl'
appr_fname='/tmp/larch_tracks.trk'


print 'Loading trackvis file...'
streams,hdr=tv.read(fname)

print 'Copying tracks...'
tracks=[i[0] for i in streams]

#tracks=tracks[:1000]

#print 'Deleting unnecessary data...'
del streams#,hdr

if not os.path.isfile(C_fname):

    print 'Starting LARCH ...'
    tim=time.clock()
    C,atracks=tl.larch(tracks,[50.**2,20.**2,5.**2],True,True)
    #tracks=[tm.downsample(t,3) for t in tracks]
    #C=pf.local_skeleton_clustering(tracks,20.)
    print 'Done in total of ',time.clock()-tim,'seconds.'

    print 'Saving result...'
    pkl.save_pickle(C_fname,C)
    
    streams=[(i,None,None)for i in atracks]
    tv.write(appr_fname,streams,hdr)

else:

    print 'Loading result...'
    C=pkl.load_pickle(C_fname)

skel=[]
for c in C:
    skel.append(C[c]['repz'])
    
print 'Showing dataset after clustering...'
r=fos.ren()
fos.clear(r)
colors=np.zeros((len(skel),3))
for (i,s) in enumerate(skel):

    color=np.random.rand(1,3)
    colors[i]=color

fos.add(r,fos.line(skel,colors,opacity=1))
fos.show(r)
