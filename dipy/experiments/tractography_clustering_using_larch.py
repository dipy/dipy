from dipy.viz import fos
from dipy.io import trackvis as tv
from dipy.io import pickle as pkl
from dipy.core import track_learning as tl

import time
import numpy as np
import os

fname='/home/eg01/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
C_fname='/home/eg01/Data/tmp/larch_tree.pkl'
appr_fname='/home/eg01/Data/tmp/larch_tracks.trk'


print 'Loading trackvis file...'
streams,hdr=tv.read(fname)

print 'Copying tracks...'
tracks=[i[0] for i in streams]

tracks=tracks[:1000]

#print 'Deleting unnecessary data...'
del streams#,hdr

if not os.path.isfile(C_fname):

    print 'Starting LARCH ...'
    tim=time.clock()
    C,atracks=tl.larch(tracks,[50.**2,20.**2,5.**2],True,True)
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
