import time,os
import numpy as np
import dipy.core.performance as pf
import dipy.io.pickle as pkl
import dipy.core.track_metrics as tm
import dipy.core.track_learning as tl
import dipy.io.trackvis as tv
from dipy.viz import fos


#============================================================

d_thr=20.
fname_ts='/home/eg01/Data/tmp/pbc_training_set.pkl'
fname='/home/eg01/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
cfname='/home/eg01/Data/tmp/C_brain1_scan1.pkl'

#============================================================

print 'Loading PBC training set data.'
TS=pkl.load_pickle(fname_ts)

print 'Reducing the number of points...'
TS=[pf.approximate_ei_trajectory(t) for t in TS]

print 'Reducing to 3 points...'
TS2=[tm.downsample(t,3) for t in TS]

print 'Clustering ...'
CTS=pf.local_skeleton_clustering(TS2,d_thr)

print 'Showing unclustered and clustered TS together'
r=fos.ren()
fos.add(r,fos.line(TS,fos.white,opacity=1))
TS=[t + np.array([0,-120,0]) for t in TS]
colors=np.zeros((len(TS),3))

colormap=np.zeros((len(CTS.keys()),3))
for c in CTS:
    color=np.random.rand(1,3)
    colormap[c]=color
    for i in CTS[c]['indices']:
        colors[i]=color
fos.add(r,fos.line(TS,colors,opacity=1))
fos.show(r)

print 'Some statistics about the clusters in TS'
print 'Number of clusters',len(CTS.keys())
lens=[len(CTS[c]['indices']) for c in CTS]
print 'max ',max(lens), 'min ',min(lens)
    
print 'singletons ',lens.count(1)
print 'doubletons ',lens.count(2)
print 'tripletons ',lens.count(3)


#============================================================

print 'Loading Brain1 Scan1 ...'
streams,hdr=tv.read(fname)
print 'Copying tracks...'
T=[i[0] for i in streams]
print 'Reducing the number of points...'
T=[pf.approximate_ei_trajectory(t) for t in T]
print 'Deleting unnecessary data...'
del streams,hdr

print 'Check if file already exists'
if os.path.isfile(cfname):

    print cfname + ' exists.'
    print 'Loading ...'

    C=pkl.load_pickle(cfname)
    
else:
    
    print cfname + ' doesn\'t exists. Creating ...' 

    print 'Representing tracks using only 3 pts...'
    T2=[tm.downsample(t,3) for t in T]

    print 'Dimensionality Reduction'
    now=time.clock()
    C=pf.local_skeleton_clustering(T2,d_thr)
    print 'Done in ', time.clock()-now,'s.'
    print 'Saving Result...'
    pkl.save_pickle(cfname,C)


#============================================================

print 'Finding near_clusters ...'

near=[]
for cts in CTS:

    near.append(tl.near_clusters(cts,CTS,C,1))

print near




print colormap

for cts in CTS:
    for n in near[cts]:

        vizT=[T[i]+ np.array([120,-120,0]) for i in C[n]['indices']]

        fos.add(r,fos.line(vizT,colormap[cts]))
        #fos.add(r,fos.line(vizT,fos.blue,opacity=0.1))

    fos.show(r)


#fos.show(r)

