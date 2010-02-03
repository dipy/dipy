import dipy.core.track_performance as pf
import dipy.io.pickle as pkl
import dipy.core.track_metrics as tm
from dipy.viz import fos
import numpy as np


fname='/home/eg01/Data/tmp/pbc_training_set.pkl'

T=pkl.load_pickle(fname)

print 'Reducing the number of points...'
T=[pf.approximate_ei_trajectory(t) for t in T]

print 'Reducing further to 3pts...'
T2=[tm.downsample(t,3) for t in T]

print 'Clustering ...'
C=pf.local_skeleton_clustering(T2,20.)


print 'Showing initial dataset.'
r=fos.ren()
fos.add(r,fos.line(T,fos.white,opacity=1))
fos.show(r)

T=[t + np.array([0,-120,0]) for t in T]

print 'Showing dataset after clustering.'
#fos.clear(r)
colors=np.zeros((len(T),3))
for c in C:
    color=np.random.rand(1,3)
    for i in C[c]['indices']:
        colors[i]=color
fos.add(r,fos.line(T,colors,opacity=1))
fos.show(r)

print 'Some statistics about the clusters'
print 'Number of clusters',len(C.keys())
lens=[len(C[c]['indices']) for c in C]
print 'max ',max(lens), 'min ',min(lens)
    
print 'singletons ',lens.count(1)
print 'doubletons ',lens.count(2)
print 'tripletons ',lens.count(3)

