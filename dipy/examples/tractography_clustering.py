import dipy.core.track_learning as tl
from dipy.core import track_metrics as tm
from dipy.viz import fos
from dipy.io import trackvis as tv
from dipy.io import pickle as pkl
from dipy.core import performance as pf




#fname='/home/eg01/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'

fname='/home/eg309/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'

print 'Loading file...'
streams,hdr=tv.read(fname)

print 'Copying tracks...'
T=[i[0] for i in streams]

print 'Representing tracks using only 3 pts...'
tracks=[tm.downsample(t,3) for t in T]

print 'Deleting unnecessary data...'
del T,streams,hdr

print 'Local Skeleton Clustering...'
C=pf.local_skeleton_clustering(tracks,d_thr=20)
#C=tl.local_skeleton_clustering(tracks,d_thr=20)

print 'Saving Result...'
pkl.save_pickle('local_skeleton_20.pkl',C)

print 'All done.'
