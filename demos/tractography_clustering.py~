import pbc
import dipy.core.track_learning as tl
from dipy.core import track_metrics as tm
from dipy.viz import fos
from dipy.io import trackvis as tv
from dipy.core import performance as pf

fname='/home/eg309/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'

streams,hdr=tv.read(fname)

T=[i[0] for i in streams]

tracks=[tm.downsample(t,3) for t in T]

C=pf.local_skeleton_clustering(tracks,d_thr=20)

pbc.save_pickle('local_skeleton_20.pkl',C)

