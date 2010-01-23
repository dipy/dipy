


#===================================
import pbc
xyz=np.vstack((x,y,z)).T
xyz=pbc.helix()
import dipy.core.performance as pf
xyzf=pf.approximate_ei_trajectory(xyz)
len(xyz)
len(xyzf)
r=fos.ren()
from dipy.viz import fos
r=fos.ren()
fos.add(r,fos.line(xyz,fos.red))
fos.add(r,fos.line(xyzf,fos.yellow))
fos.show(r)

#======================================
r=fos.ren()
fos.clear(r)
xyz=pbc.sine()
xyzf=pf.approximate_ei_trajectory(xyz)
fos.add(r,fos.line(xyz,fos.red))
colorsf=np.random.rand(len(xyzf)-1,3)
#fos.add(r,fos.line(xyzf,fos.yellow))

for i in range(1,len(xyzf)):
    fos.add(r,fos.line(xyzf[i-1:i+1],colorsf[i-1]))    
   
#=======================================

from dipy.viz import fos
from dipy.core import performance as pf
from dipy.core import track_metrics as tm
from dipy.io import trackvis as tv

r=fos.ren()
fname='/home/eg01/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
streams,hdr=tv.read(fname)

brain1=[i[0] for i in streams]

#fos.clear(r)
#fos.add(r,fos.line(brain1,fos.red))

brain1z=[pf.approximate_ei_trajectory(i,alpha=0.392) for i in brain1]

from dipy.core import track_metrics as tm
b1s=[tm.length(i) for i in brain1]
sum(b1s)
b1zs=[tm.length(i) for i in brain1z]
sum(b1zs)
sum(b1zs)/sum(b1s)

b1sl=[len(i) for i in brain1]
sum(b1sl)
b1zsl=[len(i) for i in brain1z]
sum(b1zsl)

sum(b1zsl)/float(sum(b1sl))

brain1d=[tm.downsample(i,10)+np.array([100,0,0]) for i in brain1]
fos.add(r,fos.line(brain1z[:1000],fos.red))
fos.add(r,fos.line(brain1d[:1000],fos.yellow))

fos.show(r)


#=================================

x=np.linspace(0,100,100)
y=np.zeros(x.shape)
z=y

xyz=np.vstack((x,y,z)).T
r=fos.ren()

xyzn=pf.approximate_ei_trajectory(xyz)
from dipy.viz import fos
r=fos.ren()
fos.add(r,fos.line(xyz,fos.red))
fos.add(r,fos.line(xyzn,fos.yellow))
fos.show(r)

#==================================

from dipy.viz import fos
from dipy.core import performance as pf
from dipy.core import track_metrics as tm
from dipy.io import trackvis as tv

r=fos.ren()
fname = '/home/eg01/Data/PBC/pbc2009icdm/fornix.pkl'
import pbc
a=pbc.load_pickle(fname)
an=pf.approximate_ei_trajectory(a[1])
fos.add(r,fos.line(a[1],fos.yellow))
fos.add(r,fos.line(an,fos.red))
fos.show(r)
len(a[1])
len(an)
fos.show(r)
an=[pf.approximate_ei_trajectory(i) for i in a]
fos.clear(r)
fos.add(r,fos.line(an,fos.yellow))
fos.add(r,fos.line(a,fos.red))
fos.show(r)

#==================================
from dipy.core import performance as pf

p=np.array([0,0,0],dtype=float32)
q=np.array([1,0,0],dtype=float32)
r=0.5

sa=np.array([0.5,1 ,0],dtype=float32)
sb=np.array([0.5,-1,0],dtype=float32)
pf.segment_inside_cylinder(sa,sb,p,q,r)

sa=np.array([0.5,1 ,0],dtype=float32)
sb=np.array([1,1,0],dtype=float32)
pf.segment_inside_cylinder(sa,sb,p,q,r)


sa=np.array([0.,0.2,0],dtype=float32)
sb=np.array([1.,0.2,0],dtype=float32)
pf.segment_inside_cylinder(sa,sb,p,q,r)


sa=np.array([0.2,0.2,0],dtype=float32)
sb=np.array([0.8,0.2,0],dtype=float32)
pf.segment_inside_cylinder(sa,sb,p,q,r)


sa=np.array([0.2,0.2,0],dtype=float32)
sb=np.array([1.8,0.2,0],dtype=float32)
pf.segment_inside_cylinder(sa,sb,p,q,r)


sa=np.array([0.2,0.2,0],dtype=float32)
sb=np.array([1.,0.,0],dtype=float32)
pf.segment_inside_cylinder(sa,sb,p,q,r)

sa=np.array([0.5,0.,0],dtype=float32)
sb=np.array([1.5,0.,0],dtype=float32)
pf.segment_inside_cylinder(sa,sb,p,q,r)


sa=np.array([0.,1.,0],dtype=float32)
sb=np.array([1.,0.,0],dtype=float32)
pf.segment_inside_cylinder(sa,sb,p,q,r)

#=================================

from dipy.viz import fos
from dipy.core import performance as pf
from dipy.core import track_metrics as tm
from dipy.core import track_learning as tl
from dipy.io import trackvis as tv

r=fos.ren()
fname='/home/eg01/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'
streams,hdr=tv.read(fname)

T=[i[0] for i in streams]

T=[tm.downsample(t,3).reshape((1,9)) for t in T]

T=np.concatenate(T)

T=T[:10000]

C,H=tl.local_skeleton(T)

#===================================
import pbc
import dipy.core.track_learning as tl
from dipy.core import track_metrics as tm
from dipy.viz import fos


def test():
    fname='/home/eg309/Data/PBC/pbc2009icdm/fornix.pkl'
    T=pbc.load_pickle(fname)

    tracks=[tm.downsample(t,3) for t in T]

    C=tl.local_skeleton_clustering(tracks,d_thr=10)


import pstats, cProfile
cProfile.run("test()", "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()

'''
r=fos.ren()

for c in C:
    color=np.random.rand(3)
    for i in C[c]['indices']:
        fos.add(r,fos.line(T[i],color))

fos.show(r)
'''

#===================================HERE

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

C=tl.local_skeleton_clustering(tracks,d_thr=10)

pbc.save_pickle('local_skeleton.pkl',C)

'''
r=fos.ren()

for c in C:
    color=np.random.rand(3)
    for i in C[c]['indices']:
        fos.add(r,fos.line(pf.approximate_ei_trajectory(T[i]),color))

fos.show(r)
'''
r=fos.ren()

for c in C:
    color=np.random.rand(3)
    for i in C[c]['indices']:
        fos.add(r,fos.line(T2[i],color))

fos.show(r)

#========================================

import pbc
import dipy.core.track_learning as tl
from dipy.core import track_metrics as tm
from dipy.viz import fos
from dipy.io import trackvis as tv
from dipy.core import performance as pf

fname='/home/eg309/Data/PBC/pbc2009icdm/brain1/brain1_scan1_fiber_track_mni.trk'

streams,hdr=tv.read(fname)

T=[i[0] for i in streams]

T=[pf.approximate_ei_trajectory(t) for t in T]

C=pbc.load_pickle('local_skeleton.pkl')

r=fos.ren()

T2=[]
color2=np.zeros((len(C.keys()),3))
for c in C:
    color=np.random.rand(3)
    for i in C[c]['indices']:
        #fos.add(r,fos.line(T[i],color))
        T2.append(T[i])
        color2[c]=color
        
        
fos.add(r,fos.line(T2,color2))
        

fos.show(r)

