import numpy as np
from dipy.viz import fos        
from dipy.core import track_performance as pf

tracks=[np.array([[0,0,0],[1,0,0,],[2,0,0]]),            
        np.array([[3,0,0],[3.5,1,0],[4,2,0]]),
        np.array([[3.2,0,0],[3.7,1,0],[4.4,2,0]]),
        np.array([[3.4,0,0],[3.9,1,0],[4.6,2,0]]),
        np.array([[0,0.2,0],[1,0.2,0],[2,0.2,0]]),
        np.array([[2,0.2,0],[1,0.2,0],[0,0.2,0]]),
        np.array([[0,0,0],[0,1,0],[0,2,0]]),
        np.array([[0.2,0,0],[0.2,1,0],[0.2,2,0]]),
        np.array([[-0.2,0,0],[-0.2,1,0],[-0.2,2,0]]),
        np.array([[0,1.5,0],[1,1.5,0,],[6,1.5,0]]),
        np.array([[0,1.8,0],[1,1.8,0,],[6,1.8,0]]),
        np.array([[0,0,0],[2,2,0],[4,4,0]])]

                                    
tracks=[t.astype(np.float32) for t in tracks]

C=pf.larch_3split(tracks,None,0.5)

r=fos.ren()
fos.add(r,fos.line(tracks,fos.red))
#fos.show(r)

for c in C:
    color=np.random.rand(3)
    for i in C[c]['indices']:
        fos.add(r,fos.line(tracks[i]+np.array([8.,0.,0.]),color))
        fos.add(r,fos.line(tracks[i]+np.array([16.,0.,0.]),color))

    fos.add(r,fos.line(C[c]['rep3']/C[c]['N']+np.array([16.,0.,0.]),fos.white))
        
    
fos.show(r)

"""

print len(C)

C=pf.larch_3merge(C,0.5)

print len(C)

for c in C:
    color=np.random.rand(3)
    for i in C[c]['indices']:
        fos.add(r,fos.line(tracks[i]+np.array([14.,0.,0.]),color))
#fos.show(r)

for c in C:    
    fos.add(r,fos.line(C[c]['rep3']/C[c]['N']+np.array([14.,0.,0.]),fos.white))
fos.show(r)
"""





