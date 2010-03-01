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

#C=pf.larch_fast_split(tracks,None,0.5**2)        
C=pf.larch_fast_split(tracks,None,True,0.5)        

r=fos.ren()
fos.add(r,fos.line(tracks,fos.red))
fos.show(r)

for c in C:
    color=np.random.rand(3)
    for i in C[c]['indices']:
        fos.add(r,fos.line(tracks[i],color))
fos.show(r)

for c in C:    
    fos.add(r,fos.line(C[c]['rep3']/C[c]['N'],fos.white))
fos.show(r)

