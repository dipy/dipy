import time
import numpy as np
from dipy.io import dicomreaders as dcm
import dipy.core.generalized_q_sampling as gq
import dipy.core.dti as dt
import dipy.core.track_propagation as tp
from dipy.viz import fos

dname ='/home/eg01/Data_Backup/Data/Eleftherios/CBU090134_METHODS/20090227_154122/Series_003_CBU_DTI_64D_iso_1000'

#dname =  '/home/eg01/Data_Backup/Data/Frank_Eleftherios/frank/20100511_m030y_cbu100624/08_ep2d_advdiff_101dir_DSI'

t1=time.clock()
data,affine,bvals,gradients=dcm.read_mosaic_dir(dname)

t2=time.clock()
print ('load data in %d secs' %(t2-t1))

x,y,z,g = data.shape
print('data shape is ',data.shape)

#calculate QA
gqs=gq.GeneralizedQSampling(data,bvals,gradients)

t3=time.clock()
print ('Generate QA in %d secs' %(t3-t2))

T=tp.FACT_Delta(gqs.QA,gqs.IN,seeds_no=10000).tracks

t4=time.clock()
print ('Create %d QA tracks in %d secs' %(len(T),t4-t3))

#'''
#calculate single tensor
ten=dt.Tensor(data,gradients.T,bvals,thresh=50)
t5=time.clock()
print('Create FA in %d secs' %(t5-t4))
     

IN2=dt.quantize_evecs(ten.evecs)
FA=ten.fa()

T2=tp.FACT_Delta(FA,IN2,seeds_no=10000,qa_thr=0.2).tracks
t6=time.clock()
print ('Create %d FA tracks in %d secs' %(len(T2),t6-t5))

T2=[t+np.array([100,0,0]) for t in T2]

print('Red tracks propagated based on QA')
print('Green tracks propagated based on FA')
r=fos.ren()
fos.add(r,fos.line(T,fos.red))
fos.add(r,fos.line(T2,fos.green))
fos.show(r)
#'''



