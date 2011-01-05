import time
import numpy as np
import nibabel as nib
import dipy.reconst.gqi as gqi
import dipy.reconst.dti as dti
from dipy.data import get_data

#
#The datasets need to have isotropic voxel size.
#

fimg,fbvals,fbvecs=get_data('small_101D')
img=nib.load(fimg)
data=img.get_data()
affine=img.get_affine()
bvals=np.loadtxt(fbvals)
gradients=np.loadtxt(fbvecs).T

#calculate QA
gqs=gqi.GeneralizedQSampling(data,bvals,gradients)
print('gqs.QA.shape ',gqs.qa().shape)

#calculate single tensor
ten=dti.Tensor(data,bvals,gradients,thresh=50)
print('ten.FA.shape ',ten.fa().shape)
    
''' 
T2=tp.FACT_Delta(ten.FA,ten.IN,seeds_no=10000,qa_thr=0.2).tracks
t6=time.clock()
print ('Create %d FA tracks in %d secs' %(len(T2),t6-t5))

T2=[t+np.array([100,0,0]) for t in T2]

print('Red tracks propagated based on QA')
print('Green tracks propagated based on FA')
r=fos.ren()
fos.add(r,fos.line(T,fos.red))
fos.add(r,fos.line(T2,fos.green))
fos.show(r)
'''



