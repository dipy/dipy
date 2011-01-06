import time
import numpy as np
import nibabel as nib
import dipy.reconst.gqi as gqi
import dipy.reconst.dti as dti
from dipy.tracking.propagation import EuDX
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


#calculate single tensor
ten=dti.Tensor(data,bvals,gradients,thresh=50)
FA=ten.fa()
print('FA.shape (%d,%d,%d)' % FA.shape)

eu=EuDX(a=FA,ind=ten.ind(),seed_no=10000,a_low=.2)

ten_tracks=[track for track in eu]
print('Number of FA tracks %d' % len(ten_tracks))

ten_tracks_asobj=np.array(ten_tracks,dtype=np.object)
np.save('ten_tracks.npy',ten_tracks_asobj)
print('FA tracks saved in ten_tracks.npy')

#calculate QA
gqs=gqi.GeneralizedQSampling(data,bvals,gradients)
QA=gqs.qa()
print('QA.shape (%d,%d,%d,%d)' % QA.shape)

eu2=EuDX(a=QA,ind=gqs.ind(),seed_no=10000,a_low=.0239)

gqs_tracks=[track for track in eu2]
print('Number of QA tracks %d' % len(gqs_tracks))

gqs_tracks_asobj=np.array(gqs_tracks,dtype=np.object)
np.save('gqs_tracks.npy',ten_tracks_asobj)
print('QA tracks saved in gqs_tracks.npy')




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



