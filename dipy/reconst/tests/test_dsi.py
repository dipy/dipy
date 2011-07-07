import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

import nibabel as nib
from dipy.viz import fvtk
from dipy.data import get_data, get_sphere
from dipy.reconst.recspeed import peak_finding
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrumImaging
from dipy.reconst.sims import SticksAndBall

def test_dsi():
       
    #'515_32':['/home/eg309/Data/tp2/NIFTI',\
    #                  'dsi_bvals.txt','dsi_bvects.txt','DSI.nii']}
    pass



if __name__ == '__main__':
    
    bvals=np.loadtxt('/home/eg309/Data/tp2/NIFTI/dsi_bvals.txt')
    bvecs=np.loadtxt('/home/eg309/Data/tp2/NIFTI/dsi_bvects.txt')
    bvecs[-1]=np.array([0,0,0])
    
    swap=bvals[0];bvals[0]=bvals[-1];bvals[-1]=swap
    swap=bvecs[0];bvecs[0]=bvecs[-1];bvecs[-1]=swap
    
    S,stics=SticksAndBall(bvals,bvecs,snr=None)
    
    #duplicate
    S=np.concatenate([S,S[1:]])
    bvals=np.concatenate([bvals,bvals[1:]])
    bvecs=np.concatenate([bvecs,-bvecs[1:]])
    
    qtable=np.vstack((bvals,bvals,bvals)).T * bvecs
    
    bmin=np.sort(bvals)[1]
    bvals=bvals/bmin
           
    #r=fvtk.ren()
    #fvtk.add(r,fvtk.point(qtable/qtable.max(),fvtk.red,1,0.02,6,6))
    #fvtk.show(r)
    
    r = np.sqrt(qtable[1:,0]**2+qtable[1:,1]**2+qtable[1:,2]**2)
    filter_width=1028.
    hanning=np.cos(2.*r*np.pi/filter_width)
    
    s=hanning*S[1:]/S[0]
    E=np.vstack((s,s,s)).T
        
    pdf=np.fft.ifftn(E)
    
    from scipy import fftpack
    pdf2=fftpack.ifftn(E)
    pdf3=fftpack.ifftshift(pdf2)
    
    











