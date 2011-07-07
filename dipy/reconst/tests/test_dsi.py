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
from scipy.fftpack import fftn, fftshift

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
        
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[0,0,0], snr=None)
    
    #duplicate
    S=np.concatenate([S,S[1:]])
    
    bvals=np.concatenate([bvals,bvals[1:]])
    bvecs=np.concatenate([bvecs,-bvecs[1:]])
    
    bmin=np.sort(bvals)[1]
    bv=np.sqrt(np.floor(bvals/bmin +0.5))
    
    qtable=np.vstack((bv,bv,bv)).T * bvecs
    qtable=np.floor(qtable+0.5)
       
    r = np.sqrt(qtable[:,0]**2+qtable[:,1]**2+qtable[:,2]**2)
    
    #r=fvtk.ren()
    #fvtk.add(r,fvtk.point(qtable,fvtk.red,1,0.1,6,6))
    #fvtk.show(r)
    
    filter_width=16        
    hanning=.5*np.cos(2*r*np.pi/filter_width)
    
    q=qtable+9
    q=q.astype('i8')
    n=515+514
    
    value=S*hanning
        
    Sq=np.zeros((16,16,16))
    for i in range(n):
        Sq[q[i]]=value[i]
        
    Pr=fftshift(np.abs(np.real(fftn(fftshift(Sq),(16,16,16)))))

    #"""
    ren=fvtk.ren()
    vol=fvtk.volume(Pr)
    fvtk.add(ren,vol)
    fvtk.show(ren)
    #"""
    
    #from enthought.mayavi import mlab    
    #Pr=np.interp(Pr,[Pr.min(),Pr.max()],[0,1])    
    #mlab.pipeline.volume(mlab.pipeline.scalar_field(Pr))
    #mlab.show()
    
    from dipy.core.triangle_subdivide import create_unit_sphere, create_half_unit_sphere    
    vertices, edges, faces  = create_unit_sphere(5)
    
    from scipy.ndimage import map_coordinates
    #sp.ndimage.map_coordinates(a, inds, order=1, cval=0, output=bool)
    
    odf = np.zeros(len(vertices))
    #r=np.arange(2.1,6,.2)
    r=np.arange(.1,6,.1)
    
    origin=9
    
    for m in range(len(vertices)):
        
        xi=origin+r*vertices[m,0]
        yi=origin+r*vertices[m,1]
        zi=origin+r*vertices[m,2]        
        
        PrI=map_coordinates(Pr,np.vstack((xi,yi,zi)),order=1)
        for i in range(len(r)):
            odf[m]=odf[m]+PrI[i]*r[i]**2
        

    









