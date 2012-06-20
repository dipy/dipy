import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

import nibabel as nib
from dipy.viz import fvtk
from dipy.data import get_data, get_sphere
from dipy.reconst.recspeed import peak_finding
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dni import DiffusionNabla
from dipy.sims.voxel import SticksAndBall
from scipy.fftpack import fftn, fftshift, ifftn,ifftshift
from dipy.core.triangle_subdivide import create_unit_sphere, create_half_unit_sphere 
from scipy.ndimage import map_coordinates
from dipy.utils.spheremakers import sphere_vf_from

def sim_data(bvals,bvecs,d=0.0015,S0=100,snr=None):
    
    data=np.zeros((14,len(bvals)))
    #isotropic
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[0,0,0], snr=snr)
    data[0]=S.copy()
    #one fiber    
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(30, 0),(90,0),(90,90)], 
                          fractions=[100,0,0], snr=snr)
    data[1]=S.copy()
    #two fibers
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[50,50,0], snr=snr)
    data[2]=S.copy()
    #three fibers
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[33,33,33], snr=snr)
    data[3]=S.copy()
    #three fibers iso
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[23,23,23], snr=snr)
    data[4]=S.copy()
    #three fibers more iso
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[13,13,13], snr=snr)
    data[5]=S.copy()
    #three fibers one at 60
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,33], snr=snr)
    data[6]=S.copy()    
    
    #three fibers one at 90,90 one smaller
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,23], snr=snr)
    data[7]=S.copy()
    
    #three fibers one at 90,90 one even smaller
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,13], snr=snr)
    data[8]=S.copy()
    
    #two fibers one at 60 one even smaller
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[50,30,0], snr=snr)
    data[9]=S.copy()
    
    #two fibers one at 30 one at 60 even smaller
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(20,0),(90,90)], 
                          fractions=[50,50,0], snr=snr)
    data[10]=S.copy()
    
    #one fiber one at 30 but small
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(30, 0),(60,0),(90,90)], 
                          fractions=[60,0,0], snr=snr)
    data[11]=S.copy()
    
    #one fiber one at 30 but even smaller
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[30,0,0], snr=snr)
    data[12]=S.copy()
    
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(30, 0),(60,0),(90,90)], 
                          fractions=[0,0,0], snr=snr)
    data[13]=S.copy()
    
    return data

if __name__ == '__main__':
#def test_dni():
 
    btable=np.loadtxt(get_data('dsi515btable'))    
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    data=sim_data(bvals,bvecs)
    
    dn=DiffusionNabla(data,bvals,bvecs,save_odfs=True)
    pks=dn.pk()
    #assert_array_equal(np.sum(pks>0,axis=1),
    #                   np.array([0, 1, 2, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 0]))
    
    odfs=dn.odfs()
    peaks,inds=peak_finding(odfs[10],dn.odf_faces)
    

    





