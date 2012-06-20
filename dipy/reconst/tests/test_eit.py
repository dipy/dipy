import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.reconst.eit import DiffusionNablaModel
from dipy.sims.voxel import SticksAndBall
from dipy.utils.spheremakers import sphere_vf_from
from dipy.data import get_data
from dipy.core.geometry import reduce_antipodal, unique_edges

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

def test_dni():
 
    btable=np.loadtxt(get_data('dsi515btable'))    
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    data=sim_data(bvals,bvecs)   
    #load odf sphere
    vertices,faces = sphere_vf_from('symmetric724')
    edges = unique_edges(faces)
    half_vertices,half_edges,half_faces=reduce_antipodal(vertices,faces)
    #create the sphere
    odf_sphere=(vertices,faces)
    dn=DiffusionNablaModel(bvals,bvecs,odf_sphere)
    dnfit=dn.fit(data)
    pks=dnfit.peak_values
    
    #assert_array_equal(np.sum(pks>0,axis=1),
    #                   np.array([0, 1, 2, 3, 3, 3, 3, 3, 3, 2, 2, 1, 1, 0]))


