from __future__ import division, print_function, absolute_import

import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.reconst.eit import DiffusionNablaModel, EquatorialInversionModel
from dipy.sims.voxel import SticksAndBall
from dipy.utils.spheremakers import sphere_vf_from
from dipy.data import get_data
from dipy.core.sphere import unique_edges

def sim_data(bvals,bvecs,d=0.0015,S0=100,snr=None):

    descr=np.zeros(13).tolist()
    data=np.zeros((13,len(bvals)))

    descr[0]=('isotropic',0)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[0,0,0], snr=snr)
    data[0]=S.copy()
    descr[1]=('one fiber',1)    
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(30, 0),(90,0),(90,90)], 
                          fractions=[100,0,0], snr=snr)
    data[1]=S.copy()
    descr[2]=('two fibers',2)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[50,50,0], snr=snr)
    data[2]=S.copy()
    descr[3]=('three fibers',3)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[33,33,33], snr=snr)
    data[3]=S.copy()
    descr[4]=('three fibers iso',3)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[23,23,23], snr=snr)
    data[4]=S.copy()
    descr[5]=('three fibers more iso',3)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(90,0),(90,90)], 
                          fractions=[13,13,13], snr=snr)
    data[5]=S.copy()
    descr[6]=('three fibers one at 60',3)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,33], snr=snr)
    data[6]=S.copy()        
    descr[7]=('three fibers one at 90,90 one smaller',3)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,23], snr=snr)
    data[7]=S.copy()    
    descr[8]=('three fibers one at 60, one at 90 and one smaller',3)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[33,33,13], snr=snr)
    data[8]=S.copy()    
    descr[9]=('two fibers at 60 one smaller',2)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[50,30,0], snr=snr)
    data[9]=S.copy()    
    descr[10]=('two fibers one at 30',2)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(45,0)], 
                          fractions=[50,50], snr=snr)
    data[10]=S.copy()    
    descr[11]=('one fiber one at 30 but small iso',1)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(30, 0),(60,0),(90,90)], 
                          fractions=[60,0,0], snr=snr)
    data[11]=S.copy()    
    descr[12]=('one fiber one at 30 but even smaller iso',1)
    S,stics=SticksAndBall(bvals, bvecs, d, S0, 
                          angles=[(0, 0),(60,0),(90,90)], 
                          fractions=[30,0,0], snr=snr)
    data[12]=S.copy()
        
    return data, descr

def test_dni_eit():
 
    btable=np.loadtxt(get_data('dsi515btable'))    
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    data,descr=sim_data(bvals,bvecs)   
    #load odf sphere
    vertices,faces = sphere_vf_from('symmetric724')
    edges = unique_edges(faces)
    #create the sphere
    odf_sphere=(vertices,faces)
    dn=DiffusionNablaModel(bvals,bvecs,odf_sphere)
    dn.relative_peak_threshold = 0.5
    dn.angular_distance_threshold = 20
    dnfit=dn.fit(data)
    print('DiffusionNablaModel')
    for i,d in enumerate(data):
        print(descr[i], np.sum(dnfit.peak_values[i]>0))
    ei=EquatorialInversionModel(bvals,bvecs,odf_sphere)
    ei.relative_peak_threshold = 0.3
    ei.angular_distance_threshold = 15
    ei.set_operator('laplacian')
    eifit = ei.fit(data,return_odf=True)
    print('EquatorialInversionModel')
    for i,d in enumerate(data):
        print(descr[i], np.sum(eifit.peak_values[i]>0))
        assert_equal(descr[i][1], np.sum(eifit.peak_values[i]>0))
