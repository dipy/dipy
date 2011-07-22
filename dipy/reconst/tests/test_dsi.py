import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

import nibabel as nib
from dipy.viz import fvtk
from dipy.data import get_data, get_sphere
from dipy.reconst.recspeed import peak_finding
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.sims.voxel import SticksAndBall
from scipy.fftpack import fftn, fftshift, ifftn,ifftshift
from dipy.core.triangle_subdivide import create_unit_sphere, create_half_unit_sphere 
from scipy.ndimage import map_coordinates
from dipy.utils.spheremakers import sphere_vf_from


def standard_dsi_algorithm(S,bvals,bvecs):
    #volume size
    sz=16
    #shifting
    origin=8
    #hanning width
    filter_width=32.
    #number of signal sampling points
    n=515

    #odf radius
    #radius=np.arange(2.1,30,.1)
    radius=np.arange(2.1,6,.2)
    #radius=np.arange(.1,6,.1)   
    
    bv=bvals
    bmin=np.sort(bv)[1]
    bv=np.sqrt(bv/bmin)
    qtable=np.vstack((bv,bv,bv)).T*bvecs
    qtable=np.floor(qtable+.5)
   
    #calculate radius for the hanning filter
    r = np.sqrt(qtable[:,0]**2+qtable[:,1]**2+qtable[:,2]**2)
        
    #setting hanning filter width and hanning
    hanning=.5*np.cos(2*np.pi*r/filter_width)
    
    #center and index in q space volume
    q=qtable+origin
    q=q.astype('i8')
    
    #apply the hanning filter
    values=S*hanning
    
    #create the signal volume    
    Sq=np.zeros((sz,sz,sz))
    for i in range(n):        
        Sq[q[i][0],q[i][1],q[i][2]]+=values[i]
    
    #apply fourier transform
    Pr=fftshift(np.abs(np.real(fftn(fftshift(Sq),(sz,sz,sz)))))

    #vertices, edges, faces  = create_unit_sphere(5)    
    vertices, faces = sphere_vf_from('symmetric362')           
    odf = np.zeros(len(vertices))
        
    for m in range(len(vertices)):
        
        xi=origin+radius*vertices[m,0]
        yi=origin+radius*vertices[m,1]
        zi=origin+radius*vertices[m,2]
        
        PrI=map_coordinates(Pr,np.vstack((xi,yi,zi)),order=1)
        for i in range(len(radius)):
            odf[m]=odf[m]+PrI[i]*radius[i]**2
   
    peaks,inds=peak_finding(odf.astype('f8'),faces.astype('uint16'))

    return Pr,odf,peaks

def test_dsi():
 
    btable=np.loadtxt(get_data('dsi515btable'))    
    bvals=btable[:,0]
    bvecs=btable[:,1:]        
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[50,50,0], snr=None)    
    pdf0,odf0,peaks0=standard_dsi_algorithm(S,bvals,bvecs)    
    S2=S.copy()
    S2=S2.reshape(1,len(S))    
    ds=DiffusionSpectrum(S2,bvals,bvecs)    
    assert_almost_equal(np.sum(ds.pdf(S)-pdf0),0)
    assert_almost_equal(np.sum(ds.odf(ds.pdf(S))-odf0),0)
    
    #compare gfa
    psi=odf0/odf0.max()
    numer=len(psi)*np.sum((psi-np.mean(psi))**2)
    denom=(len(psi)-1)*np.sum(psi**2) 
    GFA=np.sqrt(numer/denom)    
    assert_almost_equal(ds.gfa()[0],GFA)
    
    #compare indices
    #print ds.ind()    
    #print peak_finding(odf0,odf_faces)
    #print peaks0
    data=np.zeros((3,3,3,515))
    data[:,:,:]=S    
    ds=DiffusionSpectrum(data,bvals,bvecs)
    
    ds2=DiffusionSpectrum(data,bvals,bvecs,auto=False)
    r = np.sqrt(ds2.qtable[:,0]**2+ds2.qtable[:,1]**2+ds2.qtable[:,2]**2)    
    ds2.filter=.5*np.cos(2*np.pi*r/32)
    ds2.fit()
    assert_almost_equal(np.sum(ds2.qa()-ds.qa()),0)
    
    #1 fiber
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[100,0,0], snr=None)   
    ds=DiffusionSpectrum(S.reshape(1,len(S)),bvals,bvecs)
    QA=ds.qa()
    assert_equal(np.sum(QA>0),1)
    
    #2 fibers
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[50,50,0], snr=None)   
    ds=DiffusionSpectrum(S.reshape(1,len(S)),bvals,bvecs)
    QA=ds.qa()
    assert_equal(np.sum(QA>0),2)
    
    #3 fibers
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[33,33,33], snr=None)   
    ds=DiffusionSpectrum(S.reshape(1,len(S)),bvals,bvecs)
    QA=ds.qa()
    assert_equal(np.sum(QA>0),3)
    
    #isotropic
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[0,0,0], snr=None)   
    ds=DiffusionSpectrum(S.reshape(1,len(S)),bvals,bvecs)
    QA=ds.qa()
    assert_equal(np.sum(QA>0),0)
    
    

    
if __name__ == '__main__':

    #fname='/home/eg309/Data/project01_dsi/connectome_0001/tp1/RAWDATA/OUT/mr000001.nii.gz'
    #fname='/home/eg309/Data/project02_dsi/PH0005/tp1/RAWDATA/OUT/PH0005_1.MR.5_100.ima.nii.gz'
    fname='/home/eg309/Data/project03_dsi/tp2/RAWDATA/OUT/mr000001.nii.gz'
    
    
    import nibabel as nib
    from dipy.reconst.dsi import DiffusionSpectrum
    from dipy.reconst.dti import Tensor
    from dipy.data import get_data
    
    btable=np.loadtxt(get_data('dsi515btable'))
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    img=nib.load(fname)
    data=img.get_data()
    print data.shape   
    
    mask=data[:,:,:,0]>50
    #D=data[20:90,20:90,18:22]
    #D=data[40:44,40:44,18:22]    
    #del data
    D=data
    
    from time import time
    
    t0=time()    
    ds=DiffusionSpectrum(D,bvals,bvecs,mask=mask)
    t1=time()
    print t1-t0,' secs'
    
    GFA=ds.gfa()
    
    t2=time()
    ten=Tensor(D,bvals,bvecs,mask=mask)
    t3=time()
    print t3-t2,' secs'
    
    FA=ten.fa()
    
    from dipy.tracking.propagation import EuDX
    
    IN=ds.ind()
    
    eu=EuDX(ten.fa(),IN[:,:,:,0],seeds=10000,a_low=0.2)
    tracks=[e for e in eu]
    
    #FAX=np.zeros(IN.shape)
    #for i in range(FAX.shape[-1]):
    #    FAX[:,:,:,i]=GFA
    
    eu2=EuDX(ds.gfa(),IN[:,:,:,0],seeds=10000,a_low=0.2)
    tracks2=[e for e in eu2]
    
    """
    from dipy.viz import fvtk
    r=fvtk.ren()
    fvtk.add(r,fvtk.line(tracks,fvtk.red))
    fvtk.add(r,fvtk.line(tracks2,fvtk.green))
    fvtk.show(r)
    """




