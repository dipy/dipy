import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

import nibabel as nib
from dipy.viz import fvtk
from dipy.data import get_data, get_sphere
from dipy.reconst.recspeed import peak_finding
from dipy.reconst.gqi import GeneralizedQSampling
from dipy.reconst.dsi import DiffusionSpectrum
from dipy.reconst.sims import SticksAndBall
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
    assert_equal(np.sum(ds.pdf(S)-pdf0),0)
    assert_equal(np.sum(ds.odf(ds.pdf(S))-odf0),0)
    
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
    
    

if __name__ == '__main__':

    
    test_dsi()
    #stop
    
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
    
    matrix=np.loadtxt('/home/eg309/Devel/dipy/dipy/data/grad_514.txt')
    img=nib.load('/home/eg309/Data/project01_dsi/connectome_0001/tp1/RAWDATA/OUT/mr000001.nii.gz')    
    btable=np.loadtxt('/home/eg309/Devel/dipy/dipy/data/dsi515_b_table.txt')
    
    bv=btable[:,0]
    bmin=np.sort(bv)[1]
    bv=np.sqrt(bv/bmin)
    qtable=np.vstack((bv,bv,bv)).T*btable[:,1:]
    qtable=np.floor(qtable+.5)

    #data=img.get_data()    
    #S=img.get_data()[38,50,20]#[96/2,96/2,20]
    bvals=btable[:,0]
    bvecs=btable[:,1:]
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(60,0),(90,90)], fractions=[23,23,23], snr=None)
    
    S2=S.copy()
    S2=S2.reshape(1,len(S))
    
    ds=DiffusionSpectrum(S2,bvals,bvecs)    
    tpr=ds.pdf(S)
    todf=ds.odf(tpr)
        
    """
    #show projected signal
    Bvecs=np.concatenate([bvecs[1:],-bvecs[1:]])
    X0=np.dot(np.diag(np.concatenate([S[1:],S[1:]])),Bvecs)    
    ren=fvtk.ren()
    fvtk.add(ren,fvtk.point(X0,fvtk.yellow,1,2,16,16))    
    fvtk.show(ren)
    """
    #qtable=5*matrix[:,1:]
    
    #calculate radius for the hanning filter
    r = np.sqrt(qtable[:,0]**2+qtable[:,1]**2+qtable[:,2]**2)
        
    #setting hanning filter width and hanning
    hanning=.5*np.cos(2*np.pi*r/filter_width)
    
    #center and index in q space volume
    q=qtable+origin
    q=q.astype('i8')
    
    #apply the hanning filter
    values=S*hanning
    
    """
    #plot q-table
    ren=fvtk.ren()
    colors=fvtk.colors(values,'jet')
    fvtk.add(ren,fvtk.point(q,colors,1,0.1,6,6))
    fvtk.show(ren)
    """    
    
    #create the signal volume    
    Sq=np.zeros((sz,sz,sz))
    for i in range(n):
        print q[i],values[i]
        Sq[q[i][0],q[i][1],q[i][2]]+=values[i]
    
    #apply fourier transform
    Pr=fftshift(np.abs(np.real(fftn(fftshift(Sq),(sz,sz,sz)))))
    
    #"""
    ren=fvtk.ren()
    vol=fvtk.volume(Pr)
    fvtk.add(ren,vol)
    fvtk.show(ren)
    #"""
    
    """
    from enthought.mayavi import mlab
    mlab.pipeline.volume(mlab.pipeline.scalar_field(Sq))
    mlab.show()
    """
    
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
    
    #"""
    ren=fvtk.ren()
    colors=fvtk.colors(odf,'jet')
    fvtk.add(ren,fvtk.point(vertices,colors,point_radius=.05,theta=8,phi=8))
    fvtk.show(ren)
    #"""
    
    #odf=odf*hanning
    
    peaks,inds=peak_finding(odf.astype('f8'),faces.astype('uint16'))
    
    print peaks
    print peaks/peaks.min()
    a=vertices[inds[0]]
    b=vertices[inds[1]]
    print np.rad2deg(np.arccos(np.dot(a,b)))    
    
    
    ismallp=np.where(peaks/peaks.min()<2.)
    print ismallp                                               
    l=ismallp[0][0]
    print l
    
    
    """
    smoothing=np.dot(vertices,vertices.T)
    
    odf2=np.dot(smoothing**10,odf)
    odf2=odf2/odf2.min()
    
    peaks,inds=peak_finding(odf2.astype('f8'),faces.astype('uint16'))
    print peaks
    
    ren=fvtk.ren()
    colors=fvtk.colors(odf2,'jet')
    fvtk.add(ren,fvtk.point(vertices,colors,point_radius=.05,theta=8,phi=8))
    fvtk.show(ren)    
    """    
    
    #Pr[Pr<500]=0
    #r=fvtk.ren()
    #fvtk.add(r,fvtk.volume(Pr))
    #fvtk.show(r)
    







