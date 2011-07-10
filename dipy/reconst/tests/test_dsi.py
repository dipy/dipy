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
from scipy.fftpack import fftn, fftshift, ifftn,ifftshift
from dipy.core.triangle_subdivide import create_unit_sphere, create_half_unit_sphere 
from scipy.ndimage import map_coordinates
from dipy.utils.spheremakers import sphere_vf_from

def test_dsi():
       
    #'515_32':['/home/eg309/Data/tp2/NIFTI',\
    #                  'dsi_bvals.txt','dsi_bvects.txt','DSI.nii']}
    pass



if __name__ == '__main__':
    
    #volume size
    sz=64
    #shifting
    origin=32
    #hanning width
    filter_width=64.
    #number of signal sampling points
    n=515
    #odf radius
    radius=np.arange(2.1,25,.1) 
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
    S,stics=SticksAndBall(bvals, bvecs, d=0.0040, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[33,33,33], snr=None)
    
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
            odf[m]=odf[m]+PrI[i]*r[i]**2
    
    #"""
    ren=fvtk.ren()
    colors=fvtk.colors(odf,'jet')
    fvtk.add(ren,fvtk.point(vertices,colors,point_radius=.05,theta=8,phi=8))
    fvtk.show(ren)
    #"""
    
    peaks,inds=peak_finding(odf.astype('f8'),faces.astype('uint16'))
    
    print peaks
    a=vertices[inds[0]]
    b=vertices[inds[1]]
    print np.min(np.rad2deg(np.arccos(np.dot(a,b))),90-np.rad2deg(np.arccos(np.dot(a,b))))
    

    







