import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_almost_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.data import get_data
from dipy.reconst.recspeed import peak_finding
from dipy.reconst.dsi import DiffusionSpectrumModel
from dipy.sims.voxel import SticksAndBall
from scipy.fftpack import fftn, fftshift
from scipy.ndimage import map_coordinates
from dipy.core.sphere import unique_edges
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
    #vertices, faces = sphere_vf_from('symmetric362')           
    vertices, faces = sphere_vf_from('symmetric724')           
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





    #load odf sphere
    vertices,faces = sphere_vf_from('symmetric724')
    edges = unique_edges(faces)

    #load bvals and gradients
    btable=np.loadtxt(get_data('dsi515btable'))    
    bvals=btable[:,0]
    bvecs=btable[:,1:]        
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[50,50,0], snr=None)    
    #pdf0,odf0,peaks0=standard_dsi_algorithm(S,bvals,bvecs)    
    S2=S.copy()
    S2=S2.reshape(1,len(S)) 
    
    odf_sphere=(vertices,faces)
    ds=DiffusionSpectrumModel( bvals, bvecs, odf_sphere)    
    dsfit=ds.fit(S)
    assert_equal((dsfit.peak_values>0).sum(),3)

    #change thresholds
    ds.relative_peak_threshold = 0.5
    ds.angular_distance_threshold = 30
    dsfit = ds.fit(S)
    assert_equal((dsfit.peak_values>0).sum(),2)

    #assert_almost_equal(np.sum(ds.pdf(S)-pdf0),0)
    #assert_almost_equal(np.sum(ds.odf(ds.pdf(S))-odf0),0)

    assert_almost_equal(dsfit.gfa,np.array([0.5749720469955439]))
    
    #1 fiber
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[100,0,0], snr=None)   
    ds=DiffusionSpectrumModel(bvals,bvecs,odf_sphere)
    dsfit=ds.fit(S)
    QA=dsfit.qa
    assert_equal(np.sum(QA>0),1)
    
    #2 fibers
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[50,50,0], snr=None)   
    ds=DiffusionSpectrumModel(bvals,bvecs,odf_sphere)
    ds.relative_peak_threshold = 0.5
    ds.angular_distance_threshold = 20
    dsfit=ds.fit(S)
    QA=dsfit.qa
    assert_equal(np.sum(QA>0),2)
     
    #Give me 2 directions
    assert_equal(len(dsfit.get_directions()),2)
   
    #3 fibers
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[33,33,33], snr=None)   
    ds=DiffusionSpectrumModel(bvals,bvecs,odf_sphere)
    ds.relative_peak_threshold = 0.5
    dsfit=ds.fit(S,return_odf=True)
    QA=dsfit.qa
    assert_equal(np.sum(QA>0),3)

    #Give me 3 directions
    assert_equal(len(dsfit.get_directions()),3)

    #Recalculate the odf with a different sphere.

    vertices, faces = sphere_vf_from('symmetric724') 
    
    odf1=dsfit.odf()
    print len(odf1)
    
    odf2=dsfit.odf((vertices,faces))
    print len(odf2)

    assert_array_almost_equal(odf1,odf2)

    #isotropic
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[0,0,0], snr=None)   
    ds=DiffusionSpectrumModel(bvals,bvecs,odf_sphere)
    dsfit=ds.fit(S)
    QA=dsfit.qa
    assert_equal(np.sum(QA>0),0)

    

if __name__ == '__main__':

    #ds,dsfit=test_dsi()
    pass
