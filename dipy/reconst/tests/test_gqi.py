import time
import numpy as np
import nibabel as nib

from .. import recspeed as rp
from .. import gqi as gq
from .. import dti as dt
from ...core import meshes
from ...data import get_data, get_sphere

from dipy.sims.voxel import SticksAndBall
from dipy.reconst.gqi import GeneralizedQSamplingModel
from dipy.core.sphere import reduce_antipodal, unique_edges
from dipy.utils.spheremakers import sphere_vf_from

from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal


def test_gqi():

    #load odf sphere
    vertices,faces = sphere_vf_from('symmetric724')
    edges = unique_edges(faces)
    half_vertices,half_edges,half_faces=reduce_antipodal(vertices,faces)

    #load bvals and gradients
    btable=np.loadtxt(get_data('dsi515btable'))    
    bvals=btable[:,0]
    bvecs=btable[:,1:]        
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[50,50,0], snr=None)    
    #pdf0,odf0,peaks0=standard_dsi_algorithm(S,bvals,bvecs)    
    S2=S.copy()
    S2=S2.reshape(1,len(S)) 
    
    odf_sphere=(vertices,faces)
    ds=GeneralizedQSamplingModel( bvals, bvecs, odf_sphere)    
    dsfit=ds.fit(S)
    assert_equal((dsfit.peak_values>0).sum(),3)

    #change thresholds
    ds.relative_peak_threshold = 0.5
    ds.angular_distance_threshold = 30
    dsfit = ds.fit(S)
    assert_equal((dsfit.peak_values>0).sum(),2)
    
    #1 fiber
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[100,0,0], snr=None)   
    ds=GeneralizedQSamplingModel(bvals,bvecs,odf_sphere)
    ds.relative_peak_threshold = 0.5
    ds.angular_distance_threshold = 20
    dsfit=ds.fit(S)
    QA=dsfit.qa
    #1/0
    assert_equal(np.sum(QA>0),1)
    
    #2 fibers
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[50,50,0], snr=None)   
    ds=GeneralizedQSamplingModel(bvals,bvecs,odf_sphere)
    ds.relative_peak_threshold = 0.5
    ds.angular_distance_threshold = 20
    dsfit=ds.fit(S)
    QA=dsfit.qa
    assert_equal(np.sum(QA>0),2)
    
    #3 fibers
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[33,33,33], snr=None)   
    ds=GeneralizedQSamplingModel(bvals,bvecs,odf_sphere)
    ds.relative_peak_threshold = 0.5
    dsfit=ds.fit(S)
    QA=dsfit.qa
    assert_equal(np.sum(QA>0),3)
    
    #isotropic
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[0,0,0], snr=None)   
    ds=GeneralizedQSamplingModel(bvals,bvecs,odf_sphere)
    dsfit=ds.fit(S)
    QA=dsfit.qa
    assert_equal(np.sum(QA>0),0)

    #3 fibers DSI2
    S,stics=SticksAndBall(bvals, bvecs, d=0.0015, S0=100, angles=[(0, 0),(90,0),(90,90)], fractions=[33,33,33], snr=None)   
    ds=GeneralizedQSamplingModel(bvals,bvecs,odf_sphere,squared=True)
    ds.relative_peak_threshold = 0.5
    dsfit=ds.fit(S,gfa_thr=0.05)
    QA=dsfit.qa

    #3 fibers DSI2 with a 3D volume
    data=np.zeros((3,3,3,len(S)))
    data[...,:]= S.copy()
    dsfit=ds.fit(data,gfa_thr=0.05)
    #1/0
    assert_array_almost_equal(np.sum(dsfit.peak_values>0,axis=-1),3*np.ones((3,3,3)))
 

def upper_hemi_map(v):
    '''
    maps a 3-vector into the z-upper hemisphere
    '''
    return np.sign(v[2])*v


def equatorial_maximum(vertices, odf, pole, width):
    eqvert = meshes.equatorial_vertices(vertices, pole, width)
    '''
    need to test for whether eqvert is empty or not
    '''
    if len(eqvert) == 0:
        print 'empty equatorial band at pole', pole, 'with width', width
        return Null, Null

    eqvals = [odf[i] for i in eqvert]
    eqargmax = np.argmax(eqvals)
    eqvertmax = eqvert[eqargmax]
    eqvalmax = eqvals[eqargmax]
    return eqvertmax, eqvalmax


def patch_vertices(vertices,pole, width):
    '''
    find 'vertices' within the cone of 'width' around 'pole'
    '''
    return [i for i,v in enumerate(vertices) if np.dot(v,pole) > 1- width]


def patch_maximum(vertices, odf, pole, width):
    eqvert = patch_vertices(vertices, pole, width)
    '''
    need to test for whether eqvert is empty or not
    '''
    if len(eqvert) == 0:
        print 'empty cone around pole', pole, 'with width', width
        return Null, Null

    eqvals = [odf[i] for i in eqvert]
    eqargmax = np.argmax(eqvals)
    eqvertmax = eqvert[eqargmax]
    eqvalmax = eqvals[eqargmax]
    return eqvertmax, eqvalmax


def triple_odf_maxima(vertices, odf, width):
    #proton density already include from the scaling b_table[0][0] and s[0]
    #find local maxima
    peak=odf.copy()
    # where the smallest odf values in the vertices of a face remove the
    # two smallest vertices 

    for face in odf_faces:
        i, j, k = face
        check=np.array([odf[i],odf[j],odf[k]])
        zeroing=check.argsort()
        peak[face[zeroing[0]]]=0
        peak[face[zeroing[1]]]=0

    #for later testing expecting peak.max 794595.94774980657 and
    #np.where(peak>0) (array([166, 347]),)
    #we just need the first half of peak
    peak=peak[0:len(peak)/2]
    #find local maxima and give fiber orientation (inds) and magnitute
    #peaks in a descending order
    inds=np.where(peak>0)[0]
    pinds=np.argsort(peak[inds])
    peaks=peak[inds[pinds]][::-1]
    return peaks, inds[pinds][::-1]


if __name__ == "__main__":
    pass









