import os
import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
import time
#import dipy.core.reconstruction_performance as rp
import dipy.reconst.recspeed as rp
from os.path import join as opj
import nibabel as ni
#import dipy.core.generalized_q_sampling as gq
import dipy.reconst.gqi as gq
#import dipy.core.track_propagation as tp
import dipy.core.dti as dt
import dipy.core.meshes as meshes 

def test_gqiodf():

    #read bvals,gradients and data
    bvals=np.load(opj(os.path.dirname(__file__), \
                          'data','small_64D.bvals.npy'))
    gradients=np.load(opj(os.path.dirname(__file__), \
                              'data','small_64D.gradients.npy'))    
    img =ni.load(os.path.join(os.path.dirname(__file__),\
                                  'data','small_64D.nii'))
    data=img.get_data()    

    #print(bvals.shape)
    #print(gradients.shape)
    #print(data.shape)


    # t1=time.clock()
    
    gq.GeneralizedQSampling(data,bvals,gradients)
    ten = dt.Tensor(data,bvals,gradients,thresh=50)

    
    ten.fa()

    x,y,z,a,b=ten.evecs.shape
    evecs=ten.evecs
    xyz=x*y*z
    evecs = evecs.reshape(xyz,3,3)
    #vs = np.sign(evecs[:,2,:])
    #print vs.shape
    #print np.hstack((vs,vs,vs)).reshape(1000,3,3).shape
    #evecs = np.hstack((vs,vs,vs)).reshape(1000,3,3)
    #print evecs.shape
    evals=ten.evals
    evals = evals.reshape(xyz,3)
    #print evals.shape

    #print('GQS in %d' %(t2-t1))
        
    eds=np.load(opj(os.path.dirname(__file__),\
                        '..','matrices',\
                        'evenly_distributed_sphere_362.npz'))

    
    odf_vertices=eds['vertices']
    odf_faces=eds['faces']

    #Yeh et.al, IEEE TMI, 2010
    #calculate the odf using GQI

    scaling=np.sqrt(bvals*0.01506) # 0.01506 = 6*D where D is the free
    #water diffusion coefficient 
    #l_values sqrt(6 D tau) D free water
    #diffusion coefficiet and tau included in the b-value

    tmp=np.tile(scaling,(3,1))
    b_vector=gradients.T*tmp
    Lambda = 1.2 # smoothing parameter - diffusion sampling length
    
    q2odf_params=np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi)
    #implements equation no. 9 from Yeh et.al.

    S=data.copy()

    x,y,z,g=S.shape
    S=S.reshape(x*y*z,g)
    QA = np.zeros((x*y*z,5))
    IN = np.zeros((x*y*z,5))

    fwd = 0
    
    #Calculate Quantitative Anisotropy and find the peaks and the indices
    #for every voxel

    summary = {}

    summary['vertices'] = odf_vertices
    v = odf_vertices.shape[0]
    summary['faces'] = odf_faces
    f = odf_faces.shape[0]

    '''
    If e = number_of_edges
    the Euler formula says f-e+v = 2 for a mesh on a sphere
    Here, assuming we have a healthy triangulation
    every face is a triangle, all 3 of whose edges should belong to
    exactly two faces = so 2*e = 3*f
    to avoid division we test whether 2*f - 3*f + 2*v == 4
    or equivalently 2*v - f == 4
    '''

    assert_equal(2*v-f, 4,'Direct Euler test fails')
    assert_true(meshes.euler_characteristic_check(odf_vertices, odf_faces,chi=2),'euler_characteristic_check fails')
    
    coarse = meshes.coarseness(odf_faces)
    print 'coarseness: ', coarse

    for (i,s) in enumerate(S):

        #print 'Volume %d' % i

        istr = str(i)

        summary[istr] = {}

        odf = Q2odf(s,q2odf_params)
        peaks,inds=rp.peak_finding(odf,odf_faces)
        fwd=max(np.max(odf),fwd)
        peaks = peaks - np.min(odf)
        l=min(len(peaks),5)
        QA[i][:l] = peaks[:l]
        IN[i][:l] = inds[:l]

        summary[istr]['odf'] = odf
        summary[istr]['peaks'] = peaks
        summary[istr]['inds'] = inds
        summary[istr]['evecs'] = evecs[i,:,:]
        summary[istr]['evals'] = evals[i,:]
   
    QA /= fwd
    # QA=QA.reshape(x,y,z,5)
    # IN=IN.reshape(x,y,z,5)
    
    #print('Old %d secs' %(time.clock() - t2))
    # assert_equal((gqs.QA-QA).max(),0.,'Frank QA different than our QA')

    # assert_equal((gqs.QA.shape),QA.shape, 'Frank QA shape is different')
       
    # assert_equal((gqs.QA-QA).max(), 0.)

    #import dipy.core.track_propagation as tp

    #tp.FACT_Delta(QA,IN)

    #return tp.FACT_Delta(QA,IN,seeds_no=10000).tracks

    peaks_1 = [i for i in range(1000) if len(summary[str(i)]['inds'])==1]
    peaks_2 = [i for i in range(1000) if len(summary[str(i)]['inds'])==2]
    peaks_3 = [i for i in range(1000) if len(summary[str(i)]['inds'])==3]

    # correct numbers of voxels with respectively 1,2,3 ODF/QA peaks
    assert_array_equal((len(peaks_1),len(peaks_2),len(peaks_3)), (790,196,14),
                       'error in numbers of QA/ODF peaks')

    # correct indices of odf directions for voxels 0,10,44
    # with respectively 1,2,3 ODF/QA peaks
    assert_array_equal(summary['0']['inds'],[116],
                       'wrong peak indices for voxel 0')
    assert_array_equal(summary['10']['inds'],[105, 78],
                       'wrong peak indices for voxel 10')
    assert_array_equal(summary['44']['inds'],[95, 84, 108],
                       'wrong peak indices for voxel 44')

    assert_equal(np.argmax(summary['0']['odf']), 116)
    assert_equal(np.argmax(summary['10']['odf']), 105)
    assert_equal(np.argmax(summary['44']['odf']), 95)

    # pole_1 = summary['vertices'][116]
    #print 'pole_1', pole_1
    # pole_2 = summary['vertices'][105]
    #print 'pole_2', pole_2
    # pole_3 = summary['vertices'][95]
    #print 'pole_3', pole_3

    vertices = summary['vertices']

    width = 0.02#0.3 #0.05
    
    '''
    print 'pole_1 equator contains:', len([i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole_1)) < width])
    print 'pole_2 equator contains:', len([i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole_2)) < width])
    print 'pole_3 equator contains:', len([i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole_3)) < width])
    '''
    
    #print 'pole_1 equator contains:', len(meshes.equatorial_vertices(vertices,pole_1,width))
    #print 'pole_2 equator contains:', len(meshes.equatorial_vertices(vertices,pole_2,width))
    #print 'pole_3 equator contains:', len(meshes'equatorial_vertices(vertices,pole_3,width))

    #print triple_odf_maxima(vertices,summary['0']['odf'],width)
    #print triple_odf_maxima(vertices,summary['10']['odf'],width)
    #print triple_odf_maxima(vertices,summary['44']['odf'],width)
    #print summary['0']['evals']
    '''

    pole=np.array([0,0,1])

    from dipy.viz import fos
    r=fos.ren()
    fos.add(r,fos.point(pole,fos.green))
    for i,ev in enumerate(vertices):        
        if np.abs(np.dot(ev,pole))<width:
            fos.add(r,fos.point(ev,fos.red))
    fos.show(r)

    '''

    triple = triple_odf_maxima(vertices, summary['10']['odf'], width)
    
    indmax1, odfmax1 = triple[0]
    indmax2, odfmax2 = triple[1]
    indmax3, odfmax3 = triple[2] 

    '''
    from dipy.viz import fos
    r=fos.ren()
    for v in vertices:
        fos.add(r,fos.point(v,fos.cyan))
    fos.add(r,fos.sphere(upper_hemi_map(vertices[indmax1]),radius=0.1,color=fos.red))
    #fos.add(r,fos.line(np.array([0,0,0]),vertices[indmax1]))
    fos.add(r,fos.sphere(upper_hemi_map(vertices[indmax2]),radius=0.05,color=fos.green))
    fos.add(r,fos.sphere(upper_hemi_map(vertices[indmax3]),radius=0.025,color=fos.blue))
    fos.add(r,fos.sphere(upper_hemi_map(summary['0']['evecs'][:,0]),radius=0.1,color=fos.red,opacity=0.7))
    fos.add(r,fos.sphere(upper_hemi_map(summary['0']['evecs'][:,1]),radius=0.05,color=fos.green,opacity=0.7))
    fos.add(r,fos.sphere(upper_hemi_map(summary['0']['evecs'][:,2]),radius=0.025,color=fos.blue,opacity=0.7))
    fos.add(r,fos.sphere([0,0,0],radius=0.01,color=fos.white))
    fos.show(r)
    '''
    
    mat = np.vstack([vertices[indmax1],vertices[indmax2],vertices[indmax3]])

    print np.dot(mat,np.transpose(mat))
    # this is to assess how othogonal the triple is/are
    print np.dot(summary['0']['evecs'],np.transpose(mat))
    
    #return summary

def upper_hemi_map(v):
    '''
    maps a 3-vector into the z-upper hemisphere
    '''
    return np.sign(v[2])*v

def equatorial_maximum(vertices, odf, pole, width):

    eqvert = meshes.equatorial_zone_vertices(vertices, pole, width)

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

    indmax1 = np.argmax([odf[i] for i,v in enumerate(vertices)])
    odfmax1 = odf[indmax1]
    indmax2, odfmax2 = equatorial_maximum(vertices, odf, vertices[indmax1], width)
    cross12 = np.cross(vertices[indmax1],vertices[indmax2])
    indmax3, odfmax3 = patch_maximum(vertices, odf, cross12, width)
    return [(indmax1, odfmax1),(indmax2, odfmax2),(indmax3, odfmax3)]
    
def test_gqi_small():

    #read bvals,gradients and data
    bvals=np.load(opj(os.path.dirname(__file__), \
                          'data','small_64D.bvals.npy'))
    gradients=np.load(opj(os.path.dirname(__file__), \
                              'data','small_64D.gradients.npy'))    
    img =ni.load(os.path.join(os.path.dirname(__file__),\
                                  'data','small_64D.nii'))
    data=img.get_data()    

    print(bvals.shape)
    print(gradients.shape)
    print(data.shape)


    t1=time.clock()
    
    gqs = gq.GeneralizedQSampling(data,bvals,gradients)

    t2=time.clock()
    print('GQS in %d' %(t2-t1))
        
    eds=np.load(opj(os.path.dirname(__file__),\
                        '..','matrices',\
                        'evenly_distributed_sphere_362.npz'))

    
    odf_vertices=eds['vertices']
    odf_faces=eds['faces']

    #Yeh et.al, IEEE TMI, 2010
    #calculate the odf using GQI

    scaling=np.sqrt(bvals*0.01506) # 0.01506 = 6*D where D is the free
    #water diffusion coefficient 
    #l_values sqrt(6 D tau) D free water
    #diffusion coefficiet and tau included in the b-value

    tmp=np.tile(scaling,(3,1))
    b_vector=gradients.T*tmp
    Lambda = 1.2 # smoothing parameter - diffusion sampling length
    
    q2odf_params=np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi)
    #implements equation no. 9 from Yeh et.al.

    S=data.copy()

    x,y,z,g=S.shape
    S=S.reshape(x*y*z,g)
    QA = np.zeros((x*y*z,5))
    IN = np.zeros((x*y*z,5))

    fwd = 0
    
    #Calculate Quantitative Anisotropy and find the peaks and the indices
    #for every voxel

    for (i,s) in enumerate(S):

        odf = Q2odf(s,q2odf_params)
        peaks,inds=rp.peak_finding(odf,odf_faces)
        fwd=max(np.max(odf),fwd)
        peaks = peaks - np.min(odf)
        l=min(len(peaks),5)
        QA[i][:l] = peaks[:l]
        IN[i][:l] = inds[:l]

    QA/=fwd
    QA=QA.reshape(x,y,z,5)    
    IN=IN.reshape(x,y,z,5)
    
    print('Old %d secs' %(time.clock() - t2))
    
    assert_equal((gqs.QA-QA).max(),0.,'Frank QA different than dipy QA')
    assert_equal((gqs.QA.shape),QA.shape, 'Frank QA shape is different')  

    assert_equal(len(tp.FACT_Delta(QA,IN,seeds_no=100).tracks),100,
                 'FACT_Delta is not generating the right number of '
                 'tracks for this dataset')



def Q2odf(s,q2odf_params):

    odf=np.dot(s,q2odf_params)

    return odf

def peak_finding(odf,odf_faces):

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

    #T=test_gqiodf()
    T=test_gqi_small()
    









