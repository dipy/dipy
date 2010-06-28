import os
import numpy as np
from nose.tools import assert_true, assert_false, assert_equal, assert_raises
from numpy.testing import assert_array_equal, assert_array_almost_equal
import time
import dipy.core.reconstruction_performance as rp
from os.path import join as opj
import nibabel as ni
import dipy.core.generalized_q_sampling as gq
from dipy.testing import parametric
import dipy.core.track_propagation as tp

@parametric
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


    t1=time.clock()
    
    gqs = gq.GeneralizedQSampling(data,bvals,gradients)

    t2=time.clock()
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

    yield assert_equal(2*v-f, 4,'Euler test fails')
    
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
   
    QA/=fwd
    QA=QA.reshape(x,y,z,5)    
    IN=IN.reshape(x,y,z,5)
    
    #print('Old %d secs' %(time.clock() - t2))
    #yield assert_equal((gqs.QA-QA).max(),0.,'Frank QA different than our QA')

    #yield assert_equal((gqs.QA.shape),QA.shape, 'Frank QA shape is different')
       
    #yield assert_equal((gqs.QA-QA).max(), 0.)

    #import dipy.core.track_propagation as tp

    #tp.FACT_Delta(QA,IN)

    #return tp.FACT_Delta(QA,IN,seeds_no=10000).tracks

    peaks_1 = [i for i in range(1000) if len(summary[str(i)]['inds'])==1]
    peaks_2 = [i for i in range(1000) if len(summary[str(i)]['inds'])==2]
    peaks_3 = [i for i in range(1000) if len(summary[str(i)]['inds'])==3]

    # correct numbers of voxels with respectively 1,2,3 ODF/QA peaks
    yield assert_array_equal((len(peaks_1),len(peaks_2),len(peaks_3)), (790,196,14),
                             'error in numbers of QA/ODF peaks')

    # correct indices of odf directions for voxels 0,10,44
    # with respectively 1,2,3 ODF/QA peaks
    yield assert_array_equal(summary[str(0)]['inds'],[116],
                             'wrong peak indices for voxel 0')
    yield assert_array_equal(summary[str(10)]['inds'],[105, 78],
                             'wrong peak indices for voxel 10')
    yield assert_array_equal(summary[str(44)]['inds'],[95, 84, 108],
                             'wrong peak indices for voxel 44')

    yield assert_equal(np.argmax(summary['0']['odf']), 116)
    yield assert_equal(np.argmax(summary['10']['odf']), 105)
    yield assert_equal(np.argmax(summary['44']['odf']), 95)

    pole_1 = summary['vertices'][116]
    print 'pole_1', pole_1
    pole_2 = summary['vertices'][105]
    print 'pole_2', pole_2
    pole_3 = summary['vertices'][95]
    print 'pole_3', pole_3

    vertices = summary['vertices']

    width = 0.05
    
    print 'pole_1 equator contains:', len([i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole_1)) < width])
    print 'pole_2 equator contains:', len([i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole_2)) < width])
    print 'pole_3 equator contains:', len([i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole_3)) < width])

    print [len([i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole)) < width]) for pv, pole in enumerate(vertices)]
    
    #return summary

@parametric
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
    
    yield assert_equal((gqs.QA-QA).max(),0.,'Frank QA different than dipy QA')
    yield assert_equal((gqs.QA.shape),QA.shape, 'Frank QA shape is different')  

    yield assert_equal(len(tp.FACT_Delta(QA,IN,seeds_no=100).tracks),100,'FACT_Delta is not generating the right number of tracks for this dataset')



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

    T=test_gqiodf()
    









