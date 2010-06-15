import numpy as np

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

import time

import dipy.core.reconstruction_performance as rp


def test_gqi():

    from scipy.io import loadmat
    
    phantom=loadmat('/home/eg01/Desktop/phantom_test_data.mat',struct_as_record=True)

    all=phantom['all']
    b_table=phantom['b_table']
    
    #odf_vertices=phantom['odf_vertices']
    #odf_faces=phantom['odf_faces']

    #np.savez('/home/eg01/Devel/dipy/dipy/core/matrices/evenly_distributed_sphere_362.npz',vertices=odf_vertices.T,faces=odf_faces.T)

    eds=np.load('/home/eg01/Devel/dipy/dipy/core/matrices/evenly_distributed_sphere_362.npz')

    odf_vertices=eds['vertices']
    odf_faces=eds['faces']

    before=time.clock()

    #s = all[14,14,1]

    #Yeh et.al, IEEE TMI, 2010
    #calculate the odf using GQI


    scaling=np.sqrt(b_table[0]*0.01506) # 0.01506 = 6*D where D is the free
    #water diffusion coefficient 
    #l_values sqrt(6 D tau) D free water
    #diffusion coefficiet and tau included in the b-value

    tmp=np.tile(scaling,(3,1))

    print 'tmp.shape',tmp.shape

    b_vector=b_table[1:4,:]*tmp

    Lambda = 1.2 # smoothing parameter - diffusion sampling length


    
    q2odf_params=np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi)
    #implements equation no. 9 from Yeh et.al.


    S=all.copy()

    x,y,z,g=S.shape

    S=S.reshape(x*y*z,g)

    QA = np.zeros((x*y*z,5))

    IN = np.zeros((x*y*z,5))

    fwd = 0


    
    #Calculate Quantitative Anisotropy and find the peaks and the indices
    #for every voxel

    #test = np.zeros(len(S),)

    for (i,s) in enumerate(S):

        odf = Q2odf(s,q2odf_params)

        #print odf.shape
        #peak = odf.copy()

        peaks,inds=rp.peak_finding(odf,odf_faces)
        
        #peaks2,inds2=peak_finding(odf,odf_faces)

        #test[i]= np.sum(peaks2-peaks)
                
        fwd=max(np.max(odf),fwd)

        peaks = peaks - np.min(odf)

        l=min(len(peaks),5)

        QA[i][:l] = peaks[:l]

        IN[i][:l] = inds[:l]

        
    
    QA/=fwd

    QA=QA.reshape(x,y,z,5)
    
    IN=IN.reshape(x,y,z,5)

    #print np.sum(test)
    
    print time.clock() - before,' secs.'

    import dipy.core.generalized_q_sampling as gq

    #now test with class
    
    g=gq.GeneralizedQSampling(all,b_table[0],b_table[1:4,:].T)

    print time.clock() - before,' secs.'
        
    #yield assert_equal( (g.QA-QA).max(), 0.0)

    yield assert_equal((g.QA-QA).max(), 0.)


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













