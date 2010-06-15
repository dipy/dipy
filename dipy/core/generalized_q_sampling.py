import numpy as np
import dipy.core.reconstruction_performance as rp
import dipy

class GeneralizedQSampling():


    def __init__(self,data,bvals,gradients,Lambda=1.2):

        ''' Implements equation 9 from Generalized Q-Sampling as
        described in Yeh et.al, IEEE TMI, 2010 .

        Parameters:
        -----------
        data: array, shape(X,Y,Z,D)        
        
        bvals: array, shape (N,)
        
        gradients: array, shape (N,3) also known as bvecs
        
        Lambda: float, smoothing parameter - diffusion sampling length

        Key Properties:
        ---------------
        QA : array, shape(X,Y,Z,5), quantitative anisotropy               

        IN : array, shape(X,Y,Z,5), indices of QA

        '''

        eds=np.load(dipy.__path__[0] + '/core/matrices/evenly_distributed_sphere_362.npz')

        odf_vertices=eds['vertices']

        odf_faces=eds['faces']

        # 0.01506 = 6*D where D is the free water diffusion coefficient 
        # l_values sqrt(6 D tau) D free water diffusion coefficient and
        # tau included in the b-value
        
        scaling = np.sqrt(bvals*0.01506)

        tmp=np.tile(scaling, (3,1))

        print 'tmp.shape',tmp.shape

        #the b vectors might have nan values where they correspond to b
        #value equals with 0
        gradients[np.isnan(gradients)]= 0.
        
        gradsT = gradients.T
        
        b_vector=gradsT*tmp # element-wise also known as the Hadamard product

        print 'b_vector.shape', b_vector.shape

        print 'odf_vertices.shape', odf_vertices.shape

        print 'odf_vertices.dtype', odf_vertices.dtype
        
        #q2odf_params=np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi)              

        q2odf_params=np.real(np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi))

        #q2odf_params[np.isnan(q2odf_params)]= 1.
        
        S=data

        x,y,z,g=S.shape

        S=S.reshape(x*y*z,g)

        QA = np.zeros((x*y*z,5))

        IN = np.zeros((x*y*z,5))

        fwd = 0

        self.q2odf_params=q2odf_params

        #Calculate Quantitative Anisotropy and find the peaks and the indices
        #for every voxel
        
        for (i,s) in enumerate(S):

            #Q to ODF

            odf=np.dot(s,q2odf_params)

            peaks,inds=rp.peak_finding(odf,odf_faces)

            fwd=max(np.max(odf),fwd)

            peaks = peaks - np.min(odf)

            l=min(len(peaks),5)

            QA[i][:l] = peaks[:l]

            IN[i][:l] = inds[:l]


    
        QA/=fwd

        print 'fwd',fwd

        self.QA=QA.reshape(x,y,z,5)
    
        self.IN=IN.reshape(x,y,z,5)

        
