import numpy as np
import dipy.core.reconstruction_performance as rp
import os
from os.path import join as opj

class GeneralizedQSampling():


    def __init__(self,data,bvals,gradients,Lambda=1.2,odfsphere=None):

        ''' Generates a model-free description for every voxel that can
        be used from simple to very complicated configurations like
        quintuple crossings if your datasets support them.

        You can use this class for every kind of DWI image but it will
        perform better when you have a balanced sampling scheme.

        Implements equation [9] from Generalized Q-Sampling as
        described in Fang-Cheng Yeh, Van J. Wedeen, Wen-Yih Isaac Tseng.
        Generalized Q-Sampling Imaging. IEEE TMI, 2010.
        

        Parameters
        -----------
        data: array, shape(X,Y,Z,D)        
        
        bvals: array, shape (N,)
        
        gradients: array, shape (N,3) also known as bvecs
        
        Lambda: float, smoothing parameter - diffusion sampling length

        Key Properties
        ---------------
        QA : array, shape(X,Y,Z,5), quantitative anisotropy               

        IN : array, shape(X,Y,Z,5), indices of QA, qa unit directions

        fwd : float, normalization parameter

        Notes
        -----
        In order to reconstruct the spin distribution function  a nice symmetric evenly distributed sphere is provided using 362 points. This is usually
        sufficient for most of the datasets. 

        See also
        --------
        FACT_Delta, Tensor

        '''
        if odfsphere == None:
            eds=np.load(opj(os.path.dirname(__file__),'matrices','evenly_distributed_sphere_362.npz'))
        else:
            eds=np.load(opj(os.path.dirname(__file__),'matrices',odfsphere))
            # e.g. odfsphere = evenly_distributed_sphere_642.npz

        odf_vertices=eds['vertices']
        odf_faces=eds['faces']

        # 0.01506 = 6*D where D is the free water diffusion coefficient 
        # l_values sqrt(6 D tau) D free water diffusion coefficient and
        # tau included in the b-value
        scaling = np.sqrt(bvals*0.01506)
        tmp=np.tile(scaling, (3,1))

        #the b vectors might have nan values where they correspond to b
        #value equals with 0
        gradients[np.isnan(gradients)]= 0.
        gradsT = gradients.T
        b_vector=gradsT*tmp # element-wise also known as the Hadamard product

        #q2odf_params=np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi)              

        q2odf_params=np.real(np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi))
        
        #q2odf_params[np.isnan(q2odf_params)]= 1.

        #define total mask 
        #tot_mask = (mask > 0) & (data[...,0] > thresh)

        
        S=data

        datashape=S.shape #initial shape

        if len(datashape)==4:

            x,y,z,g=S.shape        
            S=S.reshape(x*y*z,g)
            QA = np.zeros((x*y*z,5))
            IN = np.zeros((x*y*z,5))

        if len(datashape)==2:

            x,g=S.shape
            QA = np.zeros((x,5))
            IN = np.zeros((x,5))      
            

        glob_norm_param = 0

        self.q2odf_params=q2odf_params

        #Calculate Quantitative Anisotropy and find the peaks and the indices
        #for every voxel
        
        for (i,s) in enumerate(S):

            #Q to ODF
            odf=np.dot(s,q2odf_params)            
            peaks,inds=rp.peak_finding(odf,odf_faces)            
            glob_norm_param=max(np.max(odf),glob_norm_param)
            #remove the isotropic part
            peaks = peaks - np.min(odf)
            l=min(len(peaks),5)
            QA[i][:l] = peaks[:l]
            IN[i][:l] = inds[:l]

        #normalize
        QA/=glob_norm_param

       
        if len(datashape) == 4:

            self.QA=QA.reshape(x,y,z,5)    
            self.IN=IN.reshape(x,y,z,5)

        if len(datashape) == 2:

            self.QA=QA
            self.IN=IN
            
        self.glob_norm_param = glob_norm_param
        


    def odf(self,s):
        '''
        Parameters
        ----------
        s: array, shape(D) diffusion signal for one point in the dataset

        Returns
        -------
        odf: array, shape(len(odf_vertices)), orientation distribution function

        '''

        return np.dot(s,self.q2odf_params)

