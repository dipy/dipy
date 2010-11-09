import numpy as np
import dipy.core.reconstruction_performance as rp
import os
from os.path import join as opj

class GeneralizedQSampling():


    def __init__(self,data,bvals,gradients,Lambda=1.2,odfsphere=None,mask=None):

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

        self.odf_vertices=odf_vertices

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
        msk=None #tmp mask

        if len(datashape)==4:

            x,y,z,g=S.shape        
            S=S.reshape(x*y*z,g)
            QA = np.zeros((x*y*z,5))
            IN = np.zeros((x*y*z,5))

            if mask != None:
                if mask.shape[:3]==datashape[:3]:
                    msk=mask.ravel().copy()
                    #print 'msk.shape',msk.shape

        if len(datashape)==2:

            x,g= S.shape
            QA = np.zeros((x,5))
            IN = np.zeros((x,5))  
            
        glob_norm_param = 0

        self.q2odf_params=q2odf_params

        #Calculate Quantitative Anisotropy and 
        #find the peaks and the indices
        #for every voxel
        
        if mask !=None:
            for (i,s) in enumerate(S):                            
                if msk[i]>0:
                    #Q to ODF
                    odf=np.dot(s,q2odf_params)            
                    peaks,inds=rp.peak_finding(odf,odf_faces)            
                    glob_norm_param=max(np.max(odf),glob_norm_param)
                    #remove the isotropic part
                    peaks = peaks - np.min(odf)
                    l=min(len(peaks),5)
                    QA[i][:l] = peaks[:l]
                    IN[i][:l] = inds[:l]

        if mask==None:
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

    def npa(self,s,width=0.5):
        '''
        '''   
        odf=self.odf(s)
        t0,t1,t2=triple_odf_maxima(self.odf_vertices, odf, width)

        #print 'tom >>>> ',t0,t1,t2

        return t0,t1,t2

    

        


def equatorial_zone_vertices(vertices, pole, width=.02):
    '''
    finds the 'vertices' in the equatorial zone conjugate
    to 'pole' with width half 'width' radians
    '''
    return [i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole)) < width]

def polar_zone_vertices(vertices, pole, width=0.02):
    '''
    finds the 'vertices' in the equatorial band around
    the 'pole' of radius 'width' radians (np.arcsin(width)*180/np.pi degrees)
    '''
    return [i for i,v in enumerate(vertices) if np.dot(v,pole) > 1-width]


def upper_hemi_map(v):
    '''
    maps a 3-vector into the z-upper hemisphere
    '''
    return np.sign(v[2])*v

def equatorial_maximum(vertices, odf, pole, width):
    eqvert = equatorial_zone_vertices(vertices, pole, width)
    #need to test for whether eqvert is empty or not
    if len(eqvert) == 0:
        print('empty equatorial band at %s  pole with width %f' % (np.array_str(pole), width))
        return Null, Null
    eqvals = [odf[i] for i in eqvert]
    eqargmax = np.argmax(eqvals)
    eqvertmax = eqvert[eqargmax]
    eqvalmax = eqvals[eqargmax]

    return eqvertmax, eqvalmax

#'''
def patch_vertices(vertices,pole, width):
    '''
    find 'vertices' within the cone of 'width' around 'pole'
    '''
    return [i for i,v in enumerate(vertices) if np.dot(v,pole) > 1- width]
#'''

def patch_maximum(vertices, odf, pole, width):
    eqvert = patch_vertices(vertices, pole, width)    
    #need to test for whether eqvert is empty or not    
    if len(eqvert) == 0:
        print('empty cone around pole %s with with width %f' % (np.array_str(pole), width))
        return Null, Null
    eqvals = [odf[i] for i in eqvert]
    eqargmax = np.argmax(eqvals)
    eqvertmax = eqvert[eqargmax]
    eqvalmax = eqvals[eqargmax]
    return eqvertmax, eqvalmax

def triple_odf_maxima(vertices, odf, width):

    indmax1 = np.argmax([odf[i] for i,v in enumerate(vertices)])
    odfmax1 = odf[indmax1]
    indmax2, odfmax2 = equatorial_maximum(vertices,\
                                              odf, vertices[indmax1], width)
    cross12 = np.cross(vertices[indmax1],vertices[indmax2])
    cross12 = cross12/np.sqrt(np.sum(cross12**2))    
    indmax3, odfmax3 = patch_maximum(vertices, odf, cross12, width)
    return [(indmax1, odfmax1),(indmax2, odfmax2),(indmax3, odfmax3)]


