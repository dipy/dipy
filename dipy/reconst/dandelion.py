import numpy as np
from dipy.reconst.recspeed import peak_finding
import os
from os.path import join as opj
from dipy.data import get_sphere
import warnings

warnings.warn("This module is most likely to change both as a name and in structure in the future",FutureWarning)

class SphericalDandelion():
    ''' Garyfallidis E., Nimmo-Smith I. TMI 2011 (to appear)        
    HIGHLY EXPERIMENTAL - PLEASE DO NOT USE.    
    '''
    def __init__(self,data,bvals,gradients,smoothing=1.,odfsphere=None,mask=None):
        '''
        Parameters
        -----------
        data : array, shape(X,Y,Z,D)        
        bvals : array, shape (N,)
        gradients : array, shape (N,3) also known as bvecs
        smoothing : float, smoothing parameter

        See also
        ----------
        dipy.reconst.dti.Tensor, dipy.reconst.gqi.GeneralizedQSampling

        '''
        
        if odfsphere == None:            
            eds=np.load(get_sphere('symmetric362'))
        else:
            eds=np.load(odfsphere)
            # e.g. odfsphere = evenly_distributed_sphere_642.npz

        odf_vertices=eds['vertices']
        odf_faces=eds['faces']

        self.odf_vertices=odf_vertices
        self.bvals=bvals
        
        # 0.01506 = 6*D where D is the free water diffusion coefficient 
        # l_values sqrt(6 D tau) D free water diffusion coefficient and
        # tau included in the b-value
        #scaling = np.sqrt(bvals*0.01506)
        #tmp=np.tile(scaling, (3,1))
        #the b vectors might have nan values where they correspond to b
        #value equals with 0
        gradients[np.isnan(gradients)]= 0.
        
        #gradsT = gradients.T
        #b_vector=gradsT*tmp # element-wise also known as the Hadamard product
        #q2odf_params=np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi)
        #q2odf_params=np.real(np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi))        
        #q2odf_params[np.isnan(q2odf_params)]= 1.
               
        self.gradients=gradients
        self.weighting=np.abs(np.dot(gradients,self.odf_vertices.T))     
        #self.weighting=self.weighting/np.sum(self.weighting,axis=0)
        
        '''
        r=[[],[],[]]
        for si in range(len(sL)):
            for i in range(len(L)):
                if sL[si]-L[i]==0: 
                    r[si].append(i)
        '''            
        S=data
        datashape=S.shape #initial shape
        msk=None #tmp mask

        if len(datashape)==4:
            x,y,z,g=S.shape        
            S=S.reshape(x*y*z,g)
            #QA = np.zeros((x*y*z,5))
            #IN = np.zeros((x*y*z,5))
            if mask != None:
                if mask.shape[:3]==datashape[:3]:
                    msk=mask.ravel().copy()
                    #print 'msk.shape',msk.shape
        if len(datashape)==2:
            x,g= S.shape
            #QA = np.zeros((x,5))
            #IN = np.zeros((x,5))            
        #glob_norm_param = 0
        #self.q2odf_params=q2odf_params
        
        if mask !=None:
            for (i,s) in enumerate(S):                            
                if msk[i]>0:
                    pass
                    #Q to ODF
                    #odf=np.dot(s,q2odf_params)            
                    #peaks,inds=peak_finding(odf,odf_faces)            
                    #glob_norm_param=max(np.max(odf),glob_norm_param)
                    #remove the isotropic part
                    #peaks = peaks - np.min(odf)
                    #l=min(len(peaks),5)
                    #QA[i][:l] = peaks[:l]
                    #IN[i][:l] = inds[:l]

        if mask==None:
            for (i,s) in enumerate(S):                            
                #Q to ODF
                pass
                #odf=np.dot(s,q2odf_params)            
                #peaks,inds=rp.peak_finding(odf,odf_faces)            
                #glob_norm_param=max(np.max(odf),glob_norm_param)
                #remove the isotropic part
                #peaks = peaks - np.min(odf)
                #l=min(len(peaks),5)
                #QA[i][:l] = peaks[:l]
                #IN[i][:l] = inds[:l]

        #normalize
        #QA/=glob_norm_param
       
        if len(datashape) == 4:
            pass
            #self.QA=QA.reshape(x,y,z,5)    
            #self.IN=IN.reshape(x,y,z,5)            
        if len(datashape) == 2:
            pass
            #self.QA=QA
            #self.IN=IN            
        #self.glob_norm_param = glob_norm_param
        
    def spherical_signal(self,s):
        #signal distribution function                
        #sr=s[1:].reshape(1,len(shape))
        sr=s[1:].reshape(1,len(s))        
        return np.abs(np.dot(sr,self.mapping))
    
    def spherical_diffusivity(self,s):
        ob=-1/self.bvals[1:]
        
        #print 'ob'
        #print ob.shape
        #print ob
        lg=np.log(s[1:])-np.log(s[0])
        #print 'lg'
        #print lg.shape
        #print lg
        
        d=ob*(np.log(s[1:])-np.log(s[0]))        
        d=d.reshape(1,len(d)) 
        #print 'd'
        #print d
        #print d.min(),d.mean(),d.max(),d.shape
        res=np.dot(d,self.weighting[1:,:])
        #print 'res'
        #print res
        #print res.min(),res.mean(),res.max(),res.shape
        #print 'tmp'
        #tmp=d.max()*(res/res.max())
        #print tmp
        #print tmp.min(),tmp.mean(),tmp.max(),tmp.shape
        #sum weighting
        #print 'axis=1'
        #print np.sum(self.weighting[1:,:],axis=1)
        #print 'axis=0'
        #print np.sum(self.weighting[1:,:],axis=0)

        #print np.round(10000*np.abs(np.dot(d,self.weighting[1:,:]))).astype('i8')               
        return np.dot(d,self.weighting[1:,:])
        
            
    def odf(self,s):
        '''
        Parameters
        ----------
        s: array, shape(D) diffusion signal for one point in the dataset

        Returns
        -------
        odf: array, shape(len(odf_vertices)), orientation distribution function

        '''
        #return np.dot(s,self.q2odf_params)
        pass
    
    
    
    
    

