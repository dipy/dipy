import warnings

import numpy as np

from dipy.reconst.recspeed import peak_finding
from dipy.utils.spheremakers import sphere_vf_from

warnings.warn("This module is most likely to change both as a name and in structure in the future",FutureWarning)

class SphericalDandelion(object):
    '''
    HIGHLY EXPERIMENTAL - PLEASE DO NOT USE.
    '''
    def __init__(self, data, bvals, gradients, smoothing=1.,
                 odf_sphere='symmetric362', mask=None):
        '''
        Parameters
        -----------
        data : array, shape(X,Y,Z,D)
        bvals : array, shape (N,)
        gradients : array, shape (N,3) also known as bvecs
        smoothing : float, smoothing parameter
        odf_sphere : str or tuple, optional
            If str, then load sphere of given name using ``get_sphere``.
            If tuple, gives (vertices, faces) for sphere.

        See also
        ----------
        dipy.reconst.dti.Tensor, dipy.reconst.gqi.GeneralizedQSampling
        '''
        odf_vertices, odf_faces = sphere_vf_from(odf_sphere)
        self.odf_vertices=odf_vertices
        self.bvals=bvals

        gradients[np.isnan(gradients)] = 0.
        self.gradients=gradients
        self.weighting=np.abs(np.dot(gradients,self.odf_vertices.T))     
        #self.weighting=self.weighting/np.sum(self.weighting,axis=0)

        S=data
        datashape=S.shape #initial shape
        msk=None #tmp mask

        if len(datashape)==4:
            x,y,z,g=S.shape        
            S=S.reshape(x*y*z,g)
            XA = np.zeros((x*y*z,5))
            IN = np.zeros((x*y*z,5))
            if mask != None:
                if mask.shape[:3]==datashape[:3]:
                    msk=mask.ravel().copy()
                    
        if len(datashape)==2:
            x,g= S.shape
            XA = np.zeros((x,5))
            IN = np.zeros((x,5))
        
        if mask !=None:
            for (i,s) in enumerate(S):                            
                if msk[i]>0:               
                    
                    odf=self.spherical_diffusivity(s)                    
                    peaks,inds=peak_finding(odf,odf_faces)            
                    l=min(len(peaks),5)
                    XA[i][:l] = peaks[:l]
                    IN[i][:l] = inds[:l]

        if mask==None:
            for (i,s) in enumerate(S):                            

                odf=self.spherical_diffusivity(s)
                peaks,inds=peak_finding(odf,odf_faces)            
                l=min(len(peaks),5)
                XA[i][:l] = peaks[:l]
                IN[i][:l] = inds[:l]
                
        if len(datashape) == 4:
            self.XA=XA.reshape(x,y,z,5)    
            self.IN=IN.reshape(x,y,z,5)
                    
        if len(datashape) == 2:
            self.XA=XA
            self.IN=IN            

            
    def spherical_diffusivity(self,s):
        ob=-1/self.bvals[1:]
        lg=np.log(s[1:])-np.log(s[0])
        d=ob*(np.log(s[1:])-np.log(s[0]))        
        #d=d.reshape(1,len(d)) 
        #return np.squeeze(np.dot(d,self.weighting[1:,:]))
        
        '''
        final_sphere=np.zeros(self.odf_vertices.shape[0])
        for i in range(len(d)):
            final_sphere+=d[i]*np.abs(np.dot(self.gradients[i+1],self.odf_vertices.T)**(2))
        #return (final_sphere-final_sphere.min())/float(len(d))
        return final_sphere/float(len(d))
        '''
        d=np.zeros(d.shape)
        d[0]=12
        #d=12*np.ones(d.shape)
        o=np.ones(d.shape)
        
        finald=self.koukou(d)
        finalo=self.koukou(o)
        #print finald.shape,finalo.shape
        return finald/finalo
        
    def koukou(self,d):
        width=1
        final_sphere=np.zeros((len(d),self.odf_vertices.shape[0]))
        for i in range(len(d)):
            #f=np.abs(np.dot(self.gradients[i+1],self.odf_vertices.T)**(2))
            cos2=np.dot(self.gradients[i+1],self.odf_vertices.T)**(2)
            sin2=1-cos2
            Sinc=np.sinc(width*sin2)**2
            final_sphere[i]=d[i]*Sinc
        #return final_sphere
        return np.sum(final_sphere,axis=0)#/float(len(d))
    
    def xa(self):
        return self.XA
    
    def ind(self):
        return self.IN

