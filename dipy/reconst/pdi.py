import warnings

import numpy as np

from dipy.reconst.recspeed import peak_finding
from dipy.utils.spheremakers import sphere_vf_from

warnings.warn("This module is most likely to change both as a name and in structure in the future",FutureWarning)

class ProjectiveDiffusivity(object):
    '''
    HIGHLY EXPERIMENTAL - PLEASE DO NOT USE.
    '''
    def __init__(self, data, bvals, gradients,dotpow=10,width=1,sincpow=2,
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
        
        self.dotpow=dotpow
        self.width=width
        self.sincpow=sincpow
        
        odf_vertices, odf_faces = sphere_vf_from(odf_sphere)
        self.odf_vertices=odf_vertices
        self.bvals=bvals

        gradients[np.isnan(gradients)] = 0.
        self.gradients=gradients
        
        S=data
        datashape=S.shape #initial shape
        msk=None #tmp mask

        if len(datashape)==4:
            x,y,z,g=S.shape        
            S=S.reshape(x*y*z,g)
            XA = np.zeros((x*y*z,5))
            IN = np.zeros((x*y*z,5))
            self.normalization=self.mapping(np.ones(g-1))            
            if mask != None:
                if mask.shape[:3]==datashape[:3]:
                    msk=mask.ravel().copy()
                    
        if len(datashape)==2:
            x,g= S.shape
            XA = np.zeros((x,5))
            IN = np.zeros((x,5))
            self.normalization=self.mapping(np.ones(g-1))
        
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
        d=ob*(np.log(s[1:])-np.log(s[0]))        
        #d=np.zeros(d.shape)
        #d[0]=20
        #d[60]=40
        #d[61]=40
        #d[75]=40
        #d[57]=40        
        return self.mapping(d)/self.normalization
    
    def transfer(self,i):
        cos2=np.dot(self.gradients[i+1],self.odf_vertices.T)**(self.dotpow)
        sin2=1-cos2
        Sinc=np.sinc(self.width*sin2)**self.sincpow
        return Sinc
        
    def mapping(self,d):
        final_sphere=np.zeros((len(d),self.odf_vertices.shape[0]))        
        for i in range(len(d)):
            final_sphere[i]=d[i]*self.transfer(i)
        return np.sum(final_sphere,axis=0)
    
    def xa(self):
        return self.XA
    
    def ind(self):
        return self.IN

