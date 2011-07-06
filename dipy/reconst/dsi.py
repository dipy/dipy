import warnings

import numpy as np

from dipy.reconst.recspeed import peak_finding
from dipy.utils.spheremakers import sphere_vf_from

#warnings.warn("This module is most likely to change both as a name and in structure in the future",FutureWarning)

class DiffusionSpectrumImaging(object):
    ''' Calculate the PDF and ODF using Diffusion Spectrum Imaging
    
    Based on the paper "Mapping Complex Tissue Architecture With Diffusion Spectrum Magnetic Resonance Imaging"
    by Van J. Wedeen,Patric Hagmann,Wen-Yih Isaac Tseng,Timothy G. Reese, and Robert M. Weisskoff, MRM 2005
    
    
    
    '''
    def __init__(self, data, bvals, gradients,odf_sphere='symmetric362', mask=None):
        '''
        Parameters
        -----------
        data : array, shape(X,Y,Z,D)
        bvals : array, shape (N,)
        gradients : array, shape (N,3) also known as bvecs        
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
        
        S=data
        datashape=S.shape #initial shape
        msk=None #tmp mask

        if len(datashape)==4:
            x,y,z,g=S.shape        
            S=S.reshape(x*y*z,g)  
            if mask != None:
                if mask.shape[:3]==datashape[:3]:
                    msk=mask.ravel().copy()
                    
        if len(datashape)==2:
            x,g= S.shape            
        
        if mask !=None:
            for (i,s) in enumerate(S):                            
                if msk[i]>0:                   
                    odf=self.odf(s)                    
                    peaks,inds=peak_finding(odf,odf_faces)            
                    l=min(len(peaks),5)
                    #XA[i][:l] = peaks[:l]
                    #IN[i][:l] = inds[:l]

        if mask==None:
            for (i,s) in enumerate(S):
                odf=self.spherical_diffusivity(s)
                peaks,inds=peak_finding(odf,odf_faces)            
                l=min(len(peaks),5)
                #XA[i][:l] = peaks[:l]
                #IN[i][:l] = inds[:l]
                
        if len(datashape) == 4:
            #self.XA=XA.reshape(x,y,z,5)
            #self.IN=IN.reshape(x,y,z,5)
            pass

        if len(datashape) == 2:
            #self.XA=XA
            #self.IN=IN
            pass     

            
    def odf(self,s):
        pass
    
    def gfa(self):
        pass
    
    def ind(self):
        pass

