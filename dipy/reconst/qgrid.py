import numpy as np
from dipy.reconst.recspeed import peak_finding
from dipy.utils.spheremakers import sphere_vf_from


class NonParametricCartesian(object):
    ''' Generic class for non-parametric Cartesian grid reconstructions  
    
    '''
    def __init__(self, data, bvals, gradients,odf_sphere='symmetric362', 
                 mask=None,
                 half_sphere_grads=False,
                 auto=True,
                 save_odfs=False):
        '''
        Parameters
        -----------
        data : array, shape (X,Y,Z,D), or (X,D)
        bvals : array, shape (N,)
        gradients : array, shape (N,3) also known as bvecs        
        odf_sphere : str or tuple, optional
            If str, then load sphere of given name using ``get_sphere``.
            If tuple, gives (vertices, faces) for sphere.
        mask : array, shape (X,Y,Z)
        half_sphere_grads : boolean Default(False) 
            in order to create the q-space we use the bvals and gradients. 
            If the gradients are only one hemisphere then 
        auto : boolean, default True 
            if True then the processing of all voxels will start automatically 
            with the class constructor,if False then you will have to call .fit()
            in order to do the heavy duty processing for every voxel
        save_odfs : boolean, default False
            save odfs, which is memory expensive

        See also
        ----------
        dipy.reconst.dti.Tensor, dipy.reconst.dsi.DiffusionSpectrum
        '''
        
        #read the vertices and faces for the odf sphere
        odf_vertices, odf_faces = sphere_vf_from(odf_sphere)
        self.odf_vertices=np.ascontiguousarray(odf_vertices)
        self.odf_faces=np.ascontiguousarray(odf_faces)
        self.odfn=len(self.odf_vertices)
        self.save_odfs=save_odfs

        #make the shape of any data array to a 2D array
        self.data=data.reshape(-1,2)
        
        #if bvectors are provided only on a hemisphere
        #then create the full q-space
        if half_sphere_grads==True:
            bvals=np.append(bvals.copy(),bvals[1:].copy())
            gradients=np.append(gradients.copy(),-gradients[1:].copy(),axis=0)
            data=np.append(data.copy(),data[...,1:].copy(),axis=-1)
        
        #load bvals and bvecs
        self.bvals=bvals
        gradients[np.isnan(gradients)] = 0.
        self.gradients=gradients
        #save number of total diffusion volumes
        self.dn=data.shape[-1]        
        self.data=data
        self.datashape=data.shape #initial shape  
        self.mask=mask
           
    def update(self):
        pass        

    def precompute_angular(self,smooth):
        if smooth==None:            
            self.E=None
            return        
        self.W=np.dot(self.odf_vertices,self.odf_vertices.T)
        self.W=self.W.astype('f8')
        E=np.exp(self.W/smooth)
        self.E=E/np.sum(E,axis=1)[:,None]            
        
    def fit(self):
        x,g= self.datashape
        S=self.data
        GFA=np.zeros(x)
        IN=np.zeros((x,5))
        NFA=np.zeros((x,5))
        QA=np.zeros((x,5))
        PK=np.zeros((x,5))
        if self.save_odfs:
            ODF=np.zeros((x,self.odfn))                
        if self.mask != None:
            if self.mask.shape[0]==self.datashape[0]:
                msk=self.mask.ravel().copy()
        if self.mask == None:
            self.mask=np.ones(self.datashape[:1])
            msk=self.mask.ravel().copy()
    #find the global normalization parameter 
        #useful for quantitative anisotropy
        glob_norm_param = 0.
        
        odf_func=self.odf
        
        #loop over all voxels
        for (i,s) in enumerate(S):
            if msk[i]>0:
                #calculate the orientation distribution function        
                #odf=self.odf(s)
                odf=odf_func(s)#self.odf(s)
                                                
                if self.save_odfs:
                    ODF[i]=odf                
                #normalization for QA
                glob_norm_param=max(np.max(odf),glob_norm_param)
                #calculate the generalized fractional anisotropy
                GFA[i]=self.std_over_rms(odf)                
                odf_max=odf.max()
                peaks,inds=peak_finding(odf,self.odf_faces)
                ibigp=np.where(peaks>self.peak_thr*peaks[0])[0]
                l=len(ibigp)                
                if l>3:
                    l=3               
                if l==0:
                    IN[i][l] = inds[l]
                    NFA[i][l] = GFA[i]
                    QA[i][l] = peaks[l]-np.min(odf)
                    PK[i][l] = peaks[l]/np.float(peaks[0])                         
                if l>0:                    
                    IN[i][:l] = inds[:l]
                    NFA[i][:l] = GFA[i]
                    QA[i][:l] = peaks[:l]-np.min(odf)
                    PK[i][:l] = peaks[:l]/np.float(peaks[0])
    
        if len(self.datashape) == 4:
            self.GFA=GFA.reshape(x,y,z)
            self.NFA=NFA.reshape(x,y,z,5)
            self.QA=QA.reshape(x,y,z,5)/glob_norm_param
            self.PK=PK.reshape(x,y,z,5)
            self.IN=IN.reshape(x,y,z,5)
            if self.save_odfs:
                self.ODF=ODF.reshape(x,y,z,ODF.shape[-1])                            
            self.QA_norm=glob_norm_param            
        if len(self.datashape) == 2:
            self.GFA=GFA
            self.NFA=NFA
            self.QA=QA
            self.PK=PK
            self.IN=IN
            if self.save_odfs:
                self.ODF=ODF
            self.QA_norm=None
       
    
    def odfs(self):
        return self.ODF    
           
    def angular_weighting(self,odf):
        if self.E==None:
            return odf
        else:
            return np.dot(odf[None,:],self.E).ravel()                  
    
    def std_over_rms(self,odf):
        numer=len(odf)*np.sum((odf-np.mean(odf))**2)
        denom=(len(odf)-1)*np.sum(odf**2)
        return np.sqrt(numer/denom)
    
    def gfa(self):
        """ Generalized Fractional Anisotropy
        Defined as the std/rms of the odf values.
        """
        return self.GFA
    def nfa(self):
        return self.NFA
    def qa(self):
        return self.QA
    def pk(self):
        return self.PK
    def ind(self):
        """ peak indices
        """
        return self.IN



