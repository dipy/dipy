import warnings
import numpy as np
from scipy.ndimage import map_coordinates
from dipy.reconst.recspeed import peak_finding, le_to_odf
from dipy.utils.spheremakers import sphere_vf_from
from scipy.fftpack import fftn, fftshift, ifftn,ifftshift
from dipy.reconst.dsi import project_hemisph_bvecs
from scipy.ndimage.filters import laplace
from dipy.core.geometry import sphere2cart,cart2sphere,vec2vec_rotmat

import warnings
warnings.warn("This module is most likely to change both as a name and in structure in the future",FutureWarning)


class DiffusionNabla(object):
    ''' Reconstruct the signal using Diffusion Nabla Imaging  
    
    As described in E.Garyfallidis PhD thesis, 2011.           
    '''
    def __init__(self, data, bvals, gradients,odf_sphere='symmetric362', 
                 mask=None,
                 half_sphere_grads=False,
                 auto=True,
                 save_odfs=False,laplacian=True):
        '''
        Parameters
        -----------
        data : array, shape(X,Y,Z,D), or (X,D)
        bvals : array, shape (N,)
        gradients : array, shape (N,3) also known as bvecs        
        odf_sphere : str or tuple, optional
            If str, then load sphere of given name using ``get_sphere``.
            If tuple, gives (vertices, faces) for sphere.
        filter : array, shape(len(vertices),) 
            default is None (using standard hanning filter for DSI)
        half_sphere_grad : boolean Default(False) 
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
        self.odf_vertices=odf_vertices
        self.odf_faces=odf_faces
        self.odfn=len(self.odf_vertices)
        self.save_odfs=save_odfs
        
        #check if bvectors are provided only on a hemisphere
        if half_sphere_grads==True:
            bvals=np.append(bvals.copy(),bvals[1:].copy())
            bvecs=np.append(bvecs.copy(),-bvecs[1:].copy(),axis=0)
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
        #odf sampling radius  
        self.radius=np.arange(0,6,.2)
        self.radiusn=len(self.radius)
        self.create_qspace(bvals,gradients,16,8)
        #peak threshold
        self.peak_thr=3.
        #calculate coordinates of equators
        self.radon_params()
        #precompute coordinates for pdf interpolation
        self.precompute_interp_coords()        
        #calculate laplacian
        self.laplacian=laplacian
        
        if auto:
            self.fit() 
    
    def create_qspace(self,bvals,gradients,size,origin):
        bv=bvals
        bmin=np.sort(bv)[1]
        bv=np.sqrt(bv/bmin)
        qtable=np.vstack((bv,bv,bv)).T*gradients
        qtable=np.floor(qtable+.5)
        self.qtable=qtable
        self.q=qtable+origin
        self.q=self.q.astype('i8')
        self.origin=origin
        self.sz=size
        
    def radon_params(self,ang_res=64):
        #calculate radon integration parameters
        phis=np.linspace(0,2*np.pi,ang_res)
        planars=[]
        for phi in phis:
            planars.append(sphere2cart(1,np.pi/2,phi))
        planars=np.array(planars)
        planarsR=[]
        for v in self.odf_vertices:
            R=vec2vec_rotmat(np.array([0,0,1]),v)  
            planarsR.append(np.dot(R,planars.T).T)        
        self.equators=planarsR
        self.equatorn=len(phis)        
        
    def fit(self):
        #memory allocations for 4D volumes 
        if len(self.datashape)==4:
            x,y,z,g=self.datashape        
            S=self.data.reshape(x*y*z,g)
            GFA=np.zeros((x*y*z))
            IN=np.zeros((x*y*z,5))
            NFA=np.zeros((x*y*z,5))
            QA=np.zeros((x*y*z,5))
            PK=np.zeros((x*y*z,5))
            if self.save_odfs:
                ODF=np.zeros((x*y*z,self.odfn))        
            if self.mask != None:
                if self.mask.shape[:3]==self.datashape[:3]:
                    msk=self.mask.ravel().copy()
            if self.mask == None:
                self.mask=np.ones(self.datashape[:3])
                msk=self.mask.ravel().copy()
        #memory allocations for a series of voxels       
        if len(self.datashape)==2:
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
                if mask.shape[0]==self.datashape[0]:
                    msk=self.mask.ravel().copy()
            if self.mask == None:
                self.mask=np.ones(self.datashape[:1])
                msk=self.mask.ravel().copy()
        #find the global normalization parameter 
        #useful for quantitative anisotropy
        glob_norm_param = 0.
        #loop over all voxels
        for (i,s) in enumerate(S):
            if msk[i]>0:
                #calculate the orientation distribution function        
                odf=self.odf(s)
                #odf=odf/self.odf(np.ones(s.shape))
                if self.save_odfs:
                    ODF[i]=odf
                #normalization for QA
                glob_norm_param=max(np.max(odf),glob_norm_param)
                #calculate the generalized fractional anisotropy
                GFA[i]=self.std_over_rsm(odf)
                #find peaks
                peaks,inds=peak_finding(odf,self.odf_faces)
                #remove small peaks                
                if np.var(peaks)>5000:
                    if self.laplacian:
                        ismallp=np.where(peaks[0]/peaks>self.peak_thr)
                    else:
                        ismallp=np.where(peaks[0]/peaks<self.peak_thr)
                    l=ismallp[0][0]
                    #print ismallp[0][0]
                    if l<5:
                        IN[i][:l] = inds[:l]
                        NFA[i][:l] = GFA[i]
                        QA[i][:l] = peaks[:l]-np.min(odf)
                        PK[i][:l] = peaks[:l]
                    
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
        
    def odf(self,s):
        """ Calculate the orientation distribution function 
        """        
        odf = np.zeros(self.odfn)
        Eq=np.zeros((self.sz,self.sz,self.sz))
        for i in range(self.dn):
            Eq[self.q[i][0],self.q[i][1],self.q[i][2]]+=s[i]/s[0]
        if self.laplacian:
            LEq=laplace(Eq)         
        else:
            LEq=Eq
        self.Eq=Eq
        LEs=map_coordinates(LEq,self.Xs,order=1)        
        le_to_odf(odf,LEs,self.radius,self.odfn,self.radiusn,self.equatorn)
        if self.laplacian:
            return odf
        else:
            return -odf#-(odf-odf.max())
    
    def odfs(self):
        return self.ODF
    
    def precompute_interp_coords(self):        
        Xs=[]        
        for m in range(self.odfn):
            for q in self.radius:           
                    #print disk.shape
                    xi=self.origin + q*self.equators[m][:,0]
                    yi=self.origin + q*self.equators[m][:,1]
                    zi=self.origin + q*self.equators[m][:,2]        
                    Xs.append(np.vstack((xi,yi,zi)).T)
        self.Xs=np.concatenate(Xs).T
        
    
    def std_over_rsm(self,odf):
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




