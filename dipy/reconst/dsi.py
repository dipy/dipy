import warnings
import numpy as np
from scipy.ndimage import map_coordinates
from dipy.reconst.recspeed import peak_finding, pdf_to_odf
from dipy.utils.spheremakers import sphere_vf_from
from scipy.fftpack import fftn, fftshift, ifftn,ifftshift

class DiffusionSpectrum(object):
    ''' Calculate the PDF and ODF using Diffusion Spectrum Imaging
    
    Based on the paper "Mapping Complex Tissue Architecture With Diffusion Spectrum Magnetic Resonance Imaging"
    by Van J. Wedeen,Patric Hagmann,Wen-Yih Isaac Tseng,Timothy G. Reese, and Robert M. Weisskoff, MRM 2005
        
    '''
    def __init__(self, data, bvals, gradients,odf_sphere='symmetric362', mask=None,half_sphere_grads=False,auto=True):
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

        See also
        ----------
        dipy.reconst.dti.Tensor, dipy.reconst.gqi.GeneralizedQSampling
        '''
        
        #read the vertices and faces for the odf sphere
        odf_vertices, odf_faces = sphere_vf_from(odf_sphere)
        self.odf_vertices=odf_vertices
        self.odf_faces=odf_faces
        self.odfn=len(self.odf_vertices)
        
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
        #3d volume for Sq
        self.sz=16
        #necessary shifting for centering
        self.origin=8
        #hanning filter width
        self.filter_width=32.        
        #create the q-table from bvecs and bvals        
        bv=bvals
        bmin=np.sort(bv)[1]
        bv=np.sqrt(bv/bmin)
        qtable=np.vstack((bv,bv,bv)).T*gradients
        qtable=np.floor(qtable+.5)
        self.qtable=qtable             
        #odf collecting radius
        self.radius=np.arange(2.1,6,.2)
        self.radiusn=len(self.radius)         
        #calculate r - hanning filter free parameter
        r = np.sqrt(qtable[:,0]**2+qtable[:,1]**2+qtable[:,2]**2)    
        #setting hanning filter width and hanning        
        self.filter=.5*np.cos(2*np.pi*r/self.filter_width)        
        #center and index in qspace volume
        self.q=qtable+self.origin
        self.q=self.q.astype('i8')
        #peak threshold
        self.peak_thr=2.        
        #precompute coordinates for pdf interpolation
        self.Xs=self.precompute_interp_coords()        
        #
        if auto:
            self.fit()        
        
    def fit(self):
        #memory allocations for 4D volumes 
        if len(self.datashape)==4:
            x,y,z,g=self.datashape        
            S=self.data.reshape(x*y*z,g)
            GFA=np.zeros((x*y*z))
            IN=np.zeros((x*y*z,5))
            NFA=np.zeros((x*y*z,5))
            QA=np.zeros((x*y*z,5))            
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
                #calculate the diffusion propagator or spectrum                   
                Pr=self.pdf(s)           
                #calculate the orientation distribution function        
                odf=self.odf(Pr)
                #normalization for QA
                glob_norm_param=max(np.max(odf),glob_norm_param)
                #calculate the generalized fractional anisotropy
                GFA[i]=self.std_over_rsm(odf)
                #find peaks
                peaks,inds=peak_finding(odf,self.odf_faces)
                #remove small peaks
                if len(peaks)>0:
                    ismallp=np.where(peaks/peaks.min()<self.peak_thr)                                                                        
                    l=ismallp[0][0]
                    if l<5:                                        
                        IN[i][:l] = inds[:l]
                        NFA[i][:l] = GFA[i]
                        QA[i][:l] = peaks[:l]-np.min(odf)
        if len(self.datashape) == 4:
            self.GFA=GFA.reshape(x,y,z)
            self.NFA=NFA.reshape(x,y,z,5)
            self.QA=QA.reshape(x,y,z,5)/glob_norm_param
            self.IN=IN.reshape(x,y,z,5)
            self.QA_norm=glob_norm_param            
        if len(self.datashape) == 2:
            self.GFA=GFA
            self.NFA=NFA
            self.QA=QA
            self.IN=IN
            self.QA_norm=None
        
    def pdf(self,s):
        values=s*self.filter
        #create the signal volume    
        Sq=np.zeros((self.sz,self.sz,self.sz))
        #fill q-space
        for i in range(self.dn):
            qx,qy,qz=self.q[i]            
            Sq[qx,qy,qz]+=values[i]
        #apply fourier transform
        Pr=fftshift(np.abs(np.real(fftn(fftshift(Sq),(self.sz,self.sz,self.sz)))))
        return Pr
        
    def odf(self,Pr):
        """ fill the odf by sampling radially on the pdf
        
        crucial parameter here is self.radius
        """
        odf = np.zeros(self.odfn)        
        """ 
        #for all odf vertices        
        for m in range(self.odfn):
            xi=self.origin+self.radius*self.odf_vertices[m,0]
            yi=self.origin+self.radius*self.odf_vertices[m,1]
            zi=self.origin+self.radius*self.odf_vertices[m,2]
            #apply linear 3d interpolation (trilinear)
            PrI=map_coordinates(Pr,np.vstack((xi,yi,zi)),order=1)
            for i in range(self.radiusn):
                odf[m]=odf[m]+PrI[i]*self.radius[i]**2
        """
        PrIs=map_coordinates(Pr,self.Xs,order=1)        
        #print PrIs.shape
        """ in pdf_to_odf an optimized version of the function below
        for m in range(self.odfn):
            for i in range(self.radiusn):
                odf[m]=odf[m]+PrIs[m*self.radiusn+i]*self.radius[i]**2
        """
        pdf_to_odf(odf,PrIs, self.radius,self.odfn,self.radiusn) 
        return odf
    
    def precompute_interp_coords(self):
        Xs=[]
        for m in range(self.odfn):
            xi=self.origin+self.radius*self.odf_vertices[m,0]
            yi=self.origin+self.radius*self.odf_vertices[m,1]
            zi=self.origin+self.radius*self.odf_vertices[m,2]
            Xs.append(np.vstack((xi,yi,zi)).T)
        return np.concatenate(Xs).T
    
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
    def ind(self):
        """ peak indices
        """
        return self.IN


def project_hemisph_bvecs(bvals,bvecs):
    """ project any near identical bvecs to the other hemisphere
    
    Notes
    -------
    Very useful when working with dsi data because the full q-space needs to be mapped.
    """
    bvs=bvals[1:]
    bvcs=bvecs[1:]
    b=bvs[:,None]*bvcs
    bb=np.zeros((len(bvs),len(bvs)))    
    pairs=[]
    for (i,vec) in enumerate(b):
        for (j,vec2) in enumerate(b):
            bb[i,j]=np.sqrt(np.sum((vec-vec2)**2))            
        I=np.argsort(bb[i])
        for j in I:
            if j!=i:
                break
        if (j,i) in pairs:
            pass
        else:
            pairs.append((i,j))
    bvecs2=bvecs.copy()
    for (i,j) in pairs:
        bvecs2[1+j]=-bvecs2[1+j]    
    return bvecs2,pairs

