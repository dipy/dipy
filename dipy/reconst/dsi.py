import numpy as np
from scipy.ndimage import map_coordinates
from dipy.reconst.recspeed import peak_finding, pdf_to_odf
from scipy.fftpack import fftn, fftshift
from .odf import OdfModel, OdfFit
from dipy.utils.spheremakers import sphere_vf_from

class DiffusionSpectrumModel(OdfModel):
    ''' Calculate the PDF and ODF using Diffusion Spectrum Imaging
    
    Based on the paper "Mapping Complex Tissue Architecture With Diffusion Spectrum Magnetic Resonance Imaging"
    by Van J. Wedeen,Patric Hagmann,Wen-Yih Isaac Tseng,Timothy G. Reese, and Robert M. Weisskoff, MRM 2005
        
    '''
    def __init__(self, bvals, gradients, odf_sphere='symmetric362',
                 deconv=False, half_sphere_grads=False):
        '''
        Parameters
        -----------
        bvals : array, shape (N,)
        gradients : array, shape (N,3) also known as bvecs        
        odf_sphere : tuple, (verts, faces, edges)
        deconv : bool, use deconvolution
        half_sphere_grad : boolean Default(False) 
            in order to create the q-space we use the bvals and gradients. 
            If the gradients are only one hemisphere then 
        See also
        ----------
        dipy.reconst.dti.Tensor, dipy.reconst.gqi.GeneralizedQSampling
        '''
        b0 = 0
        self.bvals=bvals
        self.gradients=gradients
        #3d volume for Sq
        self.sz=16
        #necessary shifting for centering
        self.origin=8
        #hanning filter width
        self.filter_width=32.                     
        #odf collecting radius
        self.radius=np.arange(2.1,6,.2)
        #odf sphere
        odf_vertices, odf_faces = sphere_vf_from(odf_sphere)
        self.set_odf_vertices(odf_vertices,None,odf_faces)
        self.odfn=len(self._odf_vertices)
        #number of single sampling points
        self.dn = (bvals > b0).sum()
        self.num_b0 = len(bvals) - self.dn
        self.create_qspace()
    
    def create_qspace(self):
        
        #create the q-table from bvecs and bvals        
        bv=self.bvals
        bmin=np.sort(bv)[1]
        bv=np.sqrt(bv/bmin)
        qtable=np.vstack((bv,bv,bv)).T*self.gradients
        qtable=np.floor(qtable+.5)
        self.qtable=qtable
        
        self.radiusn=len(self.radius)
        #calculate r - hanning filter free parameter
        r = np.sqrt(qtable[:,0]**2+qtable[:,1]**2+qtable[:,2]**2)    
        #setting hanning filter width and hanning        
        self.filter=.5*np.cos(2*np.pi*r/self.filter_width)                
        #center and index in qspace volume
        self.q=qtable+self.origin
        self.q=self.q.astype('i8')
        #peak threshold
        self.peak_thr=.7
        self.iso_thr=.4  

        #precompute coordinates for pdf interpolation
        self.precompute_interp_coords()    
   
    def evaluate_odf(self,s):
        return self.pdf_odf(self.pdf(s))

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
    
    def odf(self,s):
        Pr=self.pdf(s)       
        #calculate the orientation distribution function        
        odf=self.pdf_odf(Pr)
        return odf        
        
    def pdf_odf(self,Pr):
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
        self.Xs=np.concatenate(Xs).T


def project_hemisph_bvecs(bvals,bvecs):
    """ Project any near identical bvecs to the other hemisphere
    
    Notes
    -------
    Useful when working with dsi data because the full q-space needs to be mapped
    in both hemi-spheres.
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

if __name__ == '__main__':

    pass
