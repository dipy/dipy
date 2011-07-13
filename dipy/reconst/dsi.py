import warnings
import numpy as np
from scipy.ndimage import map_coordinates
from dipy.reconst.recspeed import peak_finding, pdf_to_odf
from dipy.utils.spheremakers import sphere_vf_from
from scipy.fftpack import fftn, fftshift, ifftn,ifftshift


#warnings.warn("This module is most likely to change both as a name and in structure in the future",FutureWarning)

class DiffusionSpectrum(object):
    ''' Calculate the PDF and ODF using Diffusion Spectrum Imaging
    
    Based on the paper "Mapping Complex Tissue Architecture With Diffusion Spectrum Magnetic Resonance Imaging"
    by Van J. Wedeen,Patric Hagmann,Wen-Yih Isaac Tseng,Timothy G. Reese, and Robert M. Weisskoff, MRM 2005
        
    '''
    def __init__(self, data, bvals, gradients,odf_sphere='symmetric362', mask=None):
        '''
        Parameters
        -----------
        data : array, shape(X,Y,Z,D) , or (X,D)
        bvals : array, shape (N,)
        gradients : array, shape (N,3) also known as bvecs        
        odf_sphere : str or tuple, optional
            If str, then load sphere of given name using ``get_sphere``.
            If tuple, gives (vertices, faces) for sphere.

        See also
        ----------
        dipy.reconst.dti.Tensor, dipy.reconst.gqi.GeneralizedQSampling
        '''
        
        #read the vertices and faces for the odf sphere
        odf_vertices, odf_faces = sphere_vf_from(odf_sphere)
        self.odf_vertices=odf_vertices
        self.odfn=len(self.odf_vertices)
        #load bvals and bvecs
        self.bvals=bvals
        gradients[np.isnan(gradients)] = 0.
        self.gradients=gradients
        #save number of total diffusion volumes
        self.dn=data.shape[-1]        
        S=data
        datashape=S.shape #initial shape
        msk=None #tmp mask                
        #3d volume for Sq
        self.sz=16
        #necessary shifting for centering
        self.origin=8
        #hanning filter width
        self.filter_width=32.        
        #creat the q-table from bvecs and bvals
        bv=bvals
        bmin=np.sort(bv)[1]
        bv=np.sqrt(bv/bmin)
        qtable=np.vstack((bv,bv,bv)).T*gradients
        qtable=np.floor(qtable+.5)             
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
        self.Xs=self.precompute()

        if len(datashape)==4:
            x,y,z,g=S.shape        
            S=S.reshape(x*y*z,g)
            GFA=np.zeros((x*y*z))
            IN=np.zeros((x*y*z,5))  
            if mask != None:
                if mask.shape[:3]==datashape[:3]:
                    msk=mask.ravel().copy()
                    
        if len(datashape)==2:
            x,g= S.shape
            GFA=np.zeros(x)
            IN=np.zeros((x,5)) 
        
        if mask !=None:
            for (i,s) in enumerate(S):        
                if msk[i]>0:
                    #calculate the diffusion propagator or spectrum                   
                    Pr=self.pdf(s)           
                    #calculate the orientation distribution function        
                    odf=self.odf(Pr)
                    #calculate the generalized fractional anisotropy
                    GFA[i]=self.std_over_rsm(odf)                   
                    #find peaks
                    peaks,inds=peak_finding(odf,odf_faces)
                    #remove small peaks
                    #print odf
                    #print peaks
                    if len(peaks)>0:
                        ismallp=np.where(peaks/peaks.min()<self.peak_thr)                                                                        
                        l=ismallp[0][0]
                        if l<5:                                        
                            IN[i][:l] = inds[:l]

        if mask==None:
            for (i,s) in enumerate(S):
                #calculate the diffusion propagator or spectrum
                Pr=self.pdf(s)
                #calculate the orientation distribution function
                odf=self.odf(Pr)
                #calculate the generalized fractional anisotropy
                GFA[i]=self.std_over_rsm(odf) 
                #find peaks
                peaks,inds=peak_finding(odf,odf_faces)                               
                #remove small peaks
                #print odf
                #print peaks
                if len(peaks)>0:
                    ismallp=np.where(peaks/peaks.min()<self.peak_thr)
                    #keep only peaks with high values                                                    
                    l=ismallp[0][0]
                    #print l, len(peaks), len(inds)
                    if l<5:                    
                        IN[i][:l] = inds[:l]
                
        if len(datashape) == 4:
            self.GFA=GFA.reshape(x,y,z)
            self.IN=IN.reshape(x,y,z,5)           

        if len(datashape) == 2:            
            self.GFA=GFA
            self.IN=IN
            
    
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
        #fill the odf by sampling radially on the pdf
        #crucial parameter here is self.radius
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
        """
        for m in range(self.odfn):
            for i in range(self.radiusn):
                odf[m]=odf[m]+PrIs[m*self.radiusn+i]*self.radius[i]**2
        """
        pdf_to_odf(odf,PrIs, self.radius,self.odfn,self.radiusn)                 
 
        return odf
    
    def precompute(self):            
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
        return self.GFA
    def ind(self):
        return self.IN

