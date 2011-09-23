""" Classes and functions for generalized q-sampling """
import numpy as np

import dipy.reconst.recspeed as rp

from dipy.utils.spheremakers import sphere_vf_from


class GeneralizedQSampling(object):
    """ Implements Generalized Q-Sampling

    Generates a model-free description for every voxel that can
    be used from simple to very complicated configurations like
    quintuple crossings if your datasets support them.

    You can use this class for every kind of DWI image but it will
    perform much better when you have a balanced sampling scheme.

    Implements equation [9] from Generalized Q-Sampling as
    described in Fang-Cheng Yeh, Van J. Wedeen, Wen-Yih Isaac Tseng.
    Generalized Q-Sampling Imaging. IEEE TMI, 2010.

    Parameters
    -----------
    data : array,
        shape(X,Y,Z,D) or shape(X,D)
    bvals : array,
        shape (N,)
    gradients : array,
        shape (N,3) also known as bvecs
    Lambda : float,
        smoothing parameter - diffusion sampling length

    Properties
    ----------
    QA : array, shape(X,Y,Z,5), quantitative anisotropy
    IN : array, shape(X,Y,Z,5), indices of QA, qa unit directions
    fwd : float, normalization parameter

    Notes
    -----
    In order to reconstruct the spin distribution function a nice symmetric
    evenly distributed sphere is provided using 362 or 642 points. This is
    usually sufficient for most of the datasets.
    
    GQI is performing better with specific grid-like acquisition schemes. The table
    used in the scanner for 101 directions + 1 b0 volume is provided in 
    dipy.data.get_data('gqi_vectors').

    See also
    --------
    dipy.tracking.propagation.EuDX, dipy.reconst.dti.Tensor, dipy.data.get_sphere
    """
    def __init__(self, data, bvals, gradients,
                 Lambda=1.2, odf_sphere='symmetric362', mask=None,squared=False,auto=True,save_odfs=False):
        r""" Generates a model-free description for every voxel that can
        be used from simple to very complicated configurations like
        quintuple crossings if your datasets support them.

        You can use this class for every kind of DWI image but it will
        perform much better when you have a balanced sampling scheme.

        Implements equation [9] from Generalized Q-Sampling as
        described in Fang-Cheng Yeh, Van J. Wedeen, Wen-Yih Isaac Tseng.
        Generalized Q-Sampling Imaging. IEEE TMI, 2010.

        Parameters
        -----------
        data: array, shape(X,Y,Z,D)
        bvals: array, shape (N,)
        gradients: array, shape (N,3) also known as bvecs
        Lambda: float, optional
            smoothing parameter - diffusion sampling length
        odf_sphere : None or str or tuple, optional
            input that will result in vertex, face arrays for a sphere.
        mask : None or ndarray, optional
        squared : boolean, True or False
            If True it will calculate the odf using the $L^2$ weighting.
        auto : boolean, default True 
            if True then the processing of all voxels will start automatically 
            with the class constructor,if False then you will have to call .fit()
            in order to do the heavy duty processing for every voxel
        save_odfs : boolean, default False
            save odfs, which is memory expensive

        Key Properties
        ---------------
        QA : array, shape(X,Y,Z,5), quantitative anisotropy
        IN : array, shape(X,Y,Z,5), indices of QA, qa unit directions
        fwd : float, normalization parameter

        Notes
        -------
        In order to reconstruct the spin distribution function  a nice symmetric
        evenly distributed sphere is provided using 362 points. This is usually
        sufficient for most of the datasets.

        See also
        --------
        dipy.tracking.propagation.EuDX, dipy.reconst.dti.Tensor,
        dipy.data.__init__.get_sphere
        """
        odf_vertices, odf_faces = sphere_vf_from(odf_sphere)
        self.odf_vertices=odf_vertices
        self.odf_faces=odf_faces
        self.odfn=len(self.odf_vertices)
        self.mask=mask
        self.data=data
        self.save_odfs=save_odfs
        self.squared=squared
        
        # 0.01506 = 6*D where D is the free water diffusion coefficient 
        # l_values sqrt(6 D tau) D free water diffusion coefficient and
        # tau included in the b-value
        scaling = np.sqrt(bvals*0.01506)
        tmp=np.tile(scaling,(3,1))
        #the b vectors might have nan values where they correspond to b
        #value equals with 0
        gradients[np.isnan(gradients)]= 0.
        gradsT = gradients.T
        b_vector=gradsT*tmp # element-wise also known as the Hadamard product
        #q2odf_params=np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi)
        
        
        if squared==True:
            vf=np.vectorize(self.squared_radial_component)
            def H(x):
                res=(2*x*np.cos(x) + (x**2-2)*np.sin(x))/x**3
                res[np.isnan(res)]=1/3.
                return res
            
            self.input=np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi
            self.q2odf_params=np.real(vf(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi))
            #self.q2odf_params=np.real(H(1*np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi))
        else:
            self.q2odf_params=np.real(np.sinc(np.dot(b_vector.T, odf_vertices.T) * Lambda/np.pi))
                
        #q2odf_params[np.isnan(q2odf_params)]= 1.
        #define total mask 
        #tot_mask = (mask > 0) & (data[...,0] > thresh)
        self.peak_thr=.3
        self.iso_thr=.9
        
        
        if auto:
            self.fit()
    
    def fit(self):
        """ process all voxels
        """    
        S=self.data
        datashape=S.shape #initial shape       
        #memory allocations for 4D volumes
        if len(datashape)==4:
            x,y,z,g=S.shape        
            S=S.reshape(x*y*z,g)
            QA = np.zeros((x*y*z,5))
            IN = np.zeros((x*y*z,5))
            if self.save_odfs:
                ODF=np.zeros((x*y*z,self.odfn))  
            if self.mask != None:
                if self.mask.shape[:3]==datashape[:3]:
                    msk=self.mask.ravel().copy()
            if self.mask == None:
                self.mask=np.ones(datashape[:3])
                msk=self.mask.ravel().copy()
        #memory allocations for a series of voxels
        if len(datashape)==2:
            x,g= S.shape
            QA = np.zeros((x,5))
            IN = np.zeros((x,5))
            if self.save_odfs:
                ODF=np.zeros((x,self.odfn))            
            if self.mask != None:
                if self.mask.shape[0]==datashape[0]:
                    msk=self.mask.ravel().copy()
            if self.mask == None:
                self.mask=np.ones(datashape[:1])
                msk=self.mask.ravel().copy()
        glob_norm_param = 0        
        #Calculate Quantitative Anisotropy and 
        #find the peaks and the indices
        #for every voxel        
        for (i,s) in enumerate(S):                            
            if msk[i]>0:
                #Q to ODF
                odf=np.dot(s,self.q2odf_params)
                min_odf=np.min(odf)
                if self.save_odfs:
                    ODF[i]=odf#-min_odf            
                peaks,inds=rp.peak_finding(odf,self.odf_faces)            
                glob_norm_param=max(np.max(odf),glob_norm_param)                
                #print peaks,min_odf
                #remove the isotropic part
                l=self.reduce_peaks(peaks,min_odf)
                if l==0:
                    QA[i][0] = peaks[0]-min_odf
                    IN[i][0] = inds[0]
                if l>0 and l<5:
                    QA[i][:l] = peaks[:l]-min_odf
                    IN[i][:l] = inds[:l]


        #normalize QA
        QA/=glob_norm_param
        if len(datashape) == 4:
            self.QA=QA.reshape(x,y,z,5)    
            self.IN=IN.reshape(x,y,z,5)  
            if self.save_odfs:
                self.ODF=ODF.reshape(x,y,z,ODF.shape[-1])
            self.QA_norm= glob_norm_param         
        if len(datashape) == 2:
            self.QA=QA
            self.IN=IN 
            if self.save_odfs:
                self.ODF=ODF
            self.QA_norm=None           
        self.glob_norm_param = glob_norm_param
    
    def reduce_peaks(self,peaks,odf_min):
        """ helping peak_finding when too many peaks are available 
        
        """
        if len(peaks)==0:
            return -1 
        if odf_min<self.iso_thr*peaks[0]:
            #remove small peaks
            ismallp=np.where(peaks<self.peak_thr*peaks[0])
            if len(ismallp[0])>0:
                l=ismallp[0][0]
            else:
                l=len(peaks)
        else:
            return -1
        return l

    def squared_radial_component(self,x):
        """ implementing equation (8) in the referenced paper by Yeh et al. 2010
        """
        #if x < np.finfo('f4').tiny and  x > - np.finfo('f4').tiny:
        if x < 0.01 and x > -0.01:
            #print 'small'
            return 1/3.
        return (2*x*np.cos(x) + (x**2-2)*np.sin(x))/x**3
    
    def qa(self):
        """ quantitative anisotropy
        """
        return self.QA
    
    def ind(self):
        """ 
        indices on the sampling sphere
        """
        return self.IN

    def odf(self,s):
        """ spin density orientation distribution function
         
        Parameters
        -----------        
        s : array, shape(D),
            diffusion signal for one point in the dataset
        
        Returns
        ---------
        odf : array, shape(len(odf_vertices)), 
            spin density orientation distribution function        

        """
        return np.dot(s,self.q2odf_params)

    def odfs(self):
        return self.ODF

    def npa(self,s,width=5):
        """ non-parametric anisotropy
        
        Nimmo-Smith et. al  ISMRM 2011
        """   
        odf=self.odf(s)
        t0,t1,t2=triple_odf_maxima(self.odf_vertices, odf, width)
        psi0 = t0[1]**2
        psi1 = t1[1]**2
        psi2 = t2[1]**2
        npa = np.sqrt((psi0-psi1)**2+(psi1-psi2)**2+(psi2-psi0)**2)/np.sqrt(2*(psi0**2+psi1**2+psi2**2))
        #print 'tom >>>> ',t0,t1,t2,npa

        return t0,t1,t2,npa

def equatorial_zone_vertices(vertices, pole, width=5):
    """
    finds the 'vertices' in the equatorial zone conjugate
    to 'pole' with width half 'width' degrees
    """
    return [i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole)) < np.abs(np.sin(np.pi*width/180))]

def polar_zone_vertices(vertices, pole, width=5):
    """
    finds the 'vertices' in the equatorial band around
    the 'pole' of radius 'width' degrees
    """
    return [i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole)) > np.abs(np.cos(np.pi*width/180))]


def upper_hemi_map(v):
    """
    maps a 3-vector into the z-upper hemisphere
    """
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

#"""
def patch_vertices(vertices,pole, width):
    """
    find 'vertices' within the cone of 'width' degrees around 'pole'
    """
    return [i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole)) > np.abs(np.cos(np.pi*width/180))]
#"""

def patch_maximum(vertices, odf, pole, width):
    eqvert = patch_vertices(vertices, pole, width)    
    #need to test for whether eqvert is empty or not    
    if len(eqvert) == 0:
        print('empty cone around pole %s with with width %f' % (np.array_str(pole), width))
        return np.Null, np.Null
    eqvals = [odf[i] for i in eqvert]
    eqargmax = np.argmax(eqvals)
    eqvertmax = eqvert[eqargmax]
    eqvalmax = eqvals[eqargmax]
    return eqvertmax, eqvalmax

def odf_sum(odf):
    return np.sum(odf)


def patch_sum(vertices, odf, pole, width):
    eqvert = patch_vertices(vertices, pole, width)    
    #need to test for whether eqvert is empty or not    
    if len(eqvert) == 0:
        print('empty cone around pole %s with with width %f' % (np.array_str(pole), width))
        return np.Null
    return np.sum([odf[i] for i in eqvert])

def triple_odf_maxima(vertices, odf, width):

    indmax1 = np.argmax([odf[i] for i,v in enumerate(vertices)])
    odfmax1 = odf[indmax1]
    pole = vertices[indmax1]
    eqvert = equatorial_zone_vertices(vertices, pole, width)
    indmax2, odfmax2 = equatorial_maximum(vertices,\
                                              odf, pole, width)
    indmax3 = eqvert[np.argmin([np.abs(np.dot(vertices[indmax2],vertices[p])) for p in eqvert])]
    odfmax3 = odf[indmax3]
    """
    cross12 = np.cross(vertices[indmax1],vertices[indmax2])
    cross12 = cross12/np.sqrt(np.sum(cross12**2))    
    indmax3, odfmax3 = patch_maximum(vertices, odf, cross12, 2*width)
    """
    return [(indmax1, odfmax1),(indmax2, odfmax2),(indmax3, odfmax3)]


