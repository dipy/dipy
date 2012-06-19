""" Classes and functions for generalized q-sampling """
import numpy as np
import dipy.reconst.recspeed as rp
from dipy.utils.spheremakers import sphere_vf_from
from .odf import OdfModel

class GeneralizedQSampling(OdfModel):
    """ Implements Generalized Q-Sampling
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
    evenly distributed sphere is provided using 400+ points. This is
    usually sufficient for most of the datasets.
    
    GQI is performing better with specific grid-like acquisition schemes. The table used in the scanner for 101 directions + 1 b0 volume is provided in 
    dipy.data.get_data('gqi_vectors').

    See also
    --------
    dipy.tracking.propagation.EuDX, dipy.reconst.dti.Tensor, dipy.data.get_sphere
    """
    def __init__(self, bvals, gradients, Lambda=1.2, 
                 odf_sphere='symmetric642',squared=False):
        r""" Generates a model-free description for every voxel that can
        be used from simple to very complicated configurations like
        quintuple crossings if your datasets support them.

        You can use this class for every kind of DWI image but it will
        perform much better when you have a balanced sampling scheme.

        Implements equation [9] from Generalized Q-Sampling as
        described in Fang-Cheng Yeh, Van J. Wedeen, Wen-Yih Isaac Tseng.
        Generalized Q-Sampling Imaging. IEEE TMI, 2010.

        It also implement the radially squared version knowna as GQI2 as
        described in Garyfallidis et al. "Towards an accurate brain
        tractography", PhD thesis, Cambridge University, 2012.

        Parameters
        -----------
        bvals: array, shape (N,)
        gradients: array, shape (N,3) also known as bvecs
        Lambda: float, optional
            smoothing parameter - diffusion sampling length
        odf_sphere : None or str or tuple, optional
            input that will result in vertex, face arrays for a sphere.
        squared : boolean, True or False
            If True it will calculate the odf using the $L^2$ weighting. Which
            provides higher angular accuracy.

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
        
        '''
        odf_vertices, odf_faces = sphere_vf_from(odf_sphere)
        self.odf_vertices=np.ascontiguousarray(odf_vertices)
        self.odf_faces=np.ascontiguousarray(odf_faces)
        self.odfn=len(self.odf_vertices)
        self.mask=mask
        self.data=data
        self.save_odfs=save_odfs
        '''
        
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
                
        if squared==True:
            vf=np.vectorize(self.squared_radial_component)
            #which implements
            #def H(x):
            #    res=(2*x*np.cos(x) + (x**2-2)*np.sin(x))/x**3
            #    res[np.isnan(res)]=1/3.
            #    return res            
            self.input=np.dot(b_vector.T, self.odf_vertices.T) * Lambda/np.pi
            self.q2odf_params=np.real(vf(np.dot(b_vector.T, self.odf_vertices.T) * Lambda/np.pi))
        else:
            self.q2odf_params=np.real(np.sinc(np.dot(b_vector.T, self.odf_vertices.T) * Lambda/np.pi))
                
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

    def evaluate_odf(self,s):
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


def patch_vertices(vertices,pole, width):
    """
    find 'vertices' within the cone of 'width' degrees around 'pole'
    """
    return [i for i,v in enumerate(vertices) if np.abs(np.dot(v,pole)) > np.abs(np.cos(np.pi*width/180))]


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


