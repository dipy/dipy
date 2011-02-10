import numpy as np

from dipy.tracking.propspeed import eudx_both_directions
from dipy.tracking.metrics import length
from dipy.data import get_sphere

class EuDX(object):
    ''' Euler Delta Crossings

    Generates tracks with termination criteria defined by a
    delta function [1]_ and it has similarities with FACT algorithm [2]_ and Basser's method 
    but uses trilinear interpolation.

    Can be used with any reconstruction method as DTI, DSI, QBI, GQI which can
    calculate an orientation distribution function and find the local peaks of
    that function. For example a single tensor model can give you only
    one peak a dual tensor model 2 peaks and quantitative anisotropy
    method as used in GQI can give you 3,4,5 or even more peaks.    

    The parameters of the delta function are checking thresholds for the
    direction propagation magnitude and the angle of propagation.

    A specific number of seeds is defined randomly and then the tracks
    are generated for that seed if the delta function returns true.

    Trilinear interpolation is being used for defining the weights of
    the propagation.

    References
    ------------
    .. [1] Yeh. et al. Generalized Q-Sampling Imaging, TMI 2010.

    .. [2] Mori et al. Three-dimensional tracking of axonal projections
    in the brain by magnetic resonance imaging. Ann. Neurol. 1999.
    '''

    def __init__(self, a, ind,
                 seeds=10000,
                 odf_vertices=None,
                 a_low=0.0239,
                 step_sz=0.5,
                 ang_thr=60.,
                 length_thr=0.,
                 total_weight=.5):
        ''' Euler integration with multiple stopping criteria and supporting multiple peaks

        Parameters
        ------------
        a : array, shape(x,y,z,Np)
            magnitude of the peak of a scalar anisotropic function e.g. QA
            (quantitative anisotropy)  or a different function of shape(x,y,z)
            e.g FA or GFA.
        ind : array, shape(x,y,z,Np)
            indices of orientations of the scalar anisotropic peaks found on the
            resampling sphere
        seeds : int or sequence, optional
            number of random seeds or list of seeds
        odf_vertices : None or ndarray, optional
            sphere points which define a discrete representation of orientations
            for the peaks, the same for all voxels. None results in 
        a_low : float, optional
            low threshold for QA(typical 0.023)  or FA(typical 0.2) or any other
            anisotropic function
        step_sz : float, optional
            euler propagation step size
        ang_thr : float, optional
            if turning angle is bigger than this threshold then tracking stops.
        length_thr: float, optional
        total_weight : float, optional
            total weighting threshold

        Examples
        --------
        >>> import nibabel as nib
        >>> from dipy.reconst.dti import Tensor
        >>> from dipy.data import get_data
        >>> fimg,fbvals,fbvecs=get_data('small_101D')
        >>> img=nib.load(fimg)
        >>> affine=img.get_affine()
        >>> bvals=np.loadtxt(fbvals)
        >>> gradients=np.loadtxt(fbvecs).T
        >>> data=img.get_data()
        >>> ten=Tensor(data,bvals,gradients,thresh=50)
        >>> eu=EuDX(a=ten.fa(),ind=ten.ind(),seeds=100,a_low=.2)
        >>> tracks=[e for e in eu]

        Notes
        -------
        This works as an iterator class because otherwise it could fill your entire RAM if you generate many tracks. 
        Something very common as you can easily generate millions of tracks if you have many seeds.

        '''
        self.a=a.copy()
        self.ind=ind.copy()
        self.a_low=a_low
        self.ang_thr=ang_thr
        self.step_sz=step_sz
        self.length_thr=length_thr
        self.total_weight=total_weight

        if len(self.a.shape)==3:
            self.a.shape=self.a.shape+(1,)
            self.ind.shape=self.ind.shape+(1,)

        #store number of maximum peacks
        x,y,z,g=self.a.shape
        self.Np=g
        tlist=[]

        if odf_vertices==None:
            vertices, faces = get_sphere('symmetric362')
            self.odf_vertices = vertices
        '''
        print 'Shapes'
        print 'a',self.a.shape, self.a.dtype
        print 'ind',self.ind.shape, self.ind.dtype
        print 'odf_vertices',self.odf_vertices.shape, self.odf_vertices.dtype
        '''

        try:
            if len(seeds)>0:
                self.seed_list=seeds
                self.seed_no=len(seeds)
        except TypeError:
            self.seed_no=seeds
            self.seed_list=None

        self.ind=self.ind.astype(np.double)

    def __iter__(self):
        ''' This is were all the fun starts '''
        x,y,z,g=self.a.shape
        #for all seeds
        for i in range(self.seed_no):
            
            if self.seed_list==None:
                rx=(x-1)*np.random.rand()
                ry=(y-1)*np.random.rand()
                rz=(z-1)*np.random.rand()            
                seed=np.ascontiguousarray(np.array([rx,ry,rz]),dtype=np.float64)
            else:
                seed=np.ascontiguousarray(self.seed_list[i],dtype=np.float64)
                            
            #for all peaks
            for ref in range(self.a.shape[-1]): 
                #propagate up and down 
                track =eudx_both_directions(seed.copy(),ref,self.a,self.ind,self.odf_vertices,self.a_low,self.ang_thr,self.step_sz,self.total_weight)                  
                if track == None:
                    pass
                else:        
                    #return a track from that seed                                        
                    if length(track)>self.length_thr:                        
                        yield track
                        
                        






