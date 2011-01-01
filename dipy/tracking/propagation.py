import os
import numpy as np
from dipy.tracking.propspeed import eudx_propagation
from dipy.tracking.metrics import length
from dipy.data import get_sphere

class EuDX():
    ''' Euler Delta Crossings
    
    Generates tracks with termination criteria defined by a
    delta function [1]_ and it has similarities with FACT algorithm [2]_ and Basser's method 
    but uses trilinear interpolation.

    Can be used with any reconstruction method as DTI,DSI,QBI,GQI which can
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

    def __init__(self,qa,ind,seed_list=None,seed_no=10000,odf_vertices=None,qa_thr=0.0239,step_sz=0.5,ang_thr=60.,length_thr=0.):
        ''' Euler integration with multiple stopping criteria and supporting multiple peaks
        
        Parameters
        ------------
        qa : array, shape(x,y,z,Np), magnitude of the peak (QA) or
        shape(x,y,z) a scalar volume like FA.

        ind : array, shape(x,y,z,Np), indices of orientations of the QA
        peaks found at odf_vertices used in QA or, shape(x,y,z), ind

        seed_list : list of seeds
        
        seed_no : number of random seeds if seed_list is None

        odf_vertices : sphere points which define a discrete
        representation of orientations for the peaks, the same for all voxels

        qa_thr : float, threshold for QA(typical 0.023)  or FA(typical 0.2) 
        step_sz : float, propagation step

        ang_thr : float, if turning angle is bigger than this threshold
        then tracking stops.
        
        Examples
        ----------
        This works as an iterator class because otherwise it could fill your entire RAM if you generate many tracks. 
        Something very common as you can easily generate millions of tracks.

        '''
        
        self.qa=qa.copy()
        self.ind=ind.copy()
        self.qa_thr=qa_thr
        self.ang_thr=ang_thr
        self.step_sz=step_sz
        self.length_thr=length_thr
        
        if len(self.qa.shape)==3:            
            self.qa.shape=self.qa.shape+(1,)
            self.ind.shape=self.ind.shape+(1,)

        #store number of maximum peacks
        x,y,z,g=self.qa.shape
        self.Np=g
        tlist=[]      

        if odf_vertices==None:
            eds=np.load(get_sphere('symmetric362'))
            self.odf_vertices=eds['vertices']
            
        print 'Shapes'
        print 'qa',self.qa.shape, self.qa.dtype
        print 'ind',self.ind.shape, self.ind.dtype
        print 'odf_vertices',self.odf_vertices.shape, self.odf_vertices.dtype
        
        self.seed_no=seed_no
        self.seed_list=seed_list
        
        if self.seed_list!=None:
            self.seed_no=len(seed_list)
            
#        if self.seed_list==None:
#            self.seed_list=[]
#            #for all seed points    
#            for i in range(self.seed_no):
#                rx=(x-1)*np.random.rand()
#                ry=(y-1)*np.random.rand()
#                rz=(z-1)*np.random.rand()
#                self.seed_list.append(np.array([rx,ry,rz]))          

        self.ind=self.ind.astype(np.double)        
        
    def __iter__(self):
        ''' This is were all the fun starts '''
        x,y,z,g=self.qa.shape
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
            for ref in range(self.qa.shape[-1]): 
                #propagate up and down 
                track =eudx_propagation(seed.copy(),ref,self.qa,self.ind,self.odf_vertices,self.qa_thr,self.ang_thr,self.step_sz)                  
                if track == None:
                    pass
                else:        
                    #tlist.append(track.astype(np.float32))                                        
                    if length(track)>self.length_thr:                        
                        yield track
                        
                        
    '''           
    def native(self,affine):        
        print affine.shape
        print self.tracks[0].shape
        self.tracks=[np.transpose(np.dot(affine[:3,:3],np.transpose(t)))+np.transpose(affine[:3,3]) for t in self.tracks]        
    '''






