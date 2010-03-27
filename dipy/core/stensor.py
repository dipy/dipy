import math
import numpy as np
from numpy.linalg import lstsq as lsq
from numpy.linalg import eig

class STensorL():
    r'''

    Calculate a single tensor for every voxel with linear least squares
    fitting.

    bvals and bvecs must be provided as well. FA calculated from Mori
    et.al, Neuron 2006. See also David Tuch PhD thesis p. 64 and Mahnaz
    Maddah thesis p. 44 for the tensor derivation. 

    What this algorithm does? 

    Solves an overdetermined linear system of equations for every voxel $j$. 

    $g_{0}^{2}d_{00}+g_{1}^{2}d_{11}+g_{2}^{2}d_{22}+2g_{1}g_{0}d_{01}+2g_{0}g_{2}d_{02}+2g_{1}g_{2}d_{12}=-ln(S_{ij}/S0_{j})/b_{i}$
    
    where $b_{i}$ the current b-value and $g_{i}=[g_{0},g_{1},g_{2}]^{T}$
    the unit gradient direction. $d_{xx}$ are the values of the tensor
    which we assume that is a $3x3$ symmetric matrix. 

    .. math::

        D = \left(\begin{array}{ccc}
             d_{00} & d_{01} & d_{02}\\
             d_{01} & d_{11} & d_{12}\\
             d_{02} & d_{12} & d_{22}\end{array}\right)
                
    where $b_{i}$ the current b-value and the current unit gradient direction.
    $dxx$ are the values of the symmetric matrix $D.$ $dxx$ are also
    the unknown variables - coefficients that we try to estimate by the
    fitting. 
         
    '''


    def __init__(self,b,g):


        if not b[0]==0.: ValueError('first b-value needs to be the b=0')

        A=[]
        
        for i in range(1,len(b)):

            g0,g1,g2=g[i]
            A.append(np.array([[g0*g0,g1*g1,g2*g2,2*g0*g1,2*g0*g2,2*g1*g2]]))

        self.A=np.concatenate(A)
        self.b=np.asfarray(b[1:])
        self.g=np.asarray(g[1:])

        self.coeff=None
        self.tensor=None
        self.FA=None
        self.ADC=None
        self.data_shape=None
        
        

    def voxel_fit(self,d):
        ''' fit the model for a single voxel with signal values d

        '''
        
        s0=d[0]; s=d[1:]        
        
        print self.b.shape
        print np.log(s/float(s0)).shape

        d=-(1/self.b)*np.log(s/float(s0))
        #check for division by zero or inf
        inds=np.isfinite(d)
        A=self.A[inds]
        d=d[inds]
        if len(d)>=6:
            x,resids,rank,sing=lsq(A,d)
        else:
            x=np.zeros(6)
    
        return x

        

    def fit(self,data,mask=None):

        ''' Fit the single tensor with linear least squares for every
        voxel in the volume data.
        
        Parameters:
        -----------
        data: 4d array, shape (X,Y,Z,D) with the signal values where
        X,Y,Z the dimensions of the volume and D is the dimension of
        vector holding the signal values for that voxels combined from
        all directions. We assume here that the first element of the
        vector will hold the value for the signal without diffusion gradients.

        Returns:
        --------
        coeff: 4d array, shape (X,Y,Z,6) with the coefficients of the model fit in every voxel.

        '''

        
        
        self.data_shape=data.shape


        if data.ndim==4:
        
            data=data.reshape(data.shape[0]*data.shape[1]*data.shape[2],data.shape[3])

                    
        self.coeff=np.array([self.voxel_fit(d) for d in data])

        self.tensor=None
        self.FA=None
        self.ADC=None

        return self.coeff
                        
    

    def evaluate(self):

        pass

    def voxel_tensor(self,c):
        ''' calculate the tensor only in one voxel by reading the
        coefficients produced after the fit

        '''

        if c.any()==np.zeros(6).any():
            return np.zeros(3), np.zeros((3,3))

        else:

            c00,c11,c22,c01,c02,c12=c 
            C=np.array([[c00, c01, c02],[c01,c11,c12],[c02,c12,c22]])                            
            evals,evecs=eig(C)          
        
            return evals,evecs

    @property
    def tensors(self):
        ''' calculate the tensors in a volume

        '''

        if self.coeff != None:

            self.tensor=[self.voxel_tensor(c) for c in self.coeff]
            return self.tensor
        

    def voxel_fa(self,evals):

        if evals.any() < 5.96e-08:

            return 0

        else:

            l1=evals[0]; l2=evals[1]; l3=evals[2]                 
            fa=math.sqrt(((l1-l2)**2 + (l2-l3)**2 + (l3-l1)**2 )/( 2*(l1**2+l2**2+l3**2))) 
        
            return fa
        

    @property
    def fa(self):
        ''' Calculate fractional anisotropy for every voxel in the volume
    

        '''


        if self.tensor!=None:

            if self.FA == None:

                if len(self.data_shape)==4:

                    self.FA=np.array([self.voxel_fa(t[0]) for t in self.tensor]).reshape(self.data_shape[:3])
                else:

                    self.FA=np.array([self.voxel_fa(t[0]) for t in self.tensor])
                    
                return self.FA

            else:

                return self.FA

        else:

            self.tensors
            self.fa
            

            
    def voxel_adc(self,evals):

        return sum(evals)/3.
        
    @property
    def adc(self):

        if self.tensor!=None:

            if self.ADC == None:

                if len(self.data_shape)==4:

                    self.ADC=np.array([self.voxel_adc(t[0]) for t in self.tensor]).reshape(self.data_shape[:3])        
                else:

                    self.ADC=np.array([self.voxel_adc(t[0]) for t in self.tensor])
                    
                return self.ADC
        
        else:

            self.tensors
            self.adc
            

    


