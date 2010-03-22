import math
import numpy as np
from numpy.linalg import lstsq as lsq
from numpy.linalg import eigh

class sltensor():
    ''' Calculate a single tensor for every voxel with linear least squares fitting


    bvals and bvecs must be provided as well.  FA calculated from Mori
    et.al, Neuron 2006 . See also David Tuch PhD thesis p. 64 and Mahnaz Maddah thesis p. 44 for the tensor derivation.
    
    What this algorithm does? Solves a system of equations for every voxel j
    
    g0^2*d00 + g1^2*d11+g2^2*d22+ 2*g1*g0*d01+ 2*g0*g2*d02+2*g1*g2*d12 = - ln(S_ij/S0_j)/b_i
    
    where b_i the current b-value and g_i=[g0,g1,g2] the current gradient direction. dxx are the values of 
    the symmetric matrix D. dxx are also the unknown variables.
    
    D=[[d00 ,d01,d02],[d01,d11,d12],[d02,d12,d22]]

    Examples:
    ---------
   
         
    '''


    def __init__(self,b,g):


        if not b[0]==0.: ValueError('first b-value needs to be the b=0')

        A=[]

        
        for i in range(1,len(b)):

            g0,g1,g2=g[i]
            A.append(np.array([[g0*g0,g1*g1,g2*g2,2*g0*g1,2*g0*g2,2*g1*g2]]))

        self.A=np.concatenate(A)
        self.b=b
        self.g=g

        self.coeff=None
        self.tensor=None
        self.FA=None
        self.ADC=None
        self.data_shape=None
        
        

    def voxel_fit(self,d):
        ''' fit the model for a single voxel with signal values d

        '''
        
        s0=d[0]; s=d[1:]        
        #d=np.log(s)-np.log(s0)
        d=np.log(s/float(s0))
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

        
        self.data_shape=data.shape
        
        data=data.reshape(data.shape[0]*data.shape[1]*data.shape[2],data.shape[3])

                    
        self.coeff=np.array([self.voxel_fit(d) for d in data])

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
            evals,evecs=eigh(C)          
        
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

        if self.tensor!=None:

            if self.FA == None:

                self.FA=np.array([self.voxel_fa(t[0]) for t in self.tensor]).reshape(self.data_shape[:3])        

                return self.FA


    def voxel_adc(self,evals):

        return sum(evals)/3.
        
    @property
    def adc(self):

        if self.tensor!=None:

            if self.ADC == None:

                self.ADC=np.array([self.voxel_adc(t[0]) for t in self.tensor]).reshape(self.data_shape[:3])        

                return self.ADC
        
        pass

    


