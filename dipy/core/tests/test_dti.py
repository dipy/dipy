""" Testing DTI

"""

import numpy as np

import dipy.core.dti as dti
#for reading in nifti test data
import nibabel as nib
from dipy.io.bvectxt import read_bvec_file

from nose.tools import assert_true, assert_false, \
     assert_equal, assert_raises

from numpy.testing import assert_array_equal, assert_array_almost_equal

from dipy.testing import parametric

@parametric
def test_tensor_scalar_attributes():
    """
    Tests that the tensor class scalar attributes (FA, ADC, etc...) are
    calculating properly.

    """
    ### DEFINING ANALYTICAL VALUES ###
    evals = np.array([2, 1, 0])
    a = 1 / np.sqrt(2)
    #evec[:,j] is pair with eval[j]
    evecs = np.array([[a, -a, 0], [a, a, 0], [0, 0, 1]]) 
    D = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1])
    FA = np.sqrt(1./2*(1+4+1)/(1+4+0)) # 0.7745966692414834
    MD = 1
    ADC = 1

    ### CALCULATE ESTIMATE VALUES ###
    dummy_data = np.zeros((10,))
    dummy_gtab = np.zeros((3,10))
    dummy_bval = dummy_data
    tensor = dti.tensor(dummy_data,dummy_gtab,dummy_bval)
    tensor.evals = evals
    tensor.evecs = evecs
    
    ### TESTS ###
    yield assert_equal(np.abs(np.dot(evecs[:, 2], tensor.evecs[:, 2].T)), 1),
        "Calculation of third eigenvector is not right"
    yield assert_array_almost_equal(D, tensor.D(), "Recovery of self diffusion
        tensor from eigenvalues and eigenvectors is not adequate")
    yield assert_array_almost_equal(ADC, tensor.ADC(), "Calculation of ADC of 
        self diffusion tensor is not adequate")
    yield assert_array_almost_equal(FA, tensor.FA(), "Calculation of FA of 
        self diffusion tensor is not adequate")
    yield assert_array_almost_equal(MD, tensor.MD(), "Calculation of MD of 
        self diffusion tensor is not adequate")

    
    #yield assert_equal(m_list.shape, n_list.shape)
    #yield assert_equal(m_list.ndim, 2)
    #yield assert_equal(m_list.shape, (45,1))
    #yield assert_true(np.all(np.abs(m_list) <= n_list))
    #yield assert_array_equal(n_list % 2, 0)
    #yield assert_raises(ValueError, qball.sph_harm_ind_list, 1)

@parametric
def test_WLS_fit():
    """
    Tests the WLS fitting function to see if it returns the correct
    eigenvalues and eigenvectors.

    Uses data/55dir_grad.bvec as the gradient table and 3by3by56.nii 
    as the data.

    """
    
    ##Test Voxel (avoid having to depend on loading NifTI via nibabel)
    
    #Recall D = [Dxx,Dyy,Dzz,Dxy,Dxz,Dyz,log(S_0)] and D ~ 10^-4 mm^2 /s 
    D = np.array([1.,1.,1.,1.,0.,0.,np.log(1000)*10.**4])*10.**-4
    
    #Design Matrix
    gtab, bval = read_bvec_file('data/55dir_grad.bvec')
    X = dti.design_matrix(gtab,bval)
    
    #Weights of WLS
    #math: inv(W) = 1./W ...b/c W is diagonal vector
    w = np.exp(np.dot(X,D))**2
    W = np.diag(w)
    W_inv = np.diag(1./w)
    
    #Signals
    #recall: D = inv([X.T][W][X])[X.T][W][Y]
    #   ===> Y = inv(W)inv(X.T)[X.T][W][X][D]
    Y = np.dot(np.dot(np.dot(W_inv, np.dot(np.dot(np.linalg.pinv(X.T),X.T)
                                                  ,W)),X),D)

    fit_data = np.dot(np.linalg.pinv(np.dot(np.linalg.pinv(X*w),w.ravel())[:,np.newaxis]),X)
    

    fit_data = np.exp(np.dot(X,D))


    ####example from test_qball.py

    # Tests derived from tables in
    # http://en.wikipedia.org/wiki/Table_of_spherical_harmonics
    # where real spherical harmonic $Y^m_n$ is defined to be:
    #    Real($Y^m_n$) * sqrt(2) if m > 0
    #    $Y^m_n$                 if m == 0
    #    Imag($Y^m_n$) * sqrt(2) if m < 0
 
    rsh = qball.real_sph_harm
    pi = np.pi
    exp = np.exp
    sqrt = np.sqrt
    sin = np.sin
    cos = np.cos
    yield assert_array_almost_equal(rsh(0,0,0,0),
           0.5/sqrt(pi))
    yield assert_array_almost_equal(rsh(2,2,pi/3,pi/5),
           0.25*sqrt(15./(2.*pi))*
           (sin(pi/5.))**2.*cos(0+2.*pi/3)*sqrt(2))
    yield assert_array_almost_equal(rsh(-2,2,pi/3,pi/5),
           0.25*sqrt(15./(2.*pi))*
           (sin(pi/5.))**2.*sin(0-2.*pi/3)*sqrt(2))
    yield assert_array_almost_equal(rsh(2,2,pi,pi/2),
           0.25*sqrt(15/(2.*pi))*
           cos(2.*pi)*sin(pi/2.)**2.*sqrt(2))
    yield assert_array_almost_equal(rsh(-2,4,pi/4.,pi/3.),
           (3./8.)*sqrt(5./(2.*pi))*
           sin(0-2.*pi/4.)*
           sin(pi/3.)**2.*
           (7.*cos(pi/3.)**2.-1)*sqrt(2))
    yield assert_array_almost_equal(rsh(4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi))*
           cos(0+4.*pi/8.)*sin(pi/6.)**4.*sqrt(2))
    yield assert_array_almost_equal(rsh(-4,4,pi/8.,pi/6.),
           (3./16.)*sqrt(35./(2.*pi))*
           sin(0-4.*pi/8.)*sin(pi/6.)**4.*sqrt(2))
    aa = np.ones((3,1,1,1))
    bb = np.ones((1,4,1,1))
    cc = np.ones((1,1,5,1))
    dd = np.ones((1,1,1,6))
    yield assert_equal(rsh(aa, bb, cc, dd).shape, (3, 4, 5, 6))

