#!/usr/bin/python
""" Classes and functions for fitting tensors """
from __future__ import division, print_function, absolute_import

import warnings

import numpy as np

import scipy.optimize as opt

from dipy.reconst.dti import (TensorFit, fractional_anisotropy,
                              geodesic_anisotropy, mean_diffusivity,
                              axial_diffusivity, radial_diffusivity, trace,
                              color_fa, determinant, isotropic, deviatoric,
                              norm, mode, linearity, planarity, sphericity,
                              apparent_diffusion_coef, from_lower_triangular,
                              lower_triangular, decompose_tensor)

from dipy.sims.voxel import DKI_signal
from dipy.utils.six.moves import range
from dipy.data import get_sphere
from ..core.gradients import gradient_table
from ..core.geometry import vector_norm
from ..core.sphere import Sphere
from .vec_val_sum import vec_val_vect
from ..core.onetime import auto_attr
from .base import ReconstModel


def rdpython(x,y,z):
    r"""
    WIP
    """
    d1mach=np.zeros(5)
    d1mach[0]=1.1*10**(-308)
    d1mach[1]=8.9e307
    d1mach[2]=0.22204460*10**(-15)
    d1mach[3]=0.4440*10**(-15)
    d1mach[4]=np.log(2.0)
    errtol = (d1mach[2]/3.0)**(1.0/6.0)
    lolim  = 2.0/(d1mach[1])**(2.0/3.0)
    tuplim = d1mach[0]**(1.0/3.0)
    tuplim = (0.10*errtol)**(1.0/3.0)/tuplim
    uplim  = tuplim**2.0
    c1 = 3.0/14.0
    c2 = 1.0/6.0
    c3 = 9.0/22.0
    c4 = 3.0/26.0

    xn = x.copy()
    yn = y.copy()
    zn = z.copy()
    sigma = 0.0
    power4 = 1.0

    mu = (xn+yn+3.0*zn)*0.20
    xndev = (mu-xn)/mu
    yndev = (mu-yn)/mu
    zndev = (mu-zn)/mu
    epslon = np.max([np.abs(xndev), np.abs(yndev), np.abs(zndev)])
    while (epslon >= errtol):
       xnroot = np.sqrt(xn)
       ynroot = np.sqrt(yn)
       znroot = np.sqrt(zn)
       lamda = xnroot*(ynroot+znroot) + ynroot*znroot
       sigma = sigma + power4/(znroot*(zn+lamda))
       power4 = power4*0.250
       xn = (xn+lamda)*0.250
       yn = (yn+lamda)*0.250
       zn = (zn+lamda)*0.250
       mu = (xn+yn+3.0*zn)*0.20
       xndev = (mu-xn)/mu
       yndev = (mu-yn)/mu
       zndev = (mu-zn)/mu
       epslon = np.max([np.abs(xndev), np.abs(yndev), np.abs(zndev)])

    ea = xndev*yndev
    eb = zndev*zndev
    ec = ea - eb
    ed = ea - 6.0*eb
    ef = ed + ec + ec
    s1 = ed*(-c1+0.250*c3*ed-1.50*c4*zndev*ef)
    s2 = zndev*(c2*ef+zndev*(-c3*ec+zndev*c4*ea))
    drd = 3.0*sigma + power4*(1.0+s1+s2)/(mu*np.sqrt(mu))
    return drd


def rfpython(x,y,z):
    r"""
    WIP
    """
    d1mach=np.zeros(5)
    d1mach[0]=1.1*10**(-308)
    d1mach[1]=8.9e307
    d1mach[2]=0.22204460*10**(-15)
    d1mach[3]=0.4440*10**(-15)
    d1mach[4]=np.log(2.0)
    errtol = (d1mach[2]/3.0)**(1.0/6.0)
    lolim  = 2.0/(d1mach[1])**(2.0/3.0)
    tuplim = d1mach[0]**(1.0/3.0)
    tuplim = (0.10*errtol)**(1.0/3.0)/tuplim
    uplim  = tuplim**2.0
    c1 = 3.0/14.0
    c2 = 1.0/6.0
    c3 = 9.0/22.0
    c4 = 3.0/26.0

    xn = x.copy()
    yn = y.copy()
    zn = z.copy()
 
    mu = (xn+yn+zn)/3.0
    xndev = 2.0 - (mu+xn)/mu
    yndev = 2.0 - (mu+yn)/mu
    zndev = 2.0 - (mu+zn)/mu
    epslon = np.max([np.abs(xndev),np.abs(yndev),np.abs(zndev)])
    while (epslon >= errtol):
       xnroot = np.sqrt(xn)
       ynroot = np.sqrt(yn)
       znroot = np.sqrt(zn)
       lamda = xnroot*(ynroot+znroot) + ynroot*znroot
       xn = (xn+lamda)*0.250
       yn = (yn+lamda)*0.250
       zn = (zn+lamda)*0.250
       mu = (xn+yn+zn)/3.0
       xndev = 2.0 - (mu+xn)/mu
       yndev = 2.0 - (mu+yn)/mu
       zndev = 2.0 - (mu+zn)/mu
       epslon = np.max([np.abs(xndev),np.abs(yndev),np.abs(zndev)])

    e2 = xndev*yndev - zndev*zndev
    e3 = xndev*yndev*zndev
    s  = 1.0 + (c1*e2-0.10-c2*e3)*e2 + c3*e3
    drf = s/np.sqrt(mu)
    return drf


def C2222(a,b,c):
    """
    WIP
    """
    Carray=np.zeros(a.shape)
    abc= np.array((a, b, c))
    indexesxcond1=np.logical_and.reduce(abc>0)
    if np.sum(indexesxcond1)!=0:
      Carray[indexesxcond1]=((a[indexesxcond1]+2.*b[indexesxcond1])**2/(24.*b[indexesxcond1]**2))
    
    indexesxcond2=np.logical_and(np.logical_and.reduce(abc>0),(b!=c))
    if np.sum(indexesxcond2)!=0:
      Carray[indexesxcond2]=(((a[indexesxcond2]+b[indexesxcond2]+c[indexesxcond2])**2/((18.)*(b[indexesxcond2])*(b[indexesxcond2]-c[indexesxcond2])**2))*(2.*b[indexesxcond2]+((c[indexesxcond2]**2-3.*b[indexesxcond2]*c[indexesxcond2])/(np.sqrt(b[indexesxcond2]*c[indexesxcond2])))))

  ### the following condition has to be checked ###
    indexesxcond3=np.logical_or.reduce(abc<=0)
    Carray[indexesxcond3]=0   
    return Carray  


def C2233(a,b,c):
    """
    WIP
    """
    Carray=np.zeros(a.shape)
    abc= np.array((a, b, c))
    
    indexesxcond1=np.logical_and.reduce(abc>0)
    if np.sum(indexesxcond1)!=0:
      Carray[indexesxcond1]=((a[indexesxcond1]+2.*b[indexesxcond1])**2/(12.*b[indexesxcond1]**2))
      
    indexesxcond2=np.logical_and(np.logical_and.reduce(abc>0),(b!=c))
    if np.sum(indexesxcond2)!=0:
      Carray[indexesxcond2]=(((a[indexesxcond2]+b[indexesxcond2]+c[indexesxcond2])**2/((3.)*(b[indexesxcond2]-c[indexesxcond2])**2))*(((b[indexesxcond2]+c[indexesxcond2])/(np.sqrt(b[indexesxcond2]*c[indexesxcond2])))-2.))

  ### the following condition has to be checked ###
    indexesxcond3=np.logical_or.reduce(abc<=0)
    Carray[indexesxcond3]=0   
    return Carray  


def F1m(a,b,c):
    """
    Helper function that computes function $F_1$ which is required to compute
    the analytical solution of the Mean kurtosis.
    
    Parameters
    ----------
    a : ndarray (...)
        Array containing the first variable of function $F_1$ for all signal
        voxels. This variable will be one of the three diffusion tensor
        eigenvalues.
    b : ndarray (...)
        Array containing the second variable of function $F_1$ for all signal
        voxels. This variable will be one of the three diffusion tensor
        eigenvalues.
    c : ndarray (...)
        Array containing the thrid variable of function $F_1$ for all signal
        voxels. This variable will be one of the three diffusion tensor
        eigenvalues.
        
    Returns
    -------
    F1 : array
       Value of the function $F_1$ given the parameters a, b, and c

    Notes
    --------
    Function F_1 is defined as [1]_:

    \begin{multline}
    F_1(\lambda_1,\lambda_2,\lambda_3)=
    \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
    {18(\lambda_1-\lambda_2)(\lambda_1-\lambda_3)}
    [\frac{\sqrt{\lambda_2\lambda_3}}{\lambda_1}
    R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
    \frac{3\lambda_1^2-\lambda_1\lambda_2-\lambda_2\lambda_3-
    \lambda_1\lambda_3}
    {3\lambda_1 \sqrt{\lambda_2 \lambda_3}}
    R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-1 ]
    \end{multline}

    References
    ----------
    .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """
    # Float error used to compare two floats, abs(l1 - l2) < er for l1 = l2 
    er = 1e-10

    # Initialize F1
    F1 = np.empty(a.shape)

    # zero for non plausible diffusion values, i.e. a <= 0 or b <= 0 or c <= 0
    abc = np.array((a, b, c))
    cond0 = np.logical_and.reduce(abc<=0)
    if np.sum(cond0)!=0:
        F1[cond0] = 0

    # Apply formula for non problematic plaussible cases, i.e. a!=b and b!=c
    cond1 = np.logical_and(~cond0, np.logical_and(abs(a - b) > er,
                                                  abs(b - c) > er))
    if np.sum(cond1)!=0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        RFm = rfpython(L1/L2, L1/L3, np.ones(len(L1)))
        RDm = rdpython(L1/L2, L1/L3, np.ones(len(L1)))
        F1[cond1] = ((L1+L2+L3) ** 2) / (18 * (L1-L2) * (L1-L3)) * \
                    (((np.sqrt(L2*L3)) / L1) * RFm + \
                     ((3 * L1**2 - L1*L2 - L1*L3 - L2*L3) / \
                     (3 * L1 * np.sqrt(L2*L3))) * RDm - 1)

    # Resolve possible sigularity a==b
    cond2 = np.logical_and(~cond0, np.logical_and(abs(a - b) < er,
                                                  abs(b - c) > er))
    if np.sum(cond2)!=0:
        L1 = a[cond2]
        L3 = c[cond2]
        F1[cond2] = F2m(L3, L1, L1) / 2

    # Resolve possible sigularity a==c 
    cond3 = np.logical_and(~cond0, np.logical_and(abs(a - c) < er,
                                                  abs(a - b) > er))
    if np.sum(cond3)!=0:
        L1 = a[cond3]
        L2 = b[cond3]
        F1[cond3] = F2m(L2, L1, L1) / 2

    # Resolve possible sigularity a==b and a==c
    cond4 = np.logical_and(~cond0, np.logical_and(abs(a - c) < er,
                                                  abs(a - b) < er))
    if np.sum(cond4)!=0:
        F1[cond4] = 1/5.

    return F1


def F2m(a,b,c):
    """
    Helper function that computes function $F_2$ which is required to compute
    the analytical solution of the Mean kurtosis.
    
    Parameters
    ----------
    a : ndarray (...)
        Array containing the first variable of function $F_1$ for all signal
        voxels. This variable will be one of the three diffusion tensor
        eigenvalues.
    b : ndarray (...)
        Array containing the second variable of function $F_1$ for all signal
        voxels. This variable will be one of the three diffusion tensor
        eigenvalues.
    c : ndarray (...)
        Array containing the thrid variable of function $F_1$ for all signal
        voxels. This variable will be one of the three diffusion tensor
        eigenvalues.
        
    Returns
    -------
    F2 : array
       Value of the function $F_2$ given the parameters a, b, and c

    Notes
    --------
    Function F_2 is defined as [1]_:

    \begin{multline}
    F_2(\lambda_1,\lambda_2,\lambda_3)=
    \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
    {3(\lambda_2-\lambda_3)^2}
    [\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}
    R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
    \frac{2\lambda_1-\lambda_2-\lambda_3}{3\sqrt{\lambda_2 \lambda_3}}
    R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-2]
    \end{multline}

    References
    ----------
    .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """
    # Float error used to compare two floats, abs(l1 - l2) < er for l1 = l2 
    er = 1e-10

    # Initialize F2
    F2 = np.empty(a.shape)

    # zero for non plausible diffusion values, i.e. a <= 0 or b <= 0 or c <= 0
    abc = np.array((a, b, c))    
    cond0 = np.logical_and.reduce(abc<=0)
    if np.sum(cond0)!=0:
        F2[cond0] = 0

    # Apply formula for non problematic plaussible cases, i.e. b!=c
    cond1=np.logical_and(~cond0, (abs(b - c) > er))
    if np.sum(cond1)!=0:
        L1 = a[cond1]
        L2 = b[cond1]
        L3 = c[cond1]
        RF = rfpython(L1/L2, L1/L3, np.ones(len(L1)))
        RD = rdpython(L1/L2, L1/L3, np.ones(len(L1)))
        F2[cond1] = (((L1+L2+L3) ** 2) / (3. * (L2-L3) ** 2)) * \
                    (((L2+L3) / (np.sqrt(L2*L3))) * RF + \
                     ((2.*L1-L2-L3) / (3.*np.sqrt(L2*L3))) * RD - 2.)

    cond2=np.logical_and(~cond0, np.logical_and(abs(b - c) < er,
                                                abs(a - b) > er))
    if np.sum(cond2)!=0:
        L1 = a[cond2]
        L2 = b[cond2]
        L3 = c[cond2]

        # Cumpute alfa [1]_
        x = 1. - (L1/L3)
        alpha = np.zeros(len(L1))
        for xi in x:
            if xi>0:
                alpha[xi] = 1./np.sqrt(xi) * np.arctanh(np.sqrt(xi))
            else:
                alpha[xi] = 1./np.sqrt(-xi) * np.arctan(np.sqrt(-xi))

        F2[cond2] = 6. * ((L1 + 2.*L3)**2) / (144. * L3**2 * (L1-L3)**2) * \
                    (L3 * (L1 + 2.*L3) + L1 * (L1 - 4.*L3) * alpha)
   

    # Resolve possible sigularity a==b and a==c
    cond3 = np.logical_and(~cond0, np.logical_and(abs(a - c) < er,
                                                  abs(a - b) < er))
    if np.sum(cond3)!=0:
        F2[cond3] = 6/15.

    return F2


def G1m(a,b,c):
    """
    WIP
    """
    return C2222(a,b,c)


def G2m(a,b,c):
    """
    WIP
    """
    return 6*C2233(a,b,c)


def mean_kurtosis(dki_params, sphere=None):
    r"""
    Computes mean Kurtosis (MK) from the kurtosis tensor. 

    Parameters
    ----------
    dki_params : ndarray (..., 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    sphere : a Sphere class instance (optional)
        If a sphere class instance is given, MK can be estimated faster as the
        average of the directional kurtosis of the vertices in the sphere [1]_.
        Otherwise MK is computed from its analytical solution [2]_.
        
    Returns
    -------
    mk : array
        Calculated MK.

    Notes
    --------
    The MK analytical solution is calculated using the following equation [2]_:

    .. math::

    \begin{multline}
    MK=F_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{1111}+
       F_1(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{2222}+
       F_1(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{3333}+ \\
       F_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}+
       F_2(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{1133}+
       F_2(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{1122}
    \end{multline}
        
    where $\hat{W}_{ijkl}$ are the components of the $W$ tensor in the
    coordinates system defined by the eigenvectors of the diffusion tensor
    $\mathbf{D}$ and 
 
    \begin{multline}
    F_1(\lambda_1,\lambda_2,\lambda_3)=
    \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
    {18(\lambda_1-\lambda_2)(\lambda_1-\lambda_3)}
    [\frac{\sqrt{\lambda_2\lambda_3}}{\lambda_1}
    R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
    \frac{3\lambda_1^2-\lambda_1\lambda_2-\lambda_2\lambda_3-
    \lambda_1\lambda_3}
    {3\lambda_1 \sqrt{\lambda_2 \lambda_3}}
    R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-1 ]
    \end{multline}

    \begin{multline}
    F_2(\lambda_1,\lambda_2,\lambda_3)=
    \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
    {3(\lambda_2-\lambda_3)^2}
    [\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}
    R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
    \frac{2\lambda_1-\lambda_2-\lambda_3}{3\sqrt{\lambda_2 \lambda_3}}
    R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-2]
    \end{multline}
    
    where $R_f$ and $R_d$ are the Carlson's elliptic integrals.
      
    References
    ----------
    .. [1] Hui ES, Cheung MM, Qi L, Wu EX, 2008. Towards better MR
           characterization of neural tissues using directional diffusion
           kurtosis analysis. Neuroimage 42(1): 122-34
       [2] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """
    if sphere == None:
        MK = _MK_analytical_solution(dki_params)
    else:
        MK = np.mean(apparent_kurtosis_coef(dki_params, sphere), axis=-1)

    return MK


def _MK_analytical_solution(dki_params):
    r"""
    Helper function that computes the mean Kurtosis (MK) from the kurtosis
    tensor using the analyticall solution proposed in [1]_. 

    Parameters
    ----------
    dki_params : ndarray (..., 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    
    Returns
    -------
    mk : array
        Calculated MK.

    Notes
    --------
    MK is calculated with the following equation [1]_:

    .. math::

    \begin{multline}
    MK=F_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{1111}+
       F_1(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{2222}+
       F_1(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{3333}+ \\
       F_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}+
       F_2(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{1133}+
       F_2(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{1122}
    \end{multline}
        
    where $\hat{W}_{ijkl}$ are the components of the $W$ tensor in the
    coordinates system defined by the eigenvectors of the diffusion tensor
    $\mathbf{D}$ and 
 
    \begin{multline}
    F_1(\lambda_1,\lambda_2,\lambda_3)=
    \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
    {18(\lambda_1-\lambda_2)(\lambda_1-\lambda_3)}
    [\frac{\sqrt{\lambda_2\lambda_3}}{\lambda_1}
    R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
    \frac{3\lambda_1^2-\lambda_1\lambda_2-\lambda_2\lambda_3-
    \lambda_1\lambda_3}
    {3\lambda_1 \sqrt{\lambda_2 \lambda_3}}
    R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-1 ]
    \end{multline}

    \begin{multline}
    F_2(\lambda_1,\lambda_2,\lambda_3)=
    \frac{(\lambda_1+\lambda_2+\lambda_3)^2}
    {3(\lambda_2-\lambda_3)^2}
    [\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}
    R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
    \frac{2\lambda_1-\lambda_2-\lambda_3}{3\sqrt{\lambda_2 \lambda_3}}
    R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-2]
    \end{multline}
    
    where $R_f$ and $R_d$ are the Carlson's elliptic integrals.
      
    References
    ----------

    .. [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """

    # Flat parameters
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split the model parameters to three variable containing the evals, evecs,
    # and kurtosis elements

    evals, evecs, kt = split_dki_param(dki_params)

    # Rotate the kurtosis tensor from the standard Cartesian coordinate system
    # to another coordinate system in which the 3 orthonormal eigenvectors of
    # DT are the base coordinate
    Wxxxx = np.zeros((len(kt)))
    Wyyyy = np.zeros((len(kt)))
    Wzzzz = np.zeros((len(kt)))
    Wxxyy = np.zeros((len(kt)))
    Wxxzz = np.zeros((len(kt)))
    Wyyzz = np.zeros((len(kt)))

    for vox in range(len(kt)): 
        Wxxxx[vox] = Wrotate(kt[vox], evecs[vox], [0, 0, 0, 0])
        Wyyyy[vox] = Wrotate(kt[vox], evecs[vox], [1, 1, 1, 1])
        Wzzzz[vox] = Wrotate(kt[vox], evecs[vox], [2, 2, 2, 2])
        Wxxyy[vox] = Wrotate(kt[vox], evecs[vox], [0, 0, 1, 1])
        Wxxzz[vox] = Wrotate(kt[vox], evecs[vox], [0, 0, 2, 2])
        Wyyzz[vox] = Wrotate(kt[vox], evecs[vox], [1, 1, 2, 2])

    # Compute MK
    MeanKurt = F1m(evals[..., 0], evals[..., 1], evals[..., 2])*Wxxxx + \
               F1m(evals[..., 1], evals[..., 0], evals[..., 2])*Wyyyy + \
               F1m(evals[..., 2], evals[..., 1], evals[..., 0])*Wzzzz + \
               F2m(evals[..., 0], evals[..., 1], evals[..., 2])*Wyyzz + \
               F2m(evals[..., 1], evals[..., 0], evals[..., 2])*Wxxzz + \
               F2m(evals[..., 2], evals[..., 1], evals[..., 0])*Wxxyy

    MeanKurt = MeanKurt.reshape(outshape)

    return MeanKurt


def axial_kurtosis(evals, Wrotat, axis=-1):
    r"""
    (WIP)    
    
    Axial Kurtosis (AK) of a diffusion kurtosis tensor. 

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    Wrotat : array-like
        W tensor elements of interest for the evaluation of the Kurtosis 
        (W_xxxx,W_yyyy,W_zzzz,W_xxyy,W_xxzz,W_yyzz)
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    ak : array
        Calculated AK.

    Notes
    --------
    AK is calculated with the following equation:

    .. math::

     K_{||}=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{9\lambda_1^2}
     \hat{W}_{1111}

    """
    [W_xxxx,W_yyyy,W_zzzz,W_xxyy,W_xxzz,W_yyzz]=[Wrotat[...,0],Wrotat[...,1],Wrotat[...,1],Wrotat[...,3],Wrotat[...,4],Wrotat[...,5]]
    AxialKurt=((evals[...,0]+evals[...,1]+evals[...,2])**2/(9*(evals[...,0])**2))*W_xxxx
    return AxialKurt


def radial_kurtosis(evals, Wrotat, axis=-1):
    r"""
    (WIP)
    
    Radial Kurtosis (RK) of a diffusion kurtosis tensor. 

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    Wrotat : array-like
        W tensor elements of interest for the evaluation of the Kurtosis (W_xxxx,W_yyyy,W_zzzz,W_xxyy,W_xxzz,W_yyzz)
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    rk : array
        Calculated RK.

    Notes
    --------
    RK is calculated with the following equation:

    .. math::


    K_{r}=G_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2222}+G_1(\lambda_1,\lambda_3,\lambda_2)\hat{W}_{333}+G_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}

    where:
    \begin{equation}
    G_1(\lambda_1,\lambda_2,\lambda_3)=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{18\lambda_2(\lambda_2-\lambda_3)}[2\lambda_2+\frac{\lambda_3^2-3\lambda_2 \lambda_3}{\sqrt{\lambda_2\lambda_3}}]
    \end{equation}

    \begin{equation}
    G_2(\lambda_1,\lambda_2,\lambda_3)=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{(\lambda_2-\lambda_3)^2}[\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}-2]
    \end{equation}


    """
    [W_xxxx,W_yyyy,W_zzzz,W_xxyy,W_xxzz,W_yyzz]=[Wrotat[...,0],Wrotat[...,1],Wrotat[...,1],Wrotat[...,3],Wrotat[...,4],Wrotat[...,5]]

    RadKurt=G1m(evals[...,0],evals[...,1],evals[...,2])*W_yyyy+G1m(evals[...,0],evals[...,2],evals[...,1])*W_zzzz+G2m(evals[...,0],evals[...,1],evals[...,2])*W_yyzz     
    return RadKurt


def apparent_kurtosis_coef(dki_params, sphere):
    r"""
    Calculate the apparent kurtosis coefficient (AKC) in each direction of a
    sphere.

    Parameters
    ----------
    dki_params : ndarray (..., 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    sphere : a Sphere class instance
        The AKC will be calculated for each of the vertices in the sphere

    Notes
    -----
    For each sphere direction with coordinates $(n_{1}, n_{2}, n_{3})$, the
    calculation of AKC is done using formula:

    .. math ::
        AKC(n)=\frac{MD^{2}}{ADC(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}
        \sum_{k=1}^{3}\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

    where $W_{ijkl}$ are the elements of the kurtosis tensor, MD the mean
    diffusivity and ADC the apparent diffusion coefficent computed as:

    .. math ::
        ADC(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

    where $D_{ij}$ are the elements of the diffusion tensor.
    """

    # Flat parameters
    outshape = dki_params.shape[:-1]
    dki_params = dki_params.reshape((-1, dki_params.shape[-1]))

    # Split data
    evals, evecs, kt = split_dki_param(dki_params)

    # Compute MD
    MD = mean_diffusivity(evals)

    # Initialize AKC matrix
    V = sphere.vertices
    AKC = np.zeros((len(kt), len(V)))

    # loop over all voxels
    for vox in range(len(kt)):
        R = evecs[vox]
        dt = lower_triangular(np.dot(np.dot(R, np.diag(evals[vox])), R.T))
        AKC[vox] = _directional_kurtosis(dt, MD[vox], kt[vox], V)

    # reshape data according to input data 
    AKC = AKC.reshape((outshape + (len(V),)))

    return AKC


def _directional_kurtosis(dt, MD, kt, V):
    r"""
    Helper function that calculate the apparent kurtosis coefficient (AKC)
    in each direction of a sphere for a single voxel.

    Parameters
    ----------
    dt : (6,)
        elements of the diffusion tensor of the voxel.
    MD : float 
        mean diffusivity of the voxel
    kt : (15,)
        elements of the kurtosis tensor of the voxel.
    V : (N, 3)
        N of directions of a Sphere in Cartesian coordinates 

    See Also
    --------
    apparent_kurtosis_coef
    """
    ADC = V[:, 0] * V[:, 0] * dt[0] + \
    2 * V[:, 0] * V[:, 1] * dt[1] + \
    V[:, 1] * V[:, 1] * dt[2] + \
    2 * V[:, 0] * V[:, 2] * dt[3] + \
    2 * V[:, 1] * V[:, 2] * dt[4] + \
    V[:, 2] * V[:, 2] * dt[5]

    AKC = V[:, 0] * V[:, 0] * V[:, 0] * V[:, 0] * kt[0] + \
    V[:, 1] * V[:, 1] * V[:, 1] * V[:, 1] * kt[1] + \
    V[:, 2] * V[:, 2] * V[:, 2] * V[:, 2] * kt[2] + \
    4 * V[:, 0] * V[:, 0] * V[:, 0] * V[:, 1] * kt[3] + \
    4 * V[:, 0] * V[:, 0] * V[:, 0] * V[:, 2] * kt[4] + \
    4 * V[:, 0] * V[:, 1] * V[:, 1] * V[:, 1] * kt[5] + \
    4 * V[:, 1] * V[:, 1] * V[:, 1] * V[:, 2] * kt[6] + \
    4 * V[:, 0] * V[:, 2] * V[:, 2] * V[:, 2] * kt[7] + \
    4 * V[:, 1] * V[:, 2] * V[:, 2] * V[:, 2] * kt[8] + \
    6 * V[:, 0] * V[:, 0] * V[:, 1] * V[:, 1] * kt[9] + \
    6 * V[:, 0] * V[:, 0] * V[:, 2] * V[:, 2] * kt[10] + \
    6 * V[:, 1] * V[:, 1] * V[:, 2] * V[:, 2] * kt[11] + \
    12 * V[:, 0] * V[:, 0] * V[:, 1] * V[:, 2] * kt[12] + \
    12 * V[:, 0] * V[:, 1] * V[:, 1] * V[:, 2] * kt[13] + \
    12 * V[:, 0] * V[:, 1] * V[:, 2] * V[:, 2] * kt[14]

    return (MD/ADC) ** 2 * AKC


def DKI_prediction(dki_params, gtab, S0=150, snr=None):
    """
    Predict a signal given diffusion kurtosis imaging parameters.

    Parameters
    ----------
    dki_params : ndarray (..., 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    gtab : a GradientTable class instance
        The gradient table for this prediction

    S0 : float or ndarray (optional)
        The non diffusion-weighted signal in every voxel, or across all
        voxels. Default: 150

    snr : float (optional)
        Signal to noise ratio, assuming Rician noise.  If set to None, no
        noise is added.

    Returns
    --------
    S : (..., N) ndarray
        Simulated signal based on the DKI model:

    .. math::

        S=S_{0}e^{-bD+\frac{1}{6}b^{2}D^{2}K}
    """
    evals, evecs, kt = split_dki_param(dki_params)

    # Flat parameters and initialize pred_sig
    fevals = evals.reshape((-1, evals.shape[-1]))
    fevecs = evals.reshape((-1, evecs.shape[-2]))
    fkt = kt.reshape((-1, evals.shape[-1]))
    pred_sig = np.zeros((len(fevals), len(gtab.bvals)))

    # lopping for all voxels
    for v in range(len(pred_sig)):
        DT = np.dot(np.dot(fevecs[v], np.diag(fevals[v])), fevecs[v].T)
        pred_sig[v] = DKI_signal(gtab, lower_triangular(DT), fkt[v], S0, snr)

    # Reshape data according to the shape of dki_params
    pred_sig = pred_sig.reshape(dki_params.shape + len(pred_sig))

    return pred_sig


class DKIModel(ReconstModel):
    """ Diffusion Kurtosis Tensor
    """
    def __init__(self, gtab, fit_method="OLS_DKI", *args, **kwargs):
        """ Diffusion Kurtosis Tensor Model [1]

        Parameters
        ----------
        gtab : GradientTable class instance

        fit_method : str or callable
            str can be one of the following:
            'OLS_DKI' or 'ULLS_DKI' for ordinary least squares
                dki.ols_fit_dki
            'WLS_DKI' or 'UWLLS_DKI' for weighted ordinary least squares
                dki.wls_fit_dki

            callable has to have the signature:
                fit_method(design_matrix, data, *args, **kwargs)

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dki.ols_fit_dki, dki.wls_fit_dki for details

        References
        ----------
           [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
        """
        ReconstModel.__init__(self, gtab)

        if not callable(fit_method):
            try:
                self.fit_method = common_fit_methods[fit_method]
            except KeyError:
                raise ValueError('"' + str(fit_method) + '" is not a known fit '
                                 'method, the fit method should either be a '
                                 'function or one of the common fit methods')

        self.design_matrix = dki_design_matrix(self.gtab)
        self.args = args
        self.kwargs = kwargs


    def fit(self, data, mask=None):
        """ Fit method of the DKI model class

        Parameters
        ----------
        data : array
            The measured signal from one voxel.

        mask : array
            A boolean array used to mark the coordinates in the data that
            should be analyzed that has the shape data.shape[-1]
        """
        # If a mask is provided, we will use it to access the data
        if mask is not None:
            # Make sure it's boolean, so that it can be used to mask
            mask = np.array(mask, dtype=bool, copy=False)
            data_in_mask = data[mask]
        else:
            data_in_mask = data

        params_in_mask = self.fit_method(self.design_matrix, data_in_mask,
                                         *self.args, **self.kwargs)

        if mask is None:
            out_shape = data.shape[:-1] + (-1, )
            dki_params = params_in_mask.reshape(out_shape)
        else:
            dki_params = np.zeros(data.shape[:-1] + (27,))
            dki_params[mask, :] = params_in_mask

        return DKIFit(self, dki_params)

    def predict(self, dki_params, S0=1):
        """
        Predict a signal for this DKI model class instance given parameters.

        Parameters
        ----------
        dki_params : ndarray (..., 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follow:
                1) Three diffusion tensor's eingenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor

        gtab : a GradientTable class instance
            The gradient table for this prediction

        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1
        """
        return DKI_prediction(dki_params, self.gtab, S0)


class DKIFit(TensorFit):

    def __init__(self, model, model_params):
        """ Initialize a DKIFit class instance. 
        
        Since DKI is an extension of DTI, class instance is defined as a
        subclass of the TensorFit from dti.py
        """
        TensorFit.__init__(self, model, model_params)

    @property
    def kt(self):
        """
        Returns the 15 independent elements of the kurtosis tensor as an array
        """
        return self.model_params[..., 12:]

    @auto_attr
    def mk(self, sphere=None):
        r"""
        Computes mean Kurtosis (MK) from the kurtosis tensor. 

        Parameters
        ----------
        dki_params : ndarray (..., 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follow:
                1) Three diffusion tensor's eingenvalues
                2) Three lines of the eigenvector matrix each containing
                the first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor

        Returns
        -------
        mk : array
            Calculated MK.

        Notes
        --------
        MK is computed as the average of the directional kurtosis of all the
        directions of the gradient_table gtab [1]_:

        References
        ----------
        .. [1] Hui ES, Cheung MM, Qi L, Wu EX, 2008. Towards better MR
               characterization of neural tissues using directional diffusion
               kurtosis analysis. Neuroimage 42(1): 122-34.
        """
        return mean_kurtosis(self.model_params, sphere)

    @auto_attr
    def ak(self, evals, Wrotat, axis=-1):
        r"""
        (WIP)
        Axial Kurtosis (AK) of a diffusion kurtosis tensor. 

        Parameters
        ----------
        evals : array-like
            Eigenvalues of a diffusion tensor.
        Wrotat : array-like
            W tensor elements of interest for the evaluation of the Kurtosis (W_xxxx,W_yyyy,W_zzzz,W_xxyy,W_xxzz,W_yyzz)
        axis : int
            Axis of `evals` which contains 3 eigenvalues.

        Returns
        -------
        ak : array
            Calculated AK.

        Notes
        --------
        AK is calculated with the following equation:

        .. math::

        K_{||}=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{9\lambda_1^2}\hat{W}_{1111}

        """
        return axial_kurtosis(self.evals, self.Wrotat)

    @auto_attr
    def rk(self, evals, Wrotat, axis=-1):
        r"""
        (WIP)
        Radial Kurtosis (RK) of a diffusion kurtosis tensor. 

        Parameters
        ----------
        evals : array-like
            Eigenvalues of a diffusion tensor.
        Wrotat : array-like
            W tensor elements of interest for the evaluation of the Kurtosis (W_xxxx,W_yyyy,W_zzzz,W_xxyy,W_xxzz,W_yyzz)
        axis : int
            Axis of `evals` which contains 3 eigenvalues.

        Returns
        -------
        rk : array
            Calculated RK.

        Notes
        --------
        RK is calculated with the following equation:

        .. math::

        K_{r}=G_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2222}+G_1(\lambda_1,\lambda_3,\lambda_2)\hat{W}_{333}+G_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}

        where:
        \begin{equation}
        G_1(\lambda_1,\lambda_2,\lambda_3)=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{18\lambda_2(\lambda_2-\lambda_3)}[2\lambda_2+\frac{\lambda_3^2-3\lambda_2 \lambda_3}{\sqrt{\lambda_2\lambda_3}}]
        \end{equation}

        \begin{equation}
        G_2(\lambda_1,\lambda_2,\lambda_3)=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{(\lambda_2-\lambda_3)^2}[\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}-2]
        \end{equation}

        """
        return radial_kurtosis(self.evals, self.Wrotat)

    def akc(self, sphere):
        r"""
        Calculate the apparent kurtosis coefficient (AKC) in each direction on
        the sphere for each voxel in the data

        Parameters
        ----------
        sphere : Sphere class instance

        Returns
        -------
        akc : ndarray
           The estimates of the apparent kurtosis coefficient in every
           direction on the input sphere

        Notes
        -----
        The calculation of ADC, relies on the following relationship:

        .. math ::

            ADC = \vec{b} Q \vec{b}^T

        Where Q is the quadratic form of the tensor.
        """
        return apparent_kurtosis_coef(self.model_params, sphere)

    def DKI_predict(self, gtab, S0=1):
        r"""
        Given a DKI model fit, predict the signal on the vertices of a sphere  

        Parameters
        ----------
        dki_params : ndarray (..., 27)
            All parameters estimated from the diffusion kurtosis model.
            Parameters are ordered as follow:
                1) Three diffusion tensor's eingenvalues
                2) Three lines of the eigenvector matrix each containing the
                   first, second and third coordinates of the eigenvector
                3) Fifteen elements of the kurtosis tensor

        gtab : a GradientTable class instance
            The gradient table for this prediction

        S0 : float or ndarray (optional)
            The non diffusion-weighted signal in every voxel, or across all
            voxels. Default: 1

        Notes
        -----
        The predicted signal is given by:

        .. math::

            S(n,b)=S_{0}e^{-bD(n)+\frac{1}{6}b^{2}D(n)^{2}K(n)}

        $\mathbf{D(n)}$ and $\mathbf{K(n)}$ can be computed from the DT and KT
        using the following equations:

        .. math::

            D(n)=\sum_{i=1}^{3}\sum_{j=1}^{3}n_{i}n_{j}D_{ij}

        and

        .. math::

            K(n)=\frac{MD^{2}}{D(n)^{2}}\sum_{i=1}^{3}\sum_{j=1}^{3}
            \sum_{k=1}^{3}\sum_{l=1}^{3}n_{i}n_{j}n_{k}n_{l}W_{ijkl}

        where $D_{ij}$ and $W_{ijkl}$ are the elements of the second-order DT
        and the fourth-order KT tensors, respectively, and $MD$ is the mean
        diffusivity.
        """
        return DKI_prediction(self.model_params, self.gtab, S0)


def ols_fit_dki(design_matrix, data, min_signal=1):
    r"""
    Computes ordinary least squares (OLS) fit to calculate the diffusion
    tensor and kurtosis tensor using a linear regression diffusion kurtosis
    model [1]_.
    
    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array (... , g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avoid taking log(0) durring the tensor fitting.

    Returns
    -------
    dki_params : array (... , 27)
        All parameters estimated from the diffusion kurtosis model. 
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    See Also
    --------
    wls_fit_dki

    References
    ----------
       [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """

    tol = 1e-6
    if min_signal <= 0:
        raise ValueError('min_signal must be > 0')

    # preparing data and initializing parameters
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dki_params = np.empty((len(data_flat), 27))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    # lopping OLS solution on all data voxels
    for vox in range(len(data_flat)):
        dki_params[vox] = _ols_iter(inv_design, data_flat[vox], min_signal,
                                    min_diffusivity)

    # Reshape data according to the input data shape
    dki_params = dki_params.reshape((data.shape[:-1]) + (27,))

    return dki_params


def _ols_iter(inv_design, sig, min_signal, min_diffusivity):
    ''' Helper function used by ols_fit_dki - Applies OLS fit of the diffusion
    kurtosis model to single voxel signals.
    
    Parameters
    ----------
    inv_design : array (g, 22)
        Inverse of the design matrix holding the covariants used to solve for
        the regression coefficients.
    sig : array (g, ) or array ([N, ...], g)
        Diffusion-weighted signal for a single voxel data.
    min_signal : 
        All values below min_signal are repalced with min_signal. This is done
        in order to avoid taking log(0) durring the tensor fitting.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    dki_params : array (27, )
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    '''

    # removing small signals
    sig = np.maximum(sig, min_signal)

    # DKI ordinary linear least square solution
    log_s = np.log(sig)
    result = np.dot(inv_design, log_s)

    # Extracting the diffusion tensor parameters from solution
    DT_elements = result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = (evals.mean(0))**2  
    KT_elements = result[6:21] / MD_square

    # Write output  
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2], 
                                 KT_elements), axis=0)

    return dki_params


def wls_fit_dki(design_matrix, data, min_signal=1):
    r"""
    Computes weighted linear least squares (WLS) fit to calculate
    the diffusion tensor and kurtosis tensor using a weighted linear 
    regression diffusion kurtosis model [1]_.

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array (..., g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avoid taking log(0) durring the tensor fitting.

    Returns
    -------
    dki_params : array (..., 27)
        All parameters estimated from the diffusion kurtosis model for all N
        voxels. 
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor 

    See Also
    --------
    decompose_tensors


    References
    ----------
       [1] Veraart, J., Sijbers, J., Sunaert, S., Leemans, A., Jeurissen, B.,
           2013. Weighted linear least squares estimation of diffusion MRI
           parameters: Strengths, limitations, and pitfalls. Magn Reson Med 81,
           335-346.
    """

    tol = 1e-6
    if min_signal <= 0:
        raise ValueError('min_signal must be > 0')

    # preparing data and initializing parametres
    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dki_params = np.empty((len(data_flat), 27))

    # inverting design matrix and defining minimun diffusion aloud
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    # lopping WLS solution on all data voxels
    for vox in range(len(data_flat)):
        dki_params[vox] = _wls_iter(design_matrix, inv_design, data_flat[vox],
                                    min_signal, min_diffusivity)

    # Reshape data according to the input data shape
    dki_params = dki_params.reshape((data.shape[:-1]) + (27,))

    return dki_params


def _wls_iter(design_matrix, inv_design, sig, min_signal, min_diffusivity):
    """ Helper function used by wls_fit_dki - Applies WLS fit of the diffusion
    kurtosis model to single voxel signals.
    
    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients    
    inv_design : array (g, 22)
        Inverse of the design matrix.
    sig : array (g, ) or array ([N, ...], g)
        Diffusion-weighted signal for a single voxel data.
    min_signal : 
        All values below min_signal are repalced with min_signal. This is done
        in order to avoid taking log(0) durring the tensor fitting.
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    dki_params : array (27, )
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor
    """

    A = design_matrix
    
    # removing small signals
    sig = np.maximum(sig, min_signal)

    # DKI ordinary linear least square solution
    log_s = np.log(sig)
    ols_result = np.dot(inv_design, log_s)
    
    # Define weights as diag(yn**2)
    W = np.diag(np.exp(2 * np.dot(A, ols_result)))

    # DKI weighted linear least square solution
    inv_AT_W_A = np.linalg.pinv(np.dot(np.dot(A.T, W), A))
    AT_W_LS = np.dot(np.dot(A.T, W), log_s)
    wls_result = np.dot(inv_AT_W_A, AT_W_LS)

    # Extracting the diffusion tensor parameters from solution
    DT_elements = wls_result[:6]
    evals, evecs = decompose_tensor(from_lower_triangular(DT_elements),
                                    min_diffusivity=min_diffusivity)

    # Extracting kurtosis tensor parameters from solution
    MD_square = (evals.mean(0))**2  
    KT_elements = wls_result[6:21] / MD_square

    # Write output  
    dki_params = np.concatenate((evals, evecs[0], evecs[1], evecs[2], 
                                 KT_elements), axis=0)

    return dki_params


def Wrotate(kt, Basis, inds = None):
    r"""
    Rotate a kurtosis tensor from the standard Cartesian coordinate system
    to another coordinate system basis
    
    Parameters
    ----------
    kt : (15,)
        Vector with the 15 independent elements of the kurtosis tensor
    Basis : array (3, 3)
        Vectors of the basis column-wise oriented
    inds : array(..., 4) (optional)
        Array of vectors containing the four indexes of the rotated kurtosis.
        If not specified all 15 elements of the rotated kurtosis tensor are
        computed
    
    Returns
    --------
    Wrot : array (15,) or (...,)
        Vector with the 15 independent elements of the rotated kurtosis tensor.
        If 'indices' is specified only the specified elements of the rotated
        kurtosis tensor are computed.

    Note
    ------
    KT elements are assumed to be ordered as follows:
        
    .. math::
            
    \begin{matrix} ( & W_{xxxx} & W_{yyyy} & W_{zzzz} & W_{xxxy} & W_{xxxz}
                     & ... \\
                     & W_{xyyy} & W_{yyyz} & W_{xzzz} & W_{yzzz} & W_{xxyy}
                     & ... \\
                     & W_{xxzz} & W_{yyzz} & W_{xxyz} & W_{xyyz} & W_{xyzz}
                     & & )\end{matrix}

    References
    ----------
    [1] Hui ES, Cheung MM, Qi L, Wu EX, 2008. Towards better MR
    characterization of neural tissues using directional diffusion kurtosis
    analysis. Neuroimage 42(1): 122-34
    """
    if inds is None:
        inds = np.array([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2],
                         [0, 0, 0, 1], [0, 0, 0, 2], [0, 1, 1, 1],
                         [1, 1, 1, 2], [0, 2, 2, 2], [1, 2, 2, 2],
                         [0, 0, 1, 1], [0, 0, 2, 2], [1, 1, 2, 2],
                         [0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 2, 2]])
    else:
        inds = np.array(inds)
        inds = inds.reshape((-1, inds.shape[-1]))

    Wrot = np.zeros(len(inds))

    # Construct full 4D tensor
    W4D = Wcons(kt)

    for e in range(len(inds)):
        Wrot[e] = _Wrotate_element(W4D, inds[e][0], inds[e][1], inds[e][2],
                                   inds[e][3], Basis)

    return Wrot


def _Wrotate_element(W4D, indi, indj, indk, indl, B):
    r"""
    Helper function that returns the element with specified index of a rotated
    kurtosis tensor from the Cartesian coordinate system to another coordinate
    system basis

    Parameters
    ----------
    W4D : array(4,4,4,4)
        Full 4D kutosis tensor in the Cartesian coordinate system
    indi : int
        Rotated kurtosis tensor element index i (0 for x, 1 for y, 2 for z)
    indj : int
        Rotated kurtosis tensor element index j (0 for x, 1 for y, 2 for z)
    indk : int
        Rotated kurtosis tensor element index k (0 for x, 1 for y, 2 for z)
    indl: int
        Rotated kurtosis tensor element index l (0 for x, 1 for y, 2 for z)
    B: array (3, 3)
        Vectors of the basis column-wise oriented
    
    Returns
    -------
    Wre : float
          rotated kurtosis tensor element of index ind_i, ind_j, ind_k, ind_l
    
    References
    ----------
    [1] Hui ES, Cheung MM, Qi L, Wu EX, 2008. Towards better MR
    characterization of neural tissues using directional diffusion kurtosis
    analysis. Neuroimage 42(1): 122-34
    """

    Wre = 0

    # These for loops can be avoid using kt symmetry properties. If this
    # simplification is done we don't need also to reconstruct the full kt
    # tensor
    for il in range(3):
        for jl in range(3):
            for kl in range(3):
                for ll in range(3):
                    multiplyB = B[il][indi]*B[jl][indj]*B[kl][indk]*B[ll][indl]
                    Wre = Wre + multiplyB * W4D[il][jl][kl][ll]

    return Wre


def Wcons(k_elements):
    r"""
    Construct the full 4D kurtosis tensors from its 15 independent elements
    
    Parameters
    ----------
    k_elements : (15,)
        elements of the kurtosis tensor in the following order:
        
            .. math::
            
    \begin{matrix} ( & W_{xxxx} & W_{yyyy} & W_{zzzz} & W_{xxxy} & W_{xxxz}
                     & ... \\
                     & W_{xyyy} & W_{yyyz} & W_{xzzz} & W_{yzzz} & W_{xxyy}
                     & ... \\
                     & W_{xxzz} & W_{yyzz} & W_{xxyz} & W_{xyyz} & W_{xyzz}
                     & & )\end{matrix}

    Returns
    -------
    W : array(4,4,4,4)
        Full 4D kutosis tensor
    """

    # Note: The multiplication of the indexes (i+1) * (j+1) * (k+1) * (l+1)
    # for of an elements is only equal to this multiplication for another
    # element if an only if the element corresponds to an symmetry element.
    # This multiplication is therefore used to fill the other elements of the
    # full kurtosis elements
    indep_ele = {1: k_elements[0],
                 16: k_elements[1],
                 81: k_elements[2],
                 2: k_elements[3],
                 3: k_elements[4],
                 8: k_elements[5],
                 24: k_elements[6],
                 27: k_elements[7],
                 54: k_elements[8],
                 4: k_elements[9],
                 9: k_elements[10],
                 36: k_elements[11],
                 6: k_elements[12],
                 12: k_elements[13],
                 18: k_elements[14]}

    W = np.zeros((3, 3, 3, 3))

    xyz = [0, 1, 2]
    for ind_i in xyz:
        for ind_j in xyz:
            for ind_k in xyz:
                for ind_l in xyz:
                    key = (ind_i+1) * (ind_j+1) * (ind_k+1) * (ind_l+1)
                    W[ind_i][ind_j][ind_k][ind_l] = indep_ele[key]

    return W


def split_dki_param(dki_params):
    r"""
    Extract the diffusion tensor eigenvalues, the diffusion tensor eigenvector
    matrix, and the 15 independent elements of the kurtosis tensor from the
    model parameters estimated from the DKI model

    Parameters
    ----------
    dki_params : ndarray (..., 27)
        All parameters estimated from the diffusion kurtosis model.
        Parameters are ordered as follow:
            1) Three diffusion tensor's eingenvalues
            2) Three lines of the eigenvector matrix each containing the first,
               second and third coordinates of the eigenvector
            3) Fifteen elements of the kurtosis tensor

    Returns
    --------
    eigvals : array (3,)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])
    kt : array (15,)
        Fifteen elements of the kurtosis tensor
    """    
    evals = dki_params[..., :3]
    evecs = dki_params[..., 3:12].reshape(dki_params.shape[:-1] + (3, 3))
    kt = dki_params[..., 12:]
    
    return evals, evecs, kt


def dki_design_matrix(gtab):
    r""" Constructs B design matrix for DKI

    Parameters
    ---------
    gtab : GradientTable
        Measurement directions.

    Returns
    -------
    B : array (N,22)
        Design matrix or B matrix for the DKI model
        B[j, :] = (Bxx, Bxy, Bzz, Bxz, Byz, Bzz,
                   Bxxxx, Byyyy, Bzzzz, Bxxxy, Bxxxz,
                   Bxyyy, Byyyz, Bxzzz, Byzzz, Bxxyy,
                   Bxxzz, Byyzz, Bxxyz, Bxyyz, Bxyzz,
                   BlogS0)
    """
    b = gtab.bvals
    bvec = gtab.bvecs

    B = np.zeros((len(b), 22))
    B[:, 0] = -b * bvec[:, 0] * bvec[:, 0]
    B[:, 1] = -2 * b * bvec[:, 0] * bvec[:, 1]
    B[:, 2] = -b * bvec[:, 1] * bvec[:, 1]
    B[:, 3] = -2 * b * bvec[:, 0] * bvec[:, 2]
    B[:, 4] = -2 * b * bvec[:, 1] * bvec[:, 2]
    B[:, 5] = -b * bvec[:, 2] * bvec[:, 2]
    B[:, 6] = b * b * bvec[:, 0]**4 / 6
    B[:, 7] = b * b * bvec[:, 1]**4 / 6
    B[:, 8] = b * b * bvec[:, 2]**4 / 6
    B[:, 9] = 4 * b * b * bvec[:, 0]**3 * bvec[:, 1] / 6
    B[:, 10] = 4 * b * b * bvec[:, 0]**3 * bvec[:, 2] / 6
    B[:, 11] = 4 * b * b * bvec[:, 1]**3 * bvec[:, 0] / 6
    B[:, 12] = 4 * b * b * bvec[:, 1]**3 * bvec[:, 2] / 6
    B[:, 13] = 4 * b * b * bvec[:, 2]**3 * bvec[:, 0] / 6
    B[:, 14] = 4 * b * b * bvec[:, 2]**3 * bvec[:, 1] / 6
    B[:, 15] = b * b * bvec[:, 0]**2 * bvec[:, 1]**2
    B[:, 16] = b * b * bvec[:, 0]**2 * bvec[:, 2]**2
    B[:, 17] = b * b * bvec[:, 1]**2 * bvec[:, 2]**2
    B[:, 18] = 2 * b * b * bvec[:, 0]**2 * bvec[:, 1] * bvec[:, 2]
    B[:, 19] = 2 * b * b * bvec[:, 1]**2 * bvec[:, 0] * bvec[:, 2]
    B[:, 20] = 2 * b * b * bvec[:, 2]**2 * bvec[:, 0] * bvec[:, 1]
    B[:, 21] = np.ones(len(b))

    return B


common_fit_methods = {'WLS': wls_fit_dki,
                      'OLS' : ols_fit_dki,
                      'UWLLS': wls_fit_dki,
                      'ULLS' : ols_fit_dki,
                      'WLS_DKI': wls_fit_dki,
                      'OLS_DKI' : ols_fit_dki,
                      'UWLLS_DKI': wls_fit_dki,
                      'ULLS_DKI' : ols_fit_dki
                      }
