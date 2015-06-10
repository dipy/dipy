#!/usr/bin/python
""" Classes and functions for fitting tensors """
from __future__ import division, print_function, absolute_import

import warnings

import numpy as np

import scipy.optimize as opt

from dipy.utils.six.moves import range
from dipy.data import get_sphere
from ..core.gradients import gradient_table
from ..core.geometry import vector_norm
from ..core.sphere import Sphere
from .vec_val_sum import vec_val_vect
from ..core.onetime import auto_attr
from .base import ReconstModel, ReconstFit



#Definition of quantities necessary to evaluates elements of kurtosis
#   
#    All the following definitions are needed for the evaluation of the 
#    Tensor-Derived Kurtosis Measures [1]
#
#    Parameters
#    ----------
#    a,b,c: array-like
#        Eigenvalues of a diffusion tensor equivalento to (evals[0],evals[1],evals[2]). shape should be (...,3).
#
#   Returns
#    -------
#    parameters : array-like
#        Various parameters for the evaluation fo the kurtosis tensor.
#        
#        References
#        ----------
#           [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
#           Estimation of tensors and tensor-derived measures in diffusional
#           kurtosis imaging. Magn Reson Med. 65(3), 823-836
#

def rdpython(x,y,z):
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


def rotatew(winit,r,iv):
      gval=0.
      for i2 in range(3):
        for j2 in range(3):
          for k2 in range(3):
            for l2 in range(3):
              gval=gval+r[i2,iv[0]]*r[j2,iv[1]]*r[k2,iv[2]]*r[l2,iv[3]]*winit[i2,j2,k2,l2]
      
      return gval


def alpha(a):
   alph=(1./np.sqrt(abs(a))*(np.arctan(np.sqrt(abs(a)))))
   return alph

def A1111(a,b,c):
    Aarray=np.ones(a.shape)*1/5.
    abc= np.array((a, b, c))
    
    indexesxcond1=np.logical_and(np.logical_and.reduce(abc>0),np.logical_and(a!=b, b!=c))
    if np.sum(indexesxcond1)!=0:
      d=np.zeros(a.shape)
      e=np.zeros(a.shape)
      f=np.zeros(a.shape)
      g=np.zeros(a.shape)
      h=np.zeros(a.shape)
      d[indexesxcond1]=(((a[indexesxcond1]+b[indexesxcond1]+c[indexesxcond1])**2)/(18*(a[indexesxcond1]-b[indexesxcond1])*(a[indexesxcond1]-c[indexesxcond1])))
      e[indexesxcond1]=((np.sqrt(b[indexesxcond1]*c[indexesxcond1]))/a[indexesxcond1])
      f[indexesxcond1]=rfpython(a[indexesxcond1]/b[indexesxcond1],a[indexesxcond1]/c[indexesxcond1],np.ones(len(a[indexesxcond1])))
      g[indexesxcond1]=((3*a[indexesxcond1]**2-a[indexesxcond1]*b[indexesxcond1]-a[indexesxcond1]*c[indexesxcond1]-b[indexesxcond1]*c[indexesxcond1])/(3*a[indexesxcond1]*np.sqrt(b[indexesxcond1]*c[indexesxcond1])))
      h[indexesxcond1]=rdpython(a[indexesxcond1]/b[indexesxcond1],a[indexesxcond1]/c[indexesxcond1],np.ones(len(a[indexesxcond1])))
      Aarray[indexesxcond1]=d[indexesxcond1]*(e[indexesxcond1]*f[indexesxcond1]+g[indexesxcond1]*h[indexesxcond1]-1)

    indexesxcond2=np.logical_and(np.logical_and.reduce(abc>0),np.logical_and(a==b, b!=c))
    if np.sum(indexesxcond2)!=0:
      dummy2=A2233(c,a,a)
      Aarray[indexesxcond2]=3*dummy2[indexesxcond2]

    indexesxcond3=np.logical_and(np.logical_and.reduce(abc>0),np.logical_and(a==c, a!=b))
    if np.sum(indexesxcond3)!=0:
      dummy3=A2233(b,a,a)
      Aarray[indexesxcond3]=3*dummy3[indexesxcond3]

### the following condition has to be checked ###
    indexesxcond4=np.logical_or.reduce(abc<=0)
    Aarray[indexesxcond4]=0   
    return Aarray
  
def A2233(a,b,c):
    Aarray=np.ones(a.shape)*1/15.
    abc= np.array((a, b, c))
    
    indexesxcond1=np.logical_and(np.logical_and.reduce(abc>0),(b!=c))
    if np.sum(indexesxcond1)!=0:
      d=np.zeros(a.shape)
      e=np.zeros(a.shape)
      f=np.zeros(a.shape)
      g=np.zeros(a.shape)
      h=np.zeros(a.shape)
      d[indexesxcond1]=(((a[indexesxcond1]+b[indexesxcond1]+c[indexesxcond1])**2)/(3*(b[indexesxcond1]-c[indexesxcond1])**2))
      e[indexesxcond1]=((b[indexesxcond1]+c[indexesxcond1])/(np.sqrt(b[indexesxcond1]*c[indexesxcond1])))
      f[indexesxcond1]=rfpython(a[indexesxcond1]/b[indexesxcond1],a[indexesxcond1]/c[indexesxcond1],np.ones(len(a[indexesxcond1])))
      g[indexesxcond1]=((2*a[indexesxcond1]-b[indexesxcond1]-c[indexesxcond1])/(3*np.sqrt(b[indexesxcond1]*c[indexesxcond1])))
      h[indexesxcond1]=rdpython(a[indexesxcond1]/b[indexesxcond1],a[indexesxcond1]/c[indexesxcond1],np.ones(len(a[indexesxcond1])))
      Aarray[indexesxcond1]=(1/6.)*d[indexesxcond1]*(e[indexesxcond1]*f[indexesxcond1]+g[indexesxcond1]*h[indexesxcond1]-2)


    indexesxcond2=np.logical_and(np.logical_and.reduce(abc>0),np.logical_and(b==c, a!=b))
    if np.sum(indexesxcond2)!=0:
      d=np.zeros(a.shape)
      e=np.zeros(a.shape)
      f=np.zeros(a.shape)
      g=np.zeros(a.shape)
      d[indexesxcond2]=(((a[indexesxcond2]+2.*c[indexesxcond2])**2)/(144.*c[indexesxcond2]**2*(a[indexesxcond2]-c[indexesxcond2])**2))
      e[indexesxcond2]=c[indexesxcond2]*(a[indexesxcond2]+2.*c[indexesxcond2])
      f[indexesxcond2]=a[indexesxcond2]*(a[indexesxcond2]-4.*c[indexesxcond2])
      g[indexesxcond2]=alpha(1.-(a[indexesxcond2]/c[indexesxcond2]))
      Aarray[indexesxcond2]=d[indexesxcond2]*(e[indexesxcond2]+f[indexesxcond2]*g[indexesxcond2])
   
  ### the following condition has to be checked ###
    indexesxcond3=np.logical_or.reduce(abc<=0)
    Aarray[indexesxcond3]=0   
    return Aarray  

def C2222(a,b,c):
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
        A=A1111(a,b,c)
        return A

def F2m(a,b,c):
        A=6*A2233(a,b,c)
        return 6*A2233(a,b,c)

def G1m(a,b,c):
    	return C2222(a,b,c)

def G2m(a,b,c):
    	return 6*C2233(a,b,c)



def _roll_Wrotat(Wrotat, axis=-1):
    """
    Helper function to check that the values of the W tensors rotated (needed for evaluation of the kurtosis quantities) provided to functions calculating
    tensor statistics have the right shape

    Parameters
    ----------
    Wrotat : array-like
        Values of a W tensor rotated. shape should be (...,6).

    axis : int
        The axis of the array which contains the 6 values. Default: -1

    Returns
    -------
    Wrotat : array-like
        Values of a W tensor rotated, rolled so that the 6 values are
        the last axis.
    """
    if Wrotat.shape[-1] != 3:
        msg = "Expecting 6 W tensor values, got {}".format(Wrotat.shape[-1])
        raise ValueError(msg)

    Wrotat = np.rollaxis(Wrotat, axis)

    return Wrotat

def mean_kurtosis(evals, Wrotat, axis=-1):
    r"""
    Mean Kurtosis (MK) of a diffusion kurtosis tensor. 

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
    mk : array
        Calculated MK.

    Notes
    --------
    MK is calculated with the following equation:

    .. math::

    \begin{multline}
     MK=F_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{1111}+F_1(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{2222}+F_1(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{3333}+ \\
     F_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}+F_2(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{1133}+F_2(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{1122}
     \end{multline}
     where $\hat{W}_{ijkl}$ are the components of the $W$ tensor in the coordinates system defined by the eigenvectors of the diffusion tensor $\mathbf{D}$ and 
 
    \begin{multline}
     F_1(\lambda_1,\lambda_2,\lambda_3)=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{18(\lambda_1-\lambda_2)(\lambda_1-\lambda_3)}[\frac{\sqrt{\lambda_2\lambda_3}}{\lambda_1}R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
     \frac{3\lambda_1^2-\lambda_1\lambda_2-\lambda_2\lambda_3-\lambda_1\lambda_3}{3\lambda_1 \sqrt{\lambda_2 \lambda_3}}R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-1 ]
    \end{multline}

    \begin{multline}
     F_2(\lambda_1,\lambda_2,\lambda_3)=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{3(\lambda_2-\lambda_3)^2}[\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
     \frac{2\lambda_1-\lambda_2-\lambda_3}{3\sqrt{\lambda_2 \lambda_3}}R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-2]
    \end{multline}
    where $R_f$ and $R_d$ are the Carlson's elliptic integrals.


    """
    [W_xxxx,W_yyyy,W_zzzz,W_xxyy,W_xxzz,W_yyzz]=[Wrotat[...,0],Wrotat[...,1],Wrotat[...,2],Wrotat[...,3],Wrotat[...,4],Wrotat[...,5]]
    MeanKurt=F1m(evals[...,0],evals[...,1],evals[...,2])*W_xxxx+F1m(evals[...,1],evals[...,0],evals[...,2])*W_yyyy+F1m(evals[...,2],evals[...,1],evals[...,0])*W_zzzz+F2m(evals[...,0],evals[...,1],evals[...,2])*W_yyzz+F2m(evals[...,1],evals[...,0],evals[...,2])*W_xxzz+F2m(evals[...,2],evals[...,1],evals[...,0])*W_xxyy
    return MeanKurt


def axial_kurtosis(evals, Wrotat, axis=-1):
    r"""
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
    [W_xxxx,W_yyyy,W_zzzz,W_xxyy,W_xxzz,W_yyzz]=[Wrotat[...,0],Wrotat[...,1],Wrotat[...,1],Wrotat[...,3],Wrotat[...,4],Wrotat[...,5]]
    AxialKurt=((evals[...,0]+evals[...,1]+evals[...,2])**2/(9*(evals[...,0])**2))*W_xxxx
    return AxialKurt

def radial_kurtosis(evals, Wrotat, axis=-1):
    r"""
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


#End of the definitions of quantities necessary to evaluates elements of kurtosis



def _roll_evals(evals, axis=-1):
    """
    Helper function to check that the evals provided to functions calculating
    tensor statistics have the right shape

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor. shape should be (...,3).

    axis : int
        The axis of the array which contains the 3 eigenvals. Default: -1

    Returns
    -------
    evals : array-like
        Eigenvalues of a diffusion tensor, rolled so that the 3 eigenvals are
        the last axis.
    """
    if evals.shape[-1] != 3:
        msg = "Expecting 3 eigenvalues, got {}".format(evals.shape[-1])
        raise ValueError(msg)

    evals = np.rollaxis(evals, axis)

    return evals



def fractional_anisotropy(evals, axis=-1):
    r"""
    Fractional anisotropy (FA) of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    fa : array
        Calculated FA. Range is 0 <= FA <= 1.

    Notes
    --------
    FA is calculated using the following equation:

    .. math::

        FA = \sqrt{\frac{1}{2}\frac{(\lambda_1-\lambda_2)^2+(\lambda_1-
                    \lambda_3)^2+(\lambda_2-\lambda_3)^2}{\lambda_1^2+
                    \lambda_2^2+\lambda_3^2}}

    """
    evals = _roll_evals(evals, axis)
    # Make sure not to get nans
    all_zero = (evals == 0).all(axis=0)
    ev1, ev2, ev3 = evals
    fa = np.sqrt(0.5 * ((ev1 - ev2) ** 2 + (ev2 - ev3) ** 2 + (ev3 - ev1) ** 2)
                  / ((evals * evals).sum(0) + all_zero))

    return fa


def mean_diffusivity(evals, axis=-1):
    r"""
    Mean Diffusivity (MD) of a diffusion tensor. 

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    md : array
        Calculated MD.

    Notes
    --------
    MD is calculated with the following equation:

    .. math::

        MD = \frac{\lambda_1 + \lambda_2 + \lambda_3}{3}

    """
    evals = _roll_evals(evals, axis)
    return evals.mean(0)


def axial_diffusivity(evals, axis=-1):
    r"""
    Axial Diffusivity (AD) of a diffusion tensor.
    Also called parallel diffusivity.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor, must be sorted in descending order
        along `axis`.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    ad : array
        Calculated AD.

    Notes
    --------
    AD is calculated with the following equation:

    .. math::

        AD = \lambda_1

    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return ev1


def radial_diffusivity(evals, axis=-1):
    r"""
    Radial Diffusivity (RD) of a diffusion tensor.
    Also called perpendicular diffusivity.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor, must be sorted in descending order
        along `axis`.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

        Returns
    -------
    rd : array
        Calculated RD.

    Notes
    --------
    RD is calculated with the following equation:

    .. math::

        RD = \frac{\lambda_2 + \lambda_3}{2}

    """
    evals = _roll_evals(evals, axis)
    return evals[1:].mean(0)


def trace(evals, axis=-1):
    r"""
    Trace of a diffusion tensor.

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    trace : array
        Calculated trace of the diffusion tensor.

    Notes
    --------
    Trace is calculated with the following equation:

    .. math::

        Trace = \lambda_1 + \lambda_2 + \lambda_3

    """
    evals = _roll_evals(evals, axis)
    return evals.sum(0)


def color_fa(fa, evecs):
    r""" Color fractional anisotropy of diffusion tensor

    Parameters
    ----------
    fa : array-like
        Array of the fractional anisotropy (can be 1D, 2D or 3D)

    evecs : array-like
        eigen vectors from the tensor model

    Returns
    -------
    rgb : Array with 3 channels for each color as the last dimension.
        Colormap of the FA with red for the x value, y for the green
        value and z for the blue value.

    Note
    -----

    It is computed from the clipped FA between 0 and 1 using the following
    formula

    .. math::

        rgb = abs(max(\vec{e})) \times fa
    """

    if (fa.shape != evecs[..., 0, 0].shape) or ((3, 3) != evecs.shape[-2:]):
        raise ValueError("Wrong number of dimensions for evecs")

    return np.abs(evecs[..., 0]) * np.clip(fa, 0, 1)[..., None]


# The following are used to calculate the tensor mode:
def determinant(q_form):
    """
    The determinant of a tensor, given in quadratic form

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x, y, z, 3, 3) or (n, 3, 3) or (3, 3).

    Returns
    -------
    det : array
        The determinant of the tensor in each spatial coordinate
    """

    # Following the conventions used here:
    # http://en.wikipedia.org/wiki/Determinant
    aei = q_form[..., 0, 0] * q_form[..., 1, 1] * q_form[..., 2, 2]
    bfg = q_form[..., 0, 1] * q_form[..., 1, 2] * q_form[..., 2, 0]
    cdh = q_form[..., 0, 2] * q_form[..., 1, 0] * q_form[..., 2, 1]
    ceg = q_form[..., 0, 2] * q_form[..., 1, 1] * q_form[..., 2, 0]
    bdi = q_form[..., 0, 1] * q_form[..., 1, 0] * q_form[..., 2, 2]
    afh = q_form[..., 0, 0] * q_form[..., 1, 2] * q_form[..., 2, 1]
    return aei + bfg + cdh - ceg - bdi - afh


def isotropic(q_form):
    r"""
    Calculate the isotropic part of the tensor [1]_.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x,y,z,3,3) or (n, 3, 3) or (3,3).

    Returns
    -------
    A_hat: ndarray
        The isotropic part of the tensor in each spatial coordinate

    Notes
    -----
    The isotropic part of a tensor is defined as (equations 3-5 of [1]_):

    .. math ::
        \bar{A} = \frac{1}{2} tr(A) I

    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.
    """
    tr_A = q_form[..., 0, 0] + q_form[..., 1, 1] + q_form[..., 2, 2]
    n_dims = len(q_form.shape)
    add_dims = n_dims - 2  # These are the last two (the 3,3):
    my_I = np.eye(3)
    tr_AI = (tr_A.reshape(tr_A.shape + (1, 1)) * my_I)
    return (1 / 3.0) * tr_AI


def deviatoric(q_form):
    r"""
    Calculate the deviatoric (anisotropic) part of the tensor [1]_.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x,y,z,3,3) or (n, 3, 3) or (3,3).

    Returns
    -------
    A_squiggle : ndarray
        The deviatoric part of the tensor in each spatial coordinate.

    Notes
    -----
    The deviatoric part of the tensor is defined as (equations 3-5 in [1]_):

    .. math ::
         \widetilde{A} = A - \bar{A}

    Where $A$ is the tensor quadratic form and $\bar{A}$ is the anisotropic
    part of the tensor.

    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.
    """
    A_squiggle = q_form - isotropic(q_form)
    return A_squiggle


def norm(q_form):
    r"""
    Calculate the Frobenius norm of a tensor quadratic form

    Parameters
    ----------
    q_form: ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x,y,z,3,3) or (n, 3, 3) or (3,3).

    Returns
    -------
    norm : ndarray
        The Frobenius norm of the 3,3 tensor q_form in each spatial
        coordinate.

    Notes
    -----
    The Frobenius norm is defined as:

    :math:
        ||A||_F = [\sum_{i,j} abs(a_{i,j})^2]^{1/2}

    See also
    --------
    np.linalg.norm
    """
    return np.sqrt(np.sum(np.sum(np.abs(q_form ** 2), -1), -1))


def mode(q_form):
    r"""
    Mode (MO) of a diffusion tensor [1]_.

    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (x, y, z, 3, 3) or (n, 3, 3) or (3, 3).

    Returns
    -------
    mode : array
        Calculated tensor mode in each spatial coordinate.

    Notes
    -----
    Mode ranges between -1 (linear anisotropy) and +1 (planar anisotropy)
    with 0 representing orthotropy. Mode is calculated with the
    following equation (equation 9 in [1]_):

    .. math::

        Mode = 3*\sqrt{6}*det(\widetilde{A}/norm(\widetilde{A}))

    Where $\widetilde{A}$ is the deviatoric part of the tensor quadratic form.

    References
    ----------

    .. [1] Daniel B. Ennis and G. Kindlmann, "Orthogonal Tensor
        Invariants and the Analysis of Diffusion Tensor Magnetic Resonance
        Images", Magnetic Resonance in Medicine, vol. 55, no. 1, pp. 136-146,
        2006.
    """

    A_squiggle = deviatoric(q_form)
    A_s_norm = norm(A_squiggle)
    # Add two dims for the (3,3), so that it can broadcast on A_squiggle:
    A_s_norm = A_s_norm.reshape(A_s_norm.shape + (1, 1))
    return 3 * np.sqrt(6) * determinant((A_squiggle / A_s_norm))


def linearity(evals, axis=-1):
    r"""
    The linearity of the tensor [1]_

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    linearity : array
        Calculated linearity of the diffusion tensor.

    Notes
    --------
    Linearity is calculated with the following equation:

    .. math::

        Linearity = \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\lambda_3}

    Notes
    -----
    [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz F.,
        "Geometrical diffusion measures for MRI from tensor basis analysis" in
        Proc. 5th Annual ISMRM, 1997.
    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return (ev1 - ev2) / evals.sum(0)


def planarity(evals, axis=-1):
    r"""
    The planarity of the tensor [1]_

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    linearity : array
        Calculated linearity of the diffusion tensor.

    Notes
    --------
    Linearity is calculated with the following equation:

    .. math::

        Planarity = \frac{2 (\lambda_2-\lambda_3)}{\lambda_1+\lambda_2+\lambda_3}

    Notes
    -----
    [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz F.,
        "Geometrical diffusion measures for MRI from tensor basis analysis" in
        Proc. 5th Annual ISMRM, 1997.
    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return (2 * (ev2 - ev3) / evals.sum(0))


def sphericity(evals, axis=-1):
    r"""
    The sphericity of the tensor [1]_

    Parameters
    ----------
    evals : array-like
        Eigenvalues of a diffusion tensor.
    axis : int
        Axis of `evals` which contains 3 eigenvalues.

    Returns
    -------
    sphericity : array
        Calculated sphericity of the diffusion tensor.

    Notes
    --------
    Linearity is calculated with the following equation:

    .. math::

        Sphericity = \frac{3 \lambda_3)}{\lambda_1+\lambda_2+\lambda_3}

    Notes
    -----
    [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz F.,
        "Geometrical diffusion measures for MRI from tensor basis analysis" in
        Proc. 5th Annual ISMRM, 1997.
    """
    evals = _roll_evals(evals, axis)
    ev1, ev2, ev3 = evals
    return (3 * ev3) / evals.sum(0)


def apparent_diffusion_coef(q_form, sphere):
    r"""
    Calculate the apparent diffusion coefficient (ADC) in each direction of a
    sphere.
        
    Parameters
    ----------
    q_form : ndarray
        The quadratic form of a tensor, or an array with quadratic forms of
        tensors. Should be of shape (..., 3, 3)

    sphere : a Sphere class instance
        The ADC will be calculated for each of the vertices in the sphere
        
    Notes
    -----
    The calculation of ADC, relies on the following relationship:

    .. math ::
            ADC = \vec{b} Q \vec{b}^T

    Where Q is the quadratic form of the tensor.
    
    """
    bvecs = sphere.vertices
    bvals = np.ones(bvecs.shape[0])
    gtab = gradient_table(bvals, bvecs)
    D = design_matrix(gtab)[:, :6]
    return -np.dot(lower_triangular(q_form), D.T)



class TensorModel(ReconstModel):
    """ Diffusion Kurtosis Tensor
    """
    def __init__(self, gtab, fit_method="ULLS_KURT", *args, **kwargs):
        """ A Diffusion Kurtosis Tensor Model [1]

        Parameters
        ----------
        grad_table : GradientTable class instance

        fit_method : str or callable
            str can be one of the following:
            'ULLS_KURT' for unconstrained linear least squares
                dki.ulls_fit_ktensor
            'UWLLS_KURT' for unconstrained weighted linear least squares
                 dki.uwlls_fit_ktensor

            callable has to have the signature:
              fit_method(design_matrix, data, *args, **kwargs)

        args, kwargs : arguments and key-word arguments passed to the
           fit_method. See dki.ulls_fit_ktensor, dki.uwlls_fit_ktensor for details

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

        self.design_matrix = design_matrix(self.gtab)
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

        dki_params = np.zeros(data.shape[:-1] + (18,))

        dki_params[mask, :] = params_in_mask

        return KTensor(self, dki_params)



        
class KTensor(object):
    def __init__(self, model, model_params):
        """ Initialize a KTensor class instance.
        """
        self.model = model
        self.model_params = model_params

    def __getitem__(self, index):
        model_params = self.model_params
        N = model_params.ndim
        if type(index) is not tuple:
            index = (index,)
        elif len(index) >= model_params.ndim:
            raise IndexError("IndexError: invalid index")
        index = index + (slice(None),) * (N - len(index))
        return type(self)(self.model, model_params[index])

    @property
    def shape(self):
        return self.model_params.shape[:-1]

    @property
    def directions(self):
        """
        For tracking - return the primary direction in each voxel
        """
        return self.evecs[..., None, :, 0]

    @property
    def evals(self):
        """
        Returns the eigenvalues of the tensor as an array
        """
        return self.model_params[..., :3]

    @property
    def evecs(self):
        """
        Returns the eigenvectors of the tensor as an array
        """
        evecs = self.model_params[..., 3:12]
        return evecs.reshape(self.shape + (3, 3))

    @property
    def Wrotat(self):
        """
        Returns the values of the k tensors as an array
        """
        return self.model_params[..., 12:]

        
    @property
    def quadratic_form(self):
        """Calculates the 3x3 diffusion tensor for each voxel"""
        # do `evecs * evals * evecs.T` where * is matrix multiply
        # einsum does this with:
        # np.einsum('...ij,...j,...kj->...ik', evecs, evals, evecs)
        return vec_val_vect(self.evecs, self.evals)

    def lower_triangular(self, b0=None):
        return lower_triangular(self.quadratic_form, b0)

    @auto_attr
    def fa(self):
        """Fractional anisotropy (FA) calculated from cached eigenvalues."""
        return fractional_anisotropy(self.evals)

    @auto_attr
    def mode(self):
        """
        Tensor mode calculated from cached eigenvalues.
        """
        return mode(self.quadratic_form)

    @auto_attr
    def md(self):
        r"""
        Mean diffusitivity (MD) calculated from cached eigenvalues.

        Returns
        ---------
        md : array (V, 1)
            Calculated MD.

        Notes
        --------
        MD is calculated with the following equation:

        .. math::

            MD = \frac{\lambda_1+\lambda_2+\lambda_3}{3}

        """
        return self.trace / 3.0

    @auto_attr
    def rd(self):
        r"""
        Radial diffusitivity (RD) calculated from cached eigenvalues.

        Returns
        ---------
        rd : array (V, 1)
            Calculated RD.

        Notes
        --------
        RD is calculated with the following equation:

        .. math::

          RD = \frac{\lambda_2 + \lambda_3}{2}


        """
        return radial_diffusivity(self.evals)

    @auto_attr
    def ad(self):
        r"""
        Axial diffusivity (AD) calculated from cached eigenvalues.

        Returns
        ---------
        ad : array (V, 1)
            Calculated AD.

        Notes
        --------
        RD is calculated with the following equation:

        .. math::

          AD = \lambda_1


        """
        return axial_diffusivity(self.evals)

    @auto_attr
    def trace(self):
        r"""
        Trace of the tensor calculated from cached eigenvalues.

        Returns
        ---------
        trace : array (V, 1)
            Calculated trace.

        Notes
        --------
        The trace is calculated with the following equation:

        .. math::

          trace = \lambda_1 + \lambda_2 + \lambda_3
        """
        return trace(self.evals)


    @auto_attr
    def mk(self):
      r"""
      Mean Kurtosis (MK) of a diffusion kurtosis tensor. 

      Returns
      -------
      mk : array
          Calculated MK.

      Notes
      --------
      MK is calculated with the following equation:

      .. math::

      \begin{multline}
      MK=F_1(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{1111}+F_1(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{2222}+F_1(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{3333}+ \\
      F_2(\lambda_1,\lambda_2,\lambda_3)\hat{W}_{2233}+F_2(\lambda_2,\lambda_1,\lambda_3)\hat{W}_{1133}+F_2(\lambda_3,\lambda_2,\lambda_1)\hat{W}_{1122}
      \end{multline}
      where $\hat{W}_{ijkl}$ are the components of the $W$ tensor in the coordinates system defined by the eigenvectors of the diffusion tensor $\mathbf{D}$ and 
 
      \begin{multline}
      F_1(\lambda_1,\lambda_2,\lambda_3)=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{18(\lambda_1-\lambda_2)(\lambda_1-\lambda_3)}[\frac{\sqrt{\lambda_2\lambda_3}}{\lambda_1}R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
      \frac{3\lambda_1^2-\lambda_1\lambda_2-\lambda_2\lambda_3-\lambda_1\lambda_3}{3\lambda_1 \sqrt{\lambda_2 \lambda_3}}R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-1 ]
      \end{multline}

      \begin{multline}
      F_2(\lambda_1,\lambda_2,\lambda_3)=\frac{(\lambda_1+\lambda_2+\lambda_3)^2}{3(\lambda_2-\lambda_3)^2}[\frac{\lambda_2+\lambda_3}{\sqrt{\lambda_2\lambda_3}}R_F(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)+\\
      \frac{2\lambda_1-\lambda_2-\lambda_3}{3\sqrt{\lambda_2 \lambda_3}}R_D(\frac{\lambda_1}{\lambda_2},\frac{\lambda_1}{\lambda_3},1)-2]
      \end{multline}
      where $R_f$ and $R_d$ are the Carlson's elliptic integrals.

      """
      return mean_kurtosis(self.evals, self.Wrotat)

    @auto_attr
    def ak(evals, Wrotat, axis=-1):
        r"""
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
    def rk(evals, Wrotat, axis=-1):
        r"""
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


    @auto_attr
    def planarity(self):
        r"""
        Returns
        -------
        sphericity : array
            Calculated sphericity of the diffusion tensor [1]_.

        Notes
        --------
        Sphericity is calculated with the following equation:

        .. math::

            Sphericity = \frac{2 (\lambda2 - \lambda_3)}{\lambda_1+\lambda_2+\lambda_3}

        Notes
        -----
        [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz
            F., "Geometrical diffusion measures for MRI from tensor basis
            analysis" in Proc. 5th Annual ISMRM, 1997.

        """
        return planarity(self.evals)

    @auto_attr
    def linearity(self):
        r"""
        Returns
        -------
        linearity : array
            Calculated linearity of the diffusion tensor [1]_.

        Notes
        --------
        Linearity is calculated with the following equation:

        .. math::

            Linearity = \frac{\lambda_1-\lambda_2}{\lambda_1+\lambda_2+\lambda_3}

        Notes
        -----
        [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz
            F., "Geometrical diffusion measures for MRI from tensor basis
            analysis" in Proc. 5th Annual ISMRM, 1997.

        """
        return linearity(self.evals)

    @auto_attr
    def sphericity(self):
        r"""
        Returns
        -------
        sphericity : array
            Calculated sphericity of the diffusion tensor [1]_.

        Notes
        --------
        Sphericity is calculated with the following equation:

        .. math::

            Sphericity = \frac{3 \lambda_3}{\lambda_1+\lambda_2+\lambda_3}

        Notes
        -----
        [1] Westin C.-F., Peled S., Gubjartsson H., Kikinis R., Jolesz
            F., "Geometrical diffusion measures for MRI from tensor basis
            analysis" in Proc. 5th Annual ISMRM, 1997.

        """
        return sphericity(self.evals)

    def odf(self, sphere):
        """
        The diffusion orientation distribution function (dODF). This is an
        estimate of the diffusion distance in each direction

        Parameters
        ----------
        sphere : Sphere class instance.
            The dODF is calculated in the vertices of this input.

        Returns
        -------
        odf : ndarray
            The diffusion distance in every direction of the sphere in every
            voxel in the input data.
        
        """
        lower = 4 * np.pi * np.sqrt(np.prod(self.evals, -1))
        projection = np.dot(sphere.vertices, self.evecs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            projection /= np.sqrt(self.evals)
            odf = (vector_norm(projection) ** -3) / lower
        # Zero evals are non-physical, we replace nans with zeros
        any_zero = (self.evals == 0).any(-1)
        odf = np.where(any_zero, 0, odf)
        # Move odf to be on the last dimension
        odf = np.rollaxis(odf, 0, odf.ndim)
        return odf

    def adc(self, sphere):
        r"""
        Calculate the apparent diffusion coefficient (ADC) in each direction on
        the sphere for each voxel in the data

        Parameters
        ----------
        sphere : Sphere class instance

        Returns
        -------
        adc : ndarray
           The estimates of the apparent diffusion coefficient in every
           direction on the input sphere

        Notes
        -----
        The calculation of ADC, relies on the following relationship:

        .. math ::

            ADC = \vec{b} Q \vec{b}^T

        Where Q is the quadratic form of the tensor.
        """
        return apparent_diffusion_coef(self.quadratic_form, sphere)


    def predict(self, gtab, S0=1):
        r"""
        Given a model fit, predict the signal on the vertices of a sphere 

        Parameters
        ----------
        gtab : a GradientTable class instance
            This encodes the directions for which a prediction is made

        S0 : float array
           The mean non-diffusion weighted signal in each voxel. Default: 1 in
           all voxels.
           
        Notes
        -----
        The predicted signal is given by:

        .. math ::

            S(\theta, b) = S_0 * e^{-b ADC}

        Where:
        .. math ::
            ADC = \theta Q \theta^T

        $\theta$ is a unit vector pointing at any direction on the sphere for
        which a signal is to be predicted and $b$ is the b value provided in
        the GradientTable input for that direction   
        """
        # Get a sphere to pass to the object's ADC function. The b0 vectors
        # will not be on the unit sphere, but we still want them to be there,
        # so that we have a consistent index for these, so that we can fill
        # that in later on, so we suppress the warning here:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sphere = Sphere(xyz=gtab.bvecs)

        adc = self.adc(sphere)
        # Predict!
        if np.iterable(S0):
            # If it's an array, we need to give it one more dimension:
            S0 = S0[...,None] 

        pred_sig = S0 * np.exp(-gtab.bvals * adc)

        # The above evaluates to nan for the b0 vectors, so we predict the mean
        # S0 for those, which is our best guess:
        pred_sig[...,gtab.b0s_mask] = S0

        return pred_sig

def ulls_fit_ktensor(design_matrix, data, min_signal=1):
    r"""
    Computes unconstrained linear least squares (ULLS) fit to calculate kurtosis maps using a linear regression model without constrains [1].

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avaid taking log(0) durring the tensor fitting.

    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])
    Wrotat : array (..., 6)
        Values of a W tensor rotated. 

    See Also
    --------
    decompose_tensors


    References
    ----------
       [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """

    tol = 1e-6
    if min_signal <= 0:
        raise ValueError('min_signal must be > 0')

    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dki_params = np.empty((len(data_flat), 6, 3))
    
    min_diffusivity = tol / -design_matrix.min()
    inv_design = np.linalg.pinv(design_matrix)

    
    for param, sig in zip(dki_params, data_flat):
        param[0], param[1:4], param[4], param[5] = _ulls_iter(inv_design, sig,
				                  min_signal, min_diffusivity)
        
    dki_params.shape=data.shape[:-1]+(18,)
    dki_params=dki_params
    return dki_params


def _ulls_iter(inv_design, sig, min_signal, min_diffusivity):
    ''' Helper function used by ulls_fit_tensor.
    '''
    sig=np.maximum(sig,min_signal)
    log_s = np.log(sig)
    result=np.dot(inv_design,log_s)
    D=result[:6]
    tensor=from_lower_triangular(D)
    MeanD_square=((tensor[0,0]+tensor[1,1]+tensor[2,2])/3.)**2  
    K_tensor_elements=result[6:21]/MeanD_square
    return decompose_tensors(tensor, K_tensor_elements, min_diffusivity=min_diffusivity)



def uwlls_fit_ktensor(design_matrix, data, min_signal=1):
    r"""
    Computes unconstrained weighted linear least squares (UWLLS) fit to calculate kurtosis maps using a weighted linear regression model without constrains [1].

    Parameters
    ----------
    design_matrix : array (g, 22)
        Design matrix holding the covariants used to solve for the regression
        coefficients.
    data : array ([X, Y, Z, ...], g)
        Data or response variables holding the data. Note that the last
        dimension should contain the data. It makes no copies of data.
    min_signal : default = 1
        All values below min_signal are repalced with min_signal. This is done
        in order to avaid taking log(0) durring the tensor fitting.

    Returns
    -------
    eigvals : array (..., 3)
        Eigenvalues from eigen decomposition of the tensor.
    eigvecs : array (..., 3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])
    Wrotat : array (..., 6)
        Values of a W tensor rotated. 

    See Also
    --------
    decompose_tensors


    References
    ----------
       [1] Tabesh, A., Jensen, J.H., Ardekani, B.A., Helpern, J.A., 2011.
           Estimation of tensors and tensor-derived measures in diffusional
           kurtosis imaging. Magn Reson Med. 65(3), 823-836
    """

    tol = 1e-6
    if min_signal <= 0:
        raise ValueError('min_signal must be > 0')

    data = np.asarray(data)
    data_flat = data.reshape((-1, data.shape[-1]))
    dki_params = np.empty((len(data_flat), 6, 3))
    min_diffusivity = tol / -design_matrix.min()

    ols_fit = _ols_fit_matrix(design_matrix)
   
    for param, sig in zip(dki_params, data_flat):
        param[0], param[1:4], param[4], param[5] = _uwlls_iter(ols_fit, design_matrix, sig, min_signal, min_diffusivity)
        
    dki_params.shape=data.shape[:-1]+(18,)
    dki_params=dki_params
    return dki_params



def _ols_fit_matrix(design_matrix):
    """
    Helper function to calculate the ordinary least squares (OLS)
    fit as a matrix multiplication. Mainly used to calculate WLS weights. Can
    be used to calculate regression coefficients in OLS but not recommended.

    See Also:
    ---------
    wls_fit_tensor, ols_fit_tensor

    Example:
    --------
    ols_fit = _ols_fit_matrix(design_mat)
    ols_data = np.dot(ols_fit, data)
    """

    U, S, V = np.linalg.svd(design_matrix, False)
    return np.dot(U, U.T)



def _uwlls_iter(ols_fit, design_matrix, sig, min_signal, min_diffusivity):
    ''' Helper function used by wls_fit_tensor.
    '''
    sig = np.maximum(sig, min_signal)  # throw out zero signals
    log_s = np.log(sig)
    w = np.exp(np.dot(ols_fit, log_s))
    result = np.dot(np.linalg.pinv(design_matrix * w[:, None]), w * log_s)
    D=result[:6]
    tensor=from_lower_triangular(D)
    MeanD_square=((tensor[0,0]+tensor[1,1]+tensor[2,2])/3.)**2  
    K_tensor_elements=result[6:21]/MeanD_square
    return decompose_tensors(tensor, K_tensor_elements, min_diffusivity=min_diffusivity)


"""

def _ols_iter(inv_design, sig, min_signal, min_diffusivity):
    ''' Helper function used by ols_fit_tensor.
    '''
    sig = np.maximum(sig, min_signal)  # throw out zero signals
    log_s = np.log(sig)
    D = np.dot(inv_design, log_s)
    tensor = from_lower_triangular(D)
    return decompose_tensor(tensor, min_diffusivity=min_diffusivity)


def _uwlls_iter(weighted_matrix,design_matrix,inv_design,sig,min_signal,min_diffusivity):
    ''' Helper function used by uwlls_fit_ktensor to calculate the weights' matrix.
    '''
    sig=np.maximum(sig,min_signal)  # throw out zero signals
    print('sig.shape',sig.shape)
    log_s = np.log(sig)
    betaols=np.dot(inv_design,log_s)
    print('design_matrix.shape',design_matrix.shape)
    print('inv_design.shape',inv_design.shape) 
    print('betaols.shape',betaols.shape)
    muols=np.dot(design_matrix,betaols)  
    print('muols.shape',muols.shape)
    estimated_signals=np.exp(muols)  
    w=np.dot(weighted_matrix,estimated_signals**2) 
    weighted_inverseB_matrix=np.linalg.pinv(B*w[:,None])
    result=np.dot(weighted_inverseB_matrix,w*log_s) 
    D=result[:6]
    tensor=from_lower_triangular(D)
    MeanD_square=((tensor[0,0]+tensor[1,1]+tensor[2,2])/3.)**2  
    K_tensor_elements=result[6:21]/MeanD_square
    return decompose_tensors(tensor, K_tensor_elements, min_diffusivity=min_diffusivity)

def weighted_matrix_form(B):
    ''' Helper function used by uwlls_fit_ktensor to calculate the weights' matrix.
    '''
    A=np.ones((B.shape[0],B.shape[0]))
    C=np.linalg.matrix_power(A,0)
    return C

"""

def _ols_fit_matrix(design_matrix):
    """
    Helper function to calculate the ordinary least squares (OLS)
    fit as a matrix multiplication. Mainly used to calculate WLS weights. Can
    be used to calculate regression coefficients in OLS but not recommended.

    See Also:
    ---------
    wls_fit_tensor, ols_fit_tensor

    Example:
    --------
    ols_fit = _ols_fit_matrix(design_mat)
    ols_data = np.dot(ols_fit, data)
    """

    U, S, V = np.linalg.svd(design_matrix, False)
    return np.dot(U, U.T)




_lt_indices = np.array([[0, 1, 3],
                        [1, 2, 4],
                        [3, 4, 5]])


def from_lower_triangular(D):
    """ Returns a tensor given the six unique tensor elements

    Given the six unique tensor elments (in the order: Dxx, Dxy, Dyy, Dxz, Dyz,
    Dzz) returns a 3 by 3 tensor. All elements after the sixth are ignored.

    Parameters
    -----------
    D : array_like, (..., >6)
        Unique elements of the tensors

    Returns
    --------
    tensor : ndarray (..., 3, 3)
        3 by 3 tensors

    """
    return D[..., _lt_indices]


_lt_rows = np.array([0, 1, 1, 2, 2, 2])
_lt_cols = np.array([0, 0, 1, 0, 1, 2])


def lower_triangular(tensor, b0=None):
    """
    Returns the six lower triangular values of the tensor and a dummy variable
    if b0 is not None

    Parameters
    ----------
    tensor : array_like (..., 3, 3)
        a collection of 3, 3 diffusion tensors
    b0 : float
        if b0 is not none log(b0) is returned as the dummy variable

    Returns
    -------
    D : ndarray
        If b0 is none, then the shape will be (..., 6) otherwise (..., 7)

    """
    if tensor.shape[-2:] != (3, 3):
        raise ValueError("Diffusion tensors should be (..., 3, 3)")
    if b0 is None:
        return tensor[..., _lt_rows, _lt_cols]
    else:
        D = np.empty(tensor.shape[:-2] + (7,), dtype=tensor.dtype)
        D[..., :-1] = -np.log(b0)
        D[..., :6] = tensor[..., _lt_rows, _lt_cols]
        return D



def decompose_tensors(tensor, K_tensor_elements, min_diffusivity=0):
    """ Returns eigenvalues and eigenvectors given a diffusion tensor

    Computes tensor eigen decomposition to calculate eigenvalues and
    eigenvectors (Basser et al., 1994a).

    Parameters
    ----------
    tensor : array (3, 3)
        Hermitian matrix representing a diffusion tensor.
    K_tensor_elements : array(15,1)
        Independent elements of the K tensors
    min_diffusivity : float
        Because negative eigenvalues are not physical and small eigenvalues,
        much smaller than the diffusion weighting, cause quite a lot of noise
        in metrics such as fa, diffusivity values smaller than
        `min_diffusivity` are replaced with `min_diffusivity`.

    Returns
    -------
    eigvals : array (3,)
        Eigenvalues from eigen decomposition of the tensor. Negative
        eigenvalues are replaced by zero. Sorted from largest to smallest.
    eigvecs : array (3, 3)
        Associated eigenvectors from eigen decomposition of the tensor.
        Eigenvectors are columnar (e.g. eigvecs[:,j] is associated with
        eigvals[j])

    """
    #outputs multiplicity as well so need to unique
    eigenvals, eigenvecs = np.linalg.eigh(tensor)

    #need to sort the eigenvalues and associated eigenvectors
    order = eigenvals.argsort()[::-1]
    eigenvecs = eigenvecs[:, order]
    eigenvals = eigenvals[order]

    eigenvals = eigenvals.clip(min=min_diffusivity)
    # eigenvecs: each vector is columnar

    [Wxxxx,Wyyyy,Wzzzz,Wxxxy,Wxxxz,Wxyyy,Wyyyz,Wxzzz,Wyzzz,Wxxyy,Wxxzz,Wyyzz,Wxxyz,Wxyyz,Wxyzz]=K_tensor_elements
    Wrot=np.zeros([3,3,3,3])
    Wfit=np.zeros([3,3,3,3])
    Wfit[0,0,0,0]=Wxxxx
    Wfit[1,1,1,1]=Wyyyy
    Wfit[2,2,2,2]=Wzzzz
    Wfit[0,0,0,1]=Wfit[0,0,1,0]=Wfit[0,1,0,0]=Wfit[1,0,0,0]=Wxxxy
    Wfit[0,0,0,2]=Wfit[0,0,2,0]=Wfit[0,2,0,0]=Wfit[2,0,0,0]=Wxxxz
    Wfit[1,2,2,2]=Wfit[2,2,2,1]=Wfit[2,1,2,2]=Wfit[2,2,1,2]=Wyzzz
    Wfit[0,2,2,2]=Wfit[2,2,2,0]=Wfit[2,0,2,2]=Wfit[2,2,0,2]=Wxzzz
    Wfit[0,1,1,1]=Wfit[1,0,1,1]=Wfit[1,1,1,0]=Wfit[1,1,0,1]=Wxyyy
    Wfit[1,1,1,2]=Wfit[1,2,1,1]=Wfit[2,1,1,1]=Wfit[1,1,2,1]=Wyyyz
    Wfit[0,0,1,1]=Wfit[0,1,0,1]=Wfit[0,1,1,0]=Wfit[1,0,0,1]=Wfit[1,0,1,0]=Wfit[1,1,0,0]=Wxxyy 
    Wfit[0,0,2,2]=Wfit[0,2,0,2]=Wfit[0,2,2,0]=Wfit[2,0,0,2]=Wfit[2,0,2,0]=Wfit[2,2,0,0]=Wxxzz 
    Wfit[1,1,2,2]=Wfit[1,2,1,2]=Wfit[1,2,2,1]=Wfit[2,1,1,2]=Wfit[2,2,1,1]=Wfit[2,1,2,1]=Wyyzz 
    Wfit[0,0,1,2]=Wfit[0,0,2,1]=Wfit[0,1,0,2]=Wfit[0,1,2,0]=Wfit[0,2,0,1]=Wfit[0,2,1,0]=Wfit[1,0,0,2]=Wfit[1,0,2,0]=Wfit[1,2,0,0]=Wfit[2,0,0,1]=Wfit[2,0,1,0]=Wfit[2,1,0,0]=Wxxyz
    Wfit[0,1,1,2]=Wfit[0,1,2,1]=Wfit[0,2,1,1]=Wfit[1,0,1,2]=Wfit[1,1,0,2]=Wfit[1,1,2,0]=Wfit[1,2,0,1]=Wfit[1,2,1,0]=Wfit[2,0,1,1]=Wfit[2,1,0,1]=Wfit[2,1,1,0]=Wfit[1,0,2,1]=Wxyyz
    Wfit[0,1,2,2]=Wfit[0,2,1,2]=Wfit[0,2,2,1]=Wfit[1,0,2,2]=Wfit[1,2,0,2]=Wfit[1,2,2,0]=Wfit[2,0,1,2]=Wfit[2,0,2,1]=Wfit[2,1,0,2]=Wfit[2,1,2,0]=Wfit[2,2,0,1]=Wfit[2,2,1,0]=Wxyzz

    indexarray=[[0,0,0,0],[1,1,1,1],[2,2,2,2],[0,0,1,1],[0,0,2,2],[1,1,2,2]]
    Wrotat=[0,0,0,0,0,0]
    for indval in range(len(indexarray)):
         	Wrotat[indval]=rotatew(Wfit,eigenvecs,indexarray[indval])
         	[W_xxxx,W_yyyy,W_zzzz,W_xxyy,W_xxzz,W_yyzz]=Wrotat

    return eigenvals, eigenvecs, Wrotat[:3],Wrotat[3:]



def design_matrix(gtab, dtype=None):
        """  
        Constructs design matrix for DKI weighted least squares or
        least squares fitting.

        Parameters
        ----------
        gtab : A GradientTable class instance

        dtype : string
            Parameter to control the dtype of returned designed matrix

        Returns
        -------
        design_matrix : array (g,22)
            Design matrix or B matrix assuming Gaussian distributed tensor model
            design_matrix[j, :] = (Bxx,Byy,Bzz,Bxy,Bxz,Byz,Bxxxx,Byyyy,Bzzzz,
                                   Bxxxy,Bxxxz,Bxyyy,Byyyz,Bxzzz,Byzzz, Bxxyy,
                                   Bxxzz, Byyzz,Bxxyz, Bxyyz, Bxyzz, dummy)
        """

        B = np.zeros((gtab.gradients.shape[0], 22))
	B[:, 0] = gtab.bvecs[:, 0] * gtab.bvecs[:, 0] * 1. * gtab.bvals   # Bxx
	B[:, 1] = gtab.bvecs[:, 0] * gtab.bvecs[:, 1] * 2. * gtab.bvals   # Bxy
	B[:, 2] = gtab.bvecs[:, 1] * gtab.bvecs[:, 1] * 1. * gtab.bvals   # Byy
	B[:, 3] = gtab.bvecs[:, 0] * gtab.bvecs[:, 2] * 2. * gtab.bvals   # Bxz
	B[:, 4] = gtab.bvecs[:, 1] * gtab.bvecs[:, 2] * 2. * gtab.bvals   # Byz
	B[:, 5] = gtab.bvecs[:, 2] * gtab.bvecs[:, 2] * 1. * gtab.bvals   # Bzz
	B[:, 6] = -gtab.bvecs[:,0]**4*((gtab.bvals**2)/6.) 				#Bxxxx
	B[:, 7] = -gtab.bvecs[:,1]**4*((gtab.bvals**2)/6.)  				#Byyyy
	B[:, 8] = -gtab.bvecs[:,2]**4*((gtab.bvals**2)/6.)  				#Bzzzz
	B[:, 9] = -gtab.bvecs[:,0]**3*gtab.bvecs[:,1]*4.*((gtab.bvals**2)/6.)	 	#Bxxxy
	B[:,10] = -gtab.bvecs[:,0]**3*gtab.bvecs[:,2]*4.*((gtab.bvals**2)/6.) 		#Bxxxz
	B[:,11] = -gtab.bvecs[:,0]*gtab.bvecs[:,1]**3*4.*((gtab.bvals**2)/6.) 		#Bxyyy
	B[:,12] = -gtab.bvecs[:,1]**3*gtab.bvecs[:,2]*4.*((gtab.bvals**2)/6.) 		#Byyyz
	B[:,13] = -gtab.bvecs[:,0]*gtab.bvecs[:,2]**3*4.*((gtab.bvals**2)/6.) 		#Bxzzz
	B[:,14] = -gtab.bvecs[:,1]*gtab.bvecs[:,2]**3*4.*((gtab.bvals**2)/6.) 		#Byzzz
	B[:,15] = -gtab.bvecs[:,0]**2*gtab.bvecs[:,1]**2*6.*((gtab.bvals**2)/6.)  		#Bxxyy
	B[:,16] = -gtab.bvecs[:,0]**2*gtab.bvecs[:,2]**2*6.*((gtab.bvals**2)/6.) 		#Bxxzz
	B[:,17] = -gtab.bvecs[:,1]**2*gtab.bvecs[:,2]**2*6.*((gtab.bvals**2)/6.) 		#Byyzz
	B[:,18] = -gtab.bvecs[:,0]**2*gtab.bvecs[:,1]*gtab.bvecs[:,2]*12.*((gtab.bvals**2)/6.) 	#Bxxyz
	B[:,19] = -gtab.bvecs[:,0]*gtab.bvecs[:,1]**2*gtab.bvecs[:,2]*12.*((gtab.bvals**2)/6.) 	#Bxyyz
	B[:,20] = -gtab.bvecs[:,0]*gtab.bvecs[:,1]*gtab.bvecs[:,2]**2*12.*((gtab.bvals**2)/6.) 	#Bxyzz
	B[:,21] = np.ones(gtab.bvals.size)					#BlogS0
	return -B


def quantize_evecs(evecs, odf_vertices=None):
    """ Find the closest orientation of an evenly distributed sphere

    Parameters
    ----------
    evecs : ndarray
    odf_vertices : None or ndarray
        If None, then set vertices from symmetric362 sphere.  Otherwise use
        passed ndarray as vertices

    Returns
    -------
    IN : ndarray
    """
    max_evecs = evecs[..., :, 0]
    if odf_vertices == None:
        odf_vertices = get_sphere('symmetric362').vertices
    tup = max_evecs.shape[:-1]
    mec = max_evecs.reshape(np.prod(np.array(tup)), 3)
    IN = np.array([np.argmin(np.dot(odf_vertices, m)) for m in mec])
    IN = IN.reshape(tup)
    return IN
    


    
common_fit_methods = {'UWLLS_KURT': uwlls_fit_ktensor,'ULLS_KURT' : ulls_fit_ktensor}
