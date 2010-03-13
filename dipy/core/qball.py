""" Q-ball Reconstruction

We are implementing the reconstruction from the paper

@article{descoteaux2007regularized,
  title={{Regularized, fast, and robust analytical Q-ball imaging}},
  author={Descoteaux, M. and Angelino, E. and Fitzgibbons, S. and Deriche, R.},
  journal={Magnetic Resonance in Medicine},
  volume={58},
  number={3},
  pages={497--510},
  year={2007},
  publisher={New York: Academic Press,[c1984-}
}

We also took ideas from the code found in diffusion-mri toolbox by Jiang found in the link below
http://code.google.com/p/diffusion-mri/

"""

import math
import numpy as np
import scipy as sp



def test_qball_odf():

    g_fname='/home/eg309/Devel/diffusion-mri-read-only/data/81vectors.txt'

    g=np.loadtxt(g_fname)

    
    v=400*np.ones(g.shape[0])

    l=8
    
    r=0.06

    C=qball_odf(v,g,l,r)

    return C



def qball_odf(v,g,l,r):

    ''' Calculate the coefficients of the orientation distribution function (ODF) using Q-ball reconstruction as described in Descoteaux et. al. MRM 2007

    Parameters:
    -----------
    v: signal at a voxel
    
    g: gradient encoding direction
    l: order of spherical harmonics basis R=(1/2)(l+1)(l+2)
    r: regularization parameter
    
    '''

    #number of directions
    N=len(v)

    #number of spherical harmonics coefficients that we will approximate
    
    R=(1/2)*(l+1)*(l+2)

    B=np.zeros((N,R))
    
    #gradient directions (bvecs) in spherical coordinates
    #r = math.sqrt(x**2+y**2+z**2)
    #elev = math.atan2(z,math.sqrt(x**2+y**2))
    #az = math.atan2(y,x)

    gradial =np.sqrt(np.sum(g**2,axis=1))
    gelev =np.arctan2(g[:,2],np.sqrt(np.sum(g[:,:2]**2,axis=1)))
    gazimuth = np.arctan2(g[:,1],g[:,0])
    gs=np.c_[gelev,gazimuth,gradial]

    #gelev from 0 to pi
    #gazimuth from 0 to 2pi
    
    from scipy.special import sph_harm as y_lm

    kms=[]
    for k in range(0,l+1,2):
        for m in range(-k,k+1):
            kms.append((k,m))


    #Ylm(l,m,theta,phi)

    

    return kms





