import numpy as np
import scipy.stats as stats

from dipy.sims.voxel import SingleTensor
import dipy.sims.voxel as vox
from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_data    
from dipy.core.gradients import gradient_table


def add_noise(vol, snr=1.0, S0=1.0, noise_type='rician'):
    r""" Add noise of specified distribution to a 4D array.
    
    Parameters
    -----------
    vol : array, shape (X,Y,Z,W)

    snr : float
        The desired signal-to-noise ratio.

        SNR is defined here following Descoteaux et al. (2007) as S0/sigma,
        where sigma is the standard deviation of the complex noise. That is, it
        is the standard deviation of the Gaussian distributions on the
        imaginary and on the real part that are combined to derive the Rician
        distribution of the noise (see also Gudbjartson and Patz, 2008).
    
    noise_type : string
        The distribution of noise added. Can be either 'gaussian' for Gaussian
        distributed noise (default), 'rician' for Rice-distributed noise or
        'rayleigh' for a Rayleigh distribution.
        
    Returns
    --------
    vol : array, same shape as vol
        vol with added noise    

    References
    ----------

    Gudbjartson and Patz (2008). The Rician distribution of noisy MRI data. MRM
    34: 910-914.

    Descoteaux, Angelino, Fitzgibbons and Deriche (2007) Regularized, fast and
    robust q-ball imaging. MRM, 58: 497-510 
    
    Examples
    --------
    >>> signal = np.arange(800).reshape(2, 2, 2, 100)
    >>> signal_w_noise = add_noise(signal, snr=10, noise_type='rician')

    """
    orig_shape = vol.shape
    vol_flat = np.reshape(vol.copy(), (-1, vol.shape[-1]))

    for vox_idx, signal in enumerate(vol_flat):
        max_sig = np.max(signal)
        # We assume that the S0 is the maximal signal:
        vol_flat[vox_idx] = vox.add_noise(signal, snr=snr, S0=max_sig,
                                          noise_type=noise_type)

    return np.reshape(vol_flat, orig_shape)


def diff2eigenvectors(dx,dy,dz):
    """ numerical derivatives 2 eigenvectors 
    """
    basis=np.eye(3)
    u=np.array([dx,dy,dz])
    u=u/np.linalg.norm(u)
    R=vec2vec_rotmat(basis[:,0],u)
    eig0=u
    eig1=np.dot(R,basis[:,1])
    eig2=np.dot(R,basis[:,2])
    eigs=np.zeros((3,3))
    eigs[:,0]=eig0
    eigs[:,1]=eig1
    eigs[:,2]=eig2    
    return eigs, R


def orbital_phantom(gtab=None,
                    evals=np.array([1.4,.35,.35])*10**(-3),
                    func=None,
                    t=np.linspace(0,2*np.pi,1000),
                    datashape=(64,64,64,65),
                    origin=(32,32,32),
                    scale=(25,25,25),
                    angles=np.linspace(0,2*np.pi,32),
                    radii=np.linspace(0.2,2,6),
                    S0=100.,
                    snr=None,
                    snr_tol=10e-4):
    """ Create a phantom based on a 3d orbit f(t)->(x,y,z)
    
    Parameters
    -----------
    gtab : GradientTable
    evals : array, shape (3,)
        tensor eigenvalues
    func : user defined function f(t)->(x,y,z) 
        It could be desirable for -1=<x,y,z <=1 
        if None creates a circular orbit
    t : array, shape (K,)
        represents time for the orbit
        Default is np.linspace(0,2*np.pi,1000)
    datashape : array, shape (X,Y,Z,W)
        size of the output simulated data
    origin : tuple, shape (3,)
        define the center for the volume
    scale : tuple, shape (3,)
        scale the function before applying to the grid
    angles : array, shape (L,)
        density angle points, always perpendicular to the first eigen vector
        Default np.linspace(0,2*np.pi,32),
    radii : array, shape (M,)
        thickness radii    
        Default np.linspace(0.2,2,6)
        angles and radii define the total thickness options 
    S0 : double, simulated signal without diffusion gradients applied
        Default 100.
    snr : float, optional
        The signal to noise ratio set to apply Rician noise to the data.
        Default is to not add noise at all.
    snr_tol: float, optional
        How close to the requested SNR do we have to get. Default 10e-4

    Returns
    -------
    data : array, shape (datashape)
    
    Notes 
    -----
    Crossings can be created by adding multiple orbital_phantom outputs.

    In these simulations, we can ask for Rician noise to be added. In
    that case, the definition of SNR is as follows:

        SNR = mean(true signal)/RMS(true added noise).
    
    Gudbjartsson, H and Patz, S (2008). The Rician Distribution of Noisy MRI
    Data. Magnetic Resonance in Medicine 34: 910-914


    Examples
    ---------
    
    def f(t):
        x=np.sin(t)
        y=np.cos(t)        
        z=np.linspace(-1,1,len(x))
        return x,y,z
    
    data=orbital_phantom(func=f)
        
    """
    
    if gtab.bvals==None:
        fimg, fbvals, fbvecs=get_data('small_64D')
        gtab = gradient_table(fbvals, fbvecs)
        
    if func==None:
        x=np.sin(t)
        y=np.cos(t)
        z=np.zeros(t.shape)
    else:
        x,y,z=func(t)
        
    #stop 
    
    dx=np.diff(x)
    dy=np.diff(y)
    dz=np.diff(z)
    
    x=scale[0]*x+origin[0]
    y=scale[1]*y+origin[1]
    z=scale[2]*z+origin[2]
        
    bx=np.zeros(len(angles))
    by=np.sin(angles)
    bz=np.cos(angles)
    
    vol=np.zeros(datashape)
    
    for i in range(len(dx)):
        evecs,R=diff2eigenvectors(dx[i],dy[i],dz[i])        
        S=SingleTensor(gtab,S0,evals,evecs,snr=None)
        #print sigma, S0/snr, S0, snr
        vol[x[i],y[i],z[i],:]+=S
        for r in radii:
            for j in range(len(angles)):
                rb=np.dot(R,np.array([bx[j],by[j],bz[j]])) 
                vol[x[i]+r*rb[0],y[i]+r*rb[1],z[i]+r*rb[2]]+=S
        
    #ten=Tensor(vol,bvals,bvecs)
    #FA=ten.fa()
    #FA[np.isnan(FA)]=0
    #vol[np.isnan(vol)]=0

    if snr is not None:
        vol = add_noise(vol, snr, S0=S0*len(angles)/2,
                        noise_type='rician')

    return vol


if __name__ == "__main__":

    ##TODO: this can become a nice tutorial for generating phantoms
    
    def f(t):
        x=np.sin(t)
        y=np.cos(t)
        #z=np.zeros(t.shape)
        z=np.linspace(-1,1,len(x))
        return x,y,z
    
    #helix
    vol=orbital_phantom(func=f)   
    
    def f2(t):
        x=np.linspace(-1,1,len(t))
        y=np.linspace(-1,1,len(t))    
        z=np.zeros(x.shape)
        return x,y,z

    #first direction
    vol2=orbital_phantom(func=f2)
    
    def f3(t):
        x=np.linspace(-1,1,len(t))
        y=-np.linspace(-1,1,len(t))    
        z=np.zeros(x.shape)
        return x,y,z

    #second direction
    vol3=orbital_phantom(func=f3)
    #double crossing
    vol23=vol2+vol3
       
    #"""
    def f4(t):
        x=np.zeros(t.shape)
        y=np.zeros(t.shape)
        z=np.linspace(-1,1,len(t))
        return x,y,z
    
    #triple crossing
    vol4=orbital_phantom(func=f4)
    vol234=vol23+vol4
    
    voln=add_rician_noise(vol234)
    
    #"""
    
    #r=fvtk.ren()
    #fvtk.add(r,fvtk.volume(vol234[...,0]))
    #fvtk.show(r)
    #vol234n=add_rician_noise(vol234,20)
