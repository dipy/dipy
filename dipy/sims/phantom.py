import numpy as np
from dipy.sims.voxel import SingleTensor
from dipy.core.geometry import vec2vec_rotmat
from dipy.data import get_data    
#from dipy.viz import fvtk
#from dipy.reconst.dti import Tensor

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


def orbital_phantom(bvals=None,
                     bvecs=None,
                     evals=np.array([1.4,.35,.35])*10**(-3),
                     func=None,
                     t=np.linspace(0,2*np.pi,1000),
                     datashape=(64,64,64,65),
                     origin=(32,32,32),
                     scale=(25,25,25),
                     angles=np.linspace(0,2*np.pi,32),
                     radii=np.linspace(0.2,2,6),
                     S0=100.):
    """ Create a phantom based on a 3d orbit f(t)->(x,y,z)
    
    Parameters
    -----------
    bvals : array, shape (N,)
    bvecs : array, shape (N,3)
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
    snr : signal to noise ratio
        Used for applying rician noise to the data.
        Default 200. Common is 20. 
    background_noise : boolean, Default False
    
    Returns
    ---------
    data : array, shape (datashape)
    
    Notes 
    --------
    Crossings can be created by adding multiple orbitual_phantom outputs.
    
    Examples
    ---------
    
    def f(t):
        x=np.sin(t)
        y=np.cos(t)        
        z=np.linspace(-1,1,len(x))
        return x,y,z
    
    data=orbitual_phantom(func=f)
        
    """
    
    if bvals==None:
        fimg,fbvals,fbvecs=get_data('small_64D')
        bvals=np.load(fbvals)
        bvecs=np.load(fbvecs)
        bvecs[np.isnan(bvecs)]=0
        
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
        S=SingleTensor(bvals,bvecs,S0,evals,evecs,snr=None)
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
    return vol

def add_noise(vol, snr=20, noise_type='gaussian', tol=10e-4):
    r""" add gaussian noise in a 4D array with a specific snr
    
    Parameters
    -----------
    vol : array, shape (X,Y,Z,W) 
    snr : float,
        signal to noise ratio

    noise_type: string
        The distribution of noise added. Can be either 'gaussian' for Gaussian
        distributed noise (default), or 'rician' for Rice-distributed noise.

    Returns
    --------
    voln : array, same shape as vol
        vol with additional rician noise    

    Notes
    -----
    Following: http://en.wikipedia.org/wiki/Signal-to-noise_ratio

    We use the following definition of SNR:

    .. math ::

        SNR = \frac{P_{signal}}{P_{noise}} = (\frac{A_{signal}}{A_{noise}})^2

    Where:

    .. math ::

        A_x = \sqrt{\bar{(x - \bar{x})^2}} = <x>^2

    References
    ----------

    Gudbjartsson, H and Patz, S (2008). The Rician Distribution of Noisy MRI
    Data. Magnetic Resonance in Medicine 34: 910-914

    Examples
    --------
    >>> signal = np.arange(800).reshape(2,2,2,100)
    >>> signal_w_noise = add_noise(signal,snr=10,noise_type='rician')

    """

    # We estimate the power in the signal as the mean of signal
    p_signal = np.mean(vol)

    # SNR = mean(signal)/std(noise) => mean(noise) = std(signal)/SNR:
    sigma = p_signal / snr

    if noise_type == 'gaussian':
        noise_adder = _add_gaussian
        # Generate the noise with the correct standard deviation, averaged over
        # all the voxels and with the right shape:
        noise1 = np.random.normal(0, sigma, size=vol.shape)
        # In this case, we don't need another source of noise:
        noise2 = np.nan
    elif noise_type in['rician', 'rayleigh']:
        if noise_type == 'rician':
            noise_adder = _add_rician
        elif noise_type == 'rayleigh':
            noise_adder = _add_rayleigh
        # To generate rician and rayleigh noises, we combine two IID Gaussian
        # noise sources in the complex domain (see below _add_rician and
        # _add_rayleigh for the details):
        noise1 = np.random.normal(0, sigma, size=vol.shape)
        noise2 = np.random.normal(0, sigma, size=vol.shape)

    sig_w_noise = noise_adder(vol, noise1, noise2)
    est_noise = sig_w_noise - vol
    est_snr = np.mean(vol)/np.std(est_noise)
    # The resulting distribution may have variance that doesn't give us the
    # requested SNR (seems to depend on the SNR. Larger deviations for
    # smaller SNR). We adjust the variance of the underlying Gaussians
    # until the noise-level approximately reaches the desired SNR:
    while np.abs(est_snr - snr) > tol:
        if  est_snr > snr:
            noise1 = noise1 * 1.01
            noise2 = noise2 * 1.01
        elif est_snr < snr:
            noise1 = noise1 * 0.9
            noise2 = noise2 * 0.9
        sig_w_noise = noise_adder(vol, noise1, noise2)
        est_noise = sig_w_noise - vol
        est_snr = np.mean(vol)/np.std(est_noise)
    return sig_w_noise

def _add_gaussian(vol, noise1, noise2):
    """
    Helper function to add_noise

    This one simply adds one of the Gaussians to the vol and ignores the other
    one.
    """
    return vol + noise1

def _add_rician(vol, noise1, noise2):
    """
    Helper function to add_noise.

    This does the same as abs(vol + complex(noise1, noise2))

    """
    return np.sqrt((vol + noise1)**2 + noise2**2)

def _add_rayleigh(vol, noise1, noise2):
    """
    Helper function to add_noise

    The Rayleigh distribution is $\sqrt\{Gauss_1^2 + Gauss_2^2}$.

    """
    return vol + np.sqrt(noise1**2 + noise2**2)


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
