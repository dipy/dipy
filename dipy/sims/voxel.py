import numpy as np
from dipy.core.geometry import sphere2cart, cart2sphere

def SticksAndBall(bvals,gradients,d=0.0015,S0=100,angles=[(0,0),(90,0)],fractions=[35,35],snr=20):
    """ Simulating the signal for a Sticks & Ball model 
    
    Based on the paper by Tim Behrens, H.J. Berg, S. Jbabdi, "Probabilistic Diffusion Tractography with multiple fiber orientations
    what can we gain?", Neuroimage, 2007. 
    
    Parameters
    -----------
    bvals : array, shape (N,)
    gradients : array, shape (N,3) also known as bvecs
    d : diffusivity value 
    S0 : unweighted signal value
    angles : list of polar angles (in degrees) for the sticks
        or array (K,3) with sticks as Cartesian unit vectors and K the number of sticks
    fractions : percentage of each stick
    snr : signal to noise ration assuming gaussian noise. Provide None for no noise.
    
    Returns
    --------
    S : simulated signal
    sticks : sticks in cartesian coordinates 
    
    """
    
    fractions=[f/100. for f in fractions]    
    f0=1-np.sum(fractions)    
    S=np.zeros(len(gradients))
    
    angles=np.array(angles)
    if angles.shape[-1]==3:
        sticks=angles
    if angles.shape[-1]==2:
        sticks=[ sphere2cart(1,np.deg2rad(pair[0]),np.deg2rad(pair[1]))  for pair in angles]    
        sticks=np.array(sticks)
    
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=f0*np.exp(-bvals[i+1]*d)+ np.sum([fractions[j]*np.exp(-bvals[i+1]*d*np.dot(s,g)**2) for (j,s) in enumerate(sticks)])
        S[i+1]=S0*S[i+1]    
    S[0]=S0    
    if snr!=None:
        std=S0/snr
        S=S+np.random.randn(len(S))*std
    
    return S,sticks

def SingleTensor(bvals,gradients,S0,evals,evecs,snr=None):
    """ Simulated signal with a Single Tensor
     
    Parameters
    ----------- 
    bvals : array, shape (N,)
    gradients : array, shape (N,3) also known as bvecs
    S0 : double,
    evals : array, shape (3,) eigen values
    evecs : array, shape (3,3) eigen vectors
    snr : signal to noise ratio assuming gaussian noise. 
        Provide None for no noise.
    
    Returns
    --------
    S : simulated signal
    
    
    """
    S=np.zeros(len(gradients))
    D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)    
    #print D.shape
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=S0*np.exp(-bvals[i+1]*np.dot(np.dot(g.T,D),g))
    S[0]=S0
    if snr!=None:
        std=S0/snr
        S=S+np.random.randn(len(S))*std
    return S
    

