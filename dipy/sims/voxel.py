import numpy as np
from dipy.core.geometry import sphere2cart, cart2sphere
from dipy.reconst.dti import design_matrix, lower_triangular
from dipy.core.geometry import vec2vec_rotmat



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
    angles : array (K,2) list of polar angles (in degrees) for the sticks
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
    """ Alternative suggestion which works with multiple b0s
    design = design_matrix(bval, gradients.T)
    S = np.exp(np.dot(design, lower_triangular(D)))
    """
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=S0*np.exp(-bvals[i+1]*np.dot(np.dot(g.T,D),g))
    S[0]=S0
    if snr!=None:
        std=S0/snr
        S=S+np.random.randn(len(S))*std
    return S
    

def all_tensor_evecs(e0):
    ''' Principal axis to all tensor axes
    '''
    axes=np.array([[1.,0,0],[0,1.,0],[0,0,1.]])
    mat=vec2vec_rotmat(axes[2],e0)
    e1=np.dot(mat,axes[0])
    e2=np.dot(mat,axes[1])
    return np.array([e0,e1,e2])


def multi_tensor_odf(odf_verts,mf,mevals,mevecs):
    r''' Simulating a Multi-Tensor ODF

    Parameters:
    -----------
    
    odf_verts : array, shape (N,3), 
        vertices of the reconstruction sphere 
    mf : sequence of floats, bounded [0,1]
        percentages of the fractions for each Tensor
    mevals : sequence of 1D arrays,
        eigen-values for each Tensor
    mevecs : sequence of 3D arrays,
        eigen-vectors for each Tensor

    Returns:
    ---------
    ODF : array, shape (N,),
        orientation distribution function

    Examples:
    ----------
    Simulate a MultiTensor with two peaks and calcute its exact ODF.

    >>> import numpy as np
    >>> from dipy.sims.voxel import multi_tensor_odf, all_tensor_evecs
    >>> from dipy.data import get_sphere
    >>> vertices, faces = get_sphere('symmetric724')
    >>> mevals=np.array(([0.0015,0.0003,0.0003],
                    [0.0015,0.0003,0.0003]))
    >>> e0=np.array([1,0,0.])
    >>> e1=np.array([0.,1,0])
    >>> mevecs=[all_evecs(e0),all_evecs(e1)]
    >>> odf = multi_tensor_odf(vertices,[0.5,0.5],mevals,mevecs)


    '''

    odf=np.zeros(len(odf_verts))
    m=len(mf)
    for (i,v) in enumerate(odf_verts):
        for (j,f) in enumerate(mf):
            evals=mevals[j]
            evecs=mevecs[j]
            D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)
            iD=np.linalg.inv(D)
            nD=np.linalg.det(D)
            upper=(np.dot(np.dot(v.T,iD),v))**(-3/2.)
            lower=4*np.pi*np.sqrt(nD)
            odf[i]+=f*upper/lower
    return odf


