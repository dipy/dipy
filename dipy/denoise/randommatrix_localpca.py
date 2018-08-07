import time
import numpy as np
from math import floor,sqrt
from scipy.linalg import eigh

def randommatrix_localpca(arr, patch_extent=0, out_dtype=None):
    r"""Local PCA-based denoising of diffusion datasets.

    Parameters
    ----------
    arr : 4D array
        Array of data to be denoised. The dimensions are (X, Y, Z, N), where N
        are the diffusion gradient directions.   
    patch_extent : int, optional
        The diameter of the local patch to be taken around each voxel (in
        voxels). The radius will be half of this value. If not provided,
        the default will be automatically computed as:

        .. math ::

                patch_extent = max(5,\lfloor N^{1/3} \rfloor)

    out_dtype : str or dtype, optional
        The dtype for the output array. Default: output has the same dtype as
        the input.

    Returns
    -------
    denoised_arr : 4D array
        This is the denoised array of the same size as that of the input data,
        clipped to non-negative values
    noise_arr : 3D array
        Voxelwise standard deviation of the noise estimated from the data.
    sigma : float
        Mean value of noise standard deviations over all voxels (mean of 
        noise_arr).

    References    
    ----------
    .. [Veraart16] Veraart J, Fieremans E, Novikov DS (2016)
                  Diffusion MRI noise mapping using random matrix theory.
                  Magnetic resonance in Medicine 76(5), p1582-1593.
                  https://doi.org/10.1002/mrm.26059
    """

    start_time = time.time()

    if out_dtype is None:
        out_dtype = arr.dtype

    # We perform all computations in float64 precision but return the 
    # results in the original input's precision
    if arr.dtype == np.float64:
        calc_dtype = np.float64
    else:
        calc_dtype = np.float32
        arr=arr.astype(np.float64)

    if not arr.ndim == 4:
        raise ValueError("PCA denoising can only be performed on 4D arrays.",
                         arr.shape)
    if patch_extent <= 0:
        Nvols = arr.shape[-1]
        patch_extent = max(5,Nvols ** (1. / 3.)) 
  
    patch_radius=int(floor(patch_extent/2.))
    m = arr.shape[-1]
    n = (2*patch_radius+1) ** 3
    r = m if (m<n) else n
         
    noise_arr = np.zeros([arr.shape[0],arr.shape[1],arr.shape[2]],dtype=calc_dtype)
    denoised_arr = np.zeros(arr.shape, dtype=calc_dtype)        

    for k in range(patch_radius,arr.shape[2]-patch_radius):   
        kx1 = k - patch_radius
        kx2 = k + patch_radius + 1
        for j in range(patch_radius, arr.shape[1] - patch_radius):
            jx1 = j - patch_radius
            jx2 = j + patch_radius + 1
            for i in range(patch_radius, arr.shape[0] - patch_radius):
                # Shorthand for indexing variables:
                ix1 = i - patch_radius
                ix2 = i + patch_radius + 1

                X = arr[ix1:ix2, jx1:jx2, kx1:kx2].reshape(n, m).transpose()
               
                if m <= n:
                    XtX = X.dot(np.transpose(X))
                else:
                    XtX = np.transpose(X).dot(X)                    

                [vals, vecs] = eigh(XtX)
                lamr = vals[0] / n
                clam = 0.
                sigma2 = 0.
                cutoff_p = 0

                for p in range(r):
                    lam = vals[p] / n
                    clam = clam + lam
                    gam = float(m-r+p+1) / n 
                    sigsq1 = clam / (p+1) / max(gam,1.)
                    sigsq2 = (lam -lamr) / 4. / sqrt(gam)
                    if sigsq2 < sigsq1:
                        sigma2= sigsq1
                        cutoff_p = p+1                

                if cutoff_p > 0:
                    s=vals
                    s[0:cutoff_p] = 0.
                    s[cutoff_p:] = 1.
                    if m <= n:
                        nvals = vecs.dot(np.diag(s)).dot(vecs.transpose()).dot(X[:,n/2])
                    else:
                        nvals =  np.dot(X,np.dot(vecs,np.dot(np.diag(s),vecs.transpose()[:,n/2])))
                else:
                    nvals=  np.zeros(arr.shape[3],dtype=calc_dtype)
                
                denoised_arr[i,j,k,:] = nvals
                noise_arr[i,j,k] = sqrt(sigma2)               
    
    sigma = np.mean(noise_arr[np.nonzero(noise_arr)])
    print(sigma)

    print("--- %s seconds ---" % (time.time() - start_time))

    return denoised_arr.astype(out_dtype), noise_arr.astype(out_dtype), sigma

