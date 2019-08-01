import numpy as np
cimport numpy as cnp
cimport cython

cimport safe_openmp as openmp
from safe_openmp cimport have_openmp
from cython.parallel import parallel, prange, threadid
from libc.stdlib cimport malloc, free

from dipy.denoise.enhancement_kernel import EnhancementKernel
from dipy.data import get_sphere
from dipy.reconst.shm import sh_to_sf, sf_to_sh

def convolve(odfs_sh, kernel, sh_order, test_mode=False, num_threads=None, normalize=True):
    """ Perform the shift-twist convolution with the ODF data and 
    the lookup-table of the kernel.

    Parameters
    ----------
    odfs : array of double
        The ODF data in spherical harmonics format
    kernel : array of double
        The 5D lookup table
    sh_order : integer
        Maximal spherical harmonics order
    test_mode : boolean
        Reduced convolution in one direction only for testing
    num_threads : int
            Number of threads. If None (default) then all available threads
            will be used.
    normalize : boolean
        Apply max-normalization to the output such that its value range matches 
        the input ODF data.
        
    Returns
    -------
    output : array of double
        The ODF data after convolution enhancement in spherical harmonics format
        
    References
    ----------
    [Meesters2016_ISMRM] S. Meesters, G. Sanguinetti, E. Garyfallidis, 
                         J. Portegies, R. Duits. (2016) Fast implementations of 
                         contextual PDE’s for HARDI data processing in DIPY. 
                         ISMRM 2016 conference.
    [DuitsAndFranken_IJCV] R. Duits and E. Franken (2011) Left-invariant diffusions 
                        on the space of positions and orientations and their 
                        application to crossing-preserving smoothing of HARDI 
                        images. International Journal of Computer Vision, 92:231-264.
    [Portegies2015] J. Portegies, G. Sanguinetti, S. Meesters, and R. Duits. 
                    (2015) New Approximation of a Scale Space Kernel on SE(3) and
                    Applications in Neuroimaging. Fifth International
                    Conference on Scale Space and Variational Methods in
                    Computer Vision
    [Portegies2015b] J. Portegies, R. Fick, G. Sanguinetti, S. Meesters, G.Girard,
                     and R. Duits. (2015) Improving Fiber Alignment in HARDI by 
                     Combining Contextual PDE flow with Constrained Spherical 
                     Deconvolution. PLoS One.
    """

    # convert the ODFs from SH basis to DSF
    sphere = kernel.get_sphere()
    odfs_dsf = sh_to_sf(odfs_sh, sphere, sh_order=sh_order, basis_type=None)

    # perform the convolution
    output = perform_convolution(odfs_dsf, 
                                 kernel.get_lookup_table(),
                                 test_mode,
                                 num_threads)
    
    # normalize the output
    if normalize:
        output = np.multiply(output, np.amax(odfs_dsf)/np.amax(output))
    
    # convert back to SH
    output_sh = sf_to_sh(output, sphere, sh_order=sh_order)
    
    return output_sh
    
def convolve_sf(odfs_sf, kernel, test_mode=False, num_threads=None, normalize=True):
    """ Perform the shift-twist convolution with the ODF data and 
    the lookup-table of the kernel.

    Parameters
    ----------
    odfs : array of double
        The ODF data sampled on a sphere
    kernel : array of double
        The 5D lookup table
    test_mode : boolean
        Reduced convolution in one direction only for testing
    num_threads : int
            Number of threads. If None (default) then all available threads
            will be used.
    normalize : boolean
        Apply max-normalization to the output such that its value range matches 
        the input ODF data.

    Returns
    -------
    output : array of double
        The ODF data after convolution enhancement, sampled on a sphere
    """
    # perform the convolution
    output = perform_convolution(odfs_sf, 
                                 kernel.get_lookup_table(),
                                 test_mode,
                                 num_threads)

    # normalize the output
    if normalize:
        output = np.multiply(output, np.amax(odfs_sf)/np.amax(output))

    return output
    
@cython.wraparound(False)
@cython.boundscheck(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double [:, :, :, ::1] perform_convolution (double [:, :, :, ::1] odfs, 
                                                double [:, :, :, :, ::1] lut,
                                                cnp.npy_intp test_mode,
                                                num_threads=None):
    """ Perform the shift-twist convolution with the ODF data 
    and the lookup-table of the kernel.

    Parameters
    ----------
    odfs : array of double
        The ODF data sampled on a sphere
    lut : array of double
        The 5D lookup table
    test_mode : boolean
        Reduced convolution in one direction only for testing
    num_threads : int
            Number of threads. If None (default) then all available threads
            will be used.
        
    Returns
    -------
    output : array of double
        The ODF data after convolution enhancement
    """
        
    cdef:
        double [:, :, :, ::1] output = np.array(odfs, copy=True)
        cnp.npy_intp OR1 = lut.shape[0]
        cnp.npy_intp OR2 = lut.shape[1]
        cnp.npy_intp N = lut.shape[2]
        cnp.npy_intp hn = (N - 1) / 2
        double [:, :, :, :] totalval
        double [:, :, :, :] voxcount
        cnp.npy_intp nx = odfs.shape[0]
        cnp.npy_intp ny = odfs.shape[1]
        cnp.npy_intp nz = odfs.shape[2]
        cnp.npy_intp threads_to_use = -1
        cnp.npy_intp all_cores = openmp.omp_get_num_procs()
        cnp.npy_intp corient, orient, cx, cy, cz, x, y, z
        cnp.npy_intp expectedvox
        cnp.npy_intp edgeNormalization = True

    if num_threads is not None:
        threads_to_use = num_threads
    else:
        threads_to_use = all_cores

    if have_openmp:
        openmp.omp_set_dynamic(0)
        openmp.omp_set_num_threads(threads_to_use)
    
    if test_mode:
        edgeNormalization = False
        OR2 = 1

    # expected number of voxels in kernel
    totalval = np.zeros((OR1, nx, ny, nz))
    voxcount = np.zeros((OR1, nx, ny, nz))
    expectedvox = nx * ny * nz

    with nogil:

        # loop over ODFs cx,cy,cz,orient --> y and v
        for corient in prange(OR1, schedule='guided'):
            for cx in range(nx):
                for cy in range(ny):
                    for cz in range(nz):
                        # loop over kernel x,y,z,orient --> x and r
                        for x in range(int_max(cx - hn, 0), 
                                       int_min(cx + hn + 1, ny - 1)):
                             for y in range(int_max(cy - hn, 0), 
                                            int_min(cy + hn + 1, ny - 1)):
                                 for z in range(int_max(cz - hn, 0), 
                                                int_min(cz + hn + 1, nz - 1)):
                                    voxcount[corient, cx, cy, cz] += 1.0
                                    for orient in range(0, OR2):
                                        totalval[corient, cx, cy, cz] += \
                                            odfs[x, y, z, orient] * \
                                            lut[corient, orient, x - (cx - hn), y - (cy - hn), z - (cz - hn)]
                        if edgeNormalization:
                            output[cx, cy, cz, corient] = \
                                totalval[corient, cx, cy, cz] * expectedvox/voxcount[corient, cx, cy, cz]
                        else:
                            output[cx, cy, cz, corient] = \
                                totalval[corient, cx, cy, cz]

    # Reset number of OpenMP cores to default
    if have_openmp and num_threads is not None:
        openmp.omp_set_num_threads(all_cores)

    return output

cdef inline cnp.npy_intp int_max(cnp.npy_intp a, cnp.npy_intp b) nogil: 
    return a if a >= b else b
cdef inline cnp.npy_intp int_min(cnp.npy_intp a, cnp.npy_intp b) nogil: 
    return a if a <= b else b
