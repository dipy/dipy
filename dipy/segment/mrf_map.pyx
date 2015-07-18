#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
cimport cython
cimport numpy as cnp
cdef extern from "dpy_math.h" nogil:
    cdef double NPY_PI
    double sqrt(double)
    double log(double)


class FASTImageSegmenter(object):
    """ Image segmentation with constant models

    Image segmentation by maximum a-posteriori estimation of HMRF model
    with the Potts/Ising potential
    """
    def __init__(self):
        """ Initializes an instance of the FASTImageSegmenter class
        """
        pass

    def segment(self, double[:,:,:] image, int num_classes, double beta,
                int max_iter):
        """ Segments `image` into `num_class` classes

        Parameters
        ----------
        image : array, shape(X, Y, Z)
            the image to be segmented
        num_classes : int (positive)
            number of classes of the expected segmentation
        beta : float (positive)
            parameter of the Potts/Ising model
        max_iter : int (positive)
            maximum number of iterations
        """
        cdef:
            cnp.npy_intp nx = image.shape[0]
            cnp.npy_intp ny = image.shape[1]
            cnp.npy_intp nz = image.shape[2]
            int iter
            int[:,:,:] out = np.empty((nx, ny, nz), dtype=np.int32)
            double[:,:,:,:] neglogl = np.empty((nx, ny, nz, num_classes), dtype=np.float64)
            double[:] mu = np.empty((num_classes,), dtype=np.float64)
            double[:] sigmasq = np.empty((num_classes,), dtype=np.float64)
        # Initialize the means and variances (TO-DO: initialize with median otsu)
        print("Initializing parameters")
        _initialize_constant_models_uniform(image, mu, sigmasq)

        # Compute the negative log-likelihood for all classes at all voxels
        print("Computing neg-log-likelihood")
        _compute_neg_log_likelihood_gaussian(image, mu, sigmasq, neglogl)

        # Initialize segmentation with the maximum likelihood class at each voxel
        print("Initializing segmentation")
        _initialize_maximum_likelihood(neglogl, out)

        # Iterate ICM
        for iter in range(max_iter):
            print("ICM. Iter: %d"%(iter,))
            _iterate_icm_issing(neglogl, beta, out)
        return out


def initialize_constant_models_uniform(image, num_classes=3):
    """ Initializes the means and variances uniformly

    The means are initialized uniformly along the dynamic range of `image`.
    The variances are set to 1 for all classes

    Parameters
    ----------
    image : array, shape(X, Y, Z)
    num_classes : int

    Returns
    -------
    mu : array, shape(K,)
    sigmasq : array, shape(K,)
    """

    cdef:
        double[:] mu = np.empty((num_classes,), dtype=np.float64)
        double[:] sigmasq = np.empty((num_classes,), dtype=np.float64)

    _initialize_constant_models_uniform(image, mu, sigmasq)
    return np.array(mu), np.array(sigmasq)


cdef void _initialize_constant_models_uniform(double[:,:,:] image, double[:] mu, double[:] sigma) nogil:
    """ Initializes the means and variances uniformly

    The means are initialized uniformly along the dynamic range of `image`.
    The variances are set to 1 for all classes

    Parameters
    ----------
    image : array, shape(X, Y, Z)
    mu : array, shape(K,)
    sigmasq : array, shape(K,)
    """
    cdef:
        cnp.npy_intp nx = image.shape[0]
        cnp.npy_intp ny = image.shape[1]
        cnp.npy_intp nz = image.shape[2]
        int nclasses = mu.shape[0]
        int i
        double min_val
        double max_val
    min_val = image[0,0,0]
    max_val = image[0,0,0]
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if image[x,y,z] < min_val:
                    min_val = image[x,y,z]
                if image[x,y,z] > max_val:
                    max_val = image[x,y,z]
    for i in range(nclasses):
        sigma[i] = 1.0
        mu[i] = min_val + i * (max_val - min_val)/nclasses


def neg_log_likelihood_gaussian(image, mu, sigmasq):
    """ Computes the Gaussian negative log-likelihood of each class

    Computes the Gaussian negative log-likelihood of each class for each
    voxel of `image` assuming a Gaussian distribution with means and
    variances given by `mu` and `sigmasq`, respectively (constant models
    along the full volume). The negative log-likelihood will be written
    in `out`.

    Parameters
    ----------
    image : array, shape(X, Y, Z)
    mu : array, shape (K,)
    sigmasq : array, shape (K,)

    Returns
    -------
    out : array, shape (X, Y, Z, K)
    """

    out = np.empty(image.shape, dtype=np.int32)
    _compute_neg_log_likelihood_gaussian(image, mu, sigmasq, out)

    return out


cdef void _compute_neg_log_likelihood_gaussian(double[:,:,:] image,
                                               double[:] mu,
                                               double[:] sigmasq,
                                               double[:,:,:,:] out) nogil:
    """ Computes the Gaussian negative log-likelihood of each class

    Computes the Gaussian negative log-likelihood of each class for each
    voxel of `image` assuming a Gaussian distribution with means and
    variances given by `mu` and `sigmasq`, respectively (constant models
    along the full volume). The negative log-likelihood will be written
    in `out`.

    Parameters
    ----------
    image : array, shape(X, Y, Z)
    mu : array, shape (K,)
    sigmasq : array, shape (K,)
    out : array, shape (X, Y, Z, K)
    """
    cdef:
        cnp.npy_intp nx = image.shape[0]
        cnp.npy_intp ny = image.shape[1]
        cnp.npy_intp nz = image.shape[2]
        cnp.npy_intp nclasses = out.shape[3]
        cnp.npy_intp x, y, z, k
        double diff, norm_constant
    for k in range(nclasses):
        norm_constant = 0.5 * log(2.0 * NPY_PI * sigmasq[k])
        for x in range(nx):
            for y in range(ny):
                for z in range(nz):
                    diff = image[x,y,z] - mu[k]
                    out[x,y,z,k] = (diff * diff) / (2.0 * sigmasq[k]) + norm_constant


cdef void _iterate_icm_issing(double[:,:,:,:] neglogl, double beta, int[:,:,:] out) nogil:
    """ Executes one iteration of the ICM algorithm for MRF MAP estimation

    The prior distribution of the MRF is a Gibbs distribution with the
    Potts/Ising model with parameter `beta`:

    https://en.wikipedia.org/wiki/Potts_model

    Parameters
    ----------
    neglogl : array, shape(X, Y, Z, K)
        neglogl[x,y,z,k] is the negative log likelihood of class k at voxel
        (x,y,z)
    beta : float (positive)
        the parameter of the Potts/Ising model
    out : array
        initial segmentation. On output this segmentation will change by one
        iteration of the ICM algorithm
    """
    cdef:
        cnp.npy_intp nneigh = 6
        cnp.npy_intp* dX = [-1, 0, 0, 0,  0, 1]
        cnp.npy_intp* dY = [0, -1, 0, 1,  0, 0]
        cnp.npy_intp* dZ = [0,  0, 1, 0, -1, 0]
        cnp.npy_intp nx = neglogl.shape[0]
        cnp.npy_intp ny = neglogl.shape[1]
        cnp.npy_intp nz = neglogl.shape[2]
        cnp.npy_intp nclasses = neglogl.shape[3]
        cnp.npy_intp x, y, z, xx, yy, zz, i, j, k
        double min_energy, this_energy
        int best_class

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # Select the best label for this voxel (x, y, z)
                best_class = -1
                for k in range(nclasses):
                    this_energy = neglogl[x, y, z, k]
                    # Accumulate Gibbs energy for label k
                    for i in range(nneigh):
                        xx = x + dX[i]
                        if((xx < 0) or (xx >= nx)):
                            continue
                        yy = y + dY[i]
                        if((yy < 0) or (yy >= ny)):
                            continue
                        zz = z + dZ[i]
                        if((zz < 0) or (zz >= nz)):
                            continue

                        if out[xx,yy,zz] == k:
                            this_energy -= beta
                        else:
                            this_energy += beta
                    if (best_class == -1) or (this_energy < min_energy):
                        min_energy = this_energy
                        best_class = k
                out[x,y,z] = best_class





cdef void _initialize_maximum_likelihood(double[:,:,:,:] neglogl, int[:,:,:] out) nogil:
    """ Initializes the segmentation of an image with given neg-log-likelihood

    Initializes the segmentation of an image with neg-log-likelihood field
    given by `neglogl`. The class of each voxel is selected as the one with
    the minimum neg-log-likelihood (i.e. the maximum-likelihood segmentation).
    Parameters
    ----------
    neglogl : array, shape(X, Y, Z, K)
        neglogl[x,y,z,k] is the likelihhood of class k for voxel (x,y,z)
    out : array, shape(X, Y, Z)
        the buffer in which to write the initial segmentation
    """
    cdef:
        cnp.npy_intp nx = neglogl.shape[0]
        cnp.npy_intp ny = neglogl.shape[1]
        cnp.npy_intp nz = neglogl.shape[2]
        cnp.npy_intp nclasses = neglogl.shape[3]
        double min_energy
        int best_class
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # Select the best label for this voxel (x, y, z)
                best_class = -1
                for k in range(nclasses):
                    if (best_class == -1) or (neglogl[x,y,z,k] < min_energy):
                        best_class = k
                        min_energy = neglogl[x,y,z,k]
                out[x,y,z] = best_class
