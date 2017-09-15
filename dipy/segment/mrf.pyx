#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
from dipy.segment.mask import applymask
from dipy.core.ndindex import ndindex
from dipy.sims.voxel import add_noise
cimport cython
cimport numpy as cnp
cdef extern from "dpy_math.h" nogil:
    cdef double NPY_PI
    cdef double NPY_INFINITY
    double sqrt(double)
    double log(double)
    double exp(double)
    double fabs(double)


class ConstantObservationModel(object):
    r"""
    Observation model assuming that the intensity of each class is constant.
    The model parameters are the means $\mu_{k}$ and variances $\sigma_{k}$
    associated with each tissue class. According to this model, the observed
    intensity at voxel $x$ is given by $I(x) = \mu_{k} + \eta_{k}$ where $k$
    is the tissue class of voxel $x$, and $\eta_{k}$ is a Gaussian random
    variable with zero mean and variance $\sigma_{k}^{2}$. The observation
    model is responsible for computing the negative log-likelihood of
    observing any given intensity $z$ at each voxel $x$ assuming the voxel
    belongs to each class $k$. It also provides a default parameter
    initialization.
    """
    def __init__(self):
        r""" Initializes an instance of the ConstantObservationModel class
        """
        pass


    def initialize_param_uniform(self, image, nclasses):
        r""" Initializes the means and variances uniformly

        The means are initialized uniformly along the dynamic range of
        `image`. The variances are set to 1 for all classes

        Parameters
        ----------
        image : array,
                3D structural image
        nclasses : int,
                number of desired classes

        Returns
        -------
        mu : array,
                1 x nclasses, mean for each class
        sigma : array,
                1 x nclasses, standard deviation for each class.
                Set up to 1.0 for all classes.
        """
        cdef:
            double[:] mu = np.empty((nclasses,), dtype=np.float64)
            double[:] sigma = np.empty((nclasses,), dtype=np.float64)

        _initialize_param_uniform(image, mu, sigma)

        return np.array(mu), np.array(sigma)


    def seg_stats(self, input_image, seg_image, nclass):
        r""" Mean and standard variation for N desired  tissue classes

        Parameters
        ----------
        input_image : ndarray,
                 3D structural image
        seg_image : ndarray,
                 3D segmented image
        nclass : int,
                 number of classes (3 in most cases)

        Returns
        -------
        mu, std: ndarrays,
                 1 x nclasses dimension
                 Mean and standard deviation for each class
        """
        mu = np.zeros(nclass)
        std = np.zeros(nclass)
        num_vox = np.zeros(nclass)

        for index in ndindex(np.shape(input_image)):

            s = seg_image[index]
            v = input_image[index]

            for i in range(0, nclass):

                if s == i:
                    mu[i] += v
                    std[i] += v * v
                    num_vox[i] += 1

        mu = mu / num_vox
        std = np.sqrt(std/num_vox - mu**2)

        return mu, std


    def negloglikelihood(self, image, mu, sigmasq, nclasses):
        r""" Computes the gaussian negative log-likelihood of each class at
        each voxel of `image` assuming a gaussian distribution with means and
        variances given by `mu` and `sigmasq`, respectively (constant models
        along the full volume). The negative log-likelihood will be written
        in `nloglike`.

        Parameters
        ----------
        image : ndarray,
                3D gray scale structural image
        mu : ndarray,
                mean of each class
        sigmasq : ndarray,
                variance of each class
        nclasses : int
                number of classes

        Returns
        -------
        nloglike : ndarray,
                4D negloglikelihood for each class in each volume
        """
        nloglike = np.zeros(image.shape + (nclasses,), dtype=np.float64)

        for l in range(nclasses):
            _negloglikelihood(image, mu, sigmasq, l, nloglike)

        return nloglike


    def prob_image(self, img, nclasses, mu, sigmasq, P_L_N):
        r""" Conditional probability of the label given the image

        Parameters
        -----------
        img : ndarray,
            3D structural gray-scale image
        nclasses : int,
            number of tissue classes
        mu : ndarray,
            1 x nclasses, current estimate of the mean of each tissue class
        sigmasq : ndarray,
            1 x nclasses, current estimate of the variance of each
            tissue class
        P_L_N : ndarray,
            4D probability map of the label given the neighborhood.

        Previously computed by function prob_neighborhood

        Returns
        --------
        P_L_Y : ndarray,
            4D probability of the label given the input image
        """
        P_L_Y = np.zeros_like(P_L_N)
        P_L_Y_norm = np.zeros_like(img)

        for l in range(nclasses):

            g = np.zeros_like(img)

            _prob_image(img, g, mu, sigmasq, l, P_L_N, P_L_Y)
            P_L_Y_norm[:, :, :] += P_L_Y[:, :, :, l]

        for l in range(nclasses):
            P_L_Y[:, :, :, l] = P_L_Y[:, :, :, l] / P_L_Y_norm

        return P_L_Y


    def update_param(self, image, P_L_Y, mu, nclasses):
        r""" Updates the means and the variances in each iteration for all
        the labels. This is for equations 25 and 26 of Zhang et. al.,
        IEEE Trans. Med. Imag, Vol. 20, No. 1, Jan 2001.

        Parameters
        -----------
        image : ndarray,
                3D structural gray-scale image
        P_L_Y : ndarray,
                4D probability map of the label given the input image
                computed by the expectation maximization (EM) algorithm
        mu : ndarray,
                1 x nclasses, current estimate of the mean of each tissue
                class.
        nclasses : int,
                number of tissue classes

        Returns
        --------
        mu_upd : ndarray,
                1 x nclasses, updated mean of each tissue class
        var_upd : ndarray,
                1 x nclasses, updated variance of each tissue class
        """
        mu_upd = np.zeros(nclasses, dtype=np.float64)
        var_upd = np.zeros(nclasses, dtype=np.float64)
        mu_num = np.zeros(image.shape + (nclasses,), dtype=np.float64)
        var_num = np.zeros(image.shape + (nclasses,), dtype=np.float64)

        for l in range(nclasses):
            mu_num[..., l] = P_L_Y[..., l] * image
            var_num[..., l] = P_L_Y[..., l] * ((image - mu[l]) ** 2)

            mu_upd[l] = np.sum(mu_num[..., l]) / np.sum(P_L_Y[..., l])
            var_upd[l] = np.sum(var_num[..., l]) / np.sum(P_L_Y[..., l])

        return mu_upd, var_upd


    def update_param_new(self, image, P_L_Y, mu, nclasses):
        r""" Updates the means and the variances in each iteration for all
        the labels. This is for equations 25 and 26 of the Zhang et al. paper

        Parameters
        -----------
        image : ndarray,
                3D structural gray-scale image
        P_L_Y : ndarray,
                4D probability map of the label given the input image
                computed by the expectation maximization (EM) algorithm
        mu : ndarray,
                1 x nclasses, current estimate of the mean of each tissue
                class.
        nclasses : int,
                number of tissue classes

        Returns
        --------

        mu_upd : ndarray,
                1 x nclasses, updated mean of each tissue class
        var_upd : ndarray,
                1 x nclasses, updated variance of each tissue class
        """
        mu_upd = np.zeros(nclasses, dtype=np.float64)
        var_upd = np.zeros(nclasses, dtype=np.float64)
        mu_num = np.zeros(image.shape + (nclasses,), dtype=np.float64)
        var_num = np.zeros(image.shape + (nclasses,), dtype=np.float64)

        for l in range(nclasses):
            mu_num[..., l] = P_L_Y[..., l] * image
            var_num[..., l] = mu_num[..., l] * image

            mu_upd[l] = np.sum(mu_num[..., l]) / np.sum(P_L_Y[..., l])
            var_upd[l] = (np.sum(var_num[..., l]) / np.sum(P_L_Y[..., l]) -
                          mu_upd[l] ** 2)

        return mu_upd, var_upd


cdef void _initialize_param_uniform(double[:,:,:] image, double[:] mu,
                                    double[:] sigma) nogil:
    r""" Initializes the means and standard deviations uniformly

    The means are initialized uniformly along the dynamic range of `image`.
    The standard deviations are set to 1 for all classes.

    Parameters
    ----------
    image : array,
            3D structural gray-scale image
    mu : buffer array for the mean of each tissue class
    sigma : buffer array for the variance of each tissue class

    Returns
    -------
    mu : array,
        1 x nclasses, mean of each class
    sigma : array,
        1 x nclasses, standard deviation of each class
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


cdef void _negloglikelihood(double[:, :, :] image, double[:] mu,
                            double[:] sigmasq, int classid,
                            double[:, :, :, :] neglogl) nogil:
    r""" Computes the gaussian negative log-likelihood of each class at
    each voxel of `image` assuming a gaussian distribution with means and
    variances given by `mu` and `sigmasq`, respectively (constant models
    along the full volume). The negative log-likelihood will be written
    in `neglogl`.

    Parameters
    ----------
    image : array,
            3D structural gray-scale image
    mu : array,
            mean of each class
    sigmasq : array,
            variance of each class
    classid : int,
            class identifier
    neglogl : buffer for the neg-loglikelihood

    Returns
    -------
    neglogl : array,
            neg-loglikelihood for the class (l = classid)
    """
    cdef:
        cnp.npy_intp nx = image.shape[0]
        cnp.npy_intp ny = image.shape[1]
        cnp.npy_intp nz = image.shape[2]
        cnp.npy_intp l = classid
        cnp.npy_intp x, y, z
        double eps = 1e-8      # We assume images normalized to 0-1
        double eps_sq = 1e-16  # Maximum precision for double.

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):

                if sigmasq[l] < eps_sq:

                    if fabs(image[x, y, z] - mu[l]) < eps:
                        neglogl[x, y, z, l] = 1 + log(sqrt(2.0 * NPY_PI *
                                                           sigmasq[l]))
                    else:
                        neglogl[x, y, z, l] = NPY_INFINITY

                else:
                    neglogl[x, y, z, l] = (((image[x, y, z] - mu[l])**2.0) /
                                           (2.0 * sigmasq[l]))
                    neglogl[x, y, z, l] += log(sqrt(2.0 * NPY_PI *
                                                    sigmasq[l]))


cdef void _prob_image(double[:, :, :] image, double[:, :, :] gaussian,
                      double[:] mu, double[:] sigmasq, int classid,
                      double[:, :, :, :] P_L_N,
                      double[:, :, :, :] P_L_Y) nogil:
    r""" Conditional probability of the label given the image

    Parameters
    -----------
    image : array,
            3D structural gray-scale image
    gaussian : array
            3D buffer for the gaussian distribution that is multiplied by
            P_L_N to make P_L_Y
    mu : array,
            current estimate of the mean of each tissue class
    sigmasq : array,
            current estimate of the variance of each tissue
            class
    classid : int,
            tissue class identifier
    P_L_N : array,
            4D probability map of the label given the neighborhood.
            Previously computed by function prob_neighborhood
    P_L_Y : array
            4D buffer to hold P(L|Y)

    Returns
    --------
    P_L_Y : array,
            4D probability of the label given the input image P(L|Y)
    """
    cdef:
        cnp.npy_intp nx = image.shape[0]
        cnp.npy_intp ny = image.shape[1]
        cnp.npy_intp nz = image.shape[2]
        cnp.npy_intp l = classid
        cnp.npy_intp x, y, z

        double eps = 1e-8
        double eps_sq = 1e-16

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):

                if sigmasq[l] < eps_sq:
                    if fabs(image[x, y, z] - mu[l]) < eps:
                        gaussian[x, y, z] = 1
                    else:
                        gaussian[x, y, z] = 0
                else:
                    gaussian[x, y, z] = (
                        (exp(-((image[x, y, z] - mu[l]) ** 2) /
                        (2 * sigmasq[l]))) / (sqrt(2 * NPY_PI * sigmasq[l])))

                P_L_Y[x, y, z, l] = gaussian[x, y, z] * P_L_N[x, y, z, l]


class IteratedConditionalModes(object):
    def __init__(self):
        pass

    def initialize_maximum_likelihood(self, nloglike):
        r""" Initializes the segmentation of an image with given
            neg-loglikelihood

        Initializes the segmentation of an image with neglog-likelihood field
        given by `nloglike`. The class of each voxel is selected as the one
        with the minimum neglog-likelihood (i.e. maximum-likelihood
        segmentation).

        Parameters
        ----------
        nloglike : ndarray,
                4D shape, nloglike[x,y,z,k] is the likelihhood of class k
                for voxel (x, y, z)

        Returns
        --------
        seg : ndarray,
                3D initial segmentation
        """
        seg = np.zeros(nloglike.shape[:3]).astype(np.int16)

        _initialize_maximum_likelihood(nloglike, seg)

        return seg


    def icm_ising(self, nloglike, beta, seg):
        r""" Executes one iteration of the ICM algorithm for MRF MAP
        estimation. The prior distribution of the MRF is a Gibbs
        distribution with the Potts/Ising model with parameter `beta`:

        https://en.wikipedia.org/wiki/Potts_model

        Parameters
        ----------
        nloglike : ndarray,
                4D shape, nloglike[x,y,z,k] is the negative log likelihood
                of class k at voxel (x,y,z)
        beta : float,
                positive scalar, it is the parameter of the Potts/Ising
                model. Determines the smoothness of the output segmentation.
        seg : ndarray,
                3D initial segmentation. This segmentation will change by one
                iteration of the ICM algorithm

        Returns
        -------
        new_seg : ndarray,
                3D final segmentation
        energy : ndarray,
                3D final energy
        """
        energy = np.zeros(nloglike.shape[:3]).astype(np.float64)

        new_seg = np.zeros_like(seg)

        _icm_ising(nloglike, beta, seg, energy, new_seg)

        return new_seg, energy


    def prob_neighborhood(self, seg, beta, nclasses):
        r""" Conditional probability of the label given the neighborhood
        Equation 2.18 of the Stan Z. Li book (Stan Z. Li, Markov Random Field
        Modeling in Image Analysis, 3rd ed., Advances in Pattern Recognition
        Series, Springer Verlag 2009.)

        Parameters
        -----------
        seg : ndarray,
            3D tissue segmentation derived from the ICM model
        beta : float,
            scalar that determines the importance of the neighborhood and
            the spatial smoothness of the segmentation.
            Usually between 0 to 0.5
        nclasses : int,
            number of tissue classes

        Returns
        --------
        PLN : ndarray,
            4D probability map of the label given the neighborhood of the
            voxel.
        """
        cdef:
            double[:, :, :] P_L_N = np.zeros(seg.shape, dtype=np.float64)
            cnp.npy_intp classid = 0

        PLN_norm = np.zeros(seg.shape, dtype=np.float64)
        PLN = np.zeros(seg.shape + (nclasses,), dtype=np.float64)

        for classid in range(nclasses):

            P_L_N = np.zeros(seg.shape, dtype=np.float64)
            _prob_class_given_neighb(seg, beta, classid, P_L_N)

            PLN[:, :, :, classid] = np.array(P_L_N)
            PLN[:, :, :, classid] = np.exp(- PLN[:, :, :, classid])
            PLN_norm += PLN[:, :, :, classid]

        for l in range(nclasses):
            PLN[:, :, :, l] = PLN[:, :, :, l] / PLN_norm

        return PLN


cdef void _initialize_maximum_likelihood(double[:,:,:,:] nloglike,
                                         cnp.npy_short[:,:,:] seg) nogil:
    r""" Initializes the segmentation of an image with given
    neg-log-likelihood.

    Initializes the segmentation of an image with neg-log-likelihood field
    given by `nloglike`. The class of each voxel is selected as the one with
    the minimum neg-log-likelihood (i.e. the maximum-likelihood
    segmentation).

    Parameters
    ----------
    nloglike : array
        4D nloglike[x,y,z,k] is the likelihhood of class k for voxel (x,y,z)
    seg : array
        3D buffer for the initial segmentation

    Returns :
    seg : array,
        3D initial segmentation
    """
    cdef:
        cnp.npy_intp nx = nloglike.shape[0]
        cnp.npy_intp ny = nloglike.shape[1]
        cnp.npy_intp nz = nloglike.shape[2]
        cnp.npy_intp nclasses = nloglike.shape[3]
        double min_energy
        cnp.npy_short best_class

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):

                best_class = -1
                for k in range(nclasses):
                    if (best_class == -1) or (nloglike[x, y, z, k] <
                                              min_energy):
                        best_class = k
                        min_energy = nloglike[x, y, z, k]
                seg[x, y, z] = best_class


cdef void _icm_ising(double[:,:,:,:] nloglike, double beta,
                     cnp.npy_short[:,:,:] seg, double[:,:,:] energy,
                     cnp.npy_short[:,:,:] new_seg) nogil:
    r""" Executes one iteration of the ICM algorithm for MRF MAP estimation
    The prior distribution of the MRF is a Gibbs distribution with the
    Potts/Ising model with parameter `beta`:

    https://en.wikipedia.org/wiki/Potts_model

    Parameters
    ----------
    nloglike : array,
            4D nloglike[x,y,z,k] is the negative log likelihood of class k
            at voxel (x,y,z)
    beta : float,
            positive scalar, it is the parameter of the Potts/Ising model.
            Determines the smoothness of the output segmentation
    seg : array,
            3D initial segmentation.
            This segmentation will change by one iteration of the ICM algorithm
    energy : array,
            3D buffer for the energy
    new_seg : array,
            3D buffer for the final segmentation

    Returns
    -------
    energy : array,
            3D map of the energy for every voxel
    new_seg : array,
            3D new final segmentation (there is a new one after each
            iteration).
    """
    cdef:
        cnp.npy_intp nneigh = 6
        cnp.npy_intp* dX = [-1, 0, 0, 0,  0, 1]
        cnp.npy_intp* dY = [0, -1, 0, 1,  0, 0]
        cnp.npy_intp* dZ = [0,  0, 1, 0, -1, 0]
        cnp.npy_intp nx = nloglike.shape[0]
        cnp.npy_intp ny = nloglike.shape[1]
        cnp.npy_intp nz = nloglike.shape[2]
        cnp.npy_intp nclasses = nloglike.shape[3]
        cnp.npy_intp x, y, z, xx, yy, zz, i, j, k
        double min_energy = NPY_INFINITY
        double this_energy = NPY_INFINITY
        cnp.npy_short best_class

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):

                best_class = -1
                min_energy = NPY_INFINITY

                for k in range(nclasses):
                    this_energy = nloglike[x, y, z, k]

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

                        if seg[xx, yy, zz] == k:
                            this_energy -= beta
                        else:
                            this_energy += beta

                    if this_energy < min_energy:

                        min_energy = this_energy
                        best_class = k

                new_seg[x, y, z] = best_class
                energy[x, y, z] = min_energy


cdef void _prob_class_given_neighb(cnp.npy_short[:, :, :] seg, double beta,
                                   int classid, double[:, :, :] P_L_N) nogil:
    r""" Conditional probability of the label given the neighborhood
    Equation 2.18 of the Stan Z. Li book.

    Parameters
    -----------
    image : array,
            3D structural gray-scale image
    seg : array,
            3D tissue segmentation derived from the ICM model
    beta : float,
            scalar that determines the importance of the neighborhood and the
            spatial smoothness of the segmentation. Usually between 0 to 0.5
    classid : int,
            tissue class identifier
    P_L_N : buffer array for P(L|N)

    Returns
    --------
    P_L_N : array,
            3D map of the probability of the label (l) given the neighborhood
            of the voxel P(L|N)
    """
    cdef:
        cnp.npy_intp nx = seg.shape[0]
        cnp.npy_intp ny = seg.shape[1]
        cnp.npy_intp nz = seg.shape[2]
        cnp.npy_intp nneigh = 6
        cnp.npy_intp l = classid
        cnp.npy_intp x, y, z, xx, yy, zz
        double vox_prob
        cnp.npy_intp* dX = [-1, 0, 0, 0,  0, 1]
        cnp.npy_intp* dY = [0, -1, 0, 1,  0, 0]
        cnp.npy_intp* dZ = [0,  0, 1, 0, -1, 0]

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):

                vox_prob = 0

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

                    if seg[xx, yy, zz] == l:
                        vox_prob -= beta
                    else:
                        vox_prob += beta

                P_L_N[x, y, z] = vox_prob
