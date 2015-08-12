#!python
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True
import numpy as np
from dipy.segment.mask import applymask
from dipy.core.ndindex import ndindex
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
    r""" Image segmentation with constant models
    Image segmentation by maximum a-posteriori estimation of HMRF model
    with the Potts/Ising potential
    """
    # To-do: generalize to measure fields
    # To-do: create abstract class "ObservationModel" and derive
    # Constant... and Splines... from it

    def __init__(self):
        r""" Initializes an instance of the ConstantObservationModel class
        """
        pass

        # self.nclasses
        # self.mu = np.zeros(nclasses)
        # self.sigmasq = np.zeros(nclasses)

    def initialize_param_uniform(self, image, nclasses):
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
            double[:] mu = np.empty((nclasses,), dtype=np.float64)
            double[:] sigmasq = np.empty((nclasses,), dtype=np.float64)

        _initialize_param_uniform(image, mu, sigmasq)
        return np.array(mu), np.array(sigmasq)

    def seg_stats(self, input_image, seg_image, nclass):
        r""" Mean and standard variation for 3 tissue classes

        1 is CSF
        2 is grey matter
        3 is white matter

        Parameters
        ----------
        input_image : ndarray of grey level T1 image
        seg_image : ndarray of initital segmentation, also an image
        nclass : float numeber of classes (three in most cases)

        Returns
        -------
        mu, std, var : ndarray of dimensions 1x3
            Mean, standard deviation and variance for every class

        """
        mu = np.zeros(nclass)
        std = np.zeros(nclass)
        var = np.zeros(nclass)

        for i in range(0, nclass):

            H = input_image[seg_image == i]

            mu[i] = np.mean(H, -1)
            std[i] = np.std(H, -1)
            var[i] = np.var(H, -1)

        return mu, std, var


    def negloglikelihood(self, image, mu, sigmasq, nclasses):
        r""" Computes the Gaussian negative log-likelihood of each class

        Computes the Gaussian negative log-likelihood of each class for each
        voxel of `image` assuming a Gaussian distribution with means and
        variances given by `mu` and `sigmasq`, respectively (constant models
        along the full volume). The negative log-likelihood will be written
        in `nloglike`.

        """

        nloglike = np.zeros(image.shape + (nclasses,), dtype=np.float64)

        for l in range(nclasses):
            _negloglikelihood(image, mu, sigmasq, l, nloglike)

        print('negloglike:', nloglike[50,50,1,0])
        print('negloglike:', nloglike[50,50,1,1])
        print('negloglike:', nloglike[50,50,1,2])
        print('negloglike:', nloglike[50,50,1,3])

        return nloglike


    def prob_neighborhood(self, image, seg, beta, nclasses):
        r""" Conditional probability of the label given the neighborhood
        Equation 2.18 of the Stan Z. Li book.

        Parameters
        -----------
        img : 3D ndarray - masked T1 structural image
        nclasses : int - number of tissue classes
        seg : 3D ndarray - tissue segmentation derived from the ICM
                                     model. Must be padded with zeros
        beta : float - value of th importance of the neighborhood

        Returns
        --------

        P_L_N : 4D ndarray - Probability of the label given the neighborhood of
                             the voxel
        """

        cdef:
            #double[:, :, :, :] P_L_N = np.zeros(image.shape + (nclasses,),
            #                                    dtype=np.float64)
            double[:, :, :] P_L_N = np.zeros(image.shape,
                                                dtype=np.float64)
            cnp.npy_intp classid = 0

        PLN_norm = np.zeros(image.shape, dtype=np.float64)
        PLN = np.zeros(image.shape + (nclasses,), dtype=np.float64)

        for classid in range(nclasses):
            _prob_neighb_perclass(image, seg, beta, classid, P_L_N)

            # Eq 2.18 of Stan Z. Li book
            #PLN[:, :, :, classid] = np.array(P_L_N[:, :, :, classid])
            PLN[:, :, :, classid] = np.array(P_L_N)
            PLN[:, :, :, classid] = np.exp(- PLN[:, :, :, classid])
            PLN_norm += PLN[:, :, :, classid]

        for l in range(nclasses):
            PLN[:, :, :, l] = PLN[:, :, :, l] / PLN_norm

        return PLN


    def prob_image(self, img, nclasses, mu, sigmasq, P_L_N):
        r""" Conditional probability of the label given the image
        This is for equation 27 of the Zhang paper

        Parameters
        -----------
        img : ndarray 3D
            masked T1 structural image
        nclasses : int
            number of tissue classes
        mu : ndarray (1, 3)
            current estimate of mean of each tissue type
        sigmasq : ndarray (1, 3)
            current estimate of the variance of each tissue type
        P_L_N : ndarray 4D
            probability of the label given the neighborhood. Previously
            computed by function prob_neigh

        Returns
        --------
        P_L_Y : ndarray 4D
            Probability of the label given the input image

        """
        # probability of the tissue label (from the 3 classes) given the
        # voxel
        P_L_Y = np.zeros_like(P_L_N)
        P_L_Y_norm = np.zeros_like(img)
        # normal density equation 11 of the Zhang paper
        g = np.zeros_like(img)

        for l in range(nclasses):
            _prob_image(img, g, mu, sigmasq, l, P_L_N, P_L_Y)
            P_L_Y_norm[:, :, :] += P_L_Y[:, :, :, l]

        for l in range(nclasses):
            P_L_Y[:, :, :, l] = P_L_Y[:, :, :, l]/P_L_Y_norm

        return P_L_Y


    def update_param(self, image, P_L_Y, mu, nclasses):
        r""" Updates the means and the variances in each iteration for all the
        labels. This is for equations 25 and 26 of the Zhang paper

        Parameters
        -----------
        image : ndarray
            Input T1 grey scale image

        P_L_Y : ndarray
            Probability of the label given the input image computed by the
            Expectation Maximization algorithm.

        Returns
        --------
        mu_upd : 1x3 ndarray - mean of each tissue class
        var_upd : 1x3 ndarray - variance of each tissue class

        """
        # temporary mu and var files to compute the update
        mu_upd = np.zeros(nclasses, dtype=np.float64)
        var_upd = np.zeros(nclasses, dtype=np.float64)
        mu_num = np.zeros(image.shape + (nclasses,), dtype=np.float64)
        var_num = np.zeros(image.shape + (nclasses,), dtype=np.float64)

        for l in range(nclasses):

            _update_param(image, mu, l, P_L_Y, mu_num, var_num)

            mu_upd[l] = np.sum(mu_num[:, :, :, l]) / np.sum(P_L_Y[:, :, :, l])
            var_upd[l] = np.sum(var_num[:, :, :, l]) / np.sum(P_L_Y[:, :, :, l])

            print('updated means and variances per class')
            print('class: ', l)
            print('updated_mu:', mu_upd[l])
            print('updated_var:', var_upd[l])

        return mu_upd, var_upd


cdef void _initialize_param_uniform(double[:,:,:] image, double[:] mu, double[:] sigma) nogil:
        """ Initializes the mu and sigmasqiances uniformly

        The mu are initialized uniformly along the dynamic range of `image`.
        The sigmasqiances are set to 1 for all classes

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


cdef void _negloglikelihood(double[:, :, :] image, double[:] mu,
                            double[:] sigmasq, int classid,
                            double[:, :, :, :] neglogl) nogil:

    cdef:
        cnp.npy_intp nx = image.shape[0]
        cnp.npy_intp ny = image.shape[1]
        cnp.npy_intp nz = image.shape[2]
        cnp.npy_intp l = classid
        cnp.npy_intp x, y, z

        double eps = 1e-8   # We assume images normalized to 0-1
        double eps_sq = 1e-16 # Maximum precision for double.

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if sigmasq[l] < eps_sq:
                    if fabs(image[x, y, z] - mu[l]) < eps:
                        neglogl[x, y, z, l] = 1 + log(sqrt(2.0 * NPY_PI * sigmasq[l]))
                        if neglogl[x, y, z, l] == - NPY_INFINITY:
                            neglogl[x, y, z, l] = -1.7976931348623157e+100 #308

                    else:
                        neglogl[x, y, z, l] = 1.7976931348623157e+100 #308

                else:
                    neglogl[x, y, z, l] = ((image[x, y, z] - mu[l]) ** 2.0) / (2.0 * sigmasq[l])
                    neglogl[x, y, z, l] += log(sqrt(2.0 * NPY_PI * sigmasq[l]))


cdef void _prob_neighb_perclass(double[:, :, :] image, double[:, :, :] seg,
                                double beta, int classid,
                                double[:, :, :] P_L_N) nogil:

    cdef:
        cnp.npy_intp nx = image.shape[0]
        cnp.npy_intp ny = image.shape[1]
        cnp.npy_intp nz = image.shape[2]
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

                #vox_prob = P_L_N[x, y, z, l]
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

                    if seg[xx,yy,zz] == l:
                        vox_prob -= beta
                    else:
                        vox_prob += beta

                #P_L_N[x, y, z, l] = vox_prob
                P_L_N[x, y, z] = vox_prob


cdef void _prob_image(double[:, :, :] image, double[:, :, :] gaussian,
                      double[:] mu, double[:] sigmasq, int classid,
                      double[:, :, :, :] P_L_N,
                      double[:, :, :, :] P_L_Y) nogil:

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
                    gaussian[x, y, z] = (exp(-((image[x, y, z] - mu[l]) ** 2) / (2 * sigmasq[l]))) / (sqrt(2 * NPY_PI * sigmasq[l]))

                P_L_Y[x, y, z, l] = gaussian[x, y, z] * P_L_N[x, y, z, l]


cdef void _update_param(double[:, :, :] image, double[:] mu, int classid,
                        double[:, :, :, :] P_L_Y,
                        double[:, :, :, :] mu_num,
                        double[:, :, :, :] var_num) nogil:

    cdef:
        cnp.npy_intp nx = image.shape[0]
        cnp.npy_intp ny = image.shape[1]
        cnp.npy_intp nz = image.shape[2]
        cnp.npy_intp l = classid
        cnp.npy_intp x, y, z


    for x in range(nx):
        for y in range(ny):
            for z in range(nz):

                mu_num[x, y, z, l] = (P_L_Y[x, y, z, l] * image[x, y, z])
                var_num[x, y , z, l] = (P_L_Y[x, y, z, l] * (image[x, y, z] - mu[l]) ** 2)


class IteratedConditionalModes(object):
    # To-do: generalize to measure fields
    # To-do: create abstract class "PosteriorOptimizer" and derive ICM from it
    # To-do: generalize to different potentials
    # To-do: generalize to different neighborhoods

    def __init__(self):
        pass

#    def initialize_otsu(image):
#        seg = np.zeros(...)
#        ...
#        return seg

    def initialize_maximum_likelihood(self, nloglike):
        r""" Initializes the segmentation of an image with given neg-log-likelihood

        Initializes the segmentation of an image with neglog-likelihood field
        given by `nloglike`. The class of each voxel is selected as the one with
        the minimum neglog-likelihood (i.e. the maximum-likelihood segmentation).

        Parameters
        ----------
        nloglike : array, shape(X, Y, Z, K)
            nloglike[x,y,z,k] is the likelihhood of class k for voxel (x,y,z)

        Returns
        --------
        seg : array, shape(X, Y, Z)
            the buffer in which to write the initial segmentation
        """

        seg = np.zeros(nloglike.shape[:3]).astype(np.float64)

        _initialize_maximum_likelihood(nloglike, seg)

        return seg

    def icm_ising(self, nloglike, beta, seg):
        """ Executes one iteration of the ICM algorithm for MRF MAP estimation
        The prior distribution of the MRF is a Gibbs distribution with the
        Potts/Ising model with parameter `beta`:

        https://en.wikipedia.org/wiki/Potts_model

        Parameters
        ----------
        nloglike : array, shape(X, Y, Z, K)
            nloglike[x,y,z,k] is the negative log likelihood of class k at voxel
            (x,y,z)
        beta : float (positive)
            the parameter of the Potts/Ising model
        seg : array, shape (X, Y, Z)
            initial segmentation. On segput this segmentation will change by one
            iteration of the ICM algorithm
        """

        energy = np.zeros(nloglike.shape[:3]).astype(np.float64)

        _icm_ising(nloglike, beta, seg, energy)

        return seg, energy


cdef void _initialize_maximum_likelihood(double[:,:,:,:] nloglike, double[:,:,:] seg) nogil:
    r""" Initializes the segmentation of an image with given neg-log-likelihood

    Initializes the segmentation of an image with neg-log-likelihood field
    given by `nloglike`. The class of each voxel is selected as the one with
    the minimum neg-log-likelihood (i.e. the maximum-likelihood segmentation).

    Parameters
    ----------
    nloglike : array, shape(X, Y, Z, K)
        nloglike[x,y,z,k] is the likelihhood of class k for voxel (x,y,z)
    seg : array, shape(X, Y, Z)
        the buffer in which to write the initial segmentation. This is the
        inital segmentation
    """
    cdef:
        cnp.npy_intp nx = nloglike.shape[0]
        cnp.npy_intp ny = nloglike.shape[1]
        cnp.npy_intp nz = nloglike.shape[2]
        cnp.npy_intp nclasses = nloglike.shape[3]
        double min_energy
        int best_class
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # Select the best label for this voxel (x, y, z)
                best_class = -1
                for k in range(nclasses):
                    if (best_class == -1) or (nloglike[x,y,z,k] < min_energy):
                        best_class = k
                        min_energy = nloglike[x,y,z,k]
                seg[x,y,z] = best_class


cdef void _icm_ising(double[:,:,:,:] nloglike, double beta, double[:,:,:] seg, double[:,:,:] energy) nogil:
    """ Executes one iteration of the ICM algorithm for MRF MAP estimation
    The prior distribution of the MRF is a Gibbs distribution with the
    Potts/Ising model with parameter `beta`:

    https://en.wikipedia.org/wiki/Potts_model

    Parameters
    ----------
    nloglike : array, shape(X, Y, Z, K)
        nloglike[x,y,z,k] is the negative log likelihood of class k at voxel
        (x,y,z)
    beta : float (positive)
        the parameter of the Potts/Ising model
    seg : array
        initial segmentation. This segmentation will change by one
        iteration of the ICM algorithm

    Returns
    -------
    seg :  3D array. Final segmentation.
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
        double min_energy, this_energy
        int best_class

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                # Select the best label for this voxel (x, y, z)

                best_class = -1
                for k in range(nclasses):
                    this_energy = nloglike[x, y, z, k]

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

                        if seg[xx,yy,zz] == k:
                            this_energy -= beta
                        else:
                            this_energy += beta

                    if (best_class == -1) or (this_energy < min_energy):

                        min_energy = this_energy
                        best_class = k

                seg[x, y, z] = best_class
                energy[x, y, z] = min_energy


class ImageSegmenter(object):
    # To-do: generalize to quadratic measure fields measure fields
    def __init__(self):

        pass

    def segment_hmrf(self, image, nclasses, beta, max_iter):

        observationModel = ConstantObservationModel()
        posteriorMaximizer = IteratedConditionalModes()

        if image.max() > 1:
            image = np.interp(image, [0, image.max()], [0.0, 1.0])

        print("Initializing parameters")
        mu, sigma = observationModel.initialize_param_uniform(image, nclasses)
        sigmasq = sigma ** 2
        print("Computing first negative log-likelihood")
        neglogl = observationModel.negloglikelihood(image, mu, sigmasq, nclasses)
        print("Initializing segmentation")
        initial_segmentation = posteriorMaximizer.initialize_maximum_likelihood(neglogl)
        print("Calculating parameters for initital segmentation")
        mu, sigma, sigmasq = observationModel.seg_stats(image, initial_segmentation, nclasses)

        final_seg = np.empty_like(image)
        seg_init = initial_segmentation.copy()

        for iter in range(max_iter):
            print("Iteration: %d"%(iter,))

            PLN = observationModel.prob_neighborhood(image, initial_segmentation, beta, nclasses)
            PLY = observationModel.prob_image(image, nclasses, mu, sigmasq, PLN)
            mu_upd, sigmasq_upd = observationModel.update_param(image, PLY, mu, nclasses)
            negll = observationModel.negloglikelihood(image, mu_upd, sigmasq_upd, nclasses)
            final_seg, energy = posteriorMaximizer.icm_ising(negll, beta, initial_segmentation)

            initial_segmentation = final_seg.copy()
            mu = mu_upd.copy()
            sigmasq = sigmasq_upd.copy()

        return seg_init, final_seg, PLY


#if __name__ == "__main__":
#
#    image = nib.load('my_image.nii.gz')
#    nclasses = 3
#
#    segmenter = ImageSegmenter(model)
#    segmented = segmenter.segment_HMRF(image, nclasses)
