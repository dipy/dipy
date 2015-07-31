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
    double sqrt(double)
    double log(double)
    double exp(double)


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
        mask = np.where(image > 0, 1, 0)

        #if not image is None:

        for idx in ndindex(image.shape):
#            if not mask[idx]:
#                continue
            for l in range(nclasses):
                if sigmasq[l] == 0:
                    nloglike[idx + (l,)] = 0
                else:
                    nloglike[idx + (l,)] = ((image[idx] - mu[l]) ** 2.0) / (2.0 * sigmasq[l])
#                    nloglike[idx + (l,)] += np.log(2.0 * np.pi * np.sqrt(sigmasq[l]))
                    nloglike[idx + (l,)] += np.log(np.sqrt(2.0 * np.pi * sigmasq[l]))
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
            #P_L_N[np.isnan(P_L_N)] = 0

        return PLN, PLN_norm


#    def prob_image(self, img, nclasses, mu, sigmasq, P_L_N):
#        r""" Conditional probability of the label given the image
#        This is for equation 27 of the Zhang paper
#
#        Parameters
#        -----------
#        img : ndarray 3D
#            masked T1 structural image
#        nclasses : int
#            number of tissue classes
#        mu : ndarray (1, 3)
#            current estimate of mean of each tissue type
#        sigmasq : ndarray (1, 3)
#            current estimate of the variance of each tissue type
#        P_L_N : ndarray 4D
#            probability of the label given the neighborhood. Previously
#            computed by function prob_neigh
#
#        Returns
#        --------
#        P_L_Y : ndarray 4D
#            Probability of the label given the input image
#
#        """
#        # probability of the tissue label (from the 3 classes) given the
#        # voxel
#        P_L_Y = np.zeros_like(P_L_N)
#        P_L_Y_norm = np.zeros_like(img)
#        # normal density equation 11 of the Zhang paper
#        g = np.zeros_like(img)
#        # mask = np.where(img > 0, 1, 0)
#
#        for l in range(nclasses):
#            for idx in ndindex(img.shape[:3]):
#                idxl = idx + (l,)
#                g[idx] = (np.exp(-((img[idx] - mu[l]) ** 2)) / (2 * sigmasq[l])) / (np.sqrt(2 * np.pi * sigmasq[l]))
#                # P_L_Y[idx[0], idx[1], idx[2], l] = g[idx] * P_L_N[idx[0], idx[1], idx[2], l]
#                P_L_Y[idxl] = g[idx] * P_L_N[idxl]
#
#            P_L_Y_norm[:, :, :] += P_L_Y[:, :, :, l]
#
#        for l in range(nclasses):
#            P_L_Y[:, :, :, l] = P_L_Y[:, :, :, l]/P_L_Y_norm
#
#        P_L_Y[np.isnan(P_L_Y)] = 0
#        # P_L_Y[P_L_Y < 0] = 0
#
#        return P_L_Y
#
#    def update_param(self, image, P_L_Y, mu, nclasses):
#        r""" Updates the means and the variances in each iteration for all the
#        labels. This is for equations 25 and 26 of the Zhang paper
#
#        Parameters
#        -----------
#        image : ndarray
#            Input T1 grey scale image
#
#        P_L_Y : ndarray
#            Probability of the label given the input image computed by the
#            Expectation Maximization algorithm.
#
#        Returns
#        --------
#        mu_upd : 1x3 ndarray - mean of each tissue class
#        var_upd : 1x3 ndarray - variance of each tissue class
#
#        """
#        # temporary mu and var files to compute the update
#        mu_upd = np.zeros(nclasses)
#        var_upd = np.zeros(nclasses)
#        mu_num = np.zeros(image.shape + (nclasses,))
#        var_num = np.zeros(image.shape + (nclasses,))
#        denm = np.zeros(image.shape + (nclasses,))
#
#        for l in range(nclasses):
#            for idx in ndindex(image.shape[:3]):
#                idxl = idx + (l,)
#                mu_num[idxl] = (P_L_Y[idxl] * image[idx])
#                var_num[idxl] = (P_L_Y[idxl] * (image[idx] - mu[l]) ** 2)
#                denm[idxl] = P_L_Y[idxl]
#
#            mu_upd[l] = np.sum(mu_num[:, :, :, l]) / np.sum(denm[:, :, :, l])
#            var_upd[l] = np.sum(var_num[:, :, :, l]) / np.sum(denm[:, :, :, l])
#
#            print('updated means and variances per class')
#            print('class: ', l)
#            print('updated_mu:', mu_upd[l])
#            print('updated_var:', var_upd[l])
#
#        return mu_upd, var_upd


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

    #with gil: print('Hey, this is the label :', l)
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

        Initializes the segmentation of an image with neg-log-likelihood field
        given by `nloglike`. The class of each voxel is selected as the one with
        the minimum neg-log-likelihood (i.e. the maximum-likelihood segmentation).

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

        _icm_ising(nloglike, beta, seg)

        return seg


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


cdef void _icm_ising(double[:,:,:,:] nloglike, double beta, double[:,:,:] seg) nogil:
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
                seg[x,y,z] = best_class


class ImageSegmenter(object):
    # To-do: generalize to quadratic measure fields measure fields
    def __init__(self):

        pass

    def segment_HMRF(self, image, nclasses, beta, max_iter):

        observation_model = ConstantObservationModel()
        posteriorMaximizer = IteratedConditionalModes()

        print("Initializing parameters")
        mu, sigmasq = observation_model.initialize_param_uniform(image, nclasses)

        print("Computing neg-log-likelihood")
        negll = observation_model.negloglikelihood(image, mu, sigmasq, nclasses)

        print("Initializing segmentation")
        seg_init = posteriorMaximizer.initialize_maximum_likelihood(negll)

        seg = np.empty_like(image)

        for iter in range(max_iter): # Here is where the EM parts come in
            print("Iter: %d"%(iter,))

            PLN = observation_model.prob_neighborhood(image, seg_init, beta, nclasses)
            PLY = observation_model.prob_image(image, nclasses, mu, sigmasq, PLN)
            mu_upd, sigmasq_upd = observation_model.update_param(image, PLY, mu, nclasses)
            negll = observation_model.negloglikelihood(image, mu_upd, sigmasq_upd, nclasses)
            seg = posteriorMaximizer.icm_ising(negll, beta, seg_init)

        return seg


#if __name__ == "__main__":
#
#    image = nib.load('my_image.nii.gz')
#    nclasses = 3
#
#    segmenter = ImageSegmenter(model)
#    segmented = segmenter.segment_HMRF(image, nclasses)
