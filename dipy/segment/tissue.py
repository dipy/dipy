import numpy as np
from dipy.sims.voxel import add_noise
from dipy.segment.mrf import (ConstantObservationModel,
                              IteratedConditionalModes)


class TissueClassifierHMRF:
    """
    This class contains the methods for tissue classification using the
    Markov Random Fields modeling approach.
    """

    def __init__(self, save_history=False, verbose=True):
        self.save_history = save_history
        self.segmentations = []
        self.pves = []
        self.energies = []
        self.energies_sum = []
        self.verbose = verbose

    def classify(self, image, nclasses, beta, tolerance=1e-05, max_iter=100):
        """
        This method uses the Maximum a posteriori - Markov Random Field
        approach for segmentation by using the Iterative Conditional Modes
        and Expectation Maximization to estimate the parameters.

        Parameters
        ----------
        image : ndarray,
            3D structural image.
        nclasses : int,
            Number of desired classes.
        beta : float,
            Smoothing parameter, the higher this number the smoother the
            output will be.
        tolerance: float, optional
            Value that defines the percentage of change tolerated to
            prevent the ICM loop to stop. Default is 1e-05.
            If you want tolerance check to be disabled put 'tolerance = 0'.
        max_iter : int, optional
            Fixed number of desired iterations. Default is 100.
            This parameter defines the maximum number of iterations the
            algorithm will perform. The loop may terminate early if the
            change in energy sum between iterations falls below the
            threshold defined by `tolerance`. However, if `tolerance` is
            explicitly set to 0, this early stopping mechanism is disabled,
            and the algorithm will run for the specified number of
            iterations unless another stopping criterion is met.

        Returns
        -------
        initial_segmentation : ndarray,
            3D segmented image with all tissue types specified in nclasses.
        final_segmentation : ndarray,
            3D final refined segmentation containing all tissue types.
        PVE : ndarray,
            3D probability map of each tissue type.
        """
        nclasses += 1  # One extra class for the background
        energy_sum = [1e-05]

        com = ConstantObservationModel()
        icm = IteratedConditionalModes()

        if image.max() > 1:
            image = np.interp(image, [0, image.max()], [0.0, 1.0])

        mu, sigmasq = com.initialize_param_uniform(image, nclasses)
        p = np.argsort(mu)
        mu = mu[p]
        sigmasq = sigmasq[p]

        neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
        seg_init = icm.initialize_maximum_likelihood(neglogl)

        mu, sigmasq = com.seg_stats(image, seg_init, nclasses)

        zero = np.zeros_like(image) + 0.001
        zero_noise = add_noise(zero, 10000, 1, noise_type='gaussian')
        image_gauss = np.where(image == 0, zero_noise, image)

        final_segmentation = np.empty_like(image)
        initial_segmentation = seg_init

        for i in range(max_iter):
            if self.verbose:
                print(f'>> Iteration: {i}')

            PLN = icm.prob_neighborhood(seg_init, beta, nclasses)
            PVE = com.prob_image(image_gauss, nclasses, mu, sigmasq, PLN)

            mu_upd, sigmasq_upd = com.update_param(image_gauss, PVE, mu,
                                                   nclasses)
            ind = np.argsort(mu_upd)
            mu_upd = mu_upd[ind]
            sigmasq_upd = sigmasq_upd[ind]

            negll = com.negloglikelihood(image_gauss, mu_upd, sigmasq_upd,
                                         nclasses)
            final_segmentation, energy = icm.icm_ising(negll, beta, seg_init)

            energy_sum.append(energy[energy > -np.inf].sum())

            if self.save_history:
                self.segmentations.append(final_segmentation)
                self.pves.append(PVE)
                self.energies.append(energy)
                self.energies_sum.append(energy_sum[-1])

            if tolerance > 0 and i > 5:
                e_sum = np.asarray(energy_sum)
                tol = tolerance * (np.amax(e_sum) - np.amin(e_sum))

                e_end = e_sum[-5:]
                test_dist = np.abs(np.amax(e_end) - np.amin(e_end))

                if test_dist < tol:
                    break

            seg_init = final_segmentation
            mu = mu_upd
            sigmasq = sigmasq_upd

        PVE = PVE[..., 1:]

        return initial_segmentation, final_segmentation, PVE
