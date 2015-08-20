import numpy as np
from dipy.sims.voxel import add_noise
from dipy.segment.mrf import (ConstantObservationModel,
                              IteratedConditionalModes)


class TissueClassifierHMRF(object):

    def __init__(self, save_history=False, verbose=True):
        
        self.save_history = save_history
        self.segmentations = []
        self.pves = []
        self.energies = []
        self.energies_sum = []
        self.verbose = verbose

        pass

    def classify(self, image, nclasses, beta, max_iter):
        
        com = ConstantObservationModel()
        icm = IteratedConditionalModes()

        if image.max() > 1:
            image = np.interp(image, [0, image.max()], [0.0, 1.0])
        
        mu, sigma = com.initialize_param_uniform(image, nclasses)
        sigmasq = sigma ** 2
        
        neglogl = com.negloglikelihood(image, mu, sigmasq, nclasses)
        seg_init = icm.initialize_maximum_likelihood(neglogl)
        
        mu, sigma, sigmasq = com.seg_stats(image, seg_init, nclasses)
    
        zero = np.zeros_like(image) + 0.001
        zero_noise = add_noise(zero, 10000, 1, noise_type='gaussian')
        image_gauss = np.where(image == 0, zero_noise, image)
    
        final_segmentation = np.empty_like(image)
        initial_segmentation = seg_init.copy()
        energies = []
            
        for i in range(max_iter):
            
            if self.verbose:
                print('>> Iteration: ' + str(i))

            PLN = com.prob_neighborhood(image_gauss, seg_init, 
                                        beta, nclasses)
            PVE = com.prob_image(image_gauss, nclasses, mu, sigmasq, PLN)    
            mu_upd, sigmasq_upd = com.update_param(image_gauss, PVE, mu, 
                                                   nclasses)
            
            negll = com.negloglikelihood(image_gauss,
                                         mu_upd, sigmasq_upd, nclasses)

            final_segmentation, energy = icm.icm_ising(negll, beta, seg_init)

            if self.save_history:
                self.segmentations.append(final_segmentation)
                self.pves.append(PVE)
                self.energies.append(energy)
                self.energies_sum.append(energy[energy > -np.inf].sum())
            
            seg_init = final_segmentation.copy()
            mu = mu_upd.copy()
            sigmasq = sigmasq_upd.copy()
        
        return initial_segmentation, final_segmentation, PVE

