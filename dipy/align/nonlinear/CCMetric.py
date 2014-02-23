r'''
Similarity metric based on the Normalized Cross Correlation used by ANTS.
'''
import numpy as np
import scipy as sp
import CrossCorrelationFunctions as ccf
from SimilarityMetric import SimilarityMetric
import matplotlib.pyplot as plt
import RegistrationCommon as rcommon
from scipy import ndimage
class CCMetric(SimilarityMetric):
    r'''
    Similarity metric based on the Expectation-Maximization algorithm to handle
    multi-modal images. The transfer function is modeled as a set of hidden
    random variables that are estimated at each iteration of the algorithm.
    '''
    def get_default_parameters(self):
        return {'max_step_length':0.25, 'sigma_diff':3.0, 'radius':4}

    def __init__(self, dim, parameters):
        super(CCMetric, self).__init__(dim, parameters)
        self.radius=self.parameters['radius']
        self.sigma_diff=self.parameters['sigma_diff']
        self.max_step_length=self.parameters['max_step_length']

    def initialize_iteration(self):
        r'''
        Precomputes the cross-correlation factors
        '''
        self.factors = ccf.precompute_cc_factors_3d(self.fixed_image,
                                                   self.moving_image,
                                                   self.radius)
        self.factors = np.array(self.factors)
        self.gradient_moving = np.empty(
            shape = (self.moving_image.shape)+(self.dim,), dtype = np.float64)
        i = 0
        for grad in sp.gradient(self.moving_image):
            self.gradient_moving[..., i] = grad
            i += 1
        self.gradient_fixed = np.empty(
            shape = (self.fixed_image.shape)+(self.dim,), dtype = np.float64)
        i = 0
        for grad in sp.gradient(self.fixed_image):
            self.gradient_fixed[..., i] = grad
            i += 1

    def free_iteration(self):
        r'''
        Frees the resources allocated during initialization
        '''
        del self.factors
        del self.gradient_moving
        del self.gradient_fixed

    def compute_forward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the moving image towards the fixed image
        '''
        displacement, self.energy=ccf.compute_cc_forward_step_3d(self.gradient_fixed,
                                      self.gradient_moving,
                                      self.factors)
        displacement=np.array(displacement)
        displacement[..., 0] = ndimage.filters.gaussian_filter(displacement[..., 0],
                                                               self.sigma_diff)
        displacement[..., 1] = ndimage.filters.gaussian_filter(displacement[..., 1],
                                                                self.sigma_diff)
        displacement[..., 2] = ndimage.filters.gaussian_filter(displacement[..., 2],
                                                                self.sigma_diff)
        max_norm = np.sqrt(np.sum(displacement**2, -1)).max()
        displacement *= self.max_step_length/max_norm
        return displacement

    def compute_backward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the fixed image towards the moving image
        '''
        displacement, energy=ccf.compute_cc_backward_step_3d(self.gradient_fixed,
                                      self.gradient_moving,
                                      self.factors)
        displacement=np.array(displacement)
        displacement[..., 0] = ndimage.filters.gaussian_filter(displacement[..., 0],
                                                               self.sigma_diff)
        displacement[..., 1] = ndimage.filters.gaussian_filter(displacement[..., 1],
                                                                self.sigma_diff)
        displacement[..., 2] = ndimage.filters.gaussian_filter(displacement[..., 2],
                                                                self.sigma_diff)
        max_norm = np.sqrt(np.sum(displacement**2, -1)).max()
        displacement *= self.max_step_length/max_norm
        return displacement


    def get_energy(self):
        r'''
        TO-DO: implement energy computation for the EM metric
        '''
        return NotImplemented

    def use_original_fixed_image(self, original_fixed_image):
        r'''
        CCMetric computes the object mask by thresholding the original fixed
        image
        '''
        pass

    def use_original_moving_image(self, original_moving_image):
        r'''
        CCMetric computes the object mask by thresholding the original moving
        image
        '''
        pass

    def use_fixed_image_dynamics(self, original_fixed_image, transformation):
        r'''
        CCMetric takes advantage of the image dynamics by computing the
        current fixed image mask from the originalFixedImage mask (warped
        by nearest neighbor interpolation)
        '''
        self.fixed_image_mask = (original_fixed_image>0).astype(np.int32)
        if transformation == None:
            return
        self.fixed_image_mask = transformation.warp_forward_nn(self.fixed_image_mask)

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r'''
        CCMetric takes advantage of the image dynamics by computing the
        current moving image mask from the originalMovingImage mask (warped
        by nearest neighbor interpolation)
        '''
        self.moving_image_mask = (original_moving_image>0).astype(np.int32)
        if transformation == None:
            return
        self.moving_image_mask = transformation.warp_forward_nn(self.moving_image_mask)

    def report_status(self):
        r'''
        Shows the overlaid input images
        '''
        if self.dim == 2:
            plt.figure()
            rcommon.overlayImages(self.movingq_means_field,
                                  self.fixedq_means_field, False)
        else:
            fixed = self.fixed_image
            moving = self.moving_image
            shape_fixed = fixed.shape
            rcommon.overlayImages(moving[:, shape_fixed[1]//2, :],
                                  fixed[:, shape_fixed[1]//2, :])
            rcommon.overlayImages(moving[shape_fixed[0]//2, :, :],
                                  fixed[shape_fixed[0]//2, :, :])
            rcommon.overlayImages(moving[:, :, shape_fixed[2]//2],
                                  fixed[:, :, shape_fixed[2]//2])

    def get_metric_name(self):
        return "CCMetric"
