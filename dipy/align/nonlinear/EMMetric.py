r'''
Similarity metric based on the Expectation-Maximization algorithm to handle
multi-modal images. The transfer function is modeled as a set of hidden
random variables that are estimated at each iteration of the algorithm.
'''
import numpy as np
import scipy as sp
import EMFunctions as em
from SimilarityMetric import SimilarityMetric
import matplotlib.pyplot as plt
import RegistrationCommon as rcommon
import SSDMetric
class EMMetric(SimilarityMetric):
    r'''
    Similarity metric based on the Expectation-Maximization algorithm to handle
    multi-modal images. The transfer function is modeled as a set of hidden
    random variables that are estimated at each iteration of the algorithm.
    '''
    GAUSS_SEIDEL_STEP = 0
    DEMONS_STEP = 1
    SINGLECYCLE_ITER = 0
    VCYCLE_ITER = 1
    WCYCLE_ITER = 2
    def get_default_parameters(self):
        return {'lambda':1.0, 'max_inner_iter':5, 'scale':1,
                'max_step_length':0.25, 'sigma_diff':3.0, 'step_type':0,
                'q_levels':256,'use_double_gradient':True,
                'iteration_type':'v_cycle'}

    def __init__(self, dim, parameters):
        super(EMMetric, self).__init__(dim, parameters)
        self.step_type = self.parameters['step_type']
        self.q_levels = self.parameters['q_levels']
        self.use_double_gradient = self.parameters['use_double_gradient']
        self.fixed_image_mask = None
        self.moving_image_mask = None
        self.fixedq_means_field = None
        self.movingq_means_field = None
        self.movingq_levels = None
        self.fixedq_levels = None
        if self.parameters['iteration_type'] == 'single_cycle':
            self.iteration_type = EMMetric.SINGLECYCLE_ITER
        elif self.parameters['iteration_type'] == 'w_cycle':
            self.iteration_type = EMMetric.WCYCLE_ITER
        else:
            self.iteration_type = EMMetric.VCYCLE_ITER

    def __connect_functions(self):
        r'''
        Assigns the appropriate functions to be called for image quantization,
        statistics computation and multi-resolution iterations according to the
        dimension of the input images
        '''
        if self.dim == 2:
            self.quantize = em.quantize_positive_image
            self.compute_stats = em.compute_masked_image_class_stats
            if self.iteration_type == EMMetric.SINGLECYCLE_ITER:
                self.multi_resolution_iteration = SSDMetric.single_cycle_2d
            elif self.iteration_type == EMMetric.VCYCLE_ITER:
                self.multi_resolution_iteration = SSDMetric.v_cycle_2d
            else:
                self.multi_resolution_iteration = SSDMetric.w_cycle_2d
        else:
            self.quantize = em.quantize_positive_volume
            self.compute_stats = em.compute_masked_volume_class_stats
            if self.iteration_type == EMMetric.SINGLECYCLE_ITER:
                self.multi_resolution_iteration = SSDMetric.single_cycle_3d
            elif self.iteration_type == EMMetric.VCYCLE_ITER:
                self.multi_resolution_iteration = SSDMetric.v_cycle_3d
            else:
                self.multi_resolution_iteration = SSDMetric.w_cycle_3d
        if self.step_type == EMMetric.DEMONS_STEP:
            self.compute_step = self.compute_demons_step
        else:
            self.compute_step = self.compute_gauss_seidel_step

    def initialize_iteration(self):
        r'''
        Precomputes the transfer functions (hidden random variables) and
        variances of the estimators. Also precomputes the gradient of both
        input images. Note that once the images are transformed to the opposite
        modality, the gradient of the transformed images can be used with the
        gradient of the corresponding modality in the same fasion as
        diff-demons does for mono-modality images. If the flag
        self.use_double_gradient is True these garadients are averaged.
        '''
        self.__connect_functions()
        sampling_mask = self.fixed_image_mask*self.moving_image_mask
        self.sampling_mask = sampling_mask
        fixedq, self.fixedq_levels, hist = self.quantize(self.fixed_image,
                                                      self.q_levels)
        fixedq = np.array(fixedq, dtype = np.int32)
        self.fixedq_levels = np.array(self.fixedq_levels)
        fixedq_means, fixedq_variances = self.compute_stats(sampling_mask,
                                                       self.moving_image,
                                                       self.q_levels,
                                                       fixedq)
        fixedq_means[0] = 0
        fixedq_means = np.array(fixedq_means)
        fixedq_variances = np.array(fixedq_variances)
        self.fixedq_sigma_field = fixedq_variances[fixedq]
        self.fixedq_means_field = fixedq_means[fixedq]
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
        movingq, self.movingq_levels, hist = self.quantize(self.moving_image,
                                                        self.q_levels)
        movingq = np.array(movingq, dtype = np.int32)
        self.movingq_levels = np.array(self.movingq_levels)
        movingq_means, movingq_variances = self.compute_stats(
            sampling_mask, self.fixed_image, self.q_levels, movingq)
        movingq_means[0] = 0
        movingq_means = np.array(movingq_means)
        movingq_variances = np.array(movingq_variances)
        self.movingq_sigma_field = movingq_variances[movingq]
        self.movingq_means_field = movingq_means[movingq]
        if self.use_double_gradient:
            i = 0
            for grad in sp.gradient(self.fixedq_means_field):
                self.gradient_moving[..., i] += grad
                i += 1
            i = 0
            for grad in sp.gradient(self.movingq_means_field):
                self.gradient_fixed[..., i] += grad
                i += 1

    def free_iteration(self):
        r'''
        Frees the resources allocated during initialization
        '''
        del self.sampling_mask
        del self.fixedq_levels
        del self.movingq_levels
        del self.fixedq_sigma_field
        del self.fixedq_means_field
        del self.movingq_sigma_field
        del self.movingq_means_field
        del self.gradient_moving
        del self.gradient_fixed

    def compute_forward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the moving image towards the fixed image
        '''
        return self.compute_step(True)

    def compute_backward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the fixed image towards the moving image
        '''
        return self.compute_step(False)

    def compute_gauss_seidel_step(self, forward_step = True):
        r'''
        Minimizes the linearized energy function with respect to the
        regularized displacement field (this step does not require
        post-smoothing, as opposed to the demons step, which does not include
        regularization). To accelerate convergence we use the multi-grid
        Gauss-Seidel algorithm proposed by Bruhn and Weickert et al [1]
        [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion
            estimation: combining highest accuracy with real-time performance",
            10th IEEE International Conference on Computer Vision, 2005.
            ICCV 2005.
        '''
        max_inner_iter = self.parameters['max_inner_iter']
        max_step_length = self.parameters['max_step_length']
        if forward_step:
            shape = self.fixed_image.shape
        else:
            shape = self.moving_image.shape
        if forward_step:
            delta = self.fixedq_means_field - self.moving_image
            sigma_field = self.fixedq_sigma_field
        else:
            delta = self.movingq_means_field - self.fixed_image
            sigma_field = self.movingq_sigma_field
        gradient = self.gradient_moving if forward_step else self.gradient_fixed
        displacement = np.zeros(shape = (shape)+(self.dim,), dtype = np.float64)
        self.energy = self.multi_resolution_iteration(self.levels_below,
                                                      max_inner_iter, delta,
                                                      sigma_field,
                                                      gradient,
                                                      None,
                                                      self.parameters['lambda'],
                                                      displacement)
        max_norm = np.sqrt(np.sum(displacement**2, -1)).max()
        if max_norm > max_step_length:
            displacement *= max_step_length/max_norm
        return displacement

    def compute_demons_step(self, forward_step = True):
        r'''
        TO-DO: implement Demons step for EM metric
        '''
        return NotImplemented

    def get_energy(self):
        r'''
        TO-DO: implement energy computation for the EM metric
        '''
        return NotImplemented

    def use_original_fixed_image(self, original_fixed_image):
        r'''
        EMMetric computes the object mask by thresholding the original fixed
        image
        '''
        pass

    def use_original_moving_image(self, original_moving_image):
        r'''
        EMMetric computes the object mask by thresholding the original moving
        image
        '''
        pass

    def use_fixed_image_dynamics(self, original_fixed_image, transformation):
        r'''
        EMMetric takes advantage of the image dynamics by computing the
        current fixed image mask from the originalFixedImage mask (warped
        by nearest neighbor interpolation)
        '''
        self.fixed_image_mask = (original_fixed_image>0).astype(np.int32)
        if transformation == None:
            return
        self.fixed_image_mask = transformation.warp_forward_nn(self.fixed_image_mask)

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r'''
        EMMetric takes advantage of the image dynamics by computing the
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
            shape_fixed = self.fixedq_means_field.shape
            rcommon.overlayImages(moving[:, shape_fixed[1]//2, :],
                                  fixed[:, shape_fixed[1]//2, :])
            rcommon.overlayImages(moving[shape_fixed[0]//2, :, :],
                                  fixed[shape_fixed[0]//2, :, :])
            rcommon.overlayImages(moving[:, :, shape_fixed[2]//2],
                                  fixed[:, :, shape_fixed[2]//2])

    def get_metric_name(self):
        return "EMMetric"
