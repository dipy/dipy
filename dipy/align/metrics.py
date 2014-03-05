import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import abc
from scipy import gradient, ndimage
import registration_common as rcommon
import vector_fields as vfu
import ssd
import cc
import em
from dipy.align import floating


class SimilarityMetric(object):
    """
    A similarity metric is in charge of keeping track of the numerical value
    of the similarity (or distance) between the two given images. It also
    computes the update field for the forward and inverse
    displacement fields to be used in a gradient-based optimization algorithm.
    Note that this metric does not depend on any transformation (affine or
    non-linear), so it assumes the static and moving images are already warped
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self, dim):
        self.dim = dim
        self.static_image = None
        self.moving_image = None
        self.levels_above = 0
        self.levels_below = 0
        self.symmetric = False

    def set_levels_below(self, levels):
        r"""
        Informs this metric the number of pyramid levels below the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly
        """
        self.levels_below = levels

    def set_levels_above(self, levels):
        r"""
        Informs this metric the number of pyramid levels above the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly
        """
        self.levels_above = levels

    def set_static_image(self, static_image):
        """
        Sets the static image.
        """
        self.static_image = static_image

    @abc.abstractmethod
    def get_metric_name(self):
        """
        Must return the name of the metric that specializes this generic metric
        """
        pass

    @abc.abstractmethod
    def use_static_image_dynamics(self,
                                 original_static_image,
                                 transformation):
        """
        This methods provides the metric a chance to compute any useful
        information from knowing how the current static image was generated
        (as the transformation of an original static image). This method is
        called by the optimizer just after it sets the static image.
        Transformation will be an instance of SymmetricDiffeomorficMap or None if
        the originalMovingImage equals self.moving_image.
        """

    @abc.abstractmethod
    def use_original_static_image(self, original_static_image):
        """
        This methods provides the metric a chance to compute any useful
        information from the original moving image (to be used along with the
        sequence of movingImages during optimization, for example the binary
        mask delimiting the object of interest can be computed from the original
        image only and then warp this binary mask instead of thresholding
        at each iteration, which might cause artifacts due to interpolation)
        """

    def set_moving_image(self, moving_image):
        """
        Sets the moving image.
        """
        self.moving_image = moving_image

    @abc.abstractmethod
    def use_original_moving_image(self, original_moving_image):
        """
        This methods provides the metric a chance to compute any useful
        information from the original moving image (to be used along with the
        sequence of movingImages during optimization, for example the binary
        mask delimiting the object of interest can be computed from the original
        image only and then warp this binary mask instead of thresholding
        at each iteration, which might cause artifacts due to interpolation)
        """

    @abc.abstractmethod
    def use_moving_image_dynamics(self,
                               original_moving_image,
                               transformation):
        """
        This methods provides the metric a chance to compute any useful
        information from knowing how the current static image was generated
        (as the transformation of an original static image). This method is
        called by the optimizer just after it sets the static image.
        Transformation will be an instance of SymmetricDiffeomorficMap or None if
        the originalMovingImage equals self.moving_image.
        """

    @abc.abstractmethod
    def initialize_iteration(self):
        """
        This method will be called before any computeUpdate or computeInverse
        call, this gives the chance to the Metric to precompute any useful
        information for speeding up the update computations. This initialization
        was needed in ANTS because the updates are called once per voxel. In
        Python this is unpractical, though.
        """

    @abc.abstractmethod
    def free_iteration(self):
        """
        This method is called by the RegistrationOptimizer after the required
        iterations have been computed (forward and/or backward) so that the
        SimilarityMetric can safely delete any data it computed as part of the
        initialization
        """

    @abc.abstractmethod
    def compute_forward(self):
        """
        Must return the forward update field for a gradient-based optimization
        algorithm
        """

    @abc.abstractmethod
    def compute_backward(self):
        """
        Must return the inverse update field for a gradient-based optimization
        algorithm
        """

    @abc.abstractmethod
    def get_energy(self):
        """
        Must return the numeric value of the similarity between the given static
        and moving images
        """

    @abc.abstractmethod
    def report_status(self):
        """
        This function is called mostly for debugging purposes. The metric
        can for example show the overlaid images or print some statistics
        """

class CCMetric(SimilarityMetric):
    r"""
    Similarity metric based on the Expectation-Maximization algorithm to handle
    multi-modal images. The transfer function is modeled as a set of hidden
    random variables that are estimated at each iteration of the algorithm.
    """

    def __init__(self, dim, step_length = 0.25, sigma_diff = 3.0, radius = 4):
        super(CCMetric, self).__init__(dim)
        self.step_length = step_length
        self.sigma_diff = sigma_diff
        self.radius = radius

    def initialize_iteration(self):
        r"""
        Precomputes the cross-correlation factors
        """
        self.factors = cc.precompute_cc_factors_3d(self.static_image,
                                                   self.moving_image,
                                                   self.radius)
        self.factors = np.array(self.factors)
        self.gradient_moving = np.empty(
            shape = (self.moving_image.shape)+(self.dim,), dtype = floating)
        i = 0
        for grad in sp.gradient(self.moving_image):
            self.gradient_moving[..., i] = grad
            i += 1
        self.gradient_static = np.empty(
            shape = (self.static_image.shape)+(self.dim,), dtype = floating)
        i = 0
        for grad in sp.gradient(self.static_image):
            self.gradient_static[..., i] = grad
            i += 1

    def free_iteration(self):
        r"""
        Frees the resources allocated during initialization
        """
        del self.factors
        del self.gradient_moving
        del self.gradient_static

    def compute_forward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the moving image towards the static image
        """
        displacement, self.energy=cc.compute_cc_forward_step_3d(self.gradient_static,
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
        displacement *= self.step_length/max_norm
        return displacement

    def compute_backward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        displacement, energy=cc.compute_cc_backward_step_3d(self.gradient_static,
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
        displacement *= self.step_length/max_norm
        return displacement


    def get_energy(self):
        r"""
        TO-DO: implement energy computation for the EM metric
        """
        return NotImplemented

    def use_original_static_image(self, original_static_image):
        r"""
        CCMetric computes the object mask by thresholding the original static
        image
        """
        pass

    def use_original_moving_image(self, original_moving_image):
        r"""
        CCMetric computes the object mask by thresholding the original moving
        image
        """
        pass

    def use_static_image_dynamics(self, original_static_image, transformation):
        r"""
        CCMetric takes advantage of the image dynamics by computing the
        current static image mask from the originalstaticImage mask (warped
        by nearest neighbor interpolation)
        """
        self.static_image_mask = (original_static_image>0).astype(np.int32)
        if transformation == None:
            return
        self.static_image_mask = transformation.transform(self.static_image_mask,'nn')

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""
        CCMetric takes advantage of the image dynamics by computing the
        current moving image mask from the originalMovingImage mask (warped
        by nearest neighbor interpolation)
        """
        self.moving_image_mask = (original_moving_image>0).astype(np.int32)
        if transformation == None:
            return
        self.moving_image_mask = transformation.transform(self.moving_image_mask, 'nn')

    def report_status(self):
        r"""
        Shows the overlaid input images
        """
        if self.dim == 2:
            plt.figure()
            rcommon.overlayImages(self.movingq_means_field,
                                  self.staticq_means_field, False)
        else:
            static = self.static_image
            moving = self.moving_image
            shape_static = static.shape
            rcommon.overlayImages(moving[:, shape_static[1]//2, :],
                                  static[:, shape_static[1]//2, :])
            rcommon.overlayImages(moving[shape_static[0]//2, :, :],
                                  static[shape_static[0]//2, :, :])
            rcommon.overlayImages(moving[:, :, shape_static[2]//2],
                                  static[:, :, shape_static[2]//2])

    def get_metric_name(self):
        return "CCMetric"


class EMMetric(SimilarityMetric):
    r"""
    Similarity metric based on the Expectation-Maximization algorithm to handle
    multi-modal images. The transfer function is modeled as a set of hidden
    random variables that are estimated at each iteration of the algorithm.
    """
    GAUSS_SEIDEL_STEP = 0
    DEMONS_STEP = 1
    SINGLECYCLE_ITER = 0
    VCYCLE_ITER = 1
    WCYCLE_ITER = 2

    def __init__(self, dim, smooth=1.0, inner_iter=5, step_length=0.25, q_levels=256, double_gradient=True, iter_type='v_cycle'):
        super(EMMetric, self).__init__(dim, parameters)
        self.smooth
        self.inner_iter = inner_iter
        self.step_length = step_length
        self.q_levels = q_levels
        self.use_double_gradient = double_gradient
        if iter_type == 'single_cycle':
            self.iteration_type = EMMetric.SINGLECYCLE_ITER
        elif iter_type == 'w_cycle':
            self.iteration_type = EMMetric.WCYCLE_ITER
        else:
            self.iteration_type = EMMetric.VCYCLE_ITER
        self.static_image_mask = None
        self.moving_image_mask = None
        self.staticq_means_field = None
        self.movingq_means_field = None
        self.movingq_levels = None
        self.staticq_levels = None

    def _connect_functions(self):
        r"""
        Assigns the appropriate functions to be called for image quantization,
        statistics computation and multi-resolution iterations according to the
        dimension of the input images
        """
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
        self.compute_step = self.compute_gauss_seidel_step
            

    def initialize_iteration(self):
        r"""
        Precomputes the transfer functions (hidden random variables) and
        variances of the estimators. Also precomputes the gradient of both
        input images. Note that once the images are transformed to the opposite
        modality, the gradient of the transformed images can be used with the
        gradient of the corresponding modality in the same fasion as
        diff-demons does for mono-modality images. If the flag
        self.use_double_gradient is True these garadients are averaged.
        """
        self._connect_functions()
        sampling_mask = self.static_image_mask*self.moving_image_mask
        self.sampling_mask = sampling_mask
        staticq, self.staticq_levels, hist = self.quantize(self.static_image,
                                                      self.q_levels)
        staticq = np.array(staticq, dtype = np.int32)
        self.staticq_levels = np.array(self.staticq_levels)
        staticq_means, staticq_variances = self.compute_stats(sampling_mask,
                                                       self.moving_image,
                                                       self.q_levels,
                                                       staticq)
        staticq_means[0] = 0
        staticq_means = np.array(staticq_means)
        staticq_variances = np.array(staticq_variances)
        self.staticq_sigma_field = staticq_variances[staticq]
        self.staticq_means_field = staticq_means[staticq]
        self.gradient_moving = np.empty(
            shape = (self.moving_image.shape)+(self.dim,), dtype = floating)
        i = 0
        for grad in sp.gradient(self.moving_image):
            self.gradient_moving[..., i] = grad
            i += 1
        self.gradient_static = np.empty(
            shape = (self.static_image.shape)+(self.dim,), dtype = floating)
        i = 0
        for grad in sp.gradient(self.static_image):
            self.gradient_static[..., i] = grad
            i += 1
        movingq, self.movingq_levels, hist = self.quantize(self.moving_image,
                                                        self.q_levels)
        movingq = np.array(movingq, dtype = np.int32)
        self.movingq_levels = np.array(self.movingq_levels)
        movingq_means, movingq_variances = self.compute_stats(
            sampling_mask, self.static_image, self.q_levels, movingq)
        movingq_means[0] = 0
        movingq_means = np.array(movingq_means)
        movingq_variances = np.array(movingq_variances)
        self.movingq_sigma_field = movingq_variances[movingq]
        self.movingq_means_field = movingq_means[movingq]
        if self.use_double_gradient:
            i = 0
            for grad in sp.gradient(self.staticq_means_field):
                self.gradient_moving[..., i] += grad
                i += 1
            i = 0
            for grad in sp.gradient(self.movingq_means_field):
                self.gradient_static[..., i] += grad
                i += 1

    def free_iteration(self):
        r"""
        Frees the resources allocated during initialization
        """
        del self.sampling_mask
        del self.staticq_levels
        del self.movingq_levels
        del self.staticq_sigma_field
        del self.staticq_means_field
        del self.movingq_sigma_field
        del self.movingq_means_field
        del self.gradient_moving
        del self.gradient_static

    def compute_forward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the moving image towards the static image
        """
        return self.compute_step(True)

    def compute_backward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        return self.compute_step(False)

    def compute_gauss_seidel_step(self, forward_step = True):
        r"""
        Minimizes the linearized energy function with respect to the
        regularized displacement field (this step does not require
        post-smoothing, as opposed to the demons step, which does not include
        regularization). To accelerate convergence we use the multi-grid
        Gauss-Seidel algorithm proposed by Bruhn and Weickert et al [1]
        [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion
            estimation: combining highest accuracy with real-time performance",
            10th IEEE International Conference on Computer Vision, 2005.
            ICCV 2005.
        """
        max_inner_iter = self.inner_iter
        max_step_length = self.step_length
        if forward_step:
            shape = self.static_image.shape
        else:
            shape = self.moving_image.shape
        if forward_step:
            delta = self.staticq_means_field - self.moving_image
            sigma_field = self.staticq_sigma_field
        else:
            delta = self.movingq_means_field - self.static_image
            sigma_field = self.movingq_sigma_field
        gradient = self.gradient_moving if forward_step else self.gradient_static
        displacement = np.zeros(shape = (shape)+(self.dim,), dtype = floating)
        self.energy = self.multi_resolution_iteration(self.levels_below,
                                                      max_inner_iter, delta,
                                                      sigma_field,
                                                      gradient,
                                                      None,
                                                      self.smooth,
                                                      displacement)
        max_norm = np.sqrt(np.sum(displacement**2, -1)).max()
        if max_norm > max_step_length:
            displacement *= max_step_length/max_norm
        return displacement

    def compute_demons_step(self, forward_step = True):
        r"""
        TO-DO: implement Demons step for EM metric
        """
        return NotImplemented

    def get_energy(self):
        r"""
        TO-DO: implement energy computation for the EM metric
        """
        return NotImplemented

    def use_original_static_image(self, original_static_image):
        r"""
        EMMetric computes the object mask by thresholding the original static
        image
        """
        pass

    def use_original_moving_image(self, original_moving_image):
        r"""
        EMMetric computes the object mask by thresholding the original moving
        image
        """
        pass

    def use_static_image_dynamics(self, original_static_image, transformation):
        r"""
        EMMetric takes advantage of the image dynamics by computing the
        current static image mask from the originalstaticImage mask (warped
        by nearest neighbor interpolation)
        """
        self.static_image_mask = (original_static_image>0).astype(np.int32)
        if transformation == None:
            return
        self.static_image_mask = transformation.transform(self.static_image_mask,'nn')

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""
        EMMetric takes advantage of the image dynamics by computing the
        current moving image mask from the originalMovingImage mask (warped
        by nearest neighbor interpolation)
        """
        self.moving_image_mask = (original_moving_image>0).astype(np.int32)
        if transformation == None:
            return
        self.moving_image_mask = transformation.transform(self.moving_image_mask,'nn')

    def report_status(self):
        r"""
        Shows the overlaid input images
        """
        if self.dim == 2:
            plt.figure()
            rcommon.overlayImages(self.movingq_means_field,
                                  self.staticq_means_field, False)
        else:
            static = self.static_image
            moving = self.moving_image
            shape_static = self.staticq_means_field.shape
            rcommon.overlayImages(moving[:, shape_static[1]//2, :],
                                  static[:, shape_static[1]//2, :])
            rcommon.overlayImages(moving[shape_static[0]//2, :, :],
                                  static[shape_static[0]//2, :, :])
            rcommon.overlayImages(moving[:, :, shape_static[2]//2],
                                  static[:, :, shape_static[2]//2])

    def get_metric_name(self):
        return "EMMetric"

class SSDMetric(SimilarityMetric):
    r"""
    Similarity metric for (monomodal) nonlinear image registration defined by
    the sum of squared differences (SSD).
    """
    GAUSS_SEIDEL_STEP = 0
    DEMONS_STEP = 1
    def get_default_parameters(self):
        return {'lambda':1.0, 'max_inner_iter':5, 'scale':1,
                'max_step_length':0.25, 'sigma_diff':3.0, 'step_type':0}

    def __init__(self, dim, smooth=3.0, inner_iter=5, step_length=0.25, step_type=0):
        super(SSDMetric, self).__init__(dim)
        self.smooth = smooth
        self.inner_iter = inner_iter
        self.step_length = step_length
        self.step_type = step_type
        self.levels_below = 0

    def initialize_iteration(self):
        r"""
        Precomputes the gradient of the input images to be used in the
        computation of the forward and backward steps.
        """
        self.gradient_moving = np.empty(
            shape = (self.moving_image.shape)+(self.dim,), dtype = floating)
        i = 0
        for grad in gradient(self.moving_image):
            self.gradient_moving[..., i] = grad
            i += 1
        i = 0
        self.gradient_static = np.empty(
            shape = (self.static_image.shape)+(self.dim,), dtype = floating)
        for grad in gradient(self.static_image):
            self.gradient_static[..., i] = grad
            i += 1

    def compute_forward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the moving image towards the static image
        """
        if self.step_type == SSDMetric.GAUSS_SEIDEL_STEP:
            return self.compute_gauss_seidel_step(True)
        elif self.step_type == SSDMetric.DEMONS_STEP:
            return self.compute_demons_step(True)
        return None

    def compute_backward(self):
        r"""
        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        if self.step_type == SSDMetric.GAUSS_SEIDEL_STEP:
            return self.compute_gauss_seidel_step(False)
        elif self.step_type == SSDMetric.DEMONS_STEP:
            return self.compute_demons_step(False)
        return None

    def compute_gauss_seidel_step(self, forward_step = True):
        r"""
        Minimizes the linearized energy function defined by the sum of squared
        differences of corresponding pixels of the input images with respect
        to the displacement field.
        """
        max_inner_iter = self.inner_iter
        lambda_param = self.smooth
        max_step_length = self.step_length
        if forward_step:
            shape = self.static_image.shape
        else:
            shape = self.moving_image.shape
        if forward_step:
            delta_field = self.static_image-self.moving_image
        else:
            delta_field = self.moving_image - self.static_image
        #gradient = self.gradient_moving+self.gradient_static
        gradient = self.gradient_moving
        displacement = np.zeros(shape = (shape)+(self.dim,), dtype = floating)
        if self.dim == 2:
            self.energy = v_cycle_2d(self.levels_below, max_inner_iter, 
                                    delta_field, None, gradient, None, 
                                    lambda_param, displacement)
        else:
            self.energy = v_cycle_3d(self.levels_below, max_inner_iter,
                                    delta_field, None, gradient, None, 
                                    lambda_param, displacement)
        max_norm = np.sqrt(np.sum(displacement**2, -1)).max()
        if max_norm > max_step_length:
            displacement *= max_step_length/max_norm
        return displacement

    def compute_demons_step(self, forward_step = True):
        r"""
        Computes the demons step proposed by Vercauteren et al.[1] for the SSD
        metric.
        [1] Tom Vercauteren, Xavier Pennec, Aymeric Perchant, Nicholas Ayache,
            "Diffeomorphic Demons: Efficient Non-parametric Image Registration",
            Neuroimage 2009
        """
        sigma_diff = self.smooth
        max_step_length = self.step_length
        scale = 1.0
        if forward_step:
            delta_field = self.static_image-self.moving_image
        else:
            delta_field = self.moving_image - self.static_image
        gradient = self.gradient_moving+self.gradient_static
        if self.dim == 2:
            forward = ssd.compute_demons_step2D(delta_field, gradient,
                                               max_step_length, scale)
            forward[..., 0] = ndimage.filters.gaussian_filter(forward[..., 0],
                                                                sigma_diff)
            forward[..., 1] = ndimage.filters.gaussian_filter(forward[..., 1],
                                                                sigma_diff)
        else:
            forward = ssd.compute_demons_step3D(delta_field, gradient,
                                               max_step_length, scale)
            forward[..., 0] = ndimage.filters.gaussian_filter(forward[..., 0],
                                                                sigma_diff)
            forward[..., 1] = ndimage.filters.gaussian_filter(forward[..., 1],
                                                                sigma_diff)
            forward[..., 2] = ndimage.filters.gaussian_filter(forward[..., 2],
                                                                sigma_diff)
        return forward

    def get_energy(self):
        return NotImplemented

    def use_original_static_image(self, originalstatic_image):
        r"""
        SSDMetric does not take advantage of the original static image, just pass
        """
        pass

    def use_original_moving_image(self, original_moving_image):
        r"""
        SSDMetric does not take advantage of the original moving image just pass
        """
        pass

    def use_static_image_dynamics(self, originalstatic_image, transformation):
        r"""
        SSDMetric does not take advantage of the image dynamics, just pass
        """
        pass

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""
        SSDMetric does not take advantage of the image dynamics, just pass
        """
        pass

    def report_status(self):
        plt.figure()
        rcommon.overlayImages(self.moving_image, self.static_image, False)

    def get_metric_name(self):
        return "SSDMetric"

    def free_iteration(self):
        pass


def v_cycle_2d(n, k, delta_field, sigma_field, gradient_field, target,
             lambda_param, displacement, depth = 0):
    r"""
    Multi-resolution Gauss-Seidel solver: solves the linear system by first
    filtering (GS-iterate) the current level, then solves for the residual
    at a coarcer resolution andfinally refines the solution at the current
    resolution. This scheme corresponds to the V-cycle proposed by Bruhn and
    Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
            combining highest accuracy with real-time performance",
            10th IEEE International Conference on Computer Vision, 2005.
            ICCV 2005.
    """
    #presmoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_SSD2D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
    if n == 0:
        try:
            energy
        except NameError:
            energy = ssd.compute_energy_SSD2D(delta_field, sigma_field, 
                                         gradient_field, lambda_param, 
                                         displacement)
        return energy
    #solve at coarcer grid
    residual = None
    residual = ssd.compute_residual_displacement_field_SSD2D(delta_field,
                                                            sigma_field,
                                                            gradient_field,
                                                            target,
                                                            lambda_param,
                                                            displacement,
                                                            residual)
    sub_residual = np.array(vfu.downsample_displacement_field2D(residual))
    del residual
    subsigma_field = None
    if sigma_field != None:
        subsigma_field = vfu.downsample_scalar_field2D(sigma_field)
    subdelta_field = vfu.downsample_scalar_field2D(delta_field)
    subgradient_field = np.array(
        vfu.downsample_displacement_field2D(gradient_field))
    shape = np.array(displacement.shape).astype(np.int32)
    #sub_displacement = np.array(vfu.downsample_displacement_field(displacement))
    sub_displacement = np.zeros(shape = ((shape[0]+1)//2, (shape[1]+1)//2, 2 ),
                               dtype = floating)
    sublambda_param = lambda_param*0.25
    v_cycle_2d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
             sub_residual, sublambda_param, sub_displacement, depth+1)
    displacement += np.array(
        vfu.upsample_displacement_field(sub_displacement, shape))
    #post-smoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_SSD2D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
    try:
        energy
    except NameError:
        energy = ssd.compute_energy_SSD2D(delta_field, sigma_field, 
                                         gradient_field, lambda_param, 
                                         displacement)
    return energy

def v_cycle_3d(n, k, delta_field, sigma_field, gradient_field, target,
             lambda_param, displacement, depth = 0):
    r"""
    Multi-resolution Gauss-Seidel solver: solves the linear system by first
    filtering (GS-iterate) the current level, then solves for the residual
    at a coarcer resolution andfinally refines the solution at the current 
    resolution. This scheme corresponds to the V-cycle proposed by Bruhn and 
    Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
        combining highest accuracy with real-time performance",
        10th IEEE International Conference on Computer Vision, 2005.
        ICCV 2005.
    """
    #presmoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_SSD3D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
    if n == 0:
        try:
            energy
        except NameError:
            energy = ssd.compute_energy_SSD3D(delta_field, sigma_field,
                                         gradient_field, lambda_param,
                                         displacement)
        return energy
    #solve at coarcer grid
    residual = ssd.compute_residual_displacement_field_SSD3D(delta_field,
                                                            sigma_field,
                                                            gradient_field,
                                                            target,
                                                            lambda_param,
                                                            displacement,
                                                            None)
    sub_residual = np.array(vfu.downsample_displacement_field3D(residual))
    del residual
    subsigma_field = None
    if sigma_field != None:
        subsigma_field = vfu.downsample_scalar_field3D(sigma_field)
    subdelta_field = vfu.downsample_scalar_field3D(delta_field)
    subgradient_field = np.array(
        vfu.downsample_displacement_field3D(gradient_field))
    shape = np.array(displacement.shape).astype(np.int32)
    sub_displacement = np.zeros(
        shape = ((shape[0]+1)//2, (shape[1]+1)//2, (shape[2]+1)//2, 3 ),
        dtype = floating)
    sublambda_param = lambda_param*0.25
    v_cycle_3d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
             sub_residual, sublambda_param, sub_displacement, depth+1)
    del subdelta_field
    del subsigma_field
    del subgradient_field
    del sub_residual
    vfu.accumulate_upsample_displacement_field3D(sub_displacement, displacement)
    del sub_displacement
    #post-smoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_SSD3D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
    try:
        energy
    except NameError:
        energy = ssd.compute_energy_SSD3D(delta_field, sigma_field,
                                         gradient_field, lambda_param,
                                         displacement)
    return energy