"""  Metrics for Symmetric Diffeomorphic Registration """

import abc
import numpy as np
from numpy import gradient
from scipy import ndimage
from dipy.align import vector_fields as vfu
from dipy.align import sumsqdiff as ssd
from dipy.align import crosscorr as cc
from dipy.align import expectmax as em
from dipy.align import floating


class SimilarityMetric(metaclass=abc.ABCMeta):
    def __init__(self, dim):
        r""" Similarity Metric abstract class

        A similarity metric is in charge of keeping track of the numerical
        value of the similarity (or distance) between the two given images. It
        also computes the update field for the forward and inverse displacement
        fields to be used in a gradient-based optimization algorithm. Note that
        this metric does not depend on any transformation (affine or
        non-linear) so it assumes the static and moving images are already
        warped

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        """
        self.dim = dim
        self.levels_above = None
        self.levels_below = None

        self.static_image = None
        self.static_affine = None
        self.static_spacing = None
        self.static_direction = None

        self.moving_image = None
        self.moving_affine = None
        self.moving_spacing = None
        self.moving_direction = None
        self.mask0 = False

    def set_levels_below(self, levels):
        r"""Informs the metric how many pyramid levels are below the current one

        Informs this metric the number of pyramid levels below the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly

        Parameters
        ----------
        levels : int
            the number of levels below the current Gaussian Pyramid level
        """
        self.levels_below = levels

    def set_levels_above(self, levels):
        r"""Informs the metric how many pyramid levels are above the current one

        Informs this metric the number of pyramid levels above the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly

        Parameters
        ----------
        levels : int
            the number of levels above the current Gaussian Pyramid level
        """
        self.levels_above = levels

    def set_static_image(self, static_image, static_affine, static_spacing,
                         static_direction):
        r"""Sets the static image being compared against the moving one.

        Sets the static image. The default behavior (of this abstract class) is
        simply to assign the reference to an attribute, but
        generalizations of the metric may need to perform other operations

        Parameters
        ----------
        static_image : array, shape (R, C) or (S, R, C)
            the static image
        """
        self.static_image = static_image
        self.static_affine = static_affine
        self.static_spacing = static_spacing
        self.static_direction = static_direction

    def use_static_image_dynamics(self, original_static_image, transformation):
        r"""This is called by the optimizer just after setting the static image.

        This method allows the metric to compute any useful
        information from knowing how the current static image was generated
        (as the transformation of an original static image). This method is
        called by the optimizer just after it sets the static image.
        Transformation will be an instance of DiffeomorficMap or None
        if the original_static_image equals self.moving_image.

        Parameters
        ----------
        original_static_image : array, shape (R, C) or (S, R, C)
            original image from which the current static image was generated
        transformation : DiffeomorphicMap object
            the transformation that was applied to original image to generate
            the current static image
        """
        pass

    def set_moving_image(self, moving_image, moving_affine, moving_spacing,
                         moving_direction):
        r"""Sets the moving image being compared against the static one.

        Sets the moving image. The default behavior (of this abstract class) is
        simply to assign the reference to an attribute, but
        generalizations of the metric may need to perform other operations

        Parameters
        ----------
        moving_image : array, shape (R, C) or (S, R, C)
            the moving image
        """
        self.moving_image = moving_image
        self.moving_affine = moving_affine
        self.moving_spacing = moving_spacing
        self.moving_direction = moving_direction

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""This is called by the optimizer just after setting the moving image

        This method allows the metric to compute any useful
        information from knowing how the current static image was generated
        (as the transformation of an original static image). This method is
        called by the optimizer just after it sets the static image.
        Transformation will be an instance of DiffeomorficMap or None if
        the original_moving_image equals self.moving_image.

        Parameters
        ----------
        original_moving_image : array, shape (R, C) or (S, R, C)
            original image from which the current moving image was generated
        transformation : DiffeomorphicMap object
            the transformation that was applied to the original image to generate
            the current moving image
        """
        pass

    @abc.abstractmethod
    def initialize_iteration(self):
        r"""Prepares the metric to compute one displacement field iteration.

        This method will be called before any compute_forward or
        compute_backward call, this allows the Metric to pre-compute any useful
        information for speeding up the update computations. This
        initialization was needed in ANTS because the updates are called once
        per voxel. In Python this is unpractical, though.
        """

    @abc.abstractmethod
    def free_iteration(self):
        r"""Releases the resources no longer needed by the metric

        This method is called by the RegistrationOptimizer after the required
        iterations have been computed (forward and / or backward) so that the
        SimilarityMetric can safely delete any data it computed as part of the
        initialization
        """

    @abc.abstractmethod
    def compute_forward(self):
        r"""Computes one step bringing the reference image towards the static.

        Computes the forward update field to register the moving image towards
        the static image in a gradient-based optimization algorithm
        """

    @abc.abstractmethod
    def compute_backward(self):
        r"""Computes one step bringing the static image towards the moving.

        Computes the backward update field to register the static image towards
        the moving image in a gradient-based optimization algorithm
        """

    @abc.abstractmethod
    def get_energy(self):
        r"""Numerical value assigned by this metric to the current image pair

        Must return the numeric value of the similarity between the given
        static and moving images
        """


class CCMetric(SimilarityMetric):

    def __init__(self, dim, sigma_diff=2.0, radius=4):
        r"""Normalized Cross-Correlation Similarity metric.

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        sigma_diff : the standard deviation of the Gaussian smoothing kernel to
            be applied to the update field at each iteration
        radius : int
            the radius of the squared (cubic) neighborhood at each voxel to be
            considered to compute the cross correlation
        """
        super(CCMetric, self).__init__(dim)
        self.sigma_diff = sigma_diff
        self.radius = radius
        self._connect_functions()

    def _connect_functions(self):
        r"""Assign the methods to be called according to the image dimension

        Assigns the appropriate functions to be called for precomputing the
        cross-correlation factors according to the dimension of the input
        images
        """
        if self.dim == 2:
            self.precompute_factors = cc.precompute_cc_factors_2d
            self.compute_forward_step = cc.compute_cc_forward_step_2d
            self.compute_backward_step = cc.compute_cc_backward_step_2d
            self.reorient_vector_field = vfu.reorient_vector_field_2d
        elif self.dim == 3:
            self.precompute_factors = cc.precompute_cc_factors_3d
            self.compute_forward_step = cc.compute_cc_forward_step_3d
            self.compute_backward_step = cc.compute_cc_backward_step_3d
            self.reorient_vector_field = vfu.reorient_vector_field_3d
        else:
            raise ValueError('CC Metric not defined for dim. %d' % self.dim)

    def initialize_iteration(self):
        r"""Prepares the metric to compute one displacement field iteration.

        Pre-computes the cross-correlation factors for efficient computation
        of the gradient of the Cross Correlation w.r.t. the displacement field.
        It also pre-computes the image gradients in the physical space by
        re-orienting the gradients in the voxel space using the corresponding
        affine transformations.
        """
        min_size = self.radius * 2 + 1

        def invalid_image_size(image):
            return any(size < min_size for size in image.shape)

        msg = ("Each image dimension should be superior to 2 * radius + 1 "
               f"({min_size}). Decrease CCMetric radius ({self.radius}) or "
               "increase your image size (shape=%(shape)s).")

        if invalid_image_size(self.static_image):
            raise ValueError("Static image size is too small. " +
                             msg % dict(shape=self.static_image.shape))
        if invalid_image_size(self.moving_image):
            raise ValueError("Moving image size is too small. " +
                             msg % dict(shape=self.moving_image.shape))

        self.factors = self.precompute_factors(self.static_image,
                                               self.moving_image,
                                               self.radius)
        self.factors = np.array(self.factors)

        self.gradient_moving = np.empty(
            shape=self.moving_image.shape+(self.dim,), dtype=floating)
        for i, grad in enumerate(gradient(self.moving_image)):
            self.gradient_moving[..., i] = grad

        # Convert moving image's gradient field from voxel to physical space
        if self.moving_spacing is not None:
            self.gradient_moving /= self.moving_spacing
        if self.moving_direction is not None:
            self.reorient_vector_field(self.gradient_moving,
                                       self.moving_direction)

        self.gradient_static = np.empty(
            shape=self.static_image.shape+(self.dim,), dtype=floating)
        for i, grad in enumerate(gradient(self.static_image)):
            self.gradient_static[..., i] = grad

        # Convert moving image's gradient field from voxel to physical space
        if self.static_spacing is not None:
            self.gradient_static /= self.static_spacing
        if self.static_direction is not None:
            self.reorient_vector_field(self.gradient_static,
                                       self.static_direction)

    def free_iteration(self):
        r"""Frees the resources allocated during initialization
        """
        del self.factors
        del self.gradient_moving
        del self.gradient_static

    def compute_forward(self):
        r"""Computes one step bringing the moving image towards the static.

        Computes the update displacement field to be used for registration of
        the moving image towards the static image
        """
        displacement, self.energy = self.compute_forward_step(
            self.gradient_static, self.factors, self.radius)
        displacement = np.array(displacement)
        for i in range(self.dim):
            displacement[..., i] = ndimage.gaussian_filter(
                displacement[..., i], self.sigma_diff)
        return displacement

    def compute_backward(self):
        r"""Computes one step bringing the static image towards the moving.

        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        displacement, energy = self.compute_backward_step(self.gradient_moving,
                                                          self.factors,
                                                          self.radius)
        displacement = np.array(displacement)
        for i in range(self.dim):
            displacement[..., i] = ndimage.gaussian_filter(
                displacement[..., i], self.sigma_diff)
        return displacement

    def get_energy(self):
        r"""Numerical value assigned by this metric to the current image pair

        Returns the Cross Correlation (data term) energy computed at the
        largest iteration
        """
        return self.energy


class EMMetric(SimilarityMetric):
    def __init__(self,
                 dim,
                 smooth=1.0,
                 inner_iter=5,
                 q_levels=256,
                 double_gradient=True,
                 step_type='gauss_newton'):
        r"""Expectation-Maximization Metric

        Similarity metric based on the Expectation-Maximization algorithm to
        handle multi-modal images. The transfer function is modeled as a set of
        hidden random variables that are estimated at each iteration of the
        algorithm.

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        smooth : float
            smoothness parameter, the larger the value the smoother the
            deformation field
        inner_iter : int
            number of iterations to be performed at each level of the multi-
            resolution Gauss-Seidel optimization algorithm (this is not the
            number of steps per Gaussian Pyramid level, that parameter must
            be set for the optimizer, not the metric)
        q_levels : number of quantization levels (equal to the number of hidden
            variables in the EM algorithm)
        double_gradient : boolean
            if True, the gradient of the expected static image under the moving
            modality will be added to the gradient of the moving image,
            similarly, the gradient of the expected moving image under the
            static modality will be added to the gradient of the static image.
        step_type : string ('gauss_newton', 'demons')
            the optimization schedule to be used in the multi-resolution
            Gauss-Seidel optimization algorithm (not used if Demons Step is
            selected)
        """
        super(EMMetric, self).__init__(dim)
        self.smooth = smooth
        self.inner_iter = inner_iter
        self.q_levels = q_levels
        self.use_double_gradient = double_gradient
        self.step_type = step_type
        self.static_image_mask = None
        self.moving_image_mask = None
        self.staticq_means_field = None
        self.movingq_means_field = None
        self.movingq_levels = None
        self.staticq_levels = None
        self._connect_functions()

    def _connect_functions(self):
        r"""Assign the methods to be called according to the image dimension

        Assigns the appropriate functions to be called for image quantization,
        statistics computation and multi-resolution iterations according to the
        dimension of the input images
        """
        if self.dim == 2:
            self.quantize = em.quantize_positive_2d
            self.compute_stats = em.compute_masked_class_stats_2d
            self.reorient_vector_field = vfu.reorient_vector_field_2d
        elif self.dim == 3:
            self.quantize = em.quantize_positive_3d
            self.compute_stats = em.compute_masked_class_stats_3d
            self.reorient_vector_field = vfu.reorient_vector_field_3d
        else:
            raise ValueError('EM Metric not defined for dim. %d' % self.dim)

        if self.step_type == 'demons':
            self.compute_step = self.compute_demons_step
        elif self.step_type == 'gauss_newton':
            self.compute_step = self.compute_gauss_newton_step
        else:
            raise ValueError('Opt. step %s not defined' % self.step_type)

    def initialize_iteration(self):
        r"""Prepares the metric to compute one displacement field iteration.

        Pre-computes the transfer functions (hidden random variables) and
        variances of the estimators. Also pre-computes the gradient of both
        input images. Note that once the images are transformed to the opposite
        modality, the gradient of the transformed images can be used with the
        gradient of the corresponding modality in the same fashion as
        diff-demons does for mono-modality images. If the flag
        self.use_double_gradient is True these gradients are averaged.
        """
        sampling_mask = self.static_image_mask*self.moving_image_mask
        self.sampling_mask = sampling_mask
        staticq, self.staticq_levels, hist = self.quantize(self.static_image,
                                                           self.q_levels)
        staticq = np.array(staticq, dtype=np.int32)
        self.staticq_levels = np.array(self.staticq_levels)
        staticq_means, staticq_vars = self.compute_stats(sampling_mask,
                                                         self.moving_image,
                                                         self.q_levels,
                                                         staticq)
        staticq_means[0] = 0
        self.staticq_means = np.array(staticq_means)
        self.staticq_variances = np.array(staticq_vars)
        self.staticq_sigma_sq_field = self.staticq_variances[staticq]
        self.staticq_means_field = self.staticq_means[staticq]

        self.gradient_moving = np.empty(
            shape=self.moving_image.shape+(self.dim,), dtype=floating)

        for i, grad in enumerate(gradient(self.moving_image)):
            self.gradient_moving[..., i] = grad

        # Convert moving image's gradient field from voxel to physical space
        if self.moving_spacing is not None:
            self.gradient_moving /= self.moving_spacing
        if self.moving_direction is not None:
            self.reorient_vector_field(self.gradient_moving,
                                       self.moving_direction)

        self.gradient_static = np.empty(
            shape=self.static_image.shape+(self.dim,), dtype=floating)

        for i, grad in enumerate(gradient(self.static_image)):
            self.gradient_static[..., i] = grad

        # Convert moving image's gradient field from voxel to physical space
        if self.static_spacing is not None:
            self.gradient_static /= self.static_spacing
        if self.static_direction is not None:
            self.reorient_vector_field(self.gradient_static,
                                       self.static_direction)

        movingq, self.movingq_levels, hist = self.quantize(self.moving_image,
                                                           self.q_levels)
        movingq = np.array(movingq, dtype=np.int32)
        self.movingq_levels = np.array(self.movingq_levels)
        movingq_means, movingq_variances = self.compute_stats(
            sampling_mask, self.static_image, self.q_levels, movingq)
        movingq_means[0] = 0
        self.movingq_means = np.array(movingq_means)
        self.movingq_variances = np.array(movingq_variances)
        self.movingq_sigma_sq_field = self.movingq_variances[movingq]
        self.movingq_means_field = self.movingq_means[movingq]
        if self.use_double_gradient:
            for i, grad in enumerate(gradient(self.staticq_means_field)):
                self.gradient_moving[..., i] += grad

            for i, grad in enumerate(gradient(self.movingq_means_field)):
                self.gradient_static[..., i] += grad

    def free_iteration(self):
        r"""
        Frees the resources allocated during initialization
        """
        del self.sampling_mask
        del self.staticq_levels
        del self.movingq_levels
        del self.staticq_sigma_sq_field
        del self.staticq_means_field
        del self.movingq_sigma_sq_field
        del self.movingq_means_field
        del self.gradient_moving
        del self.gradient_static

    def compute_forward(self):
        """Computes one step bringing the reference image towards the static.

        Computes the forward update field to register the moving image towards
        the static image in a gradient-based optimization algorithm
        """
        return self.compute_step(True)

    def compute_backward(self):
        r"""Computes one step bringing the static image towards the moving.

        Computes the update displacement field to be used for registration of
        the static image towards the moving image
        """
        return self.compute_step(False)

    def compute_gauss_newton_step(self, forward_step=True):
        r"""Computes the Gauss-Newton energy minimization step

        Computes the Newton step to minimize this energy, i.e., minimizes the
        linearized energy function with respect to the
        regularized displacement field (this step does not require
        post-smoothing, as opposed to the demons step, which does not include
        regularization). To accelerate convergence we use the multi-grid
        Gauss-Seidel algorithm proposed by Bruhn and Weickert et al [Bruhn05]

        Parameters
        ----------
        forward_step : boolean
            if True, computes the Newton step in the forward direction
            (warping the moving towards the static image). If False,
            computes the backward step (warping the static image to the
            moving image)

        Returns
        -------
        displacement : array, shape (R, C, 2) or (S, R, C, 3)
            the Newton step

        References
        ----------
        [Bruhn05] Andres Bruhn and Joachim Weickert, "Towards ultimate motion
                  estimation: combining highest accuracy with real-time
                  performance", 10th IEEE International Conference on Computer
                  Vision, 2005. ICCV 2005.
        """
        reference_shape = self.static_image.shape

        if forward_step:
            gradient = self.gradient_static
            delta = self.staticq_means_field - self.moving_image
            sigma_sq_field = self.staticq_sigma_sq_field
        else:
            gradient = self.gradient_moving
            delta = self.movingq_means_field - self.static_image
            sigma_sq_field = self.movingq_sigma_sq_field

        displacement = np.zeros(shape=reference_shape+(self.dim,),
                                dtype=floating)

        if self.dim == 2:
            self.energy = v_cycle_2d(self.levels_below,
                                     self.inner_iter, delta,
                                     sigma_sq_field,
                                     gradient,
                                     None,
                                     self.smooth,
                                     displacement)
        else:
            self.energy = v_cycle_3d(self.levels_below,
                                     self.inner_iter, delta,
                                     sigma_sq_field,
                                     gradient,
                                     None,
                                     self.smooth,
                                     displacement)
        return displacement

    def compute_demons_step(self, forward_step=True):
        r"""Demons step for EM metric

        Parameters
        ----------
        forward_step : boolean
            if True, computes the Demons step in the forward direction
            (warping the moving towards the static image). If False,
            computes the backward step (warping the static image to the
            moving image)

        Returns
        -------
        displacement : array, shape (R, C, 2) or (S, R, C, 3)
            the Demons step
        """
        sigma_reg_2 = np.sum(self.static_spacing**2)/self.dim

        if forward_step:
            gradient = self.gradient_static
            delta_field = self.static_image - self.movingq_means_field
            sigma_sq_field = self.movingq_sigma_sq_field
        else:
            gradient = self.gradient_moving
            delta_field = self.moving_image - self.staticq_means_field
            sigma_sq_field = self.staticq_sigma_sq_field

        if self.dim == 2:
            step, self.energy = em.compute_em_demons_step_2d(delta_field,
                                                             sigma_sq_field,
                                                             gradient,
                                                             sigma_reg_2,
                                                             None)
        else:
            step, self.energy = em.compute_em_demons_step_3d(delta_field,
                                                             sigma_sq_field,
                                                             gradient,
                                                             sigma_reg_2,
                                                             None)
        for i in range(self.dim):
            step[..., i] = ndimage.gaussian_filter(step[..., i], self.smooth)
        return step

    def get_energy(self):
        r"""The numerical value assigned by this metric to the current image pair

        Returns the EM (data term) energy computed at the largest
        iteration
        """
        return self.energy

    def use_static_image_dynamics(self, original_static_image, transformation):
        r"""This is called by the optimizer just after setting the static image.

        EMMetric takes advantage of the image dynamics by computing the
        current static image mask from the originalstaticImage mask (warped
        by nearest neighbor interpolation)

        Parameters
        ----------
        original_static_image : array, shape (R, C) or (S, R, C)
            the original static image from which the current static image was
            generated, the current static image is the one that was provided
            via 'set_static_image(...)', which may not be the same as the
            original static image but a warped version of it (even the static
            image changes during Symmetric Normalization, not only the moving
            one).
        transformation : DiffeomorphicMap object
            the transformation that was applied to the original_static_image
            to generate the current static image
        """
        self.static_image_mask = (original_static_image > 0).astype(np.int32)
        if transformation is None:
            return
        shape = np.array(self.static_image.shape, dtype=np.int32)
        affine = self.static_affine
        self.static_image_mask = transformation.transform(
            self.static_image_mask, 'nearest', None, shape, affine)

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r"""This is called by the optimizer just after setting the moving image.

        EMMetric takes advantage of the image dynamics by computing the
        current moving image mask from the original_moving_image mask (warped
        by nearest neighbor interpolation)

        Parameters
        ----------
        original_moving_image : array, shape (R, C) or (S, R, C)
            the original moving image from which the current moving image was
            generated, the current moving image is the one that was provided
            via 'set_moving_image(...)', which may not be the same as the
            original moving image but a warped version of it.
        transformation : DiffeomorphicMap object
            the transformation that was applied to the original_moving_image
            to generate the current moving image
        """
        self.moving_image_mask = (original_moving_image > 0).astype(np.int32)
        if transformation is None:
            return
        shape = np.array(self.moving_image.shape, dtype=np.int32)
        affine = self.moving_affine
        self.moving_image_mask = transformation.transform(
            self.moving_image_mask, 'nearest', None, shape, affine)


class SSDMetric(SimilarityMetric):

    def __init__(self, dim, smooth=4, inner_iter=10, step_type='demons'):
        r"""Sum of Squared Differences (SSD) Metric

        Similarity metric for (mono-modal) nonlinear image registration defined
        by the sum of squared differences (SSD)

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the image domain
        smooth : float
            smoothness parameter, the larger the value the smoother the
            deformation field
        inner_iter : int
            number of iterations to be performed at each level of the multi-
            resolution Gauss-Seidel optimization algorithm (this is not the
            number of steps per Gaussian Pyramid level, that parameter must
            be set for the optimizer, not the metric)
        step_type : string
            the displacement field step to be computed when 'compute_forward'
            and 'compute_backward' are called. Either 'demons' or
            'gauss_newton'
        """
        super(SSDMetric, self).__init__(dim)
        self.smooth = smooth
        self.inner_iter = inner_iter
        self.step_type = step_type
        self.levels_below = 0
        self._connect_functions()

    def _connect_functions(self):
        r"""Assign the methods to be called according to the image dimension

        Assigns the appropriate functions to be called for vector field
        reorientation and displacement field steps according to the
        dimension of the input images and the select type of step (either
        Demons or Gauss Newton)
        """
        if self.dim == 2:
            self.reorient_vector_field = vfu.reorient_vector_field_2d
        elif self.dim == 3:
            self.reorient_vector_field = vfu.reorient_vector_field_3d
        else:
            raise ValueError('SSD Metric not defined for dim. %d' % self.dim)

        if self.step_type == 'gauss_newton':
            self.compute_step = self.compute_gauss_newton_step
        elif self.step_type == 'demons':
            self.compute_step = self.compute_demons_step
        else:
            raise ValueError('Opt. step %s not defined' % self.step_type)

    def initialize_iteration(self):
        r"""Prepares the metric to compute one displacement field iteration.

        Pre-computes the gradient of the input images to be used in the
        computation of the forward and backward steps.
        """
        self.gradient_moving = np.empty(
            shape=self.moving_image.shape+(self.dim,), dtype=floating)
        for i, grad in enumerate(gradient(self.moving_image)):
            self.gradient_moving[..., i] = grad

        # Convert static image's gradient field from voxel to physical space
        if self.moving_spacing is not None:
            self.gradient_moving /= self.moving_spacing
        if self.moving_direction is not None:
            self.reorient_vector_field(self.gradient_moving,
                                       self.moving_direction)

        self.gradient_static = np.empty(
            shape=self.static_image.shape+(self.dim,), dtype=floating)
        for i, grad in enumerate(gradient(self.static_image)):
            self.gradient_static[..., i] = grad

        # Convert static image's gradient field from voxel to physical space
        if self.static_spacing is not None:
            self.gradient_static /= self.static_spacing
        if self.static_direction is not None:
            self.reorient_vector_field(self.gradient_static,
                                       self.static_direction)

    def compute_forward(self):
        r"""Computes one step bringing the reference image towards the static.

        Computes the update displacement field to be used for registration of
        the moving image towards the static image
        """
        return self.compute_step(True)

    def compute_backward(self):
        r"""Computes one step bringing the static image towards the moving.

        Computes the updated displacement field to be used for registration of
        the static image towards the moving image
        """
        return self.compute_step(False)

    def compute_gauss_newton_step(self, forward_step=True):
        r"""Computes the Gauss-Newton energy minimization step

        Minimizes the linearized energy function (Newton step) defined by the
        sum of squared differences of corresponding pixels of the input images
        with respect to the displacement field.

        Parameters
        ----------
        forward_step : boolean
            if True, computes the Newton step in the forward direction
            (warping the moving towards the static image). If False,
            computes the backward step (warping the static image to the
            moving image)

        Returns
        -------
        displacement : array, shape = static_image.shape + (3,)
            if forward_step==True, the forward SSD Gauss-Newton step,
            else, the backward step
        """
        reference_shape = self.static_image.shape

        if forward_step:
            gradient = self.gradient_static
            delta_field = self.static_image-self.moving_image
        else:
            gradient = self.gradient_moving
            delta_field = self.moving_image - self.static_image

        displacement = np.zeros(shape=reference_shape+(self.dim,),
                                dtype=floating)

        if self.dim == 2:
            self.energy = v_cycle_2d(self.levels_below, self.inner_iter,
                                     delta_field, None, gradient, None,
                                     self.smooth, displacement)
        else:
            self.energy = v_cycle_3d(self.levels_below, self.inner_iter,
                                     delta_field, None, gradient, None,
                                     self.smooth, displacement)
        return displacement

    def compute_demons_step(self, forward_step=True):
        r"""Demons step for SSD metric

        Computes the demons step proposed by Vercauteren et al.[Vercauteren09]
        for the SSD metric.

        Parameters
        ----------
        forward_step : boolean
            if True, computes the Demons step in the forward direction
            (warping the moving towards the static image). If False,
            computes the backward step (warping the static image to the
            moving image)

        Returns
        -------
        displacement : array, shape (R, C, 2) or (S, R, C, 3)
            the Demons step

        References
        ----------
        [Vercauteren09] Tom Vercauteren, Xavier Pennec, Aymeric Perchant,
                        Nicholas Ayache, "Diffeomorphic Demons: Efficient
                        Non-parametric Image Registration", Neuroimage 2009
        """
        sigma_reg_2 = np.sum(self.static_spacing**2)/self.dim

        if forward_step:
            gradient = self.gradient_static
            delta_field = self.static_image - self.moving_image
        else:
            gradient = self.gradient_moving
            delta_field = self.moving_image - self.static_image

        if self.dim == 2:
            step, self.energy = ssd.compute_ssd_demons_step_2d(delta_field,
                                                               gradient,
                                                               sigma_reg_2,
                                                               None)
        else:
            step, self.energy = ssd.compute_ssd_demons_step_3d(delta_field,
                                                               gradient,
                                                               sigma_reg_2,
                                                               None)
        for i in range(self.dim):
            step[..., i] = ndimage.gaussian_filter(step[..., i], self.smooth)
        return step

    def get_energy(self):
        r"""The numerical value assigned by this metric to the current image pair

        Returns the Sum of Squared Differences (data term) energy computed at
        the largest iteration
        """
        return self.energy

    def free_iteration(self):
        r"""
        Nothing to free for the SSD metric
        """
        pass


def v_cycle_2d(n, k, delta_field, sigma_sq_field, gradient_field, target,
               lambda_param, displacement, depth=0):
    r"""Multi-resolution Gauss-Seidel solver using V-type cycles

    Multi-resolution Gauss-Seidel solver: solves the Gauss-Newton linear system
    by first filtering (GS-iterate) the current level, then solves for the
    residual at a coarser resolution and finally refines the solution at the
    current resolution. This scheme corresponds to the V-cycle proposed by
    Bruhn and Weickert[Bruhn05].

    Parameters
    ----------
    n : int
        number of levels of the multi-resolution algorithm (it will be called
        recursively until level n == 0)
    k : int
        the number of iterations at each multi-resolution level
    delta_field : array, shape (R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)
    sigma_sq_field : array, shape (R, C)
        the variance of the gray level value at each voxel, according to the
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (R, C, 2)
        the gradient of the moving image
    target : array, shape (R, C, 2)
        right-hand side of the linear system to be solved in the Weickert's
        multi-resolution algorithm
    lambda_param : float
        smoothness parameter, the larger its value the smoother the
        displacement field
    displacement : array, shape (R, C, 2)
        the displacement field to start the optimization from

    Returns
    -------
    energy : the energy of the EM (or SSD if sigmafield[...]==1) metric at this
        iteration

    References
    ----------
    [Bruhn05] Andres Bruhn and Joachim Weickert, "Towards ultimate motion
              estimation: combining the highest accuracy with real-time
              performance", 10th IEEE International Conference on Computer
              Vision, 2005. ICCV 2005.
    """
    # pre-smoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_ssd_2d(delta_field,
                                                       sigma_sq_field,
                                                       gradient_field,
                                                       target,
                                                       lambda_param,
                                                       displacement)
    if n == 0:
        energy = ssd.compute_energy_ssd_2d(delta_field)
        return energy

    # solve at coarser grid
    residual = None
    residual = ssd.compute_residual_displacement_field_ssd_2d(delta_field,
                                                              sigma_sq_field,
                                                              gradient_field,
                                                              target,
                                                              lambda_param,
                                                              displacement,
                                                              residual)
    sub_residual = np.array(vfu.downsample_displacement_field_2d(residual))
    del residual
    subsigma_sq_field = None
    if sigma_sq_field is not None:
        subsigma_sq_field = vfu.downsample_scalar_field_2d(sigma_sq_field)
    subdelta_field = vfu.downsample_scalar_field_2d(delta_field)

    subgradient_field = np.array(
        vfu.downsample_displacement_field_2d(gradient_field))

    shape = np.array(displacement.shape).astype(np.int32)
    half_shape = ((shape[0] + 1) // 2, (shape[1] + 1) // 2, 2)
    sub_displacement = np.zeros(shape=half_shape,
                                dtype=floating)
    sublambda_param = lambda_param*0.25
    v_cycle_2d(n-1, k, subdelta_field, subsigma_sq_field, subgradient_field,
               sub_residual, sublambda_param, sub_displacement, depth+1)
    # displacement += np.array(
    #    vfu.upsample_displacement_field(sub_displacement, shape))
    displacement += vfu.resample_displacement_field_2d(sub_displacement,
                                                       np.array([0.5, 0.5]),
                                                       shape)

    # post-smoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_ssd_2d(delta_field,
                                                       sigma_sq_field,
                                                       gradient_field,
                                                       target,
                                                       lambda_param,
                                                       displacement)
    energy = ssd.compute_energy_ssd_2d(delta_field)
    return energy


def v_cycle_3d(n, k, delta_field, sigma_sq_field, gradient_field, target,
               lambda_param, displacement, depth=0):
    r"""Multi-resolution Gauss-Seidel solver using V-type cycles

    Multi-resolution Gauss-Seidel solver: solves the linear system by first
    filtering (GS-iterate) the current level, then solves for the residual
    at a coarser resolution and finally refines the solution at the current
    resolution. This scheme corresponds to the V-cycle proposed by Bruhn and
    Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
        combining highest accuracy with real-time performance",
        10th IEEE International Conference on Computer Vision, 2005.
        ICCV 2005.

    Parameters
    ----------
    n : int
        number of levels of the multi-resolution algorithm (it will be called
        recursively until level n == 0)
    k : int
        the number of iterations at each multi-resolution level
    delta_field : array, shape (S, R, C)
        the difference between the static and moving image (the 'derivative
        w.r.t. time' in the optical flow model)
    sigma_sq_field : array, shape (S, R, C)
        the variance of the gray level value at each voxel, according to the
        EM model (for SSD, it is 1 for all voxels). Inf and 0 values
        are processed specially to support infinite and zero variance.
    gradient_field : array, shape (S, R, C, 3)
        the gradient of the moving image
    target : array, shape (S, R, C, 3)
        right-hand side of the linear system to be solved in the Weickert's
        multi-resolution algorithm
    lambda_param : float
        smoothness parameter, the larger its value the smoother the
        displacement field
    displacement : array, shape (S, R, C, 3)
        the displacement field to start the optimization from

    Returns
    -------
    energy : the energy of the EM (or SSD if sigmafield[...]==1) metric at this
        iteration
    """
    # pre-smoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_ssd_3d(delta_field,
                                                       sigma_sq_field,
                                                       gradient_field,
                                                       target,
                                                       lambda_param,
                                                       displacement)
    if n == 0:
        energy = ssd.compute_energy_ssd_3d(delta_field)
        return energy
    # solve at coarser grid
    residual = ssd.compute_residual_displacement_field_ssd_3d(delta_field,
                                                              sigma_sq_field,
                                                              gradient_field,
                                                              target,
                                                              lambda_param,
                                                              displacement,
                                                              None)
    sub_residual = np.array(vfu.downsample_displacement_field_3d(residual))
    del residual
    subsigma_sq_field = None
    if sigma_sq_field is not None:
        subsigma_sq_field = vfu.downsample_scalar_field_3d(sigma_sq_field)
    subdelta_field = vfu.downsample_scalar_field_3d(delta_field)
    subgradient_field = np.array(
        vfu.downsample_displacement_field_3d(gradient_field))
    shape = np.array(displacement.shape).astype(np.int32)
    sub_displacement = np.zeros(
        shape=((shape[0]+1)//2, (shape[1]+1)//2, (shape[2]+1)//2, 3),
        dtype=floating)
    sublambda_param = lambda_param*0.25
    v_cycle_3d(n-1, k, subdelta_field, subsigma_sq_field, subgradient_field,
               sub_residual, sublambda_param, sub_displacement, depth+1)
    del subdelta_field
    del subsigma_sq_field
    del subgradient_field
    del sub_residual
    displacement += vfu.resample_displacement_field_3d(sub_displacement,
                                                       0.5 * np.ones(3),
                                                       shape)
    del sub_displacement
    # post-smoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_ssd_3d(delta_field,
                                                       sigma_sq_field,
                                                       gradient_field,
                                                       target,
                                                       lambda_param,
                                                       displacement)
    energy = ssd.compute_energy_ssd_3d(delta_field)
    return energy
