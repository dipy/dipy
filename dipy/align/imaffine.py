import numpy as np
import numpy.linalg as npl
import scipy as sp
import scipy.ndimage as ndimage
from dipy.align import floating
import dipy.align.vector_fields as vf
from dipy.align.mattes import MattesBase, sample_domain_regular
from dipy.core.optimize import Optimizer
import matplotlib.pyplot as plt
from dipy.align.transforms import (Transform)
from dipy.align.imwarp import (get_direction_and_spacings, ScaleSpace)

class IsotropicScaleSpace(ScaleSpace):
    def __init__(self, image, factors, sigmas,
                 codomain_affine=None,
                 input_spacing=None,
                 mask0=False):
        r"""
        Example:
        image = np.zeros(shape=(256, 256, 128))
        factors = [4, 2, 1]
        sigmas = [3, 1, 0]
        codomain_affine = None
        input_spacing = np.array([1., 1., 1.5])
        mask0 = False
        ss = IsotropicScaleSpace(image, factors, sigmas, codomain_affine,
                                 input_spacing, mask0)
        shape = ss.get_domain_shape(2)
        """
        self.dim = len(image.shape)
        self.num_levels = len(factors)
        if len(sigmas) != self.num_levels:
            raise ValueError("sigmas and factors must have the same length")
        input_size = np.array(image.shape)
        if mask0:
            mask = np.asarray(image>0, dtype=np.int32)

        #normalize input image to [0,1]
        img = (image - image.min())/(image.max() - image.min())
        if mask0:
            img *= mask

        #The properties are saved in separate lists. Insert input image
        #properties at the first level of the scale space
        self.images = [img.astype(floating)]
        self.domain_shapes = [input_size.astype(np.int32)]
        if input_spacing is None:
            input_spacing = np.ones((self.dim,), dtype = np.int32)
        self.spacings = [input_spacing]
        self.scalings = [np.ones(self.dim)]
        self.affines = [codomain_affine]
        self.sigmas = [np.ones(self.dim) * sigmas[self.num_levels - 1]]

        if codomain_affine is not None:
            self.affine_invs = [npl.inv(codomain_affine)]
        else:
            self.affine_invs = [None]

        #compute the rest of the levels
        min_index = np.argmin(input_spacing)
        min_spacing = input_spacing[min_index]
        for i in range(1, self.num_levels):
            factor = factors[self.num_levels - 1 - i]
            shrink_factors = np.zeros(self.dim)
            new_spacing = np.zeros(self.dim)
            shrink_factors[min_index] = factor
            new_spacing[min_index] = input_spacing[min_index] * factor
            for j in range(self.dim):
                if j != min_index:
                    #Select the factor that maximizes isotropy
                    shrink_factors[j] = factor
                    new_spacing[j] = input_spacing[j] * factor
                    min_diff = np.abs(new_spacing[j] - new_spacing[min_index])
                    for f in range(1, factor):
                        diff = np.abs(input_spacing[j] * f - new_spacing[min_index])
                        if diff < min_diff:
                            shrink_factors[j] = f
                            new_spacing[j] = input_spacing[j] * f
                            min_diff = diff

            extended = np.append(shrink_factors, [1])
            if not codomain_affine is None:
                affine = codomain_affine.dot(np.diag(extended))
            else:
                affine = np.diag(extended)
            output_size = (input_size / shrink_factors).astype(np.int32)
            new_sigmas = np.ones(self.dim) * sigmas[self.num_levels - i - 1]

            #filter along each direction with the appropriate sigma
            filtered = ndimage.filters.gaussian_filter(image, new_sigmas)
            filtered = ((filtered - filtered.min())/
                       (filtered.max() - filtered.min()))
            if mask0:
                filtered *= mask

            #Add current level to the scale space
            self.images.append(filtered.astype(floating))
            self.domain_shapes.append(output_size)
            self.spacings.append(new_spacing)
            self.scalings.append(shrink_factors)
            self.affines.append(affine)
            self.affine_invs.append(npl.inv(affine))
            self.sigmas.append(new_sigmas)


class MattesMIMetric(MattesBase):
    def __init__(self, nbins=32, sampling_proportion=30):
        r""" Initializes an instance of the Mattes MI metric

        Parameters
        ----------
        nbins : int
            the number of bins to be used for computing the intensity histograms
        sampling_proportion : int in (0,100]
            the percentage of voxels to be used for estimating the (joint and marginal)
            intensity histograms. If None, dense sampling is used.
        """
        super(MattesMIMetric, self).__init__(nbins)
        self.sampling_proportion = sampling_proportion

    def setup(self, transform, static, moving, moving_spacing, static_aff=None,
              moving_aff=None, smask=None, mmask=None, prealign=None,
              precondition=False):
        r""" Prepares the metric to compute intensity densities and gradients

        The histograms will be setup to compute probability densities of
        intensities within the minimum and maximum values of static and moving
        within the given masks

        Parameters
        ----------
        transform: instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        static : array, shape (S, R, C) or (R, C)
            static image
        moving : array, shape (S', R', C') or (R', C')
            moving image
        moving_spacing : array, shape (3,) or (2,)
            the spacing between neighboring voxels along each dimension
        static_affine : array (dim+1, dim+1)
            the grid-to-space transform of the static image
        moving_affine : array (dim+1, dim+1)
            the grid-to-space transform of the moving image
        smask : array, shape (S, R, C) or (R, C)
            the mask indicating the voxels of interest within the static image.
            If None, ones_like(smask) will be used as mask (all voxels will be
            taken).
        mmask : array, shape (S', R', C') or (R', C')
            the mask indicating the voxels of interest within the moving image
            If None, ones_like(smask) will be used as mask (all voxels will be
            taken).
        prealign : array, shape (dim+1, dim+1)
            the pre-aligning matrix (an affine transform) that roughly aligns
            the moving image towards the static image. If None, no pre-alignment
            is performed. If a pre-alignment matrix is available, it is
            recommended to directly provide the transform to the MattesMIMetric
            instead of manually warping the moving image and provide None or
            identity as prealign. This way, the metric avoids performing more
            than one interpolation.
        precondition : boolean
            if True, the gradient's components are scaled attempting to make the
            levelsets of the objective function as isotropic as possible
        """
        self.dim = len(static.shape)
        self.transform = transform
        self.static = np.array(static).astype(np.float64)
        self.moving = np.array(moving).astype(np.float64)
        self.static_aff = static_aff
        self.static_aff_inv = npl.inv(static_aff)
        self.moving_aff = moving_aff
        self.moving_aff_inv = npl.inv(moving_aff)
        self.moving_spacing = moving_spacing
        self.smask = smask
        self.mmask = mmask
        self.prealign = prealign
        self.param_scales = None
        self.precondition = precondition

        T = np.eye(self.dim + 1)
        if self.prealign is not None:
            T = T.dot(self.prealign)

        if self.dim == 2:
            self.interp_method = vf.interpolate_scalar_2d
        else:
            self.interp_method = vf.interpolate_scalar_3d

        if self.sampling_proportion is None:
            self.warped = aff_warp(self.static, self.static_aff, self.moving,
                                   self.moving_aff, T).astype(np.float64)
            self.samples = None
            self.ns = 0
        else:
            self.warped = None
            k = 100/self.sampling_proportion
            shape = np.array(static.shape, dtype=np.int32)
            samples = np.array(sample_domain_regular(k, shape, static_aff))
            ns = samples.shape[0]
            # Add a column of ones (homogeneous coordinates)
            samples = np.hstack((samples, np.ones(ns)[:,None]))
            # Sample the static image
            points_on_static = (self.static_aff_inv.dot(samples.T).T)[...,:self.dim]
            static_vals, inside = self.interp_method(self.static.astype(np.float32),
                                                     points_on_static)
            static_vals = np.array(static_vals, dtype=np.float64)
            # Sample the moving image
            sp_to_moving = self.moving_aff_inv.dot(T)
            points_on_moving = (sp_to_moving.dot(samples.T).T)[...,:self.dim]
            moving_vals, inside = self.interp_method(self.moving.astype(np.float32),
                                                     points_on_moving)
            # Store relevant information
            self.samples = samples
            self.ns = ns
            self.static_vals = static_vals
            self.moving_vals = moving_vals

        if self.mmask is not None:
            self.wmask = aff_warp(self.static, self.static_aff, self.mmask,
                                  self.moving_aff, T, True).astype(np.int32)
        else:
            self.wmask = None

        MattesBase.setup(self, self.static, self.moving, self.smask, self.wmask)

    def _update_dense(self, xopt, update_gradient = True):
        r""" Updates the marginal and joint distributions and the joint gradient

        The distributions and the gradient of the joint distribution are
        updated according to the static and warped images. The warped image
        is precisely the moving image after transforming it by the transform
        defined by the xopt parameters.

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        """
        # Get the matrix associated to the xopt parameter vector
        T = self.transform.param_to_matrix(xopt)
        if self.prealign is not None:
            T = T.dot(self.prealign)

        # Warp the moving image
        self.warped = aff_warp(self.static, self.static_aff, self.moving,
                               self.moving_aff, T).astype(np.float64)

        if self.mmask is not None:
            # Get the warped mask.
            self.wmask = aff_warp(self.static, self.static_aff, self.mmask,
                                  self.moving_aff, T, True).astype(np.int32)
        else:
            self.wmask = None

        # Update the joint and marginal intensity distributions
        self.update_pdfs_dense(self.static, self.warped, self.smask, self.wmask)

        # Compute the gradient of the joint PDF w.r.t. parameters
        if update_gradient:
            if self.static_aff is None:
                grid_to_space = T
            else:
                grid_to_space = T.dot(self.static_aff)

            # Compute the gradient of the moving image at the current transform
            grid_to_space = T.dot(self.static_aff)
            self.grad_w, inside = vf.gradient(self.moving.astype(floating), self.moving_aff_inv,
                                      self.moving_spacing, self.static.shape, grid_to_space)

            # Update the gradient of the metric
            self.update_gradient_dense(xopt, self.transform, self.static,
                                       self.warped, grid_to_space, self.grad_w,
                                       self.smask, self.wmask)

        # Evaluate the mutual information and its gradient
        # The results are in self.metric_val and self.metric_grad
        # ready to be returned from 'distance' and 'gradient'
        self.update_mi_metric(update_gradient)


    def _update_sparse(self, xopt, update_gradient = True):
        r""" Updates the marginal and joint distributions and the joint gradient

        The distributions and the gradient of the joint distribution are
        updated according to the samples taken from the static and moving images.
        The samples are points in physical space, so the static intensities are
        are always the same, but the corresponding points in the moving image
        depend on the transform defined by xopt.

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        """
        # Get the matrix associated to the xopt parameter vector
        T = self.transform.param_to_matrix(xopt)
        if self.prealign is not None:
            T = T.dot(self.prealign)

        # Sample the moving image
        sp_to_moving = self.moving_aff_inv.dot(T)
        points_on_moving = (sp_to_moving.dot(self.samples.T).T)[...,:self.dim]
        self.moving_vals, inside = self.interp_method(self.moving.astype(np.float32),
                                                      points_on_moving)
        self.moving_vals = np.array(self.moving_vals, dtype=np.float64)

        # Update the joint and marginal intensity distributions
        self.update_pdfs_sparse(self.static_vals, self.moving_vals)

        # Compute the gradient of the joint PDF w.r.t. parameters
        if update_gradient:
            # Compute the gradient of the moving image at the current transform
            mgrad, inside = vf.sparse_gradient(self.moving.astype(np.float32),
                                               sp_to_moving,
                                               self.moving_spacing,
                                               self.samples)
            self.update_gradient_sparse(xopt, self.transform, self.static_vals,
                                        self.moving_vals,
                                        self.samples[...,:self.dim],
                                        mgrad)

        # Evaluate the mutual information and its gradient
        # The results are in self.metric_val and self.metric_grad
        # ready to be returned from 'distance' and 'gradient'
        self.update_mi_metric(update_gradient)


    def distance(self, xopt):
        r""" Numeric value of the metric evaluated at the given parameters
        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        """
        if self.samples is None:
            self._update_dense(xopt, False)
        else:
            self._update_sparse(xopt, False)
        return self.metric_val

    def gradient(self, xopt):
        r""" Numeric value of the metric's gradient at the given parameters
        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        """
        if self.samples is None:
            self._update_dense(xopt, True)
        else:
            self._update_dense(xopt, True)
        if self.precondition:
            return self.metric_grad / self.param_scales
        else:
            return self.metric_grad.copy()


    def value_and_gradient(self, xopt):
        r""" Numeric value of the metric and its gradient at the given parameter

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        """
        if self.samples is None:
            self._update_dense(xopt, True)
        else:
            self._update_sparse(xopt, True)
        print(">>",self.metric_val)
        #print(">>",self.metric_val, self.param_scales, self.metric_grad.copy() / self.param_scales)
        if self.precondition:
            return self.metric_val, self.metric_grad / self.param_scales
        else:
            return self.metric_val, self.metric_grad.copy()


class AffineRegistration(object):
    def __init__(self,
                 metric=None,
                 method = 'CGGS',
                 level_iters=None,
                 opt_tol=1e-5,
                 ss_sigma_factor=1.0,
                 factors = None,
                 sigmas = None,
                 options=None):
        r""" Initializes an instance of the AffineRegistration class

        Parameters
        ----------
        metric : object
            an instance of a metric
        level_iters : list
            the number of iterations at each level of the Gaussian pyramid.
            level_iters[0] corresponds to the finest level, level_iters[n-1] the
            coarsest, where n is the length of the list
        opt_tol : float
            tolerance parameter for the optimizer
        ss_sigma_factor : float
            parameter of the scale-space smoothing kernel. For example, the
            std. dev. of the kernel will be factor*(2^i) in the isotropic case
            where i = 0, 1, ..., n_scales is the scale
        options : None or dict,
            extra optimization options.
        """

        self.metric = metric

        if self.metric is None:
            self.metric = MattesMIMetric()

        if level_iters is None:
            level_iters = [10000, 10000, 2500]
        self.level_iters = level_iters
        self.levels = len(level_iters)
        if self.levels == 0:
            raise ValueError('The iterations list cannot be empty')

        self.opt_tol = opt_tol
        self.ss_sigma_factor = ss_sigma_factor

        self.options = options
        self.method = method

        self.use_isotropic = False
        if (factors is not None) and (sigmas is not None):
            self.factors = factors
            self.sigmas = sigmas
            self.use_isotropic = True


    def _init_optimizer(self, static, moving, transform, x0,
                        static_affine, moving_affine, prealign):
        r"""Initializes the registration optimizer

        Initializes the optimizer by computing the scale space of the input
        images

        Parameters
        ----------
        static: array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.
        moving: array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed to
            be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the 'prealign' matrix
        transform: instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        x0: array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.
        static_affine: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the static image
        moving_affine: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the moving image
        prealign: array, shape (dim+1, dim+1)
            the affine transformation (operating on the physical space)
            pre-aligning the moving image towards the static

        """
        self.dim = len(static.shape)
        self.transform = transform
        self.nparams = transform.get_number_of_parameters()

        if x0 is None:
            x0 = self.transform.get_identity_parameters()
        self.x0 = x0
        if prealign is None:
            self.prealign = np.eye(self.dim + 1)
        elif prealign == 'mass':
            self.prealign = aff_centers_of_mass(static, static_affine, moving,
                                                moving_affine)
        elif prealign == 'origins':
            self.prealign = aff_origins(static, static_affine, moving,
                                        moving_affine)
        elif prealign == 'centers':
            self.prealign = aff_geometric_centers(static, static_affine, moving,
                                                  moving_affine)
        else:
            self.prealign = prealign
        #Extract information from the affine matrices to create the scale space
        static_direction, static_spacing = \
            get_direction_and_spacings(static_affine, self.dim)
        moving_direction, moving_spacing = \
            get_direction_and_spacings(moving_affine, self.dim)

        static = (static - static.min())/(static.max() - static.min())
        moving = (moving - moving.min())/(moving.max() - moving.min())
        #Build the scale space of the input images

        if self.use_isotropic:
            self.moving_ss = IsotropicScaleSpace(moving, self.factors, self.sigmas,
                                    moving_affine, moving_spacing, False)

            self.static_ss = IsotropicScaleSpace(static, self.factors, self.sigmas,
                                        static_affine, static_spacing, False)
        else:
            self.moving_ss = ScaleSpace(moving, self.levels, moving_affine,
                                    moving_spacing, self.ss_sigma_factor,
                                    False)

            self.static_ss = ScaleSpace(static, self.levels, static_affine,
                                    static_spacing, self.ss_sigma_factor,
                                    False)


    def optimize(self, static, moving, transform, x0, static_affine=None,
                 moving_affine=None, smask=None, mmask=None, prealign=None):
        r'''
        Parameters
        ----------
        transform: string
        prealign: string, or matrix, or None
            If string:
                'mass': align centers of gravity
                'origins': align physical coordinates of voxel (0,0,0)
                'centers': align physical coordinates of central voxels
            If matrix:
                array, shape (dim+1, dim+1)
            If None:
                Start from identity
        '''
        self._init_optimizer(static, moving, transform, x0, static_affine,
                             moving_affine, prealign)
        del prealign # now we must refer to self.prealign

        # Multi-resolution iterations
        original_static_affine = self.static_ss.get_affine(0)
        original_moving_affine = self.moving_ss.get_affine(0)
        original_moving_spacing = self.moving_ss.get_spacing(0)

        original_smask = smask
        original_mmask = mmask

        for level in range(self.levels - 1, -1, -1):
            self.current_level = level
            max_iter = self.level_iters[level]
            print('Optimizing level %d [max iter: %d]'%(level, max_iter))

            # Resample the smooth static image to the shape of this level
            smooth_static = self.static_ss.get_image(level)
            current_static_shape = self.static_ss.get_domain_shape(level)
            current_static_aff = self.static_ss.get_affine(level)

            current_static = aff_warp(tuple(current_static_shape),
                                      current_static_aff, smooth_static,
                                      original_static_affine, None, False)
            if original_smask is not None:
                current_smask = aff_warp(tuple(current_static_shape),
                                         current_static_aff, original_smask,
                                         original_static_affine, None, True)
            else:
                current_smask = None

            # The moving image is full resolution
            current_moving_aff = original_moving_affine
            current_moving_spacing = self.moving_ss.get_spacing(level)

            current_moving = self.moving_ss.get_image(level)
            current_mmask = original_mmask

            # Prepare the metric for iterations at this resolution
            if self.method == 'CGGS':
                precondition = True
            else:
                precondition = False
            self.metric.setup(transform, current_static, current_moving,
                              current_moving_spacing, current_static_aff, current_moving_aff,
                              current_smask, current_mmask, self.prealign,
                              precondition)

            if precondition:
                grid_to_space = self.prealign.dot(current_static_aff)
                scales = estimate_param_scales(self.transform_type, self.dim,
                                               current_static.shape,
                                               grid_to_space)
                self.metric.param_scales = scales

            #optimize this level
            if self.options is None:
                self.options = {'gtol':1e-4, 'maxiter': max_iter, 'disp':False}
            if self.method == 'CGGS':
                domain_shape = current_static.shape
                domain_affine = self.prealign.dot(current_static_aff)

                xopt = nonlinear_cg(self.transform_type, self.dim, domain_shape,
                                    domain_affine,
                                    self.metric.distance,
                                    self.metric.gradient,
                                    self.metric.value_and_gradient,
                                    self.x0, self.options)
            else:
                opt = Optimizer(self.metric.value_and_gradient, self.x0,
                                method=self.method, jac = True,
                                options=self.options)
                xopt = opt.xopt

            # Update prealign matrix with optimal parameters
            T = self.transform.param_to_matrix(xopt)
            self.prealign = T.dot(self.prealign)

            # Start next iteration at identity
            self.x0 = self.transform.get_identity_parameters()

            # Update the metric to the current solution
            self.metric._update_dense(xopt, False)
        return self.prealign


def estimate_param_scales(transform_type, dim, domain_shape, domain_affine):
    r""" Estimate the parameter scales of the given affine transform

    If we vary only one of the parameters of the parameter vector p of the
    given parametric transformation T (where p corresponds to the parameters of
    the identity transform), this defines a trajectory T(p[i]; x0), which
    transforms an initial point x0 according to the parameter p[i]. The
    magnitude of the tangent vector of that trajectory is a measure of its
    'velocity', which tells us how 'sensitive' the transformation is to each of
    its parameters at the specific point x0. Thus, we define the 'scale' of a
    parameter as the maximum, over all possible starting points x0, of the
    aforementioned tangent vector's squared norm.

    Parameters
    ----------
    transform_type : int
        the type of the transformation (use transform_type dictionary from
        the transforms module to map transformation name to int)
    dim : int (either 2 or 3)
        the domain dimension of the transformation (either 2 or 3)
    domain_shape : array, shape (dim,)
        the shape of the grid on which the transform is applied
    domain_affine : array, shape (dim + 1, dim + 1)
        the grid-to-space transform associated to the grid of shape domain_shape
    """
    h = 0.01
    n = number_of_parameters(transform_type, dim)
    theta = np.empty(n)
    X = np.empty((2 ** dim, dim + 1)) # All 2^dim corners of the grid
    T = np.ndarray((dim + 1, dim + 1))

    # Generate all corners of the given domain
    X[:, dim] = 1 # Homogeneous coordinate
    for i in range(2 ** dim):
        ii = i
        for j in range(dim):
            if (ii % 2) == 0:
                X[i, j] = 0
            else:
                X[i, j] = domain_shape[j] - 1
            ii = ii // 2

    # Transform grid points to physical space
    if domain_affine is not None:
        X = X.dot(domain_affine.transpose())

    # Compute the scale of each parameter
    scales = np.zeros(n)
    for i in range(n):
        get_identity_parameters(transform_type, dim, theta)
        theta[i] += h
        param_to_matrix(transform_type, dim, theta, T)
        transformed = X.dot(T.transpose())
        sq_norms = np.sum((transformed - X) ** 2, 1)
        max_shift_sq = sq_norms.max()
        scales[i] = max_shift_sq

    # Avoid zero scales to prevent division by zero
    scales[scales == 0] = scales[scales > 0].min()
    scales /= h * h
    return scales


def aff_warp(static, static_affine, moving, moving_affine, transform, nn=False):
    r""" Warps the moving image towards the static using the given transform

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image: it will provide the grid and grid-to-space transform for
        the warped image
    static_affine:
        grid-to-space transform associated to the static image
    moving: array, shape(S', R', C')
        moving image
    moving_affine:
        grid-to-space transform associated to the moving image

    Returns
    -------
    warped: array, shape (S, R, C)
    """
    if type(static) is tuple:
        dim = len(static)
        shape = np.array(static, dtype=np.int32)
    else:
        dim = len(static.shape)
        shape = np.array(static.shape, dtype=np.int32)
    if nn:
        input = np.array(moving,dtype=np.int32)
        if dim == 2:
            warp_method = vf.warp_2d_affine_nn
        elif dim == 3:
            warp_method = vf.warp_3d_affine_nn
    else:
        input = np.array(moving,dtype=floating)
        if dim == 2:
            warp_method = vf.warp_2d_affine
        elif dim == 3:
            warp_method = vf.warp_3d_affine

    m_aff_inv = npl.inv(moving_affine)
    if transform is None:
        composition = m_aff_inv.dot(static_affine)
    else:
        composition = m_aff_inv.dot(transform.dot(static_affine))

    warped = warp_method(input, shape, composition)

    return np.array(warped)


def aff_centers_of_mass(static, static_affine, moving, moving_affine):
    r""" Transformation to align the center of mass of the input images

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image
    static_affine: array, shape (4, 4)
        the voxel-to-space transformation of the static image
    moving: array, shape(S, R, C)
        moving image
    moving_affine: array, shape (4, 4)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape(4, 4)
        the affine transformation (translation only, in this case) aligning
        the center of mass of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    c_static = ndimage.measurements.center_of_mass(np.array(static))
    c_static = static_affine.dot(c_static+(1,))
    c_moving = ndimage.measurements.center_of_mass(np.array(moving))
    c_moving = moving_affine.dot(c_moving+(1,))
    transform = np.eye(dim + 1)
    transform[:dim,dim] = (c_moving - c_static)[:dim]
    return transform


def aff_geometric_centers(static, static_affine, moving, moving_affine):
    r""" Transformation to align the geometric center of the input images

    With "geometric center" of a volume we mean the physical coordinates of
    its central voxel

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image
    static_affine: array, shape (4, 4)
        the voxel-to-space transformation of the static image
    moving: array, shape(S, R, C)
        moving image
    moving_affine: array, shape (4, 4)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape(4, 4)
        the affine transformation (translation only, in this case) aligning
        the geometric center of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    c_static = tuple((np.array(static.shape, dtype = np.float64))*0.5)
    c_static = static_affine.dot(c_static+(1,))
    c_moving = tuple((np.array(moving.shape, dtype = np.float64))*0.5)
    c_moving = moving_affine.dot(c_moving+(1,))
    transform = np.eye(dim + 1)
    transform[:dim,dim] = (c_moving - c_static)[:dim]
    return transform


def aff_origins(static, static_affine, moving, moving_affine):
    r""" Transformation to align the origins of the input images

    With "origin" of a volume we mean the physical coordinates of
    voxel (0,0,0)

    Parameters
    ----------
    static: array, shape(S, R, C)
        static image
    static_affine: array, shape (4, 4)
        the voxel-to-space transformation of the static image
    moving: array, shape(S, R, C)
        moving image
    moving_affine: array, shape (4, 4)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape(4, 4)
        the affine transformation (translation only, in this case) aligning
        the origin of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    c_static = static_affine[:dim, dim]
    c_moving = moving_affine[:dim, dim]
    transform = np.eye(dim + 1)
    transform[:dim,dim] = (c_moving - c_static)[:dim]
    return transform
