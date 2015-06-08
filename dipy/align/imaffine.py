import numpy as np
import numpy.linalg as npl
import scipy.ndimage as ndimage
from ..core.optimize import Optimizer
from ..core.optimize import SCIPY_LESS_0_12
from . import floating
from . import vector_fields as vf
from .mattes import MattesBase, sample_domain_regular
from .imwarp import (get_direction_and_spacings, ScaleSpace)
from .scalespace import IsotropicScaleSpace


class MattesMIMetric(MattesBase):
    def __init__(self, nbins=32, sampling_proportion=None):
        r""" Initializes an instance of the Mattes MI metric

        This class implements the methods required by Optimizer to drive the
        registration process by making calls to the low level methods defined
        in MattesBase.

        Parameters
        ----------
        nbins : int, optional
            the number of bins to be used for computing the intensity
            histograms. The default is 32.
        sampling_proportion : float in (0,1], optional
            There are two types of sampling: dense and sparse. Dense sampling
            uses all voxels for estimating the (joint and marginal) intensity
            histograms, while sparse sampling uses a subset of them. If
            sampling_proportion is None, then dense sampling is
            used. If sampling_proportion is a floating point value in (0,1]
            then sparse sampling is used, where sampling_proportion specifies
            the proportion of voxels to be used. The default is None.

        Notes
        -----
        Since we use linear interpolation, images are not, in general,
        differentiable at exact voxel coordinates, but they are differentiable
        between voxel coordinates. When using sparse sampling, selected voxels
        are slightly moved by adding a small random displacement within one
        voxel to prevent sampling points to be located exactly at voxel
        coordinates. When using dense sampling, this random displacement is
        not applied.
        """
        super(MattesMIMetric, self).__init__(nbins)
        self.sampling_proportion = sampling_proportion

    def setup(self, transform, static, moving, static_grid2world=None,
              moving_grid2world=None, prealign=None):
        r""" Prepares the metric to compute intensity densities and gradients

        The histograms will be setup to compute probability densities of
        intensities within the minimum and maximum values of `static` and
        `moving`

        Parameters
        ----------
        transform: instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        static : array, shape (S, R, C) or (R, C)
            static image
        moving : array, shape (S', R', C') or (R', C')
            moving image
        static_grid2world : array (dim+1, dim+1), optional
            the grid-to-space transform of the static image. The default is
            None, implying the transform is the identity.
        moving_grid2world : array (dim+1, dim+1)
            the grid-to-space transform of the moving image. The default is
            None, implying the spacing along all axes is 1.
        prealign : array, shape (dim+1, dim+1), optional
            the pre-aligning matrix (an affine transform) that roughly aligns
            the moving image towards the static image. If None, no
            pre-alignment is performed. If a pre-alignment matrix is available,
            it is recommended to directly provide the transform to the
            MattesMIMetric instead of manually warping the moving image and
            provide None or identity as prealign. This way, the metric avoids
            performing more than one interpolation. The default is None,
            implying no pre-alignment is performed.
        """
        self.dim = len(static.shape)
        if moving_grid2world is None:
            moving_grid2world = np.eye(self.dim + 1)
        if static_grid2world is None:
            static_grid2world = np.eye(self.dim + 1)
        self.transform = transform
        self.static = np.array(static).astype(np.float64)
        self.moving = np.array(moving).astype(np.float64)
        self.static_grid2world = static_grid2world
        self.static_world2grid = npl.inv(static_grid2world)
        self.moving_grid2world = moving_grid2world
        self.moving_world2grid = npl.inv(moving_grid2world)
        self.static_direction, self.static_spacing = \
            get_direction_and_spacings(static_grid2world, self.dim)
        self.moving_direction, self.moving_spacing = \
            get_direction_and_spacings(moving_grid2world, self.dim)
        self.prealign = prealign

        P = np.eye(self.dim + 1) if self.prealign is None else self.prealign
        if self.dim == 2:
            self.interp_method = vf.interpolate_scalar_2d
        else:
            self.interp_method = vf.interpolate_scalar_3d

        if self.sampling_proportion is None:
            self.warped = transform_image(self.static, self.static_grid2world,
                                   self.moving, self.moving_grid2world, P)
            self.warped = self.warped.astype(np.float64)
            self.samples = None
            self.ns = 0
        else:
            static_32 = np.array(static).astype(np.float32)
            moving_32 = np.array(moving).astype(np.float32)
            self.warped = None
            k = int(np.ceil(1.0 / self.sampling_proportion))
            shape = np.array(static.shape, dtype=np.int32)
            self.samples = sample_domain_regular(k, shape, static_grid2world)
            self.samples = np.array(self.samples)
            self.ns = self.samples.shape[0]
            # Add a column of ones (homogeneous coordinates)
            self.samples = np.hstack((self.samples, np.ones(self.ns)[:, None]))
            # Sample the static image
            static_p = self.static_world2grid.dot(self.samples.T).T
            static_p = static_p[..., :self.dim]
            self.static_vals, inside = self.interp_method(static_32, static_p)
            self.static_vals = np.array(self.static_vals, dtype=np.float64)
            # Sample the moving image
            sp_to_moving = self.moving_world2grid.dot(P)
            moving_p = sp_to_moving.dot(self.samples.T).T
            moving_p = moving_p[..., :self.dim]
            self.moving_vals, inside = self.interp_method(moving_32,
                                                          moving_p)

        MattesBase.setup(self, self.static, self.moving)

    def _update_dense(self, xopt, update_gradient=True):
        r""" Updates marginal and joint distributions and the joint gradient

        The distributions are updated according to the static and warped
        images. The warped image is precisely the moving image after
        transforming it by the transform defined by the xopt parameters.

        The gradient of the joint PDF is computed only if update_gradient
        is True.

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        update_gradient : Boolean, optional
            if True, the gradient of the joint PDF will also be computed,
            otherwise, only the marginal and joint PDFs will be computed.
            The default is True.
        """
        # Get the matrix associated with the xopt parameter vector
        T = self.transform.param_to_matrix(xopt)
        if self.prealign is not None:
            T = T.dot(self.prealign)

        # Warp the moving image
        self.warped = transform_image(self.static, self.static_grid2world,
                               self.moving, self.moving_grid2world, T)
        self.warped = self.warped.astype(np.float64)

        # Update the joint and marginal intensity distributions
        self.update_pdfs_dense(self.static, self.warped, None, None)

        # Compute the gradient of the joint PDF w.r.t. parameters
        if update_gradient:
            if self.static_grid2world is None:
                grid_to_world = T
            else:
                grid_to_world = T.dot(self.static_grid2world)

            # Compute the gradient of the moving image at the current transform
            grid_to_world = T.dot(self.static_grid2world)
            self.grad_w, inside = vf.gradient(self.moving.astype(np.float32),
                                              self.moving_world2grid,
                                              self.moving_spacing,
                                              self.static.shape,
                                              grid_to_world)

            # Update the gradient of the metric
            self.update_gradient_dense(xopt, self.transform, self.static,
                                       self.warped, grid_to_world, self.grad_w,
                                       None, None)

        # Evaluate the mutual information and its gradient
        # The results are in self.metric_val and self.metric_grad
        # ready to be returned from 'distance' and 'gradient'
        self.update_mi_metric(update_gradient)

    def _update_sparse(self, xopt, update_gradient=True):
        r""" Updates the marginal and joint distributions and the joint gradient

        The distributions are updated according to the samples taken from the
        static and moving images. The samples are points in physical space,
        so the static intensities are always the same, but the corresponding
        points in the moving image depend on the transform defined by xopt.

        The gradient of the joint PDF is computed only if update_gradient
        is True.

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        update_gradient : Boolean, optional
            if True, the gradient of the joint PDF will also be computed,
            otherwise, only the marginal and joint PDFs will be computed.
            The default is True.
        """
        # Get the matrix associated with the xopt parameter vector
        T = self.transform.param_to_matrix(xopt)
        if self.prealign is not None:
            T = T.dot(self.prealign)

        # Sample the moving image
        sp_to_moving = self.moving_world2grid.dot(T)
        points_on_moving = sp_to_moving.dot(self.samples.T).T
        points_on_moving = points_on_moving[..., :self.dim]
        moving_32 = self.moving.astype(np.float32)
        self.moving_vals, inside = self.interp_method(moving_32,
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
                                        self.samples[..., :self.dim],
                                        mgrad)

        # Evaluate the mutual information and its gradient
        # The results are in self.metric_val and self.metric_grad
        # ready to be returned from 'distance' and 'gradient'
        self.update_mi_metric(update_gradient)

    def distance(self, xopt):
        r""" Numeric value of the negative Mutual Information

        We need to change the sign so we can use standard minimization
        algorithms.

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform

        Returns
        -------
        val : float
            the negative mutual information of the input images after warping
            the moving image by the currently set transform with `xopt`
            parameters
        """
        if self.samples is None:
            self._update_dense(xopt, False)
        else:
            self._update_sparse(xopt, False)
        return -1 * self.metric_val

    def gradient(self, xopt):
        r""" Numeric value of the metric's gradient at the given parameters

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform

        Returns
        -------
        grad : array, shape (n,)
            the gradient of the negative Mutual Information
        """
        if self.samples is None:
            self._update_dense(xopt, True)
        else:
            self._update_dense(xopt, True)
        return self.metric_grad * (-1)

    def value_and_gradient(self, xopt):
        r""" Numeric value of the metric and its gradient at given parameters

        Parameters
        ----------
        xopt : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform

        Returns
        -------
        val : float
            the negative mutual information of the input images after warping
            the moving image by the currently set transform with `xopt`
            parameters
        grad : array, shape (n,)
            the gradient of the negative Mutual Information
        """
        if self.samples is None:
            self._update_dense(xopt, True)
        else:
            self._update_sparse(xopt, True)
        return -1 * self.metric_val, self.metric_grad * (-1)


class AffineRegistration(object):
    def __init__(self,
                 metric=None,
                 level_iters=None,
                 sigmas=None,
                 factors=None,
                 method='L-BFGS-B',
                 ss_sigma_factor=None,
                 options=None):
        r""" Initializes an instance of the AffineRegistration class

        Parameters
        ----------
        metric : object, optional
            an instance of a metric. The default is None, implying
            the Mutual Information metric with default settings.
        level_iters : list, optional
            the number of iterations at each level of the Gaussian pyramid.
            `level_iters[0]` corresponds to the coarsest level,
            `level_iters[-1]` the finest, where n is the length of the list.
            By default, a 3-level Gaussian pyramid with iterations list
            equal to [10000, 1000, 100] will be used.
        sigmas : list of floats, optional
            custom smoothing parameter to build the scale space (one parameter
            for each scale). By default, the list of sigmas will be [3, 1, 0].
        factors : list of floats, optional
            custom scale factors to build the scale space (one factor for each
            scale). By default, the list of factors will be [4, 2, 1].
        method : string, optional
            optimization method to be used. The default is 'L-BFGS-B'.
        ss_sigma_factor : float, optional
            If None, this parameter is not used and an isotropic Gaussian
            Pyramid with the given `factors` and `sigmas` will be built.
            If not None, an anisotropic Gaussian pyramid will be used by
            automatically selecting the smoothing sigmas along each axis
            according to the voxel dimensions of the given image.
            The `ss_sigma_factor` is used to scale automatically computed
            sigmas. For example, in the isotropic case, the sigma of the
            kernel will be $factor * (2 ^ i)$ where i = 0, 1, ..., n_scales
            is the scale. The default is None.
        options : dict, optional
            extra optimization options. The default is None, implying
            no extra options are passed to the optimizer.
        """

        self.metric = metric

        if self.metric is None:
            self.metric = MattesMIMetric()

        if level_iters is None:
            level_iters = [10000, 1000, 100]
        self.level_iters = level_iters
        self.levels = len(level_iters)
        if self.levels == 0:
            raise ValueError('The iterations list cannot be empty')

        self.options = options
        self.method = method
        if ss_sigma_factor is not None:
            self.use_isotropic = False
            self.ss_sigma_factor = ss_sigma_factor
        else:
            self.use_isotropic = True
            if factors is None:
                factors = [4, 2, 1]
            if sigmas is None:
                sigmas = [3, 1, 0]
            self.factors = factors
            self.sigmas = sigmas

    def _init_optimizer(self, static, moving, transform, x0,
                        static_grid2world, moving_grid2world, prealign):
        r"""Initializes the registration optimizer

        Initializes the optimizer by computing the scale space of the input
        images

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.
        moving : array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the 'prealign'
            matrix
        transform : instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        x0 : array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.
        static_grid2world : array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated with the static image
        moving_grid2world : array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated with the moving image
        prealign : string, or matrix, or None
            If string:
                'mass': align centers of gravity
                'origins': align physical coordinates of voxel (0,0,0)
                'centers': align physical coordinates of central voxels
            If matrix:
                array, shape (dim+1, dim+1)
            If None:
                Start from identity
        """
        self.dim = len(static.shape)
        self.transform = transform
        n = transform.get_number_of_parameters()
        self.nparams = n

        if x0 is None:
            x0 = self.transform.get_identity_parameters()
        self.x0 = x0
        if prealign is None:
            self.prealign = np.eye(self.dim + 1)
        elif prealign == 'mass':
            self.prealign = align_centers_of_mass(static, static_grid2world,
                                                  moving, moving_grid2world)
        elif prealign == 'origins':
            self.prealign = align_origins(static, static_grid2world, moving,
                                          moving_grid2world)
        elif prealign == 'centers':
            self.prealign = align_geometric_centers(static, static_grid2world,
                                                    moving, moving_grid2world)
        elif isinstance(prealign, np.ndarray) and prealign.shape == (n,):
            self.prealign = prealign
        else:
            raise ValueError('Invalid prealign matrix')
        # Extract information from affine matrices to create the scale space
        static_direction, static_spacing = \
            get_direction_and_spacings(static_grid2world, self.dim)
        moving_direction, moving_spacing = \
            get_direction_and_spacings(moving_grid2world, self.dim)

        static = ((static.astype(np.float64) - static.min()) /
                  (static.max() - static.min()))
        moving = ((moving.astype(np.float64) - moving.min()) /
                  (moving.max() - moving.min()))

        # Build the scale space of the input images
        if self.use_isotropic:
            self.moving_ss = IsotropicScaleSpace(moving, self.factors,
                                                 self.sigmas,
                                                 moving_grid2world,
                                                 moving_spacing, False)

            self.static_ss = IsotropicScaleSpace(static, self.factors,
                                                 self.sigmas,
                                                 static_grid2world,
                                                 static_spacing, False)
        else:
            self.moving_ss = ScaleSpace(moving, self.levels, moving_grid2world,
                                        moving_spacing, self.ss_sigma_factor,
                                        False)

            self.static_ss = ScaleSpace(static, self.levels, static_grid2world,
                                        static_spacing, self.ss_sigma_factor,
                                        False)

    def optimize(self, static, moving, transform, x0, static_grid2world=None,
                 moving_grid2world=None, prealign=None):
        r''' Starts the optimization process

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.
        moving : array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the 'prealign'
            matrix
        transform : instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        x0 : array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.
        static_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the static
            image. The default is None, implying the transform is the
            identity.
        moving_grid2world : array, shape (dim+1, dim+1), optional
            the voxel-to-space transformation associated with the moving
            image. The default is None, implying the transform is the
            identity.
        prealign : string, or matrix, or None, optional
            If string:
                'mass': align centers of gravity
                'origins': align physical coordinates of voxel (0,0,0)
                'centers': align physical coordinates of central voxels
            If matrix:
                array, shape (dim+1, dim+1).
            If None:
                Start from identity.
            The default is None.

        Returns
        -------
        T : array, shape (dim+1, dim+1)
            the matrix representing the optimized affine transform
        '''
        self._init_optimizer(static, moving, transform, x0, static_grid2world,
                             moving_grid2world, prealign)
        del prealign  # Now we must refer to self.prealign

        # Multi-resolution iterations
        original_static_grid2world = self.static_ss.get_affine(0)
        original_moving_grid2world = self.moving_ss.get_affine(0)

        for level in range(self.levels - 1, -1, -1):
            self.current_level = level
            max_iter = self.level_iters[level]
            print('Optimizing level %d [max iter: %d]' % (level, max_iter))

            # Resample the smooth static image to the shape of this level
            smooth_static = self.static_ss.get_image(level)
            current_static_shape = self.static_ss.get_domain_shape(level)
            current_static_grid2world = self.static_ss.get_affine(level)

            current_static = transform_image(tuple(current_static_shape),
                                      current_static_grid2world, smooth_static,
                                      original_static_grid2world, None, False)

            # The moving image is full resolution
            current_moving_grid2world = original_moving_grid2world
            current_moving_spacing = self.moving_ss.get_spacing(level)

            current_moving = self.moving_ss.get_image(level)

            # Prepare the metric for iterations at this resolution
            self.metric.setup(transform, current_static, current_moving,
                              current_static_grid2world,
                              current_moving_grid2world, self.prealign)

            # Optimize this level
            if self.options is None:
                self.options = {'gtol': 1e-4,
                                'disp': False}

            if self.method == 'L-BFGS-B':
                self.options['maxfun'] = max_iter
            else:
                self.options['maxiter'] = max_iter

            if SCIPY_LESS_0_12:
                # Older versions don't expect value and gradient from
                # the same function
                opt = Optimizer(self.metric.distance, self.x0,
                                method=self.method, jac=self.metric.gradient,
                                options=self.options)
            else:
                opt = Optimizer(self.metric.value_and_gradient, self.x0,
                                method=self.method, jac=True,
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


def transform_image(static, static_grid2world, moving, moving_grid2world, transform,
             nn=False):
    r""" Warps the moving image towards the static using the given transform

    Parameters
    ----------
    static : array, shape (S, R, C)
        static image: it will provide the grid and grid-to-space transform for
        the warped image
    static_grid2world : array, shape (dim+1, dim+1)
        grid-to-space transform associated with the static image
    moving : array, shape (S', R', C')
        moving image
    moving_grid2world : array, shape (dim+1, dim+1)
        grid-to-space transform associated with the moving image
    transform : array, shape (dim+1, dim+1)
        the matrix representing the affine transform to be applied to `moving`
    nn : Boolean, optional
        if False, trilinear interpolation will be used. If True, nearest
        neighbour interpolation will be used instead. The default is False,
        implying trilinear interpolation.
    Returns
    -------
    warped : array, shape (S, R, C)
        the warped image
    """
    if type(static) is tuple:
        dim = len(static)
        shape = np.array(static, dtype=np.int32)
    else:
        dim = len(static.shape)
        shape = np.array(static.shape, dtype=np.int32)
    if nn:
        input = np.array(moving, dtype=np.int32)
        if dim == 2:
            warp_method = vf.warp_2d_affine_nn
        elif dim == 3:
            warp_method = vf.warp_3d_affine_nn
    else:
        input = np.array(moving, dtype=floating)
        if dim == 2:
            warp_method = vf.warp_2d_affine
        elif dim == 3:
            warp_method = vf.warp_3d_affine
    if moving_grid2world is not None:
        m_world2grid = npl.inv(moving_grid2world)
    else:
        m_world2grid = np.eye(dim + 1)

    if static_grid2world is None:
        static_grid2world = np.eye(dim + 1)

    if transform is None:
        composition = m_world2grid.dot(static_grid2world)
    else:
        composition = m_world2grid.dot(transform.dot(static_grid2world))

    warped = warp_method(input, shape, composition)

    return np.array(warped)


def align_centers_of_mass(static, static_grid2world, moving, moving_grid2world):
    r""" Transformation to align the center of mass of the input images

    Parameters
    ----------
    static : array, shape (S, R, C)
        static image
    static_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the static image
    moving : array, shape (S, R, C)
        moving image
    moving_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape (dim+1, dim+1)
        the affine transformation (translation only, in this case) aligning
        the center of mass of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    if static_grid2world is None:
        static_grid2world = np.eye(dim + 1)
    if moving_grid2world is None:
        moving_grid2world = np.eye(dim + 1)
    c_static = ndimage.measurements.center_of_mass(np.array(static))
    c_static = static_grid2world.dot(c_static+(1,))
    c_moving = ndimage.measurements.center_of_mass(np.array(moving))
    c_moving = moving_grid2world.dot(c_moving+(1,))
    transform = np.eye(dim + 1)
    transform[:dim, dim] = (c_moving - c_static)[:dim]
    return transform


def align_geometric_centers(static, static_grid2world, moving,
                          moving_grid2world):
    r""" Transformation to align the geometric center of the input images

    With "geometric center" of a volume we mean the physical coordinates of
    its central voxel

    Parameters
    ----------
    static : array, shape (S, R, C)
        static image
    static_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the static image
    moving : array, shape (S, R, C)
        moving image
    moving_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape (dim+1, dim+1)
        the affine transformation (translation only, in this case) aligning
        the geometric center of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    if static_grid2world is None:
        static_grid2world = np.eye(dim + 1)
    if moving_grid2world is None:
        moving_grid2world = np.eye(dim + 1)
    c_static = tuple((np.array(static.shape, dtype=np.float64)) * 0.5)
    c_static = static_grid2world.dot(c_static+(1,))
    c_moving = tuple((np.array(moving.shape, dtype=np.float64)) * 0.5)
    c_moving = moving_grid2world.dot(c_moving+(1,))
    transform = np.eye(dim + 1)
    transform[:dim, dim] = (c_moving - c_static)[:dim]
    return transform


def align_origins(static, static_grid2world, moving, moving_grid2world):
    r""" Transformation to align the origins of the input images

    With "origin" of a volume we mean the physical coordinates of
    voxel (0,0,0)

    Parameters
    ----------
    static : array, shape (S, R, C)
        static image
    static_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the static image
    moving : array, shape (S, R, C)
        moving image
    moving_grid2world : array, shape (dim+1, dim+1)
        the voxel-to-space transformation of the moving image

    Returns
    -------
    transform : array, shape (dim+1, dim+1)
        the affine transformation (translation only, in this case) aligning
        the origin of the moving image towards the one of the static
        image
    """
    dim = len(static.shape)
    if static_grid2world is None:
        static_grid2world = np.eye(dim + 1)
    if moving_grid2world is None:
        moving_grid2world = np.eye(dim + 1)
    c_static = static_grid2world[:dim, dim]
    c_moving = moving_grid2world[:dim, dim]
    transform = np.eye(dim + 1)
    transform[:dim, dim] = (c_moving - c_static)[:dim]
    return transform
