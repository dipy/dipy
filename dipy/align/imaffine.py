""" Affine image registration module consisting of the following classes:

    AffineMap: encapsulates the necessary information to perform affine
        transforms between two domains, defined by a `static` and a `moving`
        image. The `domain` of the transform is the set of points in the
        `static` image's grid, and the `codomain` is the set of points in
        the `moving` image. When we call the `transform` method, `AffineMap`
        maps each point `x` of the domain (`static` grid) to the codomain
        (`moving` grid) and interpolates the `moving` image at that point
        to obtain the intensity value to be placed at `x` in the resulting
        grid. The `transform_inverse` method performs the opposite operation
        mapping points in the codomain to points in the domain.

    ParzenJointHistogram: computes the marginal and joint distributions of
        intensities of a pair of images, using Parzen windows [Parzen62]
        with a cubic spline kernel, as proposed by Mattes et al. [Mattes03].
        It also computes the gradient of the joint histogram w.r.t. the
        parameters of a given transform.

    ParzenMutualInformation: inherits from `ParzenJointHistogram` and
        computes the Mutual Information similarity measure and its gradient
        from the joint and marginal distributions of paired intensities,
        which are computed by the super class.

    MIMetric: it is a Python interface to the ParzenMutualInformation cython
        class, which is used to compute the value and gradient the way
        `Optimizer` needs them. That is, given a set of transform parameters,
        it will use `ParzenMutualInformation` methods to compute the value and
        gradient evaluated at the given parameters.

    AffineRegistration: it runs the multi-resolution registration, putting
        all the pieces together. It needs to create the scale space of the
        images and run the multi-resolution registration by using the Metric
        and the Optimizer at each level of the Gaussian pyramid. At each
        level, it will setup the metric to compute value and gradient of the
        metric with the input images with different levels of smoothing.

    References
    ----------
    [Parzen62] E. Parzen. On the estimation of a probability density
               function and the mode. Annals of Mathematical Statistics,
               33(3), 1065-1076, 1962.
    [Mattes03] Mattes, D., Haynor, D. R., Vesselle, H., Lewellen, T. K.,
               & Eubank, W. PET-CT image registration in the chest using
               free-form deformations. IEEE Transactions on Medical
               Imaging, 22(1), 120-8, 2003.
"""

import numpy as np
import numpy.linalg as npl
import scipy.ndimage as ndimage
from ..core.optimize import Optimizer
from ..core.optimize import SCIPY_LESS_0_12
from . import vector_fields as vf
from . import VerbosityLevels
from .parzenhist import ParzenMutualInformation, sample_domain_regular
from .imwarp import (get_direction_and_spacings, ScaleSpace)
from .scalespace import IsotropicScaleSpace


class AffineMap(object):
    def __init__(self, affine, domain_grid_shape=None, domain_grid2world=None,
                 codomain_grid_shape=None, codomain_grid2world=None):
        """ AffineMap

        Implements an affine transformation whose domain is given by
        `domain_grid` and `domain_grid2world`, and whose co-domain is
        given by `codomain_grid` and `codomain_grid2world`. This domain
        and co-domain information is used as default sampling information
        to transform images. If such domain/co-domain information is not
        provided (None) then the sampling information needs to be specified
        each time the `transform` or `transform_inverse` is called to
        transform images. Note that such sampling information is not
        necessary to transform points defined in physical space, such as
        stream lines.

        Parameters
        ----------
        affine : array, shape (dim + 1, dim + 1)
            the matrix defining the affine transform, where `dim` is the
            dimension of the space this map operates in (2 for 2D images,
            3 for 3D images)
        domain_grid_shape : sequence, shape (dim,), optional
            the shape of the default domain sampling grid. When `transform`
            is called to transform an image, the resulting image will have
            this shape, unless a different sampling information is provided.
            If None, then the sampling grid shape must be specified each time
            the `transform` method is called.
        domain_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with the domain grid.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.
        codomain_grid_shape : sequence of integers, shape (dim,)
            the shape of the default co-domain sampling grid. When
            `transform_inverse` is called to transform an image, the resulting
            image will have this shape, unless a different sampling
            information is provided. If None (the default), then the sampling
            grid shape must be specified each time the `transform_inverse`
            method is called.
        codomain_grid2world : array, shape (dim + 1, dim + 1)
            the grid-to-world transform associated with the co-domain grid.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.
        """
        self.set_affine(affine)
        self.domain_shape = domain_grid_shape
        self.domain_grid2world = domain_grid2world
        self.codomain_shape = codomain_grid_shape
        self.codomain_grid2world = codomain_grid2world

    def set_affine(self, affine):
        """ Sets the affine transform (operating in physical space)

        Parameters
        ----------
        affine : array, shape (dim + 1, dim + 1)
            the matrix representing the affine transform operating in
            physical space. The domain and co-domain information
            remains unchanged
        """
        self.affine = affine
        self.affine_inv = None
        if affine is not None:
            self.affine_inv = npl.inv(affine)

    def _apply_transform(self, image, interp='linear', image_grid2world=None,
                         sampling_grid_shape=None, sampling_grid2world=None,
                         resample_only=False, apply_inverse=False):
        """ Transforms the input image applying this affine transform

        This is a generic function to transform images using either this
        (direct) transform or its inverse.

        If applying the direct transform (`apply_inverse=False`):
            by default, the transformed image is sampled at a grid defined by
            `self.domain_shape` and `self.domain_grid2world`.
        If applying the inverse transform (`apply_inverse=True`):
            by default, the transformed image is sampled at a grid defined by
            `self.codomain_shape` and `self.codomain_grid2world`.

        If the sampling information was not provided at initialization of this
        transform then `sampling_grid_shape` is mandatory.

        Parameters
        ----------
        image : array, shape (X, Y) or (X, Y, Z)
            the image to be transformed
        interp : string, either 'linear' or 'nearest'
            the type of interpolation to be used, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        image_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with `image`.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.
        sampling_grid_shape : sequence, shape (dim,), optional
            the shape of the grid where the transformed image must be sampled.
            If None (the default), then `self.domain_shape` is used instead
            (which must have been set at initialization, otherwise an exception
            will be raised).
        sampling_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with the sampling grid
            (specified by `sampling_grid_shape`, or by default
            `self.domain_shape`). If None (the default), then the
            grid-to-world transform is assumed to be the identity.
        resample_only : Boolean, optional
            If False (the default) the affine transform is applied normally.
            If True, then the affine transform is not applied, and the input
            image is just re-sampled on the domain grid of this transform.
        apply_inverse : Boolean, optional
            If False (the default) the image is transformed from the codomain
            of this transform to its domain using the (direct) affine
            transform. Otherwise, the image is transformed from the domain
            of this transform to its codomain using the (inverse) affine
            transform.
        Returns
        -------
        transformed : array, shape `sampling_grid_shape` or `self.domain_shape`
            the transformed image, sampled at the requested grid
        """
        # Obtain sampling grid
        if sampling_grid_shape is None:
            if apply_inverse:
                sampling_grid_shape = self.codomain_shape
            else:
                sampling_grid_shape = self.domain_shape
        if sampling_grid_shape is None:
            msg = 'Unknown sampling info. Provide a valid sampling_grid_shape'
            raise ValueError(msg)

        dim = len(sampling_grid_shape)
        shape = np.array(sampling_grid_shape, dtype=np.int32)

        # Obtain grid-to-world transform for sampling grid
        if sampling_grid2world is None:
            if apply_inverse:
                sampling_grid2world = self.codomain_grid2world
            else:
                sampling_grid2world = self.domain_grid2world
        if sampling_grid2world is None:
            sampling_grid2world = np.eye(dim + 1)

        # Obtain world-to-grid transform for input image
        if image_grid2world is None:
            if apply_inverse:
                image_grid2world = self.domain_grid2world
            else:
                image_grid2world = self.codomain_grid2world
            if image_grid2world is None:
                image_grid2world = np.eye(dim + 1)
        image_world2grid = npl.inv(image_grid2world)

        # Obtain the appropriate transform method
        if interp == 'nearest':
            if dim == 2:
                transform_method = vf.warp_2d_affine_nn
            elif dim == 3:
                transform_method = vf.warp_3d_affine_nn
        elif interp == 'linear':
            image = image.astype(np.float64)
            if dim == 2:
                transform_method = vf.warp_2d_affine
            elif dim == 3:
                transform_method = vf.warp_3d_affine
        else:
            raise ValueError('Unknown interpolation method: '+str(interp))

        # Compute the transform from sampling grid to input image grid
        if apply_inverse:
            aff = self.affine_inverse
        else:
            aff = self.affine

        if (aff is None) or resample_only:
            comp = image_world2grid.dot(sampling_grid2world)
        else:
            comp = image_world2grid.dot(self.affine.dot(sampling_grid2world))

        # Transform the input image
        transformed = transform_method(image, shape, comp)
        return transformed

    def transform(self, image, interp='linear', image_grid2world=None,
                  sampling_grid_shape=None, sampling_grid2world=None,
                  resample_only=False):
        """ Transforms the input image from co-domain to domain space

        By default, the transformed image is sampled at a grid defined by
        `self.domain_shape` and `self.domain_grid2world`. If such
        information was not provided then `sampling_grid_shape` is mandatory.

        Parameters
        ----------
        image : array, shape (X, Y) or (X, Y, Z)
            the image to be transformed
        interp : string, either 'linear' or 'nearest'
            the type of interpolation to be used, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        image_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with `image`.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.
        sampling_grid_shape : sequence, shape (dim,), optional
            the shape of the grid where the transformed image must be sampled.
            If None (the default), then `self.codomain_shape` is used instead
            (which must have been set at initialization, otherwise an exception
            will be raised).
        sampling_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with the sampling grid
            (specified by `sampling_grid_shape`, or by default
            `self.codomain_shape`). If None (the default), then the
            grid-to-world transform is assumed to be the identity.
        resample_only : Boolean, optional
            If False (the default) the affine transform is applied normally.
            If True, then the affine transform is not applied, and the input
            image is just re-sampled on the domain grid of this transform.
        Returns
        -------
        transformed : array, shape `sampling_grid_shape` or
                      `self.codomain_shape`
            the transformed image, sampled at the requested grid
        """
        transformed = self._apply_transform(image, interp, image_grid2world,
                                            sampling_grid_shape,
                                            sampling_grid2world,
                                            resample_only,
                                            apply_inverse=False)
        return np.array(transformed)

    def transform_inverse(self, image, interp='linear', image_grid2world=None,
                          sampling_grid_shape=None, sampling_grid2world=None,
                          resample_only=False):
        """ Transforms the input image from domain to co-domain space

        By default, the transformed image is sampled at a grid defined by
        `self.codomain_shape` and `self.codomain_grid2world`. If such
        information was not provided then `sampling_grid_shape` is mandatory.

        Parameters
        ----------
        image : array, shape (X, Y) or (X, Y, Z)
            the image to be transformed
        interp : string, either 'linear' or 'nearest'
            the type of interpolation to be used, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        image_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with `image`.
            If None (the default), then the grid-to-world transform is assumed
            to be the identity.
        sampling_grid_shape : sequence, shape (dim,), optional
            the shape of the grid where the transformed image must be sampled.
            If None (the default), then `self.codomain_shape` is used instead
            (which must have been set at initialization, otherwise an exception
            will be raised).
        sampling_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-world transform associated with the sampling grid
            (specified by `sampling_grid_shape`, or by default
            `self.codomain_shape`). If None (the default), then the
            grid-to-world transform is assumed to be the identity.
        resample_only : Boolean, optional
            If False (the default) the affine transform is applied normally.
            If True, then the affine transform is not applied, and the input
            image is just re-sampled on the domain grid of this transform.
        Returns
        -------
        transformed : array, shape `sampling_grid_shape` or
                      `self.codomain_shape`
            the transformed image, sampled at the requested grid
        """
        transformed = self._apply_transform(image, interp, image_grid2world,
                                            sampling_grid_shape,
                                            sampling_grid2world,
                                            resample_only,
                                            apply_inverse=True)
        return np.array(transformed)


class MIMetric(ParzenMutualInformation):
    def __init__(self, nbins=32, sampling_proportion=None):
        r""" Initializes an instance of the Mutual Information metric

        This class implements the methods required by Optimizer to drive the
        registration process by making calls to the low level methods defined
        in ParzenMutualInformation.

        Parameters
        ----------
        nbins : int, optional
            the number of bins to be used for computing the intensity
            histograms. The default is 32.
        sampling_proportion : None or float in interval (0, 1], optional
            There are two types of sampling: dense and sparse. Dense sampling
            uses all voxels for estimating the (joint and marginal) intensity
            histograms, while sparse sampling uses a subset of them. If
            `sampling_proportion` is None, then dense sampling is
            used. If `sampling_proportion` is a floating point value in (0,1]
            then sparse sampling is used, where `sampling_proportion`
            specifies the proportion of voxels to be used. The default is
            None.

        Notes
        -----
        Since we use linear interpolation, images are not, in general,
        differentiable at exact voxel coordinates, but they are differentiable
        between voxel coordinates. When using sparse sampling, selected voxels
        are slightly moved by adding a small random displacement within one
        voxel to prevent sampling points from being located exactly at voxel
        coordinates. When using dense sampling, this random displacement is
        not applied.
        """
        super(MIMetric, self).__init__(nbins)
        self.sampling_proportion = sampling_proportion

    def setup(self, transform, static, moving, static_grid2world=None,
              moving_grid2world=None, starting_affine=None):
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
            moving image. The dimensions of the static (S, R, C) and moving
            (S', R', C') images do not need to be the same.
        static_grid2world : array (dim+1, dim+1), optional
            the grid-to-space transform of the static image. The default is
            None, implying the transform is the identity.
        moving_grid2world : array (dim+1, dim+1)
            the grid-to-space transform of the moving image. The default is
            None, implying the spacing along all axes is 1.
        starting_affine : array, shape (dim+1, dim+1), optional
            the pre-aligning matrix (an affine transform) that roughly aligns
            the moving image towards the static image. If None, no
            pre-alignment is performed. If a pre-alignment matrix is available,
            it is recommended to provide this matrix as `starting_affine`
            instead of manually transforming the moving image to reduce
            interpolation artifacts. The default is None, implying no
            pre-alignment is performed.
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
        self.starting_affine = starting_affine

        P = np.eye(self.dim + 1)
        if self.starting_affine is not None:
            P = self.starting_affine

        self.affine_map = AffineMap(P, static.shape, static_grid2world,
                                    moving.shape, moving_grid2world)

        if self.dim == 2:
            self.interp_method = vf.interpolate_scalar_2d
        else:
            self.interp_method = vf.interpolate_scalar_3d

        if self.sampling_proportion is None:
            self.update_pdfs = self.update_pdfs_dense
            self.update_gradient = self.update_gradient_dense
            self.samples = None
            self.ns = 0
            self.transformed = self.affine_map.transform(self.moving)
            self.transformed = self.transformed.astype(np.float64)
        else:
            self.update_pdfs = self.update_pdfs_sparse
            self.update_gradient = self.update_gradient_sparse
            self.transformed = None
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
            self.static_vals, inside = self.interp_method(static, static_p)
            self.static_vals = np.array(self.static_vals, dtype=np.float64)
            # Sample the moving image
            sp_to_moving = self.moving_world2grid.dot(P)
            moving_p = sp_to_moving.dot(self.samples.T).T
            moving_p = moving_p[..., :self.dim]
            self.moving_vals, inside = self.interp_method(moving, moving_p)
        super(MIMetric, self).setup(self.static, self.moving)

    def _update(self, params, update_gradient=True):
        r""" Updates marginal and joint distributions and the joint gradient

        The distributions are updated according to the static and transformed
        images. The transformed image is precisely the moving image after
        transforming it by the transform defined by the `params` parameters.

        The gradient of the joint PDF is computed only if update_gradient
        is True.

        Parameters
        ----------
        params : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform
        update_gradient : Boolean, optional
            if True, the gradient of the joint PDF will also be computed,
            otherwise, only the marginal and joint PDFs will be computed.
            The default is True.
        """
        # Get the matrix associated with the params parameter vector
        M = self.transform.param_to_matrix(params)
        if self.starting_affine is not None:
            M = M.dot(self.starting_affine)
        try:
            self.affine_map.set_affine(M)
        except:
            raise ValueError('The transformation is not invertible')



        # Update the joint and marginal intensity distributions
        if self.samples is None:
            # Warp the moving image (dense case)
            self.transformed = self.affine_map.transform(self.moving)
            self.transformed = self.transformed.astype(np.float64)
            static_values = self.static
            moving_values = self.transformed
        else:
            # Sample the moving image (sparse case)
            sp_to_moving = self.moving_world2grid.dot(M)
            points_on_moving = sp_to_moving.dot(self.samples.T).T
            points_on_moving = points_on_moving[..., :self.dim]
            self.moving_vals, inside = self.interp_method(self.moving,
                                                          points_on_moving)
            self.moving_vals = np.array(self.moving_vals)
            static_values = self.static_vals
            moving_values = self.moving_vals
        self.update_pdfs(static_values, moving_values)

        # Compute the gradient of the joint PDF w.r.t. parameters
        if update_gradient:
            if self.samples is None:
                # Compute the gradient of moving img. at physical points
                # associated with the >>static image's grid<< cells
                grid_to_world = M.dot(self.static_grid2world)
                mgrad, inside = vf.gradient(self.moving,
                                            self.moving_world2grid,
                                            self.moving_spacing,
                                            self.static.shape,
                                            grid_to_world)
                # Dense case: we just need to provide the grid-to-world
                # transform to obtain the sampling points' world coordinates
                # because we know that all grid cells must be processed
                sampling_info = grid_to_world
            else:
                # Compute the gradient of moving at the sampling points
                # which are already given in physical space coordinates
                mgrad, inside = vf.sparse_gradient(self.moving,
                                                   sp_to_moving,
                                                   self.moving_spacing,
                                                   self.samples)
                # Sparse case: we need to provide the actual coordinates of
                # all sampling points
                sampling_info = self.samples[..., :self.dim]
            self.update_gradient(params, self.transform, static_values,
                                 moving_values, sampling_info, mgrad)

        # Evaluate the mutual information and its gradient
        # The results are in self.metric_val and self.metric_grad
        # ready to be returned from 'distance' and 'gradient'
        self.update_mi_metric(update_gradient)

    def distance(self, params):
        r""" Numeric value of the negative Mutual Information

        We need to change the sign so we can use standard minimization
        algorithms.

        Parameters
        ----------
        params : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform

        Returns
        -------
        neg_mi : float
            the negative mutual information of the input images after
            transforming the moving image by the currently set transform
            with `params` parameters
        """
        try:
            self._update(params, False)
        except:
            return np.inf
        return -1 * self.metric_val

    def gradient(self, params):
        r""" Numeric value of the metric's gradient at the given parameters

        Parameters
        ----------
        params : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform

        Returns
        -------
        grad : array, shape (n,)
            the gradient of the negative Mutual Information
        """
        try:
            self._update(params, True)
        except:
            return 0 * self.metric_grad
        return -1 * self.metric_grad

    def distance_and_gradient(self, params):
        r""" Numeric value of the metric and its gradient at given parameters

        Parameters
        ----------
        params : array, shape (n,)
            the parameter vector of the transform currently used by the metric
            (the transform name is provided when self.setup is called), n is
            the number of parameters of the transform

        Returns
        -------
        neg_mi : float
            the negative mutual information of the input images after
            transforming the moving image by the currently set transform
            with `params` parameters
        neg_mi_grad : array, shape (n,)
            the gradient of the negative Mutual Information
        """
        try:
            self._update(params, True)
        except:
            return np.inf, 0 * self.metric_grad
        return -1 * self.metric_val, -1 * self.metric_grad


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
        metric : None or object, optional
            an instance of a metric. The default is None, implying
            the Mutual Information metric with default settings.
        level_iters : sequence, optional
            the number of iterations at each scale of the scale space.
            `level_iters[0]` corresponds to the coarsest scale,
            `level_iters[-1]` the finest, where n is the length of the
            sequence. By default, a 3-level scale space with iterations
            sequence equal to [10000, 1000, 100] will be used.
        sigmas : sequence of floats, optional
            custom smoothing parameter to build the scale space (one parameter
            for each scale). By default, the sequence of sigmas will be
            [3, 1, 0].
        factors : sequence of floats, optional
            custom scale factors to build the scale space (one factor for each
            scale). By default, the sequence of factors will be [4, 2, 1].
        method : string, optional
            optimization method to be used. If Scipy version < 0.12, then
            only L-BFGS-B is available. Otherwise, `method` can be any
            gradient-based method available in `dipy.core.Optimize`: CG, BFGS,
            Newton-CG, dogleg or trust-ncg.
            The default is 'L-BFGS-B'.
        ss_sigma_factor : float, optional
            If None, this parameter is not used and an isotropic scale
            space with the given `factors` and `sigmas` will be built.
            If not None, an anisotropic scale space will be used by
            automatically selecting the smoothing sigmas along each axis
            according to the voxel dimensions of the given image.
            The `ss_sigma_factor` is used to scale the automatically computed
            sigmas. For example, in the isotropic case, the sigma of the
            kernel will be $factor * (2 ^ i)$ where
            $i = 1, 2, ..., n_scales - 1$ is the scale (the finest resolution
            image $i=0$ is never smoothed). The default is None.
        options : dict, optional
            extra optimization options. The default is None, implying
            no extra options are passed to the optimizer.
        """

        self.metric = metric

        if self.metric is None:
            self.metric = MIMetric()

        if level_iters is None:
            level_iters = [10000, 1000, 100]
        self.level_iters = level_iters
        self.levels = len(level_iters)
        if self.levels == 0:
            raise ValueError('The iterations sequence cannot be empty')

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
        self.verbosity = VerbosityLevels.STATUS

    def _init_optimizer(self, static, moving, transform, params0,
                        static_grid2world, moving_grid2world,
                        starting_affine):
        r"""Initializes the registration optimizer

        Initializes the optimizer by computing the scale space of the input
        images

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.
        moving : array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization. The
            dimensions of the static (S, R, C) and moving (S', R', C') images
            do not need to be the same.
        transform : instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        params0 : array, shape (n,)
            parameters from which to start the optimization. If None, the
            optimization will start at the identity transform. n is the
            number of parameters of the specified transformation.
        static_grid2world : array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated with the static image
        moving_grid2world : array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated with the moving image
        starting_affine : string, or matrix, or None
            If string:
                'mass': align centers of gravity
                'voxel-origin': align physical coordinates of voxel (0,0,0)
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

        if params0 is None:
            params0 = self.transform.get_identity_parameters()
        self.params0 = params0
        if starting_affine is None:
            self.starting_affine = np.eye(self.dim + 1)
        elif starting_affine == 'mass':
            affine_map = align_centers_of_mass(static,
                                               static_grid2world,
                                               moving,
                                               moving_grid2world)
            self.starting_affine = affine_map.affine
        elif starting_affine == 'voxel-origin':
            affine_map = align_origins(static, static_grid2world,
                                       moving, moving_grid2world)
            self.starting_affine = affine_map.affine
        elif starting_affine == 'centers':
            affine_map = align_geometric_centers(static,
                                                 static_grid2world,
                                                 moving,
                                                 moving_grid2world)
            self.starting_affine = affine_map.affine
        elif (isinstance(starting_affine, np.ndarray) and
              starting_affine.shape >= (self.dim, self.dim + 1)):
            self.starting_affine = starting_affine
        else:
            raise ValueError('Invalid starting_affine matrix')
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

    def optimize(self, static, moving, transform, params0,
                 static_grid2world=None, moving_grid2world=None,
                 starting_affine=None):
        r''' Starts the optimization process

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization.
        moving : array, shape (S', R', C') or (R', C')
            the image to be used as "moving" during optimization. It is
            necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the
            'starting_affine' matrix
        transform : instance of Transform
            the transformation with respect to whose parameters the gradient
            must be computed
        params0 : array, shape (n,)
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
        starting_affine : string, or matrix, or None, optional
            If string:
                'mass': align centers of gravity
                'voxel-origin': align physical coordinates of voxel (0,0,0)
                'centers': align physical coordinates of central voxels
            If matrix:
                array, shape (dim+1, dim+1).
            If None:
                Start from identity.
            The default is None.

        Returns
        -------
        affine_map : instance of AffineMap
            the affine resulting affine transformation
        '''
        self._init_optimizer(static, moving, transform, params0,
                             static_grid2world, moving_grid2world,
                             starting_affine)
        del starting_affine  # Now we must refer to self.starting_affine

        # Multi-resolution iterations
        original_static_shape = self.static_ss.get_image(0).shape
        original_static_grid2world = self.static_ss.get_affine(0)
        original_moving_shape = self.moving_ss.get_image(0).shape
        original_moving_grid2world = self.moving_ss.get_affine(0)
        affine_map = AffineMap(None,
                               original_static_shape,
                               original_static_grid2world,
                               original_moving_shape,
                               original_moving_grid2world)

        for level in range(self.levels - 1, -1, -1):
            self.current_level = level
            max_iter = self.level_iters[level]
            if self.verbosity >= VerbosityLevels.STATUS:
                print('Optimizing level %d [max iter: %d]' % (level, max_iter))

            # Resample the smooth static image to the shape of this level
            smooth_static = self.static_ss.get_image(level)
            current_static_shape = self.static_ss.get_domain_shape(level)
            current_static_grid2world = self.static_ss.get_affine(level)

            current_affine_map = AffineMap(None,
                                           current_static_shape,
                                           current_static_grid2world,
                                           original_static_shape,
                                           original_static_grid2world)
            current_static = current_affine_map.transform(smooth_static)

            # The moving image is full resolution
            current_moving_grid2world = original_moving_grid2world

            current_moving = self.moving_ss.get_image(level)

            # Prepare the metric for iterations at this resolution
            self.metric.setup(transform, current_static, current_moving,
                              current_static_grid2world,
                              current_moving_grid2world, self.starting_affine)

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
                opt = Optimizer(self.metric.distance, self.params0,
                                method=self.method, jac=self.metric.gradient,
                                options=self.options)
            else:
                opt = Optimizer(self.metric.distance_and_gradient, self.params0,
                                method=self.method, jac=True,
                                options=self.options)
            params = opt.xopt

            # Update starting_affine matrix with optimal parameters
            T = self.transform.param_to_matrix(params)
            self.starting_affine = T.dot(self.starting_affine)

            # Start next iteration at identity
            self.params0 = self.transform.get_identity_parameters()

            # Update the metric to the current solution
            self.metric._update(params, False)
        affine_map.set_affine(self.starting_affine)
        return affine_map


def align_centers_of_mass(static, static_grid2world,
                          moving, moving_grid2world):
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
    affine_map : instance of AffineMap
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
    affine_map = AffineMap(transform,
                           static.shape, static_grid2world,
                           moving.shape, moving_grid2world)
    return affine_map


def align_geometric_centers(static, static_grid2world,
                            moving, moving_grid2world):
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
    affine_map : instance of AffineMap
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
    affine_map = AffineMap(transform,
                           static.shape, static_grid2world,
                           moving.shape, moving_grid2world)
    return affine_map


def align_origins(static, static_grid2world,
                  moving, moving_grid2world):
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
    affine_map : instance of AffineMap
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
    affine_map = AffineMap(transform,
                           static.shape, static_grid2world,
                           moving.shape, moving_grid2world)
    return affine_map
