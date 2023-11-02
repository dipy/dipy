import logging
from dipy.align import floating
import numpy as np
import numpy.linalg as npl
from scipy.ndimage import gaussian_filter

logger = logging.getLogger(__name__)

class ScaleSpace:
    def __init__(self, image, num_levels,
                 image_grid2world=None,
                 input_spacing=None,
                 sigma_factor=0.2,
                 mask0=False):
        """ ScaleSpace.

        Computes the Scale Space representation of an image. The scale space is
        simply a list of images produced by smoothing the input image with a
        Gaussian kernel with increasing smoothing parameter. If the image's
        voxels are isotropic, the smoothing will be the same along all
        directions: at level L = 0, 1, ..., the sigma is given by
        $s * ( 2^L - 1 )$.
        If the voxel dimensions are not isotropic, then the smoothing is
        weaker along low resolution directions.

        Parameters
        ----------
        image : array, shape (r,c) or (s, r, c) where s is the number of
            slices, r is the number of rows and c is the number of columns of
            the input image.
        num_levels : int
            the desired number of levels (resolutions) of the scale space
        image_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-space transform of the image grid. The default is
            the identity matrix
        input_spacing : array, shape (dim,), optional
            the spacing (voxel size) between voxels in physical space. The
            default is 1.0 along all axes
        sigma_factor : float, optional
            the smoothing factor to be used in the construction of the scale
            space. The default is 0.2
        mask0 : Boolean, optional
            if True, all smoothed images will be zero at all voxels that are
            zero in the input image. The default is False.

        """
        self.dim = len(image.shape)
        self.num_levels = num_levels
        input_size = np.array(image.shape)
        if mask0:
            mask = np.asarray(image > 0, dtype=np.int32)

        # Normalize input image to [0,1]
        img = (image - np.min(image))/(np.max(image) - np.min(image))
        if mask0:
            img *= mask

        # The properties are saved in separate lists. Insert input image
        # properties at the first level of the scale space
        self.images = [img.astype(floating)]
        self.domain_shapes = [input_size.astype(np.int32)]
        if input_spacing is None:
            input_spacing = np.ones((self.dim,), dtype=np.int32)
        self.spacings = [input_spacing]
        self.scalings = [np.ones(self.dim)]
        self.affines = [image_grid2world]
        self.sigmas = [np.zeros(self.dim)]

        if image_grid2world is not None:
            self.affine_invs = [npl.inv(image_grid2world)]
        else:
            self.affine_invs = [None]

        # Compute the rest of the levels
        min_spacing = np.min(input_spacing)
        for i in range(1, num_levels):
            scaling_factor = 2 ** i
            # Note: the minimum below is present in ANTS to prevent the scaling
            # from being too large (making the sub-sampled image to be too
            # small) this makes the sub-sampled image at least 32 voxels at
            # each direction it is risky to make this decision based on image
            # size, though (we need to investigate more the effect of this)

            # scaling = np.minimum(scaling_factor * min_spacing /input_spacing,
            #                     input_size / 32)

            scaling = scaling_factor * min_spacing / input_spacing
            output_spacing = input_spacing * scaling
            extended = np.append(scaling, [1])
            if image_grid2world is not None:
                affine = image_grid2world.dot(np.diag(extended))
            else:
                affine = np.diag(extended)
            output_size = input_size * (input_spacing / output_spacing) + 0.5
            output_size = output_size.astype(np.int32)
            sigmas = sigma_factor * (output_spacing / input_spacing - 1.0)

            # Filter along each direction with the appropriate sigma
            filtered = gaussian_filter(image, sigmas)
            filtered = ((filtered - np.min(filtered)) /
                        (np.max(filtered) - np.min(filtered)))
            if mask0:
                filtered *= mask

            # Add current level to the scale space
            self.images.append(filtered.astype(floating))
            self.domain_shapes.append(output_size)
            self.spacings.append(output_spacing)
            self.scalings.append(scaling)
            self.affines.append(affine)
            self.affine_invs.append(npl.inv(affine))
            self.sigmas.append(sigmas)

    def get_expand_factors(self, from_level, to_level):
        """Ratio of voxel size from pyramid level from_level to to_level.

        Given two scale space resolutions a = from_level, b = to_level,
        returns the ratio of voxels size at level b to voxel size at level a
        (the factor that must be used to multiply voxels at level a to
        'expand' them to level b).

        Parameters
        ----------
        from_level : int, 0 <= from_level < L, (L = number of resolutions)
            the resolution to expand voxels from
        to_level : int, 0 <= to_level < from_level
            the resolution to expand voxels to

        Returns
        -------
        factors : array, shape (k,), k = 2, 3
            the expand factors (a scalar for each voxel dimension)

        """
        factors = (np.array(self.spacings[to_level]) /
                   np.array(self.spacings[from_level]))
        return factors

    def print_level(self, level):
        """Prints properties of a pyramid level.

        Prints the properties of a level of this scale space to standard output

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to be printed

        """
        logger.info('Domain shape: ' + str(self.get_domain_shape(level)))
        logger.info('Spacing: ' + str(self.get_spacing(level)))
        logger.info('Scaling: ' + str(self.get_scaling(level)))
        logger.info('Affine: ' + str(self.get_affine(level)))
        logger.info('Sigmas: ' + str(self.get_sigmas(level)))

    def _get_attribute(self, attribute, level):
        """Return an attribute from the Scale Space at a given level.

        Returns the level-th element of attribute if level is a valid level
        of this scale space. Otherwise, returns None.

        Parameters
        ----------
        attribute : list
            the attribute to retrieve the level-th element from
        level : int,
            the index of the required element from attribute.

        Returns
        -------
        attribute[level] : object
            the requested attribute if level is valid, else it raises
            a ValueError

        """
        if 0 <= level < self.num_levels:
            return attribute[level]
        raise ValueError('Invalid pyramid level: '+str(level))

    def get_image(self, level):
        """Smoothed image at a given level.

        Returns the smoothed image at the requested level in the Scale Space.

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to get the smooth image from

        Returns
        -------
            the smooth image at the requested resolution or None if an invalid
            level was requested

        """
        return self._get_attribute(self.images, level)

    def get_domain_shape(self, level):
        """Shape the sub-sampled image must have at a particular level.

        Returns the shape the sub-sampled image must have at a particular
        resolution of the scale space (note that this object does not
        explicitly subsample the smoothed images, but only provides the
        properties the sub-sampled images must have).

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to get the sub-sampled shape from

        Returns
        -------
            the sub-sampled shape at the requested resolution or None if an
            invalid level was requested

        """
        return self._get_attribute(self.domain_shapes, level)

    def get_spacing(self, level):
        """Spacings the sub-sampled image must have at a particular level.

        Returns the spacings (voxel sizes) the sub-sampled image must have at a
        particular resolution of the scale space (note that this object does
        not explicitly subsample the smoothed images, but only provides the
        properties the sub-sampled images must have).

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to get the sub-sampled shape from

        Returns
        -------
        the spacings (voxel sizes) at the requested resolution or None if an
        invalid level was requested

        """
        return self._get_attribute(self.spacings, level)

    def get_scaling(self, level):
        """Adjustment factor for input-spacing to reflect voxel sizes at level.

        Returns the scaling factor that needs to be applied to the input
        spacing (the voxel sizes of the image at level 0 of the scale space) to
        transform them to voxel sizes at the requested level.

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to get the scalings from

        Returns
        -------
        the scaling factors from the original spacing to the spacings at the
        requested level

        """
        return self._get_attribute(self.scalings, level)

    def get_affine(self, level):
        """Voxel-to-space transformation at a given level.

        Returns the voxel-to-space transformation associated with the
        sub-sampled image at a particular resolution of the scale space (note
        that this object does not explicitly subsample the smoothed images, but
        only provides the properties the sub-sampled images must have).

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to get affine transform from

        Returns
        -------
            the affine (voxel-to-space) transform at the requested resolution
            or None if an invalid level was requested
        """
        return self._get_attribute(self.affines, level)

    def get_affine_inv(self, level):
        """Space-to-voxel transformation at a given level.

        Returns the space-to-voxel transformation associated with the
        sub-sampled image at a particular resolution of the scale space (note
        that this object does not explicitly subsample the smoothed images, but
        only provides the properties the sub-sampled images must have).

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to get the inverse transform from

        Returns
        -------
        the inverse (space-to-voxel) transform at the requested resolution or
        None if an invalid level was requested

        """
        return self._get_attribute(self.affine_invs, level)

    def get_sigmas(self, level):
        """Smoothing parameters used at a given level.

        Returns the smoothing parameters (a scalar for each axis) used at the
        requested level of the scale space

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to get the smoothing parameters from

        Returns
        -------
        the smoothing parameters at the requested level

        """
        return self._get_attribute(self.sigmas, level)


class IsotropicScaleSpace(ScaleSpace):
    def __init__(self, image, factors, sigmas,
                 image_grid2world=None,
                 input_spacing=None,
                 mask0=False):
        """ IsotropicScaleSpace.

        Computes the Scale Space representation of an image using isotropic
        smoothing kernels for all scales. The scale space is simply a list
        of images produced by smoothing the input image with a Gaussian
        kernel with different smoothing parameters.

        This specialization of ScaleSpace allows the user to provide custom
        scale and smoothing factors for all scales.

        Parameters
        ----------
        image : array, shape (r,c) or (s, r, c) where s is the number of
            slices, r is the number of rows and c is the number of columns of
            the input image.
        factors : list of floats
            custom scale factors to build the scale space (one factor for each
            scale).
        sigmas : list of floats
            custom smoothing parameter to build the scale space (one parameter
            for each scale).
        image_grid2world : array, shape (dim + 1, dim + 1), optional
            the grid-to-space transform of the image grid. The default is
            the identity matrix.
        input_spacing : array, shape (dim,), optional
            the spacing (voxel size) between voxels in physical space. The
            default if 1.0 along all axes.
        mask0 : Boolean, optional
            if True, all smoothed images will be zero at all voxels that are
            zero in the input image. The default is False.

        """
        self.dim = len(image.shape)
        self.num_levels = len(factors)
        if len(sigmas) != self.num_levels:
            raise ValueError("sigmas and factors must have the same length")
        input_size = np.array(image.shape)
        if mask0:
            mask = np.asarray(image > 0, dtype=np.int32)

        # Normalize input image to [0,1]
        img = ((image.astype(np.float64) - np.min(image)) /
               (np.max(image) - np.min(image)))
        if mask0:
            img *= mask

        # The properties are saved in separate lists. Insert input image
        # properties at the first level of the scale space
        self.images = [img.astype(floating)]
        self.domain_shapes = [input_size.astype(np.int32)]
        if input_spacing is None:
            input_spacing = np.ones((self.dim,), dtype=np.int32)
        self.spacings = [input_spacing]
        self.scalings = [np.ones(self.dim)]
        self.affines = [image_grid2world]
        self.sigmas = [np.ones(self.dim) * sigmas[self.num_levels - 1]]

        if image_grid2world is not None:
            self.affine_invs = [npl.inv(image_grid2world)]
        else:
            self.affine_invs = [None]

        # Compute the rest of the levels
        min_index = np.argmin(input_spacing)
        for i in range(1, self.num_levels):
            factor = factors[self.num_levels - 1 - i]
            shrink_factors = np.zeros(self.dim)
            new_spacing = np.zeros(self.dim)
            shrink_factors[min_index] = factor
            new_spacing[min_index] = input_spacing[min_index] * factor
            for j in range(self.dim):
                if j != min_index:
                    # Select the factor that maximizes isotropy
                    shrink_factors[j] = factor
                    new_spacing[j] = input_spacing[j] * factor
                    min_diff = np.abs(new_spacing[j] - new_spacing[min_index])
                    for f in range(1, factor):
                        diff = input_spacing[j] * f - new_spacing[min_index]
                        diff = np.abs(diff)
                        if diff < min_diff:
                            shrink_factors[j] = f
                            new_spacing[j] = input_spacing[j] * f
                            min_diff = diff

            extended = np.append(shrink_factors, [1])
            if image_grid2world is not None:
                affine = image_grid2world.dot(np.diag(extended))
            else:
                affine = np.diag(extended)
            output_size = (input_size / shrink_factors).astype(np.int32)
            new_sigmas = np.ones(self.dim) * sigmas[self.num_levels - i - 1]

            # Filter along each direction with the appropriate sigma
            filtered = gaussian_filter(image.astype(np.float64), new_sigmas)
            filtered = ((filtered.astype(np.float64) - np.min(filtered)) /
                        (np.max(filtered) - np.min(filtered)))
            if mask0:
                filtered *= mask

            # Add current level to the scale space
            self.images.append(filtered.astype(floating))
            self.domain_shapes.append(output_size)
            self.spacings.append(new_spacing)
            self.scalings.append(shrink_factors)
            self.affines.append(affine)
            self.affine_invs.append(npl.inv(affine))
            self.sigmas.append(new_sigmas)
