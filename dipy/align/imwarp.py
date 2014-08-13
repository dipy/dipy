from __future__ import print_function
import abc
from dipy.utils.six import with_metaclass
import numpy as np
import numpy.linalg as npl
import scipy as sp
import nibabel as nib
import dipy.align.vector_fields as vfu
from dipy.align import floating
from dipy.align import VerbosityLevels
from dipy.align import Bunch

RegistrationStages = Bunch(INIT_START=0, 
                          INIT_END=1,
                          OPT_START=2,
                          OPT_END=3,
                          SCALE_START=4,
                          SCALE_END=5,
                          ITER_START=6,
                          ITER_END=7)
r"""Registration Stages

This enum defines the different stages which the Volumetric Registration
may be in. The value of the stage is passed as a parameter to the call-back
function so that it can react accordingly.

INIT_START: optimizer initialization starts
INIT_END: optimizer initialization ends
OPT_START: optimization starts
OPT_END: optimization ends
SCALE_START: optimization at a new scale space resolution starts
SCALE_END: optimization at the current scale space resolution ends
ITER_START: a new iteration starts
ITER_END: the current iteration ends

"""

def mult_aff(A, B):
    r"""Returns the matrix product A.dot(B) considering None as the identity

    Parameters
    ----------
    A : array, shape (n,k)
    B : array, shape (k,m)

    Returns 
    -------
    The matrix product A.dot(B). If any of the input matrices is None, it is 
    treated as the identity matrix. If both matrices are None, None is returned.
    """
    if A is None:
        return B
    elif B is None:
        return A
    return A.dot(B)


def get_direction_and_spacings(affine, dim):
    r"""Extracts the rotational and spacing components from a matrix
    
    Extracts the rotational and spacing (voxel dimensions) components from a 
    matrix. An image gradient represents the local variation of the image's gray
    values per voxel. Since we are iterating on the physical space, we need to
    compute the gradients as variation per millimeter, so we need to divide each
    gradient's component by the voxel size along the corresponding axis, that's
    what the spacings are used for. Since the image's gradients are oriented 
    along the grid axes, we also need to re-orient the gradients to be given
    in physical space coordinates.

    Parameters
    ----------
    affine : array, shape (k, k), k = 3, 4
        the matrix transforming grid coordinates to physical space.

    Returns
    -------
    direction : array, shape (k-1, k-1)
        the rotational component of the input matrix
    spacings : array, shape (k-1,)
        the scaling component (voxel size) of the matrix

    """
    if affine == None:
        return np.eye(dim), np.ones(dim)
    dim = affine.shape[1]-1
    #Temporary hack: get the zooms by building a nifti image
    affine4x4 = np.eye(4)
    empty_volume = np.zeros((0,0,0))
    affine4x4[:dim, :dim] = affine[:dim, :dim]
    affine4x4[:dim, 3] = affine[:dim, dim-1]
    nib_nifti = nib.Nifti1Image(empty_volume, affine4x4)
    scalings = np.asarray(nib_nifti.get_header().get_zooms())
    scalings = np.asarray(scalings[:dim], dtype = np.float64)
    A = affine[:dim,:dim]
    return A.dot(np.diag(1.0/scalings)), scalings


class ScaleSpace(object):
    def __init__(self, image, num_levels,
                 codomain_affine=None,
                 input_spacing=None,
                 sigma_factor=0.2,
                 mask0=False):
        r""" ScaleSpace

        Computes the Scale Space representation of an image. The scale space is
        simply a list of images produced by smoothing the input image with a
        Gaussian kernel with increasing smoothing parameter. If the image's
        voxels are isotropic, the smoothing will be the same along all 
        directions: at level L = 0,1,..., the sigma is given by s * ( 2^L - 1 ).
        If the voxel dimensions are not isotropic, then the smoothing is
        weaker along low resolution directions.

        Parameters
        ----------
        image : array, shape (r,c) or (s, r, c) where s is the number of slices,
            r is the number of rows and c is the number of columns of the input
            image.
        num_levels : int
            the desired number of levels (resolutions) of the scale space
        codomain_affine : array, shape (k, k), k=3,4 (for either 2D or 3D images)
            the matrix transforming voxel coordinates to space coordinates in
            the input image discretization
        input_spacing : array, shape (k-1,)
            the spacing (voxel size) between voxels in physical space 
        sigma_factor : float
            the smoothing factor to be used in the construction of the scale
            space.
        mask0 : Boolean
            if True, all smoothed images will be zero at all voxels that are
            zero in the input image. 

        """
        self.dim = len(image.shape)
        self.num_levels = num_levels
        input_size = np.array(image.shape)
        if mask0:
            mask = np.asarray(image>0, dtype=np.int32)
        
        #normalize input image to [0,1]
        img = (image - image.min())/(image.max() - image.min())        

        #The properties are saved in separate lists. Insert input image
        #properties at the first level of the scale space
        self.images = [img.astype(floating)]
        self.domain_shapes = [input_size.astype(np.int32)]
        if input_spacing == None:
            input_spacing = np.ones((self.dim,), dtype = np.int32)
        self.spacings = [input_spacing]
        self.scalings = [np.ones(self.dim)]
        self.affines = [codomain_affine]
        self.sigmas = [np.zeros(self.dim)]

        if codomain_affine is not None:
            self.affine_invs = [npl.inv(codomain_affine)]
        else:
            self.affine_invs = [None]

        #compute the rest of the levels
        min_spacing = np.min(input_spacing)
        for i in range(1, num_levels):
            scaling_factor = 2**i
            scaling = np.ndarray((self.dim+1,))
            #Note: the minimum below is present in ANTS to prevent the scaling
            #from being too large (making the sub-sampled image to be too small)
            #this makes the sub-sampled image at least 32 voxels at each
            #direction it is risky to make this decision based on image size,
            #though (we need to investigate more the effect of this)
            
            #scaling = np.minimum(scaling_factor * min_spacing / input_spacing, 
            #                     input_size / 32)

            scaling = scaling_factor * min_spacing / input_spacing
            output_spacing = input_spacing * scaling
            extended = np.append(scaling, [1])
            if not codomain_affine is None:
                affine = codomain_affine.dot(np.diag(extended))
            else:
                affine = np.diag(extended)
            output_size = input_size * (input_spacing / output_spacing) + 0.5
            output_size = output_size.astype(np.int32)
            sigmas = sigma_factor * (output_spacing / input_spacing - 1.0)

            #filter along each direction with the appropriate sigma
            filtered = sp.ndimage.filters.gaussian_filter(image, sigmas)
            filtered = ((filtered - filtered.min())/
                       (filtered.max() - filtered.min()))
            if mask0:
                filtered *= mask

            #Add current level to the scale space
            self.images.append(filtered.astype(floating))
            self.domain_shapes.append(output_size)
            self.spacings.append(output_spacing)
            self.scalings.append(scaling)
            self.affines.append(affine)
            self.affine_invs.append(npl.inv(affine))
            self.sigmas.append(sigmas)

    def get_expand_factors(self, from_level, to_level):
        r"""Ratio of voxel size from pyramid level from_level to to_level

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
                  np.array(self.spacings[from_level]) )
        return factors

    def print_level(self, level):
        r"""Prints properties of a pyramid level

        Prints the properties of a level of this scale space to standard output

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to be printed 
        """
        print('Domain shape: ', self.get_domain_shape(level)) 
        print('Spacing: ', self.get_spacing(level))
        print('Scaling: ', self.get_scaling(level)) 
        print('Affine: ', self.get_affine(level))
        print('Sigmas: ', self.get_sigmas(level))

    def _get_attribute(self, attribute, level):
        r"""Returns an attribute from the Scale Space at a given level

        Returns the level-th element of attribute if level is a valid level
        of this scale space. Otherwise, returns None.

        Parameters
        ----------
        attribute : list
            the attribute to retrieve the level-th element from
        level : int,
            the index of the required element from attribute. 

        """
        if 0 <= level < self.num_levels:
            return attribute[level]
        raise ValueError('Invalid pyramid level: '+str(level))

    def get_image(self, level):
        r"""Smoothed image at a given level

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
        r"""Shape the sub-sampled image must have at a particular level

        Returns the shape the sub-sampled image must have at a particular
        resolution of the scale space (note that this object does not explicitly
        subsample the smoothed images, but only provides the properties
        the sub-sampled images must have).

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
        r"""Spacings the sub-sampled image must have at a particular level

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
        r"""Adjustment factor for input-spacing to reflect voxel sizes at level

        Returns the scaling factor that needs to be applied to the input spacing
        (the voxel sizes of the image at level 0 of the scale space) to
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
        r"""Voxel-to-space transformation at a given level

        Returns the voxel-to-space transformation associated to the sub-sampled
        image at a particular resolution of the scale space (note that this 
        object does not explicitly subsample the smoothed images, but only 
        provides the properties the sub-sampled images must have).

        Parameters
        ----------
        level : int, 0 <= from_level < L, (L = number of resolutions)
            the scale space level to get affine transform from

        Returns
        -------
            the affine (voxel-to-space) transform at the requested resolution or
            None if an invalid level was requested
        """
        return self._get_attribute(self.affines, level)

    def get_affine_inv(self, level):
        r"""Space-to-voxel transformation at a given level

        Returns the space-to-voxel transformation associated to the sub-sampled
        image at a particular resolution of the scale space (note that this 
        object does not explicitly subsample the smoothed images, but only 
        provides the properties the sub-sampled images must have).

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
        r"""Smoothing parameters used at a given level

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


class DiffeomorphicMap(object):
    def __init__(self, 
                 dim,
                 disc_shape,
                 disc_affine=None,
                 domain_shape=None,
                 domain_affine=None,
                 codomain_shape=None,
                 codomain_affine=None,
                 prealign=None):
        r""" DiffeomorphicMap

        Implements a diffeomorphic transformation on the physical space. The 
        deformation fields encoding the direct and inverse transformations
        share the same domain discretization (both the discretization grid shape
        and voxel-to-space matrix). The input coordinates (physical coordinates)
        are first aligned using prealign, and then displaced using the 
        corresponding vector field interpolated at the aligned coordinates.

        Parameters
        ----------
        dim : int, 2 or 3
            the transformation's dimension
        disc_shape : array, shape (dim,)
            the number of slices (if 3D), rows and columns of the deformation
            field's discretization
        disc_affine : the voxel-to-space transformation between the deformation field's
            grid and space
        domain_shape : array, shape (dim,)
            the number of slices (if 3D), rows and columns of the default
            discretizatio of this map's domain
        domain_affine : array, shape (dim+1, dim+1)
            the default voxel-to-space transformation between this map's
            discretization and physical space
        codomain_shape : array, shape (dim,)
            the number of slices (if 3D), rows and columns of the images that
            are 'normally' warped using this transformation in the forward
            direction (this will provide default transformation parameters to
            warp images under this transformation). By default, we assume that
            the inverse transformation is 'normally' used to warp images with
            the same discretization and voxel-to-space transformation as the 
            deformation field grid.
        codomain_affine : array, shape (dim+1, dim+1)
            the voxel-to-space transformation of images that are 'normally'
            warped using this transformation (in the forward direction).
        prealign : array, shape (dim+1, dim+1)
            the linear transformation to be applied to align input images to
            the reference space before warping under the deformation field.

        """

        self.dim = dim

        if(disc_shape is None):
            raise ValueError("Invalid displacement field discretization")

        self.disc_shape = np.asarray(disc_shape, dtype = np.int32)

        # If the discretization affine is None, we assume it's the identity 
        self.disc_affine = disc_affine
        if(self.disc_affine is None):
            self.disc_affine_inv = None
        else:
            self.disc_affine_inv = npl.inv(self.disc_affine)

        # If domain_shape is not provided, we use the map's discretization shape
        if(domain_shape is None):
            self.domain_shape = self.disc_shape
        else:
            self.domain_shape = np.asarray(domain_shape, dtype = np.int32)
        self.domain_affine = domain_affine
        if(domain_affine is None):
            self.domain_affine_inv = None
        else:
            self.domain_affine_inv = npl.inv(domain_affine)

        # If codomain shape was not provided, we assume it is an endomorphism:
        # use the same domain_shape and codomain_affine as the field domain
        if codomain_shape is None:
            self.codomain_shape = self.domain_shape
        else:
            self.codomain_shape = np.asarray(codomain_shape, dtype = np.int32)
        self.codomain_affine = codomain_affine
        if codomain_affine is None:
            self.codomain_affine_inv = None
        else:
            self.codomain_affine_inv = npl.inv(codomain_affine)

        self.prealign = prealign
        if prealign is None:
            self.prealign_inv = None
        else:
            self.prealign_inv = npl.inv(prealign)

        self.is_inverse = False
        self.forward = None
        self.backward = None

    def get_forward_field(self):
        r"""Deformation field to transform an image in the forward direction

        Returns the deformation field that must be used to warp an image under
        this transformation in the forward direction (note the 'is_inverse'
        flag). 
        """
        if self.is_inverse:
            return self.backward
        else:
            return self.forward

    def get_backward_field(self):
        r"""Deformation field to transform an image in the backward direction

        Returns the deformation field that must be used to warp an image under
        this transformation in the backward direction (note the 'is_inverse'
        flag). 
        """
        if self.is_inverse:
            return self.forward
        else:
            return self.backward

    def allocate(self):
        r"""Creates a zero displacement field

        Creates a zero displacement field (the identity transformation).
        """
        self.forward = np.zeros(tuple(self.disc_shape)+(self.dim,), 
                                dtype=floating)
        self.backward = np.zeros(tuple(self.disc_shape)+(self.dim,),
                                dtype=floating)

    def _get_warping_function(self, interpolation):
        r"""Appropriate warping function for the given interpolation type

        Returns the right warping function from vector_fields that must be
        called for the specified data dimension and interpolation type
        """
        if self.dim == 2:
            if interpolation == 'linear':
                return vfu.warp_image
            else:
                return vfu.warp_image_nn
        else:
            if interpolation == 'linear':
                return vfu.warp_volume
            else:
                return vfu.warp_volume_nn

    def _warp_forward(self, image, interpolation='linear', world_to_image=-1, 
                      sampling_shape=None, sampling_affine=-1):
        r"""Warps an image in the forward direction

        Deforms the input image under this diffeomorphic map in the forward 
        direction. Since the mapping is defined in the physical space, the user
        must specify the sampling grid shape and its space-to-voxel mapping.
        By default, the transformation will use the discretization information
        given at initialization.

        Parameters
        ----------
        image : array, shape (s, r, c) if dim = 3 or (r, c) if dim = 2
            the image to be warped under this transformation in the forward 
            direction
        interpolation : string, either 'linear' or 'nearest'
            the type of interpolation to be used for warping, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        world_to_image : array, shape (dim+1, dim+1)
            the transformation bringing world (space) coordinates to voxel
            coordinates of the image given as input
        sampling_shape : array, shape (dim,)
            the number of slices, rows and columns of the desired warped image
        sampling_affine : the transformation bringing voxel coordinates of the
            warped image to physical space

        Returns
        -------
        warped : array, shape = sampling_shape or self.codomain_shape if None
            the warped image under this transformation in the forward direction

        Notes
        -----
        The default value for the affine transformations is "-1" to handle the
        case in which the user provides "None" as input meaning "identity". If
        we used None as default, we wouldn't know if the user specifically wants 
        to use the identity (specifically passing None) or if it was left
        unspecified, meaning to use the appropriate default matrix

        A diffeomorphic map must be thought as a mapping between points
        in space. Warping an image J towards an image I means transforming
        each voxel with (discrete) coordinates i in I to (floating-point) voxel
        coordinates j in J. The transformation we consider 'forward' is
        precisely mapping coordinates i from the input image to coordinates j
        from reference image, which has the effect of warping an image with
        reference discretization (typically, the "static image") "towards" an 
        image with input discretization (typically, the "moving image"). More
        precisely, the warped image is produced by the following interpolation:

        warped[i] = image[W * forward[Dinv * P * S * i] + W * P * S * i )]

        where i denotes the coordinates of a voxel in the input grid, W is
        the world-to-grid transformation of the image given as input, Dinv
        is the world-to-grid transformation of the deformation field 
        discretization, P is the pre-aligning matrix (transforming input
        points to reference points), S is the voxel-to-space transformation of
        the sampling grid (see comment below) and forward is the forward
        deformation field.
        
        If we want to warp an image, we also must specify on what grid we
        want to sample the resulting warped image (the images are considered as
        points in space and its representation on a grid depends on its
        grid-to-space transform telling us for each grid voxel what point in
        space we need to bring via interpolation). So, S is the matrix that
        converts the sampling grid (whose shape is given as parameter
        'sampling_shape' ) to space coordinates.

        """
        #if no world-to-image transform is provided, we use the codomain info
        if world_to_image is -1:
            world_to_image = self.codomain_affine_inv
        #if no sampling info is provided, we use the domain info 
        if sampling_shape is None:
            if self.domain_shape is None:
                raise ValueError('Unable to infer sampling info. Provide a valid sampling_shape.')
            sampling_shape = self.domain_shape
        if sampling_affine is -1:
            sampling_affine = self.domain_affine

        W = world_to_image
        Dinv = self.disc_affine_inv
        P = self.prealign 
        S = sampling_affine

        #this is the matrix which we need to multiply the voxel coordinates
        #to interpolate on the forward displacement field ("in"side the 
        #'forward' brackets in the expression above)
        affine_idx_in = mult_aff(Dinv, mult_aff(P, S))

        #this is the matrix which we need to multiply the voxel coordinates
        #to add to the displacement ("out"side the 'forward' brackets in the
        #expression above)
        affine_idx_out = mult_aff(W, mult_aff(P, S))

        #this is the matrix which we need to multiply the displacement vector
        #prior to adding to the transformed input point
        affine_disp = W

        #Convert the data to the required types to use the cythonized functions  
        if interpolation == 'nearest':
            if image.dtype is np.dtype('float64') and floating is np.float32:
                image = image.astype(floating)
            elif image.dtype is np.dtype('int64'):
                image = image.astype(np.int32)
        else:
            image = np.asarray(image, dtype=floating)

        warp_f = self._get_warping_function(interpolation)

        warped = warp_f(image, self.forward, affine_idx_in, affine_idx_out,
                        affine_disp, sampling_shape)
        return warped

    def _warp_backward(self, image, interpolation='linear', world_to_image=-1, 
                       sampling_shape=None, sampling_affine=-1):
        r"""Warps an image in the backward direction

        Deforms the input image under this diffeomorphic map in the backward 
        direction. Since the mapping is defined in the physical space, the user
        must specify the sampling grid shape and its space-to-voxel mapping. 
        By default, the transformation will use the discretization information 
        given at initialization.

        Parameters
        ----------
        image : array, shape (s, r, c) if dim = 3 or (r, c) if dim = 2
            the image to be warped under this transformation in the backward
            direction
        interpolation : string, either 'linear' or 'nearest'
            the type of interpolation to be used for warping, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        world_to_image : array, shape (dim+1, dim+1)
            the transformation bringing world (space) coordinates to voxel
            coordinates of the image given as input
        sampling_shape : array, shape (dim,)
            the number of slices, rows and columns of the desired warped image
        sampling_affine : the transformation bringing voxel coordinates of the
            warped image to physical space

        Returns
        -------
        warped : array, shape = sampling_shape or self.domain_shape if None
            the warped image under this transformation in the backward direction

        Notes
        -----
        The default value for the affine transformations is "-1" to handle the
        case in which the user provides "None" as input meaning "identity". If
        we used None as default, we wouldn't know if the user specifically wants
        to use the identity (specifically passing None) or if it was left
        unspecified, meaning to use the appropriate default matrix

        A diffeomorphic map must be thought as a mapping between points
        in space. Warping an image J towards an image I means transforming
        each voxel with (discrete) coordinates i in I to (floating-point) voxel
        coordinates j in J. The transformation we consider 'backward' is
        precisely mapping coordinates i from the reference grid to coordinates j
        from the input image (that's why it's "backward"), which has the effect
        of warping the input image (moving) "towards" the reference. More
        precisely, the warped image is produced by the following interpolation:

        warped[i]= image[W * Pinv * backward[Dinv * S * i] + W * Pinv * S * i )]

        where i denotes the coordinates of a voxel in the input grid, W is
        the world-to-grid transformation of the image given as input, Dinv
        is the world-to-grid transformation of the deformation field 
        discretization, Pinv is the pre-aligning matrix's inverse (transforming
        reference points to input points), S is the grid-to-space transformation
        of the sampling grid (see comment below) and backward is the backward
        deformation field.

        If we want to warp an image, we also must specify on what grid we
        want to sample the resulting warped image (the images are considered as
        points in space and its representation on a grid depends on its
        grid-to-space transform telling us for each grid voxel what point in
        space we need to bring via interpolation). So, S is the matrix that
        converts the sampling grid (whose shape is given as parameter
        'sampling_shape' ) to space coordinates.

        """
        #if no world-to-image transform is provided, we use the domain info
        if world_to_image is -1:
            world_to_image = self.domain_affine_inv

        #if no sampling info is provided, we use the codomain info
        if sampling_shape is None:
            if self.codomain_shape is None:
                raise ValueError('Unable to infer sampling info. Provide a valid sampling_shape.')
            sampling_shape = self.codomain_shape
        if sampling_affine is -1:
            sampling_affine = self.codomain_affine

        W = world_to_image
        Dinv = self.disc_affine_inv
        Pinv = self.prealign_inv
        S = sampling_affine

        #this is the matrix which we need to multiply the voxel coordinates
        #to interpolate on the backward displacement field ("in"side the 
        #'backward' brackets in the expression above)
        affine_idx_in = mult_aff(Dinv, S)

        #this is the matrix which we need to multiply the voxel coordinates
        #to add to the displacement ("out"side the 'backward' brackets in the
        #expression above)
        affine_idx_out = mult_aff(W, mult_aff(Pinv, S))

        #this is the matrix which we need to multiply the displacement vector
        #prior to adding to the transformed input point
        affine_disp = mult_aff(W, Pinv)
        
        if interpolation == 'nearest':
            if image.dtype is np.dtype('float64') and floating is np.float32:
                image = image.astype(floating)
            elif image.dtype is np.dtype('int64'):
                image = image.astype(np.int32)
        else:
            image = np.asarray(image, dtype=floating)

        warp_f = self._get_warping_function(interpolation)
        
        warped = warp_f(image, self.backward, affine_idx_in, affine_idx_out,
                        affine_disp, sampling_shape)

        return warped

    def transform(self, image, interpolation='linear', world_to_image=-1, 
                  sampling_shape=None, sampling_affine=-1):
        r"""Warps an image in the forward direction

        Transforms the input image under this transformation in the forward
        direction. It uses the "is_inverse" flag to switch between "forward"
        and "backward" (if is_inverse is False, then transform(...) warps the
        image forwards, else it warps the image backwards).

        Parameters
        ----------
        image : array, shape (s, r, c) if dim = 3 or (r, c) if dim = 2
            the image to be warped under this transformation in the forward 
            direction
        interpolation : string, either 'linear' or 'nearest'
            the type of interpolation to be used for warping, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        world_to_image : array, shape (dim+1, dim+1)
            the transformation bringing world (space) coordinates to voxel
            coordinates of the image given as input
        sampling_shape : array, shape (dim,)
            the number of slices, rows and columns of the desired warped image
        sampling_affine : the transformation bringing voxel coordinates of the
            warped image to physical space

        Returns
        -------
        warped : array, shape = sampling_shape or self.codomain_shape if None
            the warped image under this transformation in the forward direction

        Notes
        -----
        See _warp_forward and _warp_backward documentation for further 
        information.
        """
        if self.is_inverse:
            warped = self._warp_backward(image, interpolation, world_to_image, 
                                       sampling_shape, sampling_affine)
        else:
            warped = self._warp_forward(image, interpolation, world_to_image, 
                                       sampling_shape, sampling_affine)
        return np.asarray(warped)

    def transform_inverse(self, image, interpolation='linear', world_to_image=-1, 
                          sampling_shape=None, sampling_affine=-1):
        r"""Warps an image in the backward direction

        Transforms the input image under this transformation in the backward
        direction. It uses the "is_inverse" flag to switch between "forward"
        and "backward" (if is_inverse is False, then transform_inverse(...) 
        warps the image backwards, else it warps the image forwards)

        Parameters
        ----------
        image : array, shape (s, r, c) if dim = 3 or (r, c) if dim = 2
            the image to be warped under this transformation in the forward 
            direction
        interpolation : string, either 'linear' or 'nearest'
            the type of interpolation to be used for warping, either 'linear'
            (for k-linear interpolation) or 'nearest' for nearest neighbor
        world_to_image : array, shape (dim+1, dim+1)
            the transformation bringing world (space) coordinates to voxel
            coordinates of the image given as input
        sampling_shape : array, shape (dim,)
            the number of slices, rows and columns of the desired warped image
        sampling_affine : the transformation bringing voxel coordinates of the
            warped image to physical space

        Returns
        -------
        warped : array, shape = sampling_shape or self.codomain_shape if None
            the warped image under this transformation in the backward direction

        Notes
        -----
        See _warp_forward and _warp_backward documentation for further 
        information.
        """
        if self.is_inverse:
            warped = self._warp_forward(image, interpolation, world_to_image, 
                                        sampling_shape, sampling_affine)
        else:
            warped = self._warp_backward(image, interpolation, world_to_image, 
                                       sampling_shape, sampling_affine)
        return np.asarray(warped)

    def inverse(self):
        r"""Inverse of this DiffeomorphicMap instance

        Returns a diffeomorphic map object representing the inverse of this
        transformation. The internal arrays are not copied but just referenced.

        Returns
        -------
        inv : DiffeomorphicMap object
            the inverse of this diffeomorphic map.

        """
        inv = DiffeomorphicMap(self.dim,
                               self.disc_shape,
                               self.disc_affine,
                               self.domain_shape,
                               self.domain_affine,
                               self.codomain_shape,
                               self.codomain_affine,
                               self.prealign)
        inv.forward = self.forward
        inv.backward = self.backward
        inv.is_inverse = True
        return inv

    def expand_fields(self, expand_factors, new_shape):
        r"""Expands the displacement fields from current shape to new_shape

        Up-samples the discretization of the displacement fields to be of 
        new_shape shape.

        Parameters
        ----------
        expand_factors : array, shape (dim,)
            the factors scaling current spacings (voxel sizes) to spacings in
            the expanded discretization.
        new_shape : array, shape (dim,)
            the shape of the arrays holding the up-sampled discretization

        """
        if self.dim == 2:
            expand_f = vfu.expand_displacement_field_2d
        else:
            expand_f = vfu.expand_displacement_field_3d

        expanded_forward = expand_f(self.forward, expand_factors, new_shape)
        expanded_backward = expand_f(self.backward, expand_factors, new_shape)

        expand_factors = np.append(expand_factors, [1])
        expanded_affine = mult_aff(self.disc_affine, np.diag(expand_factors))
        expanded_affine_inv = npl.inv(expanded_affine)
        self.forward = expanded_forward
        self.backward = expanded_backward
        self.disc_shape = new_shape
        self.disc_affine = expanded_affine
        self.disc_affine_inv = expanded_affine_inv

    def compute_inversion_error(self):
        r"""Inversion error of the displacement fields

        Estimates the inversion error of the displacement fields by computing
        statistics of the residual vectors obtained after composing the forward
        and backward displacement fields.

        Returns
        -------
        residual : array, shape (R, C) or (S, R, C)
            the displacement field resulting from composing the forward and
            backward displacement fields of this transformation (the residual
            should be zero for a perfect diffeomorphism)
        stats : array, shape (3,)
            statistics from the norms of the vectors of the residual 
            displacement field: maximum, mean and standard deviation

        Notes
        -----
        Since the forward and backward displacement fields have the same 
        discretization, the final composition is given by

        comp[i] = forward[ i + Dinv * backward[i]]

        where Dinv is the space-to-grid transformation of the displacement
        fields

        """
        Dinv = self.disc_affine_inv
        if self.dim == 2:
            compose_f = vfu.compose_vector_fields_2d
        else:
            compose_f = vfu.compose_vector_fields_3d

        residual, stats = compose_f(self.backward, self.forward, 
                                    None, Dinv, 1.0)

        return np.asarray(residual), np.asarray(stats)

    def shallow_copy(self):
        r"""Shallow copy of this DiffeomorphicMap instance

        Creates a shallow copy of this diffeomorphic map (the arrays are not
        copied but just referenced)
        
        Returns
        -------
        new_map : DiffeomorphicMap object
            the shallow copy of this diffeomorphic map

        """
        new_map = DiffeomorphicMap(self.dim,
                                   self.disc_shape,
                                   self.disc_affine,
                                   self.domain_shape,
                                   self.domain_affine,
                                   self.codomain_shape,
                                   self.codomain_affine,
                                   self.prealign)
        new_map.forward = self.forward
        new_map.backward = self.backward
        new_map.is_inverse = self.is_inverse
        return new_map

    def warp_endomorphism(self, phi):
        r"""Composition of this DiffeomorphicMap with a given endomorphism

        Creates a new DiffeomorphicMap C with the same properties as self and
        composes its displacement fields with phi's corresponding fields. 
        The resulting diffeomorphism is of the form C(x) = phi(self(x)) with
        inverse C^{-1}(y) = self^{-1}(phi^{-1}(y)). We assume that phi is an 
        endomorphism with the same discretization and domain affine as self 
        to ensure that the composition inherits self's properties (we also
        assume that the pre-aligning matrix of phi is None or identity). 

        Parameters
        ----------
        phi : DiffeomorphicMap object
            the endomorphism to be warped by this diffeomorphic map

        Returns
        -------
        composition : the composition of this diffeomorphic map with the
            endomorphism given as input

        Notes
        -----
        The problem with our current representation of a DiffeomorphicMap is
        that the set of Diffeomorphism that can be represented this way (a 
        pre-aligning matrix followed by a non-linear endomorphism given as a
        displacement field) is not closed under the composition operation. 

        Supporting a general DiffeomorphicMap class, closed under composition, 
        may be extremely costly computationally, and the kind of transformations 
        we actually need for Avants' mid-point algorithm (SyN) are much simpler.

        """
        #Compose the forward deformation fields
        d1 = self.get_forward_field()
        d2 = phi.get_forward_field()
        d1_inv = self.get_backward_field()
        d2_inv = phi.get_backward_field()

        premult_disp = self.disc_affine_inv

        if self.dim == 2:
            compose_f = vfu.compose_vector_fields_2d
        else:
            compose_f = vfu.compose_vector_fields_3d

        forward, stats = compose_f(d1, d2, None, premult_disp, 1.0)
        backward, stats, = compose_f(d2_inv, d1_inv, None, premult_disp, 1.0)

        composition = self.shallow_copy()
        composition.forward = forward
        composition.backward = backward
        return composition


class DiffeomorphicRegistration(with_metaclass(abc.ABCMeta, object)):
    def __init__(self, metric=None):
        r""" Diffeomorphic Registration

        This abstract class defines the interface to be implemented by any
        optimization algorithm for diffeomorphic registration.

        Parameters
        ----------
        metric : SimilarityMetric object
            the object measuring the similarity of the two images. The
            registration algorithm will minimize (or maximize) the provided
            similarity.
        """
        if metric is None:
            raise ValueError('The metric cannot be None')
        self.metric = metric
        self.dim = metric.dim

    def set_level_iters(self, level_iters):
        r"""Sets the number of iterations at each pyramid level

        Establishes the maximum number of iterations to be performed at each
        level of the Gaussian pyramid, similar to ANTS.

        Parameters
        ----------
        level_iters : list
            the number of iterations at each level of the Gaussian pyramid.
            level_iters[0] corresponds to the finest level, level_iters[n-1] the
            coarsest, where n is the length of the list
        """
        self.levels = len(level_iters) if level_iters else 0
        self.level_iters = level_iters

    @abc.abstractmethod
    def optimize(self):
        r"""Starts the metric optimization

        This is the main function each specialized class derived from this must
        implement. Upon completion, the deformation field must be available from
        the forward transformation model.
        """

    @abc.abstractmethod
    def get_map(self):
        r"""
        Returns the resulting diffeomorphic map after optimization
        """


class SymmetricDiffeomorphicRegistration(DiffeomorphicRegistration):
    def __init__(self,
                 metric,
                 level_iters=None,
                 step_length=0.25,
                 ss_sigma_factor=0.2,
                 opt_tol=1e-5,
                 inv_iter=20,
                 inv_tol=1e-3,
                 callback=None):
        r""" Symmetric Diffeomorphic Registration (SyN) Algorithm

        Performs the multi-resolution optimization algorithm for non-linear
        registration using a given similarity metric.

        Parameters
        ----------
        metric : SimilarityMetric object
            the metric to be optimized
        level_iters : list of int
            the number of iterations at each level of the Gaussian Pyramid (the
            length of the list defines the number of pyramid levels to be 
            used)
        opt_tol : float
            the optimization will stop when the estimated derivative of the
            energy profile w.r.t. time falls below this threshold
        inv_iter : int
            the number of iterations to be performed by the displacement field 
            inversion algorithm
        step_length : float
            the length of the maximum displacement vector of the update 
            displacement field at each iteration
        ss_sigma_factor : float
            parameter of the scale-space smoothing kernel. For example, the 
            std. dev. of the kernel will be factor*(2^i) in the isotropic case
            where i = 0, 1, ..., n_scales is the scale
        inv_tol : float
            the displacement field inversion algorithm will stop iterating
            when the inversion error falls below this threshold
        callback : function(SymmetricDiffeomorphicRegistration)
            a function receiving a SymmetricDiffeomorphicRegistration object 
            to be called after each iteration (this optimizer will call this
            function passing self as parameter)
        """
        super(SymmetricDiffeomorphicRegistration, self).__init__(metric)
        if level_iters is None:
            level_iters = [100, 100, 25]

        if len(level_iters) == 0:
            raise ValueError('The iterations list cannot be empty')

        self.set_level_iters(level_iters)
        self.step_length = step_length
        self.ss_sigma_factor = ss_sigma_factor
        self.opt_tol = opt_tol
        self.inv_tol = inv_tol
        self.inv_iter = inv_iter
        self.energy_window = 12
        self.energy_list = []
        self.full_energy_profile = []
        self.verbosity = VerbosityLevels.STATUS
        self.callback = callback
        self.moving_ss = None
        self.static_ss = None
        self.static_direction = None
        self.moving_direction = None
        self.mask0 = metric.mask0

    def update(self, new_displacement, current_displacement, 
               affine_inv, time_scaling):
        r"""Composition of the current displacement field with the given field

        Interpolates current displacement at the locations defined by 
        new_displacement. Equivalently, computes the composition C of the given
        displacement fields as C(x) = B(A(x)), where A is new_displacement and B
        is currentDisplacement. This function is intended to be used with
        deformation fields of the same sampling (e.g. to be called by a
        registration algorithm), in this case, the pre-multiplication matrix for
        the index vector is the identity

        Parameters
        ----------
        new_displacement : array, shape (R, C, 2) or (S, R, C, 3)
            the displacement field defining where to interpolate 
            current_displacement
        current_displacement : array, shape (R', C', 2) or (S', R', C', 3)
            the displacement field to be warped by new_displacement

        Returns
        -------
        updated : array, shape (the same as new_displacement)
            the warped displacement field
        mean_norm : the mean norm of all vectors in current_displacement
        """        
        mean_norm = np.sqrt(np.sum((current_displacement ** 2), -1)).mean()
        updated, stats = self.compose(new_displacement, current_displacement,
                                      None, affine_inv, time_scaling)

        return np.array(updated), np.array(mean_norm)

    def get_map(self):
        r"""Returns the resulting diffeomorphic map
        Returns the DiffeomorphicMap registering the moving image towards
        the static image.
        """
        return self.forward_model

    def _connect_functions(self):
        r"""Assign the methods to be called according to the image dimension

        Assigns the appropriate functions to be called for displacement field
        inversion, Gaussian pyramid, and affine / dense deformation composition
        according to the dimension of the input images e.g. 2D or 3D.
        """
        if self.dim == 2:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point_2d
            self.append_affine = vfu.append_affine_to_displacement_field_2d
            self.prepend_affine = vfu.prepend_affine_to_displacement_field_2d
            self.compose = vfu.compose_vector_fields_2d
        else:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point_3d
            self.append_affine = vfu.append_affine_to_displacement_field_3d
            self.prepend_affine = vfu.prepend_affine_to_displacement_field_3d
            self.compose = vfu.compose_vector_fields_3d

    def _init_optimizer(self, static, moving, 
                        static_affine, moving_affine, prealign):
        r"""Initializes the registration optimizer

        Initializes the optimizer by computing the scale space of the input
        images and allocating the required memory for the transformation models
        at the coarsest scale.

        Parameters
        ----------
        static: array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization. The 
            displacement fields will have the same discretization as the static
            image.
        moving: array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. Since the
            deformation fields' discretization is the same as the static image, 
            it is necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed to
            be accomplished by "pre-aligning" the moving image towards the 
            static using an affine transformation given by the 'prealign' matrix
        static_affine: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the static image
        moving_affine: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the moving image
        prealign: array, shape (dim+1, dim+1)
            the affine transformation (operating on the physical space) 
            pre-aligning the moving image towards the static

        """
        self._connect_functions()
        #Extract information from the affine matrices to create the scale space
        static_direction, static_spacing = \
            get_direction_and_spacings(static_affine, self.dim)
        moving_direction, moving_spacing = \
            get_direction_and_spacings(moving_affine, self.dim)

        #the images' directions don't change with scale
        self.static_direction = static_direction
        self.moving_direction = moving_direction

        #Build the scale space of the input images
        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            print('Applying zero mask: ' + str(self.mask0))

        if self.verbosity >= VerbosityLevels.STATUS:
            print('Creating scale space from the moving image. Levels: %d. '
                  'Sigma factor: %f.' % (self.levels, self.ss_sigma_factor))
        
        self.moving_ss = ScaleSpace(moving, self.levels, moving_affine,
                                    moving_spacing, self.ss_sigma_factor,
                                    self.mask0)

        if self.verbosity >= VerbosityLevels.STATUS:
            print('Creating scale space from the static image. Levels: %d. '
                  'Sigma factor: %f.' % (self.levels, self.ss_sigma_factor))
        
        self.static_ss = ScaleSpace(static, self.levels, static_affine,
                                    static_spacing, self.ss_sigma_factor,
                                    self.mask0)

        if self.verbosity >= VerbosityLevels.DEBUG:
            print('Moving scale space:')
            for level in range(self.levels):
                self.moving_ss.print_level(level)

            print('Static scale space:')
            for level in range(self.levels):
                self.static_ss.print_level(level)
        
        #Get the properties of the coarsest level from the static image. These
        #properties will be taken as the reference discretization.
        disc_shape = self.static_ss.get_domain_shape(self.levels-1)
        disc_affine = self.static_ss.get_affine(self.levels-1)

        # The codomain discretization of both diffeomorphic maps is
        # precisely the discretization of the static image
        codomain_shape = static.shape
        codomain_affine = static_affine

        #The forward model transforms points from the static image
        #to points on the reference (which is the static as well). So the domain
        #properties are taken from the static image. Since its the same as the
        #reference, we don't need to pre-align.
        domain_shape = static.shape
        domain_affine = static_affine
        self.forward_model = DiffeomorphicMap(self.dim,
                                              disc_shape,
                                              disc_affine,
                                              domain_shape,
                                              domain_affine,
                                              codomain_shape,
                                              codomain_affine,
                                              None)
        self.forward_model.allocate()

        #The backward model transforms points from the moving image
        #to points on the reference (which is the static). So the input
        #properties are taken from the moving image, and we need to pre-align
        #points on the moving physical space to the reference physical space by
        #applying the inverse of pre-align. This is done this way to make it
        #clear for the user: the pre-align matrix is usually obtained by doing
        #affine registration of the moving image towards the static image, which
        #results in a matrix transforming points in the static physical space to
        #points in the moving physical space
        prealign_inv = None if prealign is None else npl.inv(prealign)
        domain_shape = moving.shape
        domain_affine = moving_affine
        self.backward_model = DiffeomorphicMap(self.dim,
                                               disc_shape,
                                               disc_affine,
                                               domain_shape,
                                               domain_affine,
                                               codomain_shape,
                                               codomain_affine,
                                               prealign_inv)
        self.backward_model.allocate()

    def _end_optimizer(self):
        r"""Frees the resources allocated during initialization
        """
        del self.moving_ss
        del self.static_ss

    def _iterate(self):
        r"""Performs one symmetric iteration

        Performs one iteration of the SyN algorithm:
            1.Compute forward
            2.Compute backward
            3.Update forward
            4.Update backward
            5.Compute inverses
            6.Invert the inverses
        
        Returns
        -------
        der : float
            the derivative of the energy profile, computed by fitting a
            quadratic function to the energy values at the latest T iterations,
            where T = self.energy_window. If the current iteration is less than
            T then np.inf is returned instead. 
        """
        #Acquire current resolution information from scale spaces
        current_moving = self.moving_ss.get_image(self.current_level)
        current_static = self.static_ss.get_image(self.current_level)

        current_static_affine_inv = self.static_ss.get_affine_inv(0)
        current_moving_affine_inv = self.moving_ss.get_affine_inv(0)

        current_disc_shape = \
            self.static_ss.get_domain_shape(self.current_level)
        current_disc_affine = \
            self.static_ss.get_affine(self.current_level)
        current_disc_affine_inv = \
            self.static_ss.get_affine_inv(self.current_level)
        current_disc_spacing = \
            self.static_ss.get_spacing(self.current_level)
        
        #Warp the input images (smoothed to the current scale) to the common 
        #(reference) space at the current resolution
        wstatic = self.forward_model.transform_inverse(current_static, 'linear', 
                                                       -1,
                                                       current_disc_shape,
                                                       current_disc_affine)
        wmoving = self.backward_model.transform_inverse(current_moving, 'linear',
                                                        -1,
                                                        current_disc_shape,
                                                        current_disc_affine)
        
        #Pass both images to the metric. Now both images are sampled on the
        #reference grid (equal to the static image's grid) and the direction
        #doesn't change across scales
        self.metric.set_moving_image(wmoving, current_disc_affine, 
            current_disc_spacing, self.static_direction)
        self.metric.use_moving_image_dynamics(
            current_moving, self.backward_model.inverse())

        self.metric.set_static_image(wstatic, current_disc_affine, 
            current_disc_spacing, self.static_direction)
        self.metric.use_static_image_dynamics(
            current_static, self.forward_model.inverse())

        #Initialize the metric for a new iteration
        self.metric.initialize_iteration()
        if self.callback is not None:
            self.callback(self, RegistrationStages.ITER_START)

        #Free some memory (useful when using double precision)
        del self.forward_model.backward
        del self.backward_model.backward

        #Compute the forward step (to be used to update the forward transform) 
        fw_step = np.array(self.metric.compute_forward())
        #Normalize the forward step
        nrm = np.sqrt(np.sum((fw_step/current_disc_spacing)**2, -1)).max()
        if nrm>0:
            fw_step /= nrm
        
        #Add to current total field
        self.forward_model.forward, md_forward = self.update(
            self.forward_model.forward, fw_step, 
            current_disc_affine_inv, self.step_length)
        del fw_step

        #Keep track of the forward energy
        fw_energy = self.metric.get_energy()

        #Compose the backward step (to be used to update the backward transform)
        bw_step = np.array(self.metric.compute_backward())
        #Normalize the backward step
        nrm = np.sqrt(np.sum((bw_step/current_disc_spacing)**2, -1)).max()
        if nrm>0:
            bw_step /= nrm

        #Add to current total field
        self.backward_model.forward, md_backward = self.update(
            self.backward_model.forward, bw_step, 
            current_disc_affine_inv, self.step_length)
        del bw_step

        #Keep track of the energy
        bw_energy = self.metric.get_energy()
        der = np.inf
        n_iter = len(self.energy_list)
        if len(self.energy_list) >= self.energy_window:
            der = self._get_energy_derivative()

        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            ch = '-' if np.isnan(der) else der
            print('%d:\t%0.6f\t%0.6f\t%0.6f\t%s' % 
                    (n_iter, fw_energy, bw_energy, fw_energy + bw_energy, ch))

        self.energy_list.append(fw_energy + bw_energy)

        #Invert the forward model's forward field
        self.forward_model.backward = np.array(
            self.invert_vector_field(
                self.forward_model.forward,
                current_disc_affine_inv,
                current_disc_spacing,
                self.inv_iter, self.inv_tol, None))

        #Invert the backward model's forward field
        self.backward_model.backward = np.array(
            self.invert_vector_field(
                self.backward_model.forward,
                current_disc_affine_inv,
                current_disc_spacing,
                self.inv_iter, self.inv_tol, None))

        #Invert the forward model's backward field
        self.forward_model.forward = np.array(
            self.invert_vector_field(
                self.forward_model.backward,
                current_disc_affine_inv,
                current_disc_spacing,
                self.inv_iter, self.inv_tol, self.forward_model.forward))

        #Invert the backward model's backward field
        self.backward_model.forward = np.array(
            self.invert_vector_field(
                self.backward_model.backward,
                current_disc_affine_inv,
                current_disc_spacing,
                self.inv_iter, self.inv_tol, self.backward_model.forward))

        #Free resources no longer needed to compute the forward and backward
        #steps
        if self.callback is not None:
            self.callback(self, RegistrationStages.ITER_END)
        self.metric.free_iteration()

        return der

    def _approximate_derivative_direct(self, x, y):
        r"""Derivative of the degree-2 polynomial fit of the given x, y pairs

        Directly computes the derivative of the least-squares-fit quadratic
        function estimated from (x[...],y[...]) pairs.

        Parameters
        ----------
        x : array, shape(n,)
            increasing array representing the x-coordinates of the points to
            be fit
        y : array, shape(n,)
            array representing the y-coordinates of the points to be fit

        Returns
        -------
        y0 : float
            the estimated derivative at x0 = 0.5*len(x) 
        """
        x = np.asarray(x)
        y = np.asarray(y)
        X = np.row_stack((x**2, x, np.ones_like(x)))
        XX = (X).dot(X.T)
        b = X.dot(y)
        beta = npl.solve(XX,b)
        x0 = 0.5 * len(x)
        y0 = 2.0 * beta[0] * (x0) + beta[1]
        return y0

    def _get_energy_derivative(self):
        r"""Approximate derivative of the energy profile

        Returns the derivative of the estimated energy as a function of "time"
        (iterations) at the last iteration
        """
        n_iter = len(self.energy_list)
        if n_iter < self.energy_window:
            raise ValueError('Not enough data to fit the energy profile')
        x = range(self.energy_window)
        y = self.energy_list[(n_iter - self.energy_window):n_iter]
        ss = sum(y)
        if(ss > 0):
            ss *= -1
        y = [v / ss for v in y]
        der = self._approximate_derivative_direct(x,y)
        return der
    
    def _optimize(self):
        r"""Starts the optimization

        The main multi-scale symmetric optimization algorithm
        """
        self.full_energy_profile = []
        if self.callback is not None:
            self.callback(self, RegistrationStages.OPT_START)
        for level in range(self.levels - 1, -1, -1):
            if self.verbosity >= VerbosityLevels.STATUS:
                print('Optimizing level %d'%level)

            self.current_level = level
            current_static = self.static_ss.get_image(level)
            current_moving = self.moving_ss.get_image(level)
            
            self.metric.set_levels_below(self.levels - level)
            self.metric.set_levels_above(level)

            if level < self.levels - 1:
                expand_factors = \
                    self.static_ss.get_expand_factors(level+1, level) 
                new_shape = self.static_ss.get_domain_shape(level)
                self.forward_model.expand_fields(expand_factors, new_shape)
                self.backward_model.expand_fields(expand_factors, new_shape)

            self.niter = 0
            self.energy_list = []
            derivative = np.inf

            if self.callback is not None:
                self.callback(self, RegistrationStages.SCALE_START)

            while ((self.niter < self.level_iters[self.levels - 1 - level]) and 
                   (self.opt_tol < derivative)):
                derivative = self._iterate()
                self.niter += 1

            self.full_energy_profile.extend(self.energy_list)

            if self.callback is not None:
                self.callback(self, RegistrationStages.SCALE_END)
            
        # Reporting mean and std in stats[1] and stats[2]
        residual, stats = self.forward_model.compute_inversion_error()
        
        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            print('Forward Residual error: %0.6f (%0.6f)'
                  % (stats[1], stats[2]))
        
        residual, stats = self.backward_model.compute_inversion_error()

        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            print('Backward Residual error :%0.6f (%0.6f)'
                  % (stats[1], stats[2]))

        #Compose the two partial transformations
        self.forward_model = self.backward_model.warp_endomorphism(
                                    self.forward_model.inverse()).inverse()
                
        # Report mean and std for the composed deformation field
        residual, stats = self.forward_model.compute_inversion_error()
        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            print('Final residual error: %0.6f (%0.6f)' % (stats[1], stats[2]))
        if self.callback is not None:
            self.callback(self, RegistrationStages.OPT_END)

    def optimize(self, static, moving, static_affine=None, moving_affine=None, 
                 prealign=None):
        r"""
        Starts the optimization

        Parameters
        ----------
        static: array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization. The 
            displacement fields will have the same discretization as the static
            image.
        moving: array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. Since the
            deformation fields' discretization is the same as the static image, 
            it is necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed to
            be accomplished by "pre-aligning" the moving image towards the 
            static using an affine transformation given by the 'prealign' matrix
        static_affine: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the static image
        moving_affine: array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the moving image
        prealign: array, shape (dim+1, dim+1)
            the affine transformation (operating on the physical space) 
            pre-aligning the moving image towards the static
        
        Returns
        -------
        forward_model : DiffeomorphicMap object
            the diffeomorphic map that brings the moving image towards the
            static one in the forward direction (i.e. by calling 
            forward_model.transform) and the static image towards the
            moving one in the backward direction (i.e. by calling 
            forward_model.transform_inverse). 

        """
        if self.verbosity >= VerbosityLevels.DEBUG:
            print("Pre-align:", prealign)

        self._init_optimizer(static.astype(floating), moving.astype(floating), 
                             static_affine, moving_affine, prealign)
        self._optimize()
        self._end_optimizer()
        return self.forward_model

