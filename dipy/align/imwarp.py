"""  Classes and functions for Symmetric Diffeomorphic Registration """

import logging
import abc

import numpy as np
import numpy.linalg as npl
import nibabel as nib
from nibabel.streamlines import ArraySequence as Streamlines

from dipy.align import vector_fields as vfu
from dipy.align import floating
from dipy.align import VerbosityLevels
from dipy.align import Bunch
from dipy.align.scalespace import ScaleSpace

RegistrationStages = Bunch(INIT_START=0,
                           INIT_END=1,
                           OPT_START=2,
                           OPT_END=3,
                           SCALE_START=4,
                           SCALE_END=5,
                           ITER_START=6,
                           ITER_END=7)
"""Registration Stages

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

logger = logging.getLogger(__name__)

def mult_aff(A, B):
    """Returns the matrix product A.dot(B) considering None as the identity

    Parameters
    ----------
    A : array, shape (n,k)
    B : array, shape (k,m)

    Returns
    -------
    The matrix product A.dot(B). If any of the input matrices is None, it is
    treated as the identity matrix. If both matrices are None, None is returned
    """
    if A is None:
        return B
    elif B is None:
        return A
    return A.dot(B)


def get_direction_and_spacings(affine, dim):
    """Extracts the rotational and spacing components from a matrix

    Extracts the rotational and spacing (voxel dimensions) components from a
    matrix. An image gradient represents the local variation of the image's
    gray values per voxel. Since we are iterating on the physical space, we
    need to compute the gradients as variation per millimeter, so we need to
    divide each gradient's component by the voxel size along the corresponding
    axis, that's what the spacings are used for. Since the image's gradients
    are oriented along the grid axes, we also need to re-orient the gradients
    to be given in physical space coordinates.

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
    if affine is None:
        return np.eye(dim), np.ones(dim)
    dim = affine.shape[1]-1
    # Temporary hack: get the zooms by building a nifti image
    affine4x4 = np.eye(4)
    empty_volume = np.zeros((0, 0, 0))
    affine4x4[:dim, :dim] = affine[:dim, :dim]
    affine4x4[:dim, 3] = affine[:dim, dim-1]
    nib_nifti = nib.Nifti1Image(empty_volume, affine4x4)
    scalings = np.asarray(nib_nifti.header.get_zooms())
    scalings = np.asarray(scalings[:dim], dtype=np.float64)
    A = affine[:dim, :dim]
    return A.dot(np.diag(1.0/scalings)), scalings


class DiffeomorphicMap:
    def __init__(self,
                 dim,
                 disp_shape,
                 disp_grid2world=None,
                 domain_shape=None,
                 domain_grid2world=None,
                 codomain_shape=None,
                 codomain_grid2world=None,
                 prealign=None):
        """ DiffeomorphicMap

        Implements a diffeomorphic transformation on the physical space. The
        deformation fields encoding the direct and inverse transformations
        share the same domain discretization (both the discretization grid
        shape and voxel-to-space matrix). The input coordinates (physical
        coordinates) are first aligned using prealign, and then displaced
        using the corresponding vector field interpolated at the aligned
        coordinates.

        Parameters
        ----------
        dim : int, 2 or 3
            the transformation's dimension
        disp_shape : array, shape (dim,)
            the number of slices (if 3D), rows and columns of the deformation
            field's discretization
        disp_grid2world : the voxel-to-space transform between the def. fields
            grid and space
        domain_shape : array, shape (dim,)
            the number of slices (if 3D), rows and columns of the default
            discretization of this map's domain
        domain_grid2world : array, shape (dim+1, dim+1)
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
        codomain_grid2world : array, shape (dim+1, dim+1)
            the voxel-to-space transformation of images that are 'normally'
            warped using this transformation (in the forward direction).
        prealign : array, shape (dim+1, dim+1)
            the linear transformation to be applied to align input images to
            the reference space before warping under the deformation field.

        """

        self.dim = dim

        if disp_shape is None:
            raise ValueError("Invalid displacement field discretization")

        self.disp_shape = np.asarray(disp_shape, dtype=np.int32)

        # If the discretization affine is None, we assume it's the identity
        self.disp_grid2world = disp_grid2world
        if self.disp_grid2world is None:
            self.disp_world2grid = None
        else:
            self.disp_world2grid = npl.inv(self.disp_grid2world)

        # If domain_shape isn't provided, we use the map's discretization shape
        if domain_shape is None:
            self.domain_shape = self.disp_shape
        else:
            self.domain_shape = np.asarray(domain_shape, dtype=np.int32)
        self.domain_grid2world = domain_grid2world
        if domain_grid2world is None:
            self.domain_world2grid = None
        else:
            self.domain_world2grid = npl.inv(domain_grid2world)

        # If codomain shape was not provided, we assume it is an endomorphism:
        # use the same domain_shape and codomain_grid2world as the field domain
        if codomain_shape is None:
            self.codomain_shape = self.domain_shape
        else:
            self.codomain_shape = np.asarray(codomain_shape, dtype=np.int32)
        self.codomain_grid2world = codomain_grid2world
        if codomain_grid2world is None:
            self.codomain_world2grid = None
        else:
            self.codomain_world2grid = npl.inv(codomain_grid2world)

        self.prealign = prealign
        if prealign is None:
            self.prealign_inv = None
        else:
            self.prealign_inv = npl.inv(prealign)

        self.is_inverse = False
        self.forward = None
        self.backward = None

    def interpret_matrix(self, obj):
        """ Try to interpret `obj` as a matrix

        Some operations are performed faster if we know in advance if a matrix
        is the identity (so we can skip the actual matrix-vector
        multiplication). This function returns None if the given object
        is None or the 'identity' string. It returns the same object if it is
        a numpy array. It raises an exception otherwise.

        Parameters
        ----------
        obj : object
            any object

        Returns
        -------
        obj : object
            the same object given as argument if `obj` is None or a numpy
            array. None if `obj` is the 'identity' string.
        """
        if (obj is None) or isinstance(obj, np.ndarray):
            return obj
        if isinstance(obj, str) and (obj == 'identity'):
            return None
        raise ValueError('Invalid matrix')

    def get_forward_field(self):
        """Deformation field to transform an image in the forward direction

        Returns the deformation field that must be used to warp an image under
        this transformation in the forward direction (note the 'is_inverse'
        flag).
        """
        if self.is_inverse:
            return self.backward
        else:
            return self.forward

    def get_backward_field(self):
        """Deformation field to transform an image in the backward direction

        Returns the deformation field that must be used to warp an image under
        this transformation in the backward direction (note the 'is_inverse'
        flag).
        """
        if self.is_inverse:
            return self.forward
        else:
            return self.backward

    def allocate(self):
        """Creates a zero displacement field

        Creates a zero displacement field (the identity transformation).
        """
        self.forward = np.zeros(tuple(self.disp_shape) + (self.dim,),
                                dtype=floating)
        self.backward = np.zeros(tuple(self.disp_shape) + (self.dim,),
                                 dtype=floating)

    def _get_warping_function(self, interpolation, warp_coordinates=False):
        r"""Appropriate warping function for the given interpolation type

        Returns the right warping function from vector_fields that must be
        called for the specified data dimension and interpolation type

        Parameters
        ----------
        interpolation : string, either 'linear' or 'nearest'
            specifies the type of interpolation used for image warping. It
            does not have any effect if `warp_coordinates` is True, in which
            case no interpolation is intended to be performed.
        warp_coordinates : Boolean,
            if False, then returns the right image warping function for this
            DiffeomorphicMap dimension and the specified `interpolation`. If
            True, then returns the right coordinate warping function.
        """
        if self.dim == 2:
            if warp_coordinates:
                return vfu.warp_coordinates_2d
            if interpolation == 'linear':
                return vfu.warp_2d
            else:
                return vfu.warp_2d_nn
        else:
            if warp_coordinates:
                return vfu.warp_coordinates_3d
            if interpolation == 'linear':
                return vfu.warp_3d
            else:
                return vfu.warp_3d_nn

    def _warp_coordinates_forward(self, points, coord2world=None,
                                  world2coord=None):
        r"""Warps the list of points in the forward direction

        Applies this diffeomorphic map to the list of points given by `points`.
        We assume that the points' coordinates are mapped to world coordinates
        by applying the `coord2world` affine transform. The warped coordinates
        are given in world coordinates unless `world2coord` affine transform
        is specified, in which case the warped points will be transformed
        to the corresponding coordinate system.

        Parameters
        ----------
        points :
        coord2world :
        world2coord :
        """
        warp_f = self._get_warping_function(None, warp_coordinates=True)
        coord2prealigned = mult_aff(self.prealign, coord2world)
        out = warp_f(points, self.forward, coord2prealigned, world2coord,
                     self.disp_world2grid)
        return out

    def _warp_coordinates_backward(self, points, coord2world=None,
                                   world2coord=None):
        """Warps the list of points in the backward direction

        Applies this diffeomorphic map to the list of points given by `points`.
        We assume that the points' coordinates are mapped to world coordinates
        by applying the `coord2world` affine transform. The warped coordinates
        are given in world coordinates unless `world2coord` affine transform
        is specified, in which case the warped points will be transformed
        to the corresponding coordinate system.

        Parameters
        ----------
        points :
        coord2world :
        world2coord :
        """
        warp_f = self._get_warping_function(None, warp_coordinates=True)
        world2invprealigned = mult_aff(world2coord, self.prealign_inv)
        out = warp_f(points, self.backward, coord2world, world2invprealigned,
                     self.disp_world2grid)
        return out

    def _warp_forward(self, image, interpolation='linear',
                      image_world2grid=None, out_shape=None,
                      out_grid2world=None):
        """Warps an image in the forward direction

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
        image_world2grid : array, shape (dim+1, dim+1)
            the transformation bringing world (space) coordinates to voxel
            coordinates of the image given as input
        out_shape : array, shape (dim,)
            the number of slices, rows, and columns of the desired warped image
        out_grid2world : the transformation bringing voxel coordinates of the
            warped image to physical space

        Returns
        -------
        warped : array, shape = out_shape or self.codomain_shape if None
            the warped image under this transformation in the forward direction

        Notes
        -----
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
        'out_shape' ) to space coordinates.
        """
        # if no world-to-image transform is provided, we use the codomain info
        if image_world2grid is None:
            image_world2grid = self.codomain_world2grid
        # if no sampling info is provided, we use the domain info
        if out_shape is None:
            if self.domain_shape is None:
                raise ValueError('Unable to infer sampling info. '
                                 'Provide a valid out_shape.')
            out_shape = self.domain_shape
        else:
            out_shape = np.asarray(out_shape, dtype=np.int32)
        if out_grid2world is None:
            out_grid2world = self.domain_grid2world

        W = self.interpret_matrix(image_world2grid)
        Dinv = self.disp_world2grid
        P = self.prealign
        S = self.interpret_matrix(out_grid2world)

        # this is the matrix which we need to multiply the voxel coordinates
        # to interpolate on the forward displacement field ("in"side the
        # 'forward' brackets in the expression above)
        affine_idx_in = mult_aff(Dinv, mult_aff(P, S))

        # this is the matrix which we need to multiply the voxel coordinates
        # to add to the displacement ("out"side the 'forward' brackets in the
        # expression above)
        affine_idx_out = mult_aff(W, mult_aff(P, S))

        # this is the matrix which we need to multiply the displacement vector
        # prior to adding to the transformed input point
        affine_disp = W

        # Convert the data to required types to use the cythonized functions
        if interpolation == 'nearest':
            if image.dtype is np.dtype('float64') and floating is np.float32:
                image = image.astype(floating)
            elif image.dtype is np.dtype('int64'):
                image = image.astype(np.int32)
        else:
            image = np.asarray(image, dtype=floating)

        warp_f = self._get_warping_function(interpolation)

        warped = warp_f(image, self.forward, affine_idx_in, affine_idx_out,
                        affine_disp, out_shape)
        return warped

    def _warp_backward(self, image, interpolation='linear',
                       image_world2grid=None, out_shape=None,
                       out_grid2world=None):
        """Warps an image in the backward direction

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
        image_world2grid : array, shape (dim+1, dim+1)
            the transformation bringing world (space) coordinates to voxel
            coordinates of the image given as input
        out_shape : array, shape (dim,)
            the number of slices, rows and columns of the desired warped image
        out_grid2world : the transformation bringing voxel coordinates of the
            warped image to physical space

        Returns
        -------
        warped : array, shape = out_shape or self.domain_shape if None
            the warped image under this transformation in the backward
            direction

        Notes
        -----
        A diffeomorphic map must be thought as a mapping between points
        in space. Warping an image J towards an image I means transforming
        each voxel with (discrete) coordinates i in I to (floating-point) voxel
        coordinates j in J. The transformation we consider 'backward' is
        precisely mapping coordinates i from the reference grid to coordinates
        j from the input image (that's why it's "backward"), which has the
        effect of warping the input image (moving) "towards" the reference.
        More precisely, the warped image is produced by the following
        interpolation:

        warped[i]=image[W * Pinv * backward[Dinv * S * i] + W * Pinv * S * i )]

        where i denotes the coordinates of a voxel in the input grid, W is
        the world-to-grid transformation of the image given as input, Dinv
        is the world-to-grid transformation of the deformation field
        discretization, Pinv is the pre-aligning matrix's inverse (transforming
        reference points to input points), S is the grid-to-space
        transformation of the sampling grid (see comment below) and backward is
        the backward deformation field.

        If we want to warp an image, we also must specify on what grid we
        want to sample the resulting warped image (the images are considered as
        points in space and its representation on a grid depends on its
        grid-to-space transform telling us for each grid voxel what point in
        space we need to bring via interpolation). So, S is the matrix that
        converts the sampling grid (whose shape is given as parameter
        'out_shape' ) to space coordinates.

        """
        # if no world-to-image transform is provided, we use the domain info
        if image_world2grid is None:
            image_world2grid = self.domain_world2grid

        # if no sampling info is provided, we use the codomain info
        if out_shape is None:
            if self.codomain_shape is None:
                msg = 'Unknown sampling info. Provide a valid out_shape.'
                raise ValueError(msg)
            out_shape = self.codomain_shape
        if out_grid2world is None:
            out_grid2world = self.codomain_grid2world

        W = self.interpret_matrix(image_world2grid)
        Dinv = self.disp_world2grid
        Pinv = self.prealign_inv
        S = self.interpret_matrix(out_grid2world)

        # this is the matrix which we need to multiply the voxel coordinates
        # to interpolate on the backward displacement field ("in"side the
        # 'backward' brackets in the expression above)
        affine_idx_in = mult_aff(Dinv, S)

        # this is the matrix which we need to multiply the voxel coordinates
        # to add to the displacement ("out"side the 'backward' brackets in the
        # expression above)
        affine_idx_out = mult_aff(W, mult_aff(Pinv, S))

        # this is the matrix which we need to multiply the displacement vector
        # prior to adding to the transformed input point
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
                        affine_disp, out_shape)

        return warped

    def transform(self, image, interpolation='linear', image_world2grid=None,
                  out_shape=None, out_grid2world=None):
        """Warps an image in the forward direction

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
        image_world2grid : array, shape (dim+1, dim+1)
            the transformation bringing world (space) coordinates to voxel
            coordinates of the image given as input
        out_shape : array, shape (dim,)
            the number of slices, rows and columns of the desired warped image
        out_grid2world : the transformation bringing voxel coordinates of the
            warped image to physical space

        Returns
        -------
        warped : array, shape = out_shape or self.codomain_shape if None
            the warped image under this transformation in the forward direction

        Notes
        -----
        See _warp_forward and _warp_backward documentation for further
        information.
        """
        if out_shape is not None:
            out_shape = np.asarray(out_shape, dtype=np.int32)
        if self.is_inverse:
            warped = self._warp_backward(image, interpolation,
                                         image_world2grid, out_shape,
                                         out_grid2world)
        else:
            warped = self._warp_forward(image, interpolation, image_world2grid,
                                        out_shape, out_grid2world)
        return np.asarray(warped)

    def transform_inverse(self, image, interpolation='linear',
                          image_world2grid=None, out_shape=None,
                          out_grid2world=None):
        """Warps an image in the backward direction

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
        image_world2grid : array, shape (dim+1, dim+1)
            the transformation bringing world (space) coordinates to voxel
            coordinates of the image given as input
        out_shape : array, shape (dim,)
            the number of slices, rows, and columns of the desired warped image
        out_grid2world : the transformation bringing voxel coordinates of the
            warped image to physical space

        Returns
        -------
        warped : array, shape = out_shape or self.codomain_shape if None
            warped image under this transformation in the backward direction

        Notes
        -----
        See _warp_forward and _warp_backward documentation for further
        information.
        """
        if self.is_inverse:
            warped = self._warp_forward(image, interpolation, image_world2grid,
                                        out_shape, out_grid2world)
        else:
            warped = self._warp_backward(image, interpolation,
                                         image_world2grid, out_shape,
                                         out_grid2world)
        return np.asarray(warped)

    def transform_points(self, points, coord2world=None, world2coord=None):
        """Warp the list of points in the forward direction.

        Applies this diffeomorphic map to the list of points (or streamlines)
        given by `points`. We assume that the points' coordinates are mapped
        to world coordinates by applying the `coord2world` affine transform.
        The warped coordinates are given in world coordinates unless
        `world2coord` affine transform is specified, in which case the warped
        points will be transformed to the corresponding coordinate system.

        Parameters
        ----------
        points : array, shape (N, dim) or Streamlines object

        coord2world : array, shape (dim+1, dim+1), optional
            affine matrix mapping points to world coordinates

        world2coord : array, shape (dim+1, dim+1), optional
            affine matrix mapping world coordinates to points

        """
        return self._transform_coordinates(points, coord2world, world2coord,
                                           inverse=self.is_inverse)

    def transform_points_inverse(self, points, coord2world=None,
                                 world2coord=None):
        """Warp the list of points in the backward direction.

        Applies this diffeomorphic map to the list of points (or streamlines)
        given by `points`. We assume that the points' coordinates are mapped
        to world coordinates by applying the `coord2world` affine transform.
        The warped coordinates are given in world coordinates unless
        `world2coord` affine transform is specified, in which case the warped
        points will be transformed to the corresponding coordinate system.

        Parameters
        ----------
        points : array, shape (N, dim) or Streamlines object

        coord2world : array, shape (dim+1, dim+1), optional
            affine matrix mapping points to world coordinates

        world2coord : array, shape (dim+1, dim+1), optional
            affine matrix mapping world coordinates to points

        """
        return self._transform_coordinates(points, coord2world, world2coord,
                                           inverse=not self.is_inverse)

    def _transform_coordinates(self, points, coord2world, world2coord,
                               inverse=False):

        is_streamline_obj = isinstance(points, Streamlines)
        data = points.get_data() if is_streamline_obj else points

        if inverse:
            out = self._warp_coordinates_backward(data, coord2world,
                                                  world2coord)
        else:
            out = self._warp_coordinates_forward(data, coord2world,
                                                 world2coord)

        if is_streamline_obj:
            old_data_dtype = points._data.dtype
            old_offsets_dtype = points._offsets.dtype
            streamlines = points.copy()
            streamlines._offsets = points._offsets.astype(old_offsets_dtype)
            streamlines._data = out.astype(old_data_dtype)
            return streamlines

        return out

    def inverse(self):
        """Inverse of this DiffeomorphicMap instance

        Returns a diffeomorphic map object representing the inverse of this
        transformation. The internal arrays are not copied but just referenced.

        Returns
        -------
        inv : DiffeomorphicMap object
            the inverse of this diffeomorphic map.

        """
        inv = DiffeomorphicMap(self.dim,
                               self.disp_shape,
                               self.disp_grid2world,
                               self.domain_shape,
                               self.domain_grid2world,
                               self.codomain_shape,
                               self.codomain_grid2world,
                               self.prealign)
        inv.forward = self.forward
        inv.backward = self.backward
        inv.is_inverse = True
        return inv

    def expand_fields(self, expand_factors, new_shape):
        """Expands the displacement fields from current shape to new_shape

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
            expand_f = vfu.resample_displacement_field_2d
        else:
            expand_f = vfu.resample_displacement_field_3d

        expanded_forward = expand_f(self.forward, expand_factors, new_shape)
        expanded_backward = expand_f(self.backward, expand_factors, new_shape)

        expand_factors = np.append(expand_factors, [1])
        expanded_grid2world = mult_aff(self.disp_grid2world,
                                       np.diag(expand_factors))
        expanded_world2grid = npl.inv(expanded_grid2world)
        self.forward = expanded_forward
        self.backward = expanded_backward
        self.disp_shape = new_shape
        self.disp_grid2world = expanded_grid2world
        self.disp_world2grid = expanded_world2grid

    def compute_inversion_error(self):
        """Inversion error of the displacement fields

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
        Dinv = self.disp_world2grid
        if self.dim == 2:
            compose_f = vfu.compose_vector_fields_2d
        else:
            compose_f = vfu.compose_vector_fields_3d

        residual, stats = compose_f(self.backward, self.forward,
                                    None, Dinv, 1.0, None)

        return np.asarray(residual), np.asarray(stats)

    def shallow_copy(self):
        """Shallow copy of this DiffeomorphicMap instance

        Creates a shallow copy of this diffeomorphic map (the arrays are not
        copied but just referenced)

        Returns
        -------
        new_map : DiffeomorphicMap object
            the shallow copy of this diffeomorphic map

        """
        new_map = DiffeomorphicMap(self.dim,
                                   self.disp_shape,
                                   self.disp_grid2world,
                                   self.domain_shape,
                                   self.domain_grid2world,
                                   self.codomain_shape,
                                   self.codomain_grid2world,
                                   self.prealign)
        new_map.forward = self.forward
        new_map.backward = self.backward
        new_map.is_inverse = self.is_inverse
        return new_map

    def warp_endomorphism(self, phi):
        """Composition of this DiffeomorphicMap with a given endomorphism

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
        may be extremely costly computationally, and the kind of
        transformations we actually need for Avants' mid-point algorithm (SyN)
        are much simpler.

        """
        # Compose the forward deformation fields
        d1 = self.get_forward_field()
        d2 = phi.get_forward_field()
        d1_inv = self.get_backward_field()
        d2_inv = phi.get_backward_field()

        premult_disp = self.disp_world2grid

        if self.dim == 2:
            compose_f = vfu.compose_vector_fields_2d
        else:
            compose_f = vfu.compose_vector_fields_3d

        forward, stats = compose_f(d1, d2, None, premult_disp, 1.0, None)
        backward, stats, = compose_f(d2_inv, d1_inv, None, premult_disp, 1.0,
                                     None)

        composition = self.shallow_copy()
        composition.forward = forward
        composition.backward = backward
        return composition

    def get_simplified_transform(self):
        """ Constructs a simplified version of this Diffeomorhic Map

        The simplified version incorporates the pre-align transform, as well as
        the domain and codomain affine transforms into the displacement field.
        The resulting transformation may be regarded as operating on the
        image spaces given by the domain and codomain discretization. As a
        result, self.prealign, self.disp_grid2world, self.domain_grid2world and
        self.codomain affine will be None (denoting Identity) in the resulting
        diffeomorphic map.
        """
        if self.dim == 2:
            simplify_f = vfu.simplify_warp_function_2d
        else:
            simplify_f = vfu.simplify_warp_function_3d
        # Simplify the forward transform
        D = self.domain_grid2world
        P = self.prealign
        Rinv = self.disp_world2grid
        Cinv = self.codomain_world2grid

        # this is the matrix which we need to multiply the voxel coordinates
        # to interpolate on the forward displacement field ("in"side the
        # 'forward' brackets in the expression above)
        affine_idx_in = mult_aff(Rinv, mult_aff(P, D))

        # this is the matrix which we need to multiply the voxel coordinates
        # to add to the displacement ("out"side the 'forward' brackets in the
        # expression above)
        affine_idx_out = mult_aff(Cinv, mult_aff(P, D))

        # this is the matrix which we need to multiply the displacement vector
        # prior to adding to the transformed input point
        affine_disp = Cinv

        new_forward = simplify_f(self.forward, affine_idx_in,
                                 affine_idx_out, affine_disp,
                                 self.domain_shape)

        # Simplify the backward transform
        C = self.codomain_world2grid
        Pinv = self.prealign_inv
        Dinv = self.domain_world2grid

        affine_idx_in = mult_aff(Rinv, C)
        affine_idx_out = mult_aff(Dinv, mult_aff(Pinv, C))
        affine_disp = mult_aff(Dinv, Pinv)
        new_backward = simplify_f(self.backward, affine_idx_in,
                                  affine_idx_out, affine_disp,
                                  self.codomain_shape)
        simplified = DiffeomorphicMap(self.dim,
                                      self.disp_shape,
                                      None,
                                      self.domain_shape,
                                      None,
                                      self.codomain_shape,
                                      None,
                                      None)
        simplified.forward = new_forward
        simplified.backward = new_backward
        return simplified


class DiffeomorphicRegistration(metaclass=abc.ABCMeta):
    def __init__(self, metric=None):
        """ Diffeomorphic Registration

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
        """Sets the number of iterations at each pyramid level

        Establishes the maximum number of iterations to be performed at each
        level of the Gaussian pyramid, similar to ANTS.

        Parameters
        ----------
        level_iters : list
            the number of iterations at each level of the Gaussian pyramid.
            level_iters[0] corresponds to the finest level, level_iters[n-1]
            the coarsest, where n is the length of the list
        """
        self.levels = len(level_iters) if level_iters else 0
        self.level_iters = level_iters

    @abc.abstractmethod
    def optimize(self):
        """Starts the metric optimization

        This is the main function each specialized class derived from this must
        implement. Upon completion, the deformation field must be available
        from the forward transformation model.
        """

    @abc.abstractmethod
    def get_map(self):
        """
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
        """ Symmetric Diffeomorphic Registration (SyN) Algorithm

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

    def update(self, current_displacement, new_displacement,
               disp_world2grid, time_scaling):
        """Composition of the current displacement field with the given field

        Interpolates new displacement at the locations defined by
        current_displacement. Equivalently, computes the composition C of the
        given displacement fields as C(x) = B(A(x)), where A is
        current_displacement and B is new_displacement. This function is
        intended to be used with deformation fields of the same sampling
        (e.g. to be called by a registration algorithm).

        Parameters
        ----------
        current_displacement : array, shape (R', C', 2) or (S', R', C', 3)
            the displacement field defining where to interpolate
            new_displacement
        new_displacement : array, shape (R, C, 2) or (S, R, C, 3)
            the displacement field to be warped by current_displacement
        disp_world2grid : array, shape (dim+1, dim+1)
            the space-to-grid transform associated with the displacements'
            grid (we assume that both displacements are discretized over the
            same grid)
        time_scaling : float
            scaling factor applied to d2. The effect may be interpreted as
            moving d1 displacements along a factor (`time_scaling`) of d2.

        Returns
        -------
        updated : array, shape (the same as new_displacement)
            the warped displacement field
        mean_norm : the mean norm of all vectors in current_displacement
        """
        sq_field = np.sum((np.array(current_displacement) ** 2), -1)
        mean_norm = np.sqrt(sq_field).mean()
        # We assume that both displacement fields have the same
        # grid2world transform, which implies premult_index=Identity
        # and premult_disp is the world2grid transform associated with
        # the displacements' grid
        self.compose(current_displacement, new_displacement, None,
                     disp_world2grid, time_scaling, current_displacement)

        return np.array(current_displacement), np.array(mean_norm)

    def get_map(self):
        """Return the resulting diffeomorphic map.

        Returns the DiffeomorphicMap registering the moving image towards
        the static image.

        """
        if not hasattr(self, 'static_to_ref'):
            msg = 'Diffeormorphic map can not be obtained without running '
            msg += 'the optimizer. Please call first '
            msg += 'SymmetricDiffeomorphicRegistration.optimize()'
            raise ValueError(msg)
        return self.static_to_ref

    def _connect_functions(self):
        """Assign the methods to be called according to the image dimension

        Assigns the appropriate functions to be called for displacement field
        inversion, Gaussian pyramid, and affine / dense deformation composition
        according to the dimension of the input images e.g. 2D or 3D.
        """
        if self.dim == 2:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point_2d
            self.compose = vfu.compose_vector_fields_2d
        else:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point_3d
            self.compose = vfu.compose_vector_fields_3d

    def _init_optimizer(self, static, moving,
                        static_grid2world, moving_grid2world, prealign):
        """Initializes the registration optimizer

        Initializes the optimizer by computing the scale space of the input
        images and allocating the required memory for the transformation models
        at the coarsest scale.

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization. The
            displacement fields will have the same discretization as the static
            image.
        moving : array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. Since the
            deformation fields' discretization is the same as the static image,
            it is necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the 'prealign'
            matrix
        static_grid2world : array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the static image
        moving_grid2world : array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the moving image
        prealign : array, shape (dim+1, dim+1)
            the affine transformation (operating on the physical space)
            pre-aligning the moving image towards the static

        """
        self._connect_functions()
        # Extract information from affine matrices to create the scale space
        static_direction, static_spacing = \
            get_direction_and_spacings(static_grid2world, self.dim)
        moving_direction, moving_spacing = \
            get_direction_and_spacings(moving_grid2world, self.dim)

        # the images' directions don't change with scale
        self.static_direction = np.eye(self.dim + 1)
        self.moving_direction = np.eye(self.dim + 1)
        self.static_direction[:self.dim, :self.dim] = static_direction
        self.moving_direction[:self.dim, :self.dim] = moving_direction

        # Build the scale space of the input images
        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            logger.info('Applying zero mask: ' + str(self.mask0))

        if self.verbosity >= VerbosityLevels.STATUS:
            logger.info('Creating scale space from the moving image.' +
                        ' Levels: %d. Sigma factor: %f.' %
                        (self.levels, self.ss_sigma_factor))

        self.moving_ss = ScaleSpace(moving, self.levels, moving_grid2world,
                                    moving_spacing, self.ss_sigma_factor,
                                    self.mask0)

        if self.verbosity >= VerbosityLevels.STATUS:
            logger.info('Creating scale space from the static image.' +
                        ' Levels: %d. Sigma factor: %f.' %
                        (self.levels, self.ss_sigma_factor))

        self.static_ss = ScaleSpace(static, self.levels, static_grid2world,
                                    static_spacing, self.ss_sigma_factor,
                                    self.mask0)

        if self.verbosity >= VerbosityLevels.DEBUG:
            logger.info('Moving scale space:')
            for level in range(self.levels):
                self.moving_ss.print_level(level)

            logger.info('Static scale space:')
            for level in range(self.levels):
                self.static_ss.print_level(level)

        # Get the properties of the coarsest level from the static image. These
        # properties will be taken as the reference discretization.
        disp_shape = self.static_ss.get_domain_shape(self.levels-1)
        disp_grid2world = self.static_ss.get_affine(self.levels-1)

        # The codomain discretization of both diffeomorphic maps is
        # precisely the discretization of the static image
        codomain_shape = static.shape
        codomain_grid2world = static_grid2world

        # The forward model transforms points from the static image
        # to points on the reference (which is the static as well). So the
        # domain properties are taken from the static image. Since its the same
        # as the reference, we don't need to pre-align.
        domain_shape = static.shape
        domain_grid2world = static_grid2world
        self.static_to_ref = DiffeomorphicMap(self.dim,
                                              disp_shape,
                                              disp_grid2world,
                                              domain_shape,
                                              domain_grid2world,
                                              codomain_shape,
                                              codomain_grid2world,
                                              None)
        self.static_to_ref.allocate()

        # The backward model transforms points from the moving image
        # to points on the reference (which is the static). So the input
        # properties are taken from the moving image, and we need to pre-align
        # points on the moving physical space to the reference physical space
        # by applying the inverse of pre-align. This is done this way to make
        # it clear for the user: the pre-align matrix is usually obtained by
        # doing affine registration of the moving image towards the static
        # image, which results in a matrix transforming points in the static
        # physical space to points in the moving physical space
        prealign_inv = None if prealign is None else npl.inv(prealign)
        domain_shape = moving.shape
        domain_grid2world = moving_grid2world
        self.moving_to_ref = DiffeomorphicMap(self.dim,
                                              disp_shape,
                                              disp_grid2world,
                                              domain_shape,
                                              domain_grid2world,
                                              codomain_shape,
                                              codomain_grid2world,
                                              prealign_inv)
        self.moving_to_ref.allocate()

    def _end_optimizer(self):
        """Frees the resources allocated during initialization
        """
        del self.moving_ss
        del self.static_ss

    def _iterate(self):
        """Performs one symmetric iteration

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
        # Acquire current resolution information from scale spaces
        current_moving = self.moving_ss.get_image(self.current_level)
        current_static = self.static_ss.get_image(self.current_level)

        current_disp_shape = \
            self.static_ss.get_domain_shape(self.current_level)
        current_disp_grid2world = \
            self.static_ss.get_affine(self.current_level)
        current_disp_world2grid = \
            self.static_ss.get_affine_inv(self.current_level)
        current_disp_spacing = \
            self.static_ss.get_spacing(self.current_level)

        # Warp the input images (smoothed to the current scale) to the common
        # (reference) space at the current resolution
        wstatic = self.static_to_ref.transform_inverse(current_static,
                                                       'linear',
                                                       None,
                                                       current_disp_shape,
                                                       current_disp_grid2world)
        wmoving = self.moving_to_ref.transform_inverse(current_moving,
                                                       'linear',
                                                       None,
                                                       current_disp_shape,
                                                       current_disp_grid2world)
        # Pass both images to the metric. Now both images are sampled on the
        # reference grid (equal to the static image's grid) and the direction
        # doesn't change across scales
        self.metric.set_moving_image(wmoving, current_disp_grid2world,
                                     current_disp_spacing,
                                     self.static_direction)
        self.metric.use_moving_image_dynamics(
            current_moving, self.moving_to_ref.inverse())

        self.metric.set_static_image(wstatic, current_disp_grid2world,
                                     current_disp_spacing,
                                     self.static_direction)
        self.metric.use_static_image_dynamics(
            current_static, self.static_to_ref.inverse())

        # Initialize the metric for a new iteration
        self.metric.initialize_iteration()
        if self.callback is not None:
            self.callback(self, RegistrationStages.ITER_START)

        # Compute the forward step (to be used to update the forward transform)
        fw_step = np.array(self.metric.compute_forward())

        # set zero displacements at the boundary
        fw_step = self.__set_no_boundary_displacement(fw_step)

        # Normalize the forward step
        nrm = np.sqrt(np.sum((fw_step/current_disp_spacing)**2, -1)).max()
        if nrm > 0:
            fw_step /= nrm

        # Add to current total field
        self.static_to_ref.forward, md_forward = self.update(
            self.static_to_ref.forward, fw_step,
            current_disp_world2grid, self.step_length)
        del fw_step

        # Keep track of the forward energy
        fw_energy = self.metric.get_energy()

        # Compose backward step (to be used to update the backward transform)
        bw_step = np.array(self.metric.compute_backward())

        # set zero displacements at the boundary
        bw_step = self.__set_no_boundary_displacement(bw_step)

        # Normalize the backward step
        nrm = np.sqrt(np.sum((bw_step/current_disp_spacing) ** 2, -1)).max()
        if nrm > 0:
            bw_step /= nrm

        # Add to current total field
        self.moving_to_ref.forward, md_backward = self.update(
            self.moving_to_ref.forward, bw_step,
            current_disp_world2grid, self.step_length)
        del bw_step

        # Keep track of the energy
        bw_energy = self.metric.get_energy()
        der = np.inf
        n_iter = len(self.energy_list)
        if len(self.energy_list) >= self.energy_window:
            der = self._get_energy_derivative()

        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            ch = '-' if np.isnan(der) else der
            logger.info('%d:\t%0.6f\t%0.6f\t%0.6f\t%s' %
                        (n_iter, fw_energy, bw_energy,
                         fw_energy + bw_energy, ch))

        self.energy_list.append(fw_energy + bw_energy)

        self.__invert_models(current_disp_world2grid, current_disp_spacing)

        # Free resources no longer needed to compute the forward and backward
        # steps
        if self.callback is not None:
            self.callback(self, RegistrationStages.ITER_END)
        self.metric.free_iteration()

        return der

    def __set_no_boundary_displacement(self, step):
        """ set zero displacements at the boundary

        Parameters
        ----------
        step : array, ndim 2 or 3
            displacements field

        Returns
        -------
        step : array, ndim 2 or 3
            displacements field
        """
        step[0, ...] = 0
        step[:, 0, ...] = 0
        step[-1, ...] = 0
        step[:, -1, ...] = 0
        if self.dim == 3:
            step[:, :, 0, ...] = 0
            step[:, :, -1, ...] = 0
        return step

    def __invert_models(self, current_disp_world2grid, current_disp_spacing):
        """Converting static - moving models in both direction.

        Parameters
        ----------
        current_disp_world2grid : array, shape (3, 3) or  (4, 4)
            the space-to-grid transformation associated to the displacement field
            d (transforming physical space coordinates to voxel coordinates of the
            displacement field grid)
        current_disp_spacing :array, shape (2,) or  (3,)
            the spacing between voxels (voxel size along each axis)
        """

        # Invert the forward model's forward field
        self.static_to_ref.backward = np.array(
            self.invert_vector_field(self.static_to_ref.forward,
                                     current_disp_world2grid,
                                     current_disp_spacing,
                                     self.inv_iter, self.inv_tol,
                                     self.static_to_ref.backward))

        # Invert the backward model's forward field
        self.moving_to_ref.backward = np.array(
            self.invert_vector_field(self.moving_to_ref.forward,
                                     current_disp_world2grid,
                                     current_disp_spacing,
                                     self.inv_iter, self.inv_tol,
                                     self.moving_to_ref.backward))

        # Invert the forward model's backward field
        self.static_to_ref.forward = np.array(
            self.invert_vector_field(self.static_to_ref.backward,
                                     current_disp_world2grid,
                                     current_disp_spacing,
                                     self.inv_iter, self.inv_tol,
                                     self.static_to_ref.forward))

        # Invert the backward model's backward field
        self.moving_to_ref.forward = np.array(
            self.invert_vector_field(self.moving_to_ref.backward,
                                     current_disp_world2grid,
                                     current_disp_spacing,
                                     self.inv_iter, self.inv_tol,
                                     self.moving_to_ref.forward))

    def _approximate_derivative_direct(self, x, y):
        """Derivative of the degree-2 polynomial fit of the given x, y pairs

        Directly computes the derivative of the least-squares-fit quadratic
        function estimated from (x[...],y[...]) pairs.

        Parameters
        ----------
        x : array, shape (n,)
            increasing array representing the x-coordinates of the points to
            be fit
        y : array, shape (n,)
            array representing the y-coordinates of the points to be fit

        Returns
        -------
        y0 : float
            the estimated derivative at x0 = 0.5*len(x)
        """
        x = np.asarray(x)
        y = np.asarray(y)
        X = np.row_stack((x**2, x, np.ones_like(x)))
        XX = X.dot(X.T)
        b = X.dot(y)
        beta = npl.solve(XX, b)
        x0 = 0.5 * len(x)
        y0 = 2.0 * beta[0] * x0 + beta[1]
        return y0

    def _get_energy_derivative(self):
        """Approximate derivative of the energy profile

        Returns the derivative of the estimated energy as a function of "time"
        (iterations) at the last iteration
        """
        n_iter = len(self.energy_list)
        if n_iter < self.energy_window:
            raise ValueError('Not enough data to fit the energy profile')
        x = range(self.energy_window)
        y = self.energy_list[(n_iter - self.energy_window):n_iter]
        ss = sum(y)
        if not ss == 0:  # avoid division by zero
            ss = - ss if ss > 0 else ss
            y = [v / ss for v in y]
        der = self._approximate_derivative_direct(x, y)
        return der

    def _optimize(self):
        """Starts the optimization

        The main multi-scale symmetric optimization algorithm
        """
        self.full_energy_profile = []
        if self.callback is not None:
            self.callback(self, RegistrationStages.OPT_START)
        for level in range(self.levels - 1, -1, -1):
            if self.verbosity >= VerbosityLevels.STATUS:
                logger.info('Optimizing level %d' % level)

            self.current_level = level

            self.metric.set_levels_below(self.levels - level)
            self.metric.set_levels_above(level)

            if level < self.levels - 1:
                expand_factors = \
                    self.static_ss.get_expand_factors(level+1, level)
                new_shape = self.static_ss.get_domain_shape(level)
                self.static_to_ref.expand_fields(expand_factors, new_shape)
                self.moving_to_ref.expand_fields(expand_factors, new_shape)

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
        residual, stats = self.static_to_ref.compute_inversion_error()

        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            logger.info('Static-Reference Residual error: %0.6f (%0.6f)'
                        % (stats[1], stats[2]))

        residual, stats = self.moving_to_ref.compute_inversion_error()

        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            logger.info('Moving-Reference Residual error :%0.6f (%0.6f)'
                        % (stats[1], stats[2]))

        # Compose the two partial transformations
        self.static_to_ref = self.moving_to_ref.warp_endomorphism(
            self.static_to_ref.inverse()).inverse()

        # Report mean and std for the composed deformation field
        residual, stats = self.static_to_ref.compute_inversion_error()
        if self.verbosity >= VerbosityLevels.DIAGNOSE:
            logger.info('Final residual error: %0.6f (%0.6f)' % (stats[1],
                        stats[2]))
        if self.callback is not None:
            self.callback(self, RegistrationStages.OPT_END)

    def optimize(self, static, moving, static_grid2world=None,
                 moving_grid2world=None, prealign=None):
        """
        Starts the optimization

        Parameters
        ----------
        static : array, shape (S, R, C) or (R, C)
            the image to be used as reference during optimization. The
            displacement fields will have the same discretization as the static
            image.
        moving : array, shape (S, R, C) or (R, C)
            the image to be used as "moving" during optimization. Since the
            deformation fields' discretization is the same as the static image,
            it is necessary to pre-align the moving image to ensure its domain
            lies inside the domain of the deformation fields. This is assumed
            to be accomplished by "pre-aligning" the moving image towards the
            static using an affine transformation given by the 'prealign'
            matrix
        static_grid2world : array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the static image
        moving_grid2world : array, shape (dim+1, dim+1)
            the voxel-to-space transformation associated to the moving image
        prealign : array, shape (dim+1, dim+1)
            the affine transformation (operating on the physical space)
            pre-aligning the moving image towards the static

        Returns
        -------
        static_to_ref : DiffeomorphicMap object
            the diffeomorphic map that brings the moving image towards the
            static one in the forward direction (i.e. by calling
            static_to_ref.transform) and the static image towards the
            moving one in the backward direction (i.e. by calling
            static_to_ref.transform_inverse).

        """
        if self.verbosity >= VerbosityLevels.DEBUG:
            if prealign is not None:
                logger.info("Pre-align: " + str(prealign))

        self._init_optimizer(static.astype(floating), moving.astype(floating),
                             static_grid2world, moving_grid2world, prealign)
        self._optimize()
        self._end_optimizer()
        self.static_to_ref.forward = np.array(self.static_to_ref.forward)
        self.static_to_ref.backward = np.array(self.static_to_ref.backward)
        return self.static_to_ref
