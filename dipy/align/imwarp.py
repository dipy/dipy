import numpy as np
import scipy as sp
import numpy.linalg as linalg
import abc
import vector_fields as vfu
from dipy.align import floating
import nibabel as nib
import matplotlib.pyplot as plt


def multi_aff(A, B):
    if A is None:
        return B
    elif B is None:
        return A
    return A.dot(B)


def inv_aff(A):
    if A is None:
        return None
    return np.linalg.inv(A)


def get_direction_and_scalings(affine, dim):
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
    scalings = scalings[:dim]
    A = affine[:dim,:dim]
    return A.dot(np.diag(1.0/scalings)), scalings


def scale_space(image, max_scale, input_affine=None):
    r"""
    
    Returns
    -------
    An iterator to 5 properties of the scale space for each scale:

    filtered : array, shape = image.shape
        filtered (not subsampled) images
    size : array, shape(dim,)
        the shape of each image in the scale space representation
        of the input image, where dim is the dimension of the imput 
        image (either 2 or 3)
    spacing : array, shape(dim,)
        the distance between consecutive voxels along each dimension
    scaling : array, shape(dim,)
        the scale applied along each dimension (the target scaling 
        is 2^k, where k=0,1,...,max_scale is the corresponding scale. 
        If the resolution along some dimensions is lower than
        the highest resolution dimension, then the scaling will be 
        lower along that dimension)
    affine : array, shape(dim+1, dim+1)
        the affine transformation bringing voxel coordinates to physical space

    """
    sigma_factor = 1.2
    dim = len(image.shape)
    input_direction, input_spacing = get_direction_and_scalings(input_affine, dim)
    
    input_size = np.array(image.shape)
    input_spacing = np.array(input_spacing)
    img = (image - image.min())/(image.max() - image.min())
    yield img.astype(floating), input_size.astype(np.int32), input_spacing, np.ones(dim), input_affine
    min_spacing = np.min(input_spacing)
    for i in range(max_scale):
        scaling_factor = 2**(i+1)
        scaling = np.ndarray((dim+1,))
        scaling = np.minimum(scaling_factor * min_spacing / input_spacing, input_size / 32)
        output_spacing = input_spacing * scaling
        extended = np.append(scaling, [1])
        if not input_affine is None:
            print 'extended shape:',extended.shape
            affine = input_affine.dot(np.diag(extended))
        else:
            affine = np.diag(extended)
        output_size = input_size * (input_spacing / output_spacing) + 0.5
        output_size = output_size.astype(np.int32)
        sigmas = sigma_factor * (output_spacing / input_spacing - 1.0)
        #filter along each direction with the appropriate sigma
        filtered = sp.ndimage.filters.gaussian_filter(image, sigmas)
        filtered = (filtered - filtered.min())/(filtered.max() - filtered.min())
        yield filtered.astype(floating), output_size, output_spacing, scaling, affine


def pyramid_gaussian_3D(image, max_layer, mask=None):
    r'''
    Generates a 3D Gaussian Pyramid of max_layer+1 levels from image

    Parameters
    ----------
    image : array, shape (S, R, C)
        the base image (level zero) of the pyramid
    max_layer : int
        the index of the last level of the pyramid (the final pyramid will have
        max_layer+1 levels)
    mask : array, shape (S, R, C)
        a binary mask to be applied to the base image (the upper levels of the
        pyramid will be masked using subsampled versions of this mask)

    Returns
    -------
    new_image : iterator
        the Gaussian Pyramid as an iterator over the pyramid levels
    '''
    yield image.copy().astype(floating)
    for i in range(max_layer):
        new_image=np.array(sp.ndimage.filters.gaussian_filter(image, 2.0/3.0)[::2,::2,::2], dtype = floating)
        if(mask!=None):
            mask=mask[::sc,::sc,::sc]
            new_image*=mask
        image=new_image.copy()
        yield new_image

def pyramid_gaussian_2D(image, max_layer, mask=None):
    r'''
    Generates a 3D Gaussian Pyramid of max_layer+1 levels from image

    Parameters
    ----------
    image : array, shape (R, C)
        the base image (level zero) of the pyramid
    max_layer : int
        the index of the last level of the pyramid (the final pyramid will have
        max_layer+1 levels)
    mask : array, shape (R, C)
        a binary mask to be applied to the base image (the upper levels of the
        pyramid will be masked using subsampled versions of this mask)

    Returns
    -------
    new_image : iterator
        the Gaussian Pyramid as an iterator over the pyramid levels
    '''
    yield image.copy().astype(floating)
    for i in range(max_layer):
        new_image=np.empty(shape=((image.shape[0]+1)//2, (image.shape[1]+1)//2), dtype=floating)
        new_image[...]=sp.ndimage.filters.gaussian_filter(image, 2.0/3.0)[::2,::2]
        if(mask!=None):
            mask=mask[::2,::2]
            new_image*=mask
        image=new_image
        yield new_image

def compose_displacements(new_displacement, current_displacement, affine_inv):
    r"""
    Interpolates current displacement at the locations defined by 
    new_displacement. Equivalently, computes the composition C of the given
    displacement fields as C(x) = B(A(x)), where A is new_displacement and B is 
    currentDisplacement. This function is intended to be used with deformation
    fields of the same sampling (e.g. to be called by a registration algorithm),
    in this case, the premultiplication matrix for the index vector is the 
    identity

    Parameters
    ----------
    new_displacement : array, shape (R, C, 2) or (S, R, C, 3)
        the displacement field defining where to interpolate current_displacement
    current_displacement : array, shape (R', C', 2) or (S', R', C', 3)
        the displacement field to be warped by new_displacement

    Returns
    -------
    updated : array, shape (the same as new_displacement)
        the warped displacement field
    mse : the mean norm of all vectors in current_displacement
    """
    #Compute the premultiplication matrices to be used in the composition:
    #new_displacement is evaluated first, so R1 = affine_new, and 
    #current_displacement is evaluated next, so R2^{-1} = affine_current_inv 
    dim = len(new_displacement.shape) - 1
    premult_index = None
    premult_disp = affine_inv
    
    mse = np.sqrt(np.sum((current_displacement ** 2), -1)).mean()
    if dim == 2:
        updated, stats = vfu.compose_vector_fields_2d(new_displacement,
                                                      current_displacement,
                                                      premult_index, 
                                                      premult_disp,
                                                      1.0)
    else:
        updated, stats = vfu.compose_vector_fields_3d(new_displacement,
                                                      current_displacement,
                                                      premult_index, 
                                                      premult_disp,
                                                      1.0)
    return np.array(updated), np.array(mse)


def scale_affine(affine, factor):
    r"""
    Multiplies the translation part of the affine transformation by a factor
    to be used with upsampled/downsampled images (if the affine transformation
    corresponds to an Image I and we need to apply the corresponding
    transformation to a downsampled version J of I, then the affine matrix
    is the same as for I but the translation is scaled).
    
    Parameters
    ----------
    affine : array, shape (3, 3) or (4, 4)
        the affine matrix to be scaled
    factor : float 
        the scale factor to be applied to affine

    Notes
    -----
    Internally, the affine transformation is applied component-wise instead of
    actually evaluating a matrix-vector product, so the shape of the input
    matrix may even be (2, 3) or (3, 4), since the last row is never accessed.  
    """
    scaled_affine = np.array(affine, dtype = floating)
    domain_dimension = affine.shape[1] - 1
    scaled_affine[:domain_dimension, domain_dimension] *= factor
    return scaled_affine


def compute_warping_affines(T_inv, R, R_inv, A, B):
    r"""
    Computes the affine matrices to be passed to warping functions. After 
    simplifying the domain transformation and
    physical transformation products, the final warping is of the form
    warped[i] = image[Tinv*B*A*R*i + Tinv*B*d1[Rinv*A*R*i]]
    where Tinv is the affine transformation gringing physical points to 
    image's discretization, and R, Rinv transform d1's discretization to 
    physical space and physical space to discretization respectively.
    We require affine_idx_in:=Rinv*A*R, affine_idx_out:=Tinv*B*A*R,
    and affine_disp:=Tinv*B

    """
    if A is None:
        affine_index_in = None
        if R is None:
            out_1 = None
        else:
            out_1 = R
    elif R is None:
        affine_index_in = A.astype(floating)
        out_1 = A
    else:
        affine_index_in = R_inv.dot(A.dot(R)).astype(floating)
        out_1 = A.dot(R)

    if T_inv is None:
        affine_disp = B.astype(floating) if not B is None else None
        if B is None:
            out_2 = None
        else:
            out_2 = B
    elif B is None:
        affine_disp = T_inv.astype(floating)
        out_2 = T_inv
    else:
        affine_disp = T_inv.dot(B).astype(floating)
        out_2 = T_inv.dot(B)

    if out_1 is None:
        affine_index_out = out_2.astype(floating) if not out_2 is None else None
    elif out_2 is None:
        affine_index_out = out_1.astype(floating)
    else:
        affine_index_out = out_2.dot(out_1).astype(floating)
    return affine_index_in, affine_index_out, affine_disp


def compute_composition_affines(T1, T2_inv):
    r"""
    Computes the affine matrices to be passed to displacement field composition
    functions. 

    """
    premult_disp = 


class DiffeomorphicMap(object):

    def __init__(self,
                 dim,
                 forward=None,
                 backward=None,
                 affine_pre=None,
                 affine_post=None,
                 scalings_forward=None,
                 scalings_backward=None,
                 affine_forward=None,
                 affine_backward=None):
        r""" Diffeomorphic Map

        Defines the mapping between two spaces: "reference" and "target".
        The transformations modeled are of the form B*phi(A*x), with inverse
        given by A^{-1}*phi^{-1}(B^{-1}(x)) where A and B are affine matrices 
        and phi is a deformation field. The deformation fields (forward and 
        backward) are discretized using a vector field defined over a regular
        lattice. The properties of the lattice are the following:
        
        scalings_forward, scalings_backward: a vector of dim (either 2 or 3)
            scalars specifying the size of each discrete voxel (e.g. the size
            of each voxel in millimiters). By default, the scalings are 1 along
            every axis.

        affine_forward, affine_backward: a matrix transforming voxel coordinates
            to physical space. By default, the identity is used.

        The combination of an Identity reference matrix and unit scalings (default)
        is equivalent to a diffeomorphic map operating on the voxel space at full
        resolution. 

        Internally, the individual terms of the transformation can be accessed through:
        A : self.affine_pre
        A^{-1} : self.affine_pre_inv
        B : self.affine_post
        B^{-1} : self.affine_post_inv
        phi : self.forward
        phi^{-1} : self.backward
        
        The discretization properties are accessible through:
        self.scalings_forward, 
        self.scalings_backward, 
        self.affine_forward, 
        self.affine_backward 

        Parameters
        ----------
        dim : int (either 2 or 3)
            the dimension of the mapped spaces
        forward : array, shape (R, C, 2) or (S, R, C, 3)
            the forward displacement field mapping the target towards the reference
        backward : array, shape (R', C', 2) or (S', R', C', 3)
            the backward displacement field mapping the reference towards the 
            target(denoted "phi^{-1}" above)
        affine_pre : array, shape (3, 3) or (4, 4)
            the affine matrix pre-multiplying the argument of the forward field
        affine_post : array, shape (3, 3) or (4, 4)
            the affine matrix post-multiplying the argument of the backward field
        scalings_forward : array, shape (2,) or (3,)
            the voxel scalings (dimensions) in the forward field discretization
        scalings_backward : array, shape (2,) or (3,)
            the voxel scalings (dimensions) in the backward field discretization
        affine_forward : array, shape (3, 3) or (4, 4)
            the transformation bringing voxels from the forward field discretization
            to physical space
        affine_backward : array, shape (3, 3) or (4, 4)
            the transformation bringing voxels from the backward field discretization
            to physical space
        """
        self.dim = dim
        self.set_forward(forward, affine_forward, scalings_forward)
        self.set_backward(backward, affine_backward, scalings_backward)
        self.set_affine_pre(affine_pre)
        self.set_affine_post(affine_post)

    def set_affine_pre(self, affine_pre):
        r"""
        Establishes the pre-multiplication affine matrix of this
        transformation and computes its inverse.

        Parameters
        ----------
        affine_pre : array, shape (3, 3) or (4, 4)
            the invertible affine matrix pre-multiplying the argument of the 
            forward field
        """
        if affine_pre == None:
            self.affine_pre = None
            self.affine_pre_inv = None
        else:
            self.affine_pre = np.array(affine_pre, dtype = floating)
            self.affine_pre_inv = np.array(linalg.inv(affine_pre), order='C', dtype = floating)

    def set_affine_post(self, affine_post):
        r"""
        Establishes the post-multiplication affine matrix of this
        transformation and computes its inverse

        Parameters
        ----------
        affine_post : array, shape (3, 3) or (4, 4)
            the invertible affine matrix post-multiplying the argument of the 
            backward field
        """
        if affine_post == None:
            self.affine_post = None
            self.affine_post_inv = None
        else:
            self.affine_post_inv = np.array(linalg.inv(affine_post), order='C', dtype = floating)
            self.affine_post = np.array(affine_post, dtype = floating)

    # def set_sampling_scalings(self, scalings_forward, scalings_backward):
    #     if scalings_forward is None:
    #         self.scalings_forward = None
    #     else:
    #         self.scalings_forward = np.array(scalings_forward, dtype=floating)

    #     if scalings_backward is None:
    #         self.scalings_backward = None
    #     else:
    #         self.scalings_backward = np.array(scalings_backward, dtype=floating)

    # def set_sampling_affines(self, affine_forward, affine_backward):
    #     if affine_forward is None:
    #         self.affine_forward = None
    #         self.affine_forward_inv = None
    #     else:
    #         self.affine_forward = np.array(affine_forward, dtype=floating)
    #         self.affine_forward_inv = np.array(linalg.inv(affine_forward), dtype=floating)

    #     if affine_backward is None:
    #         self.affine_backward = None
    #         self.affine_backward_inv = None
    #     else:
    #         self.affine_backward = np.array(affine_backward, dtype=floating)
    #         self.affine_backward_inv = np.array(linalg.inv(affine_backward), dtype=floating)

    def set_forward(self, forward, affine_forward, scalings_forward):
        r"""
        Establishes the forward displacement field with the sampling 
        properties given by affine_forward and scalings_forward

        Notes
        -----
        This assignment does not compute the inverse of the provided displacement
        field. The user must update the backward displacement field accordingly
        """
        self.forward = forward

        if affine_forward is None:
            self.affine_forward = None
            self.affine_forward_inv = None
        else:
            self.affine_forward = np.array(affine_forward, dtype=floating)
            self.affine_forward_inv = np.array(linalg.inv(affine_forward), dtype=floating)

        if scalings_forward is None:
            self.scalings_forward = None
        else:
            self.scalings_forward = np.array(scalings_forward, dtype=floating)

    def set_backward(self, backward, affine_backward, scalings_backward):
        r"""
        Establishes the backward displacement field

        Notes
        -----
        This assignment does not compute the inverse of the provided displacement
        field. The user must update the backward displacement field accordingly
        """
        self.backward = backward

        if affine_backward is None:
            self.affine_backward = None
            self.affine_backward_inv = None
        else:
            self.affine_backward = np.array(affine_backward, dtype=floating)
            self.affine_backward_inv = np.array(linalg.inv(affine_backward), dtype=floating)

        if scalings_backward is None:
            self.scalings_backward = None
        else:
            self.scalings_backward = np.array(scalings_backward, dtype=floating)

    def _get_full_premultiplier_matrix(self, R_inv, A, R):
        if R is None:
            return A
        else:
            if A is None:
                return None
            else:#R_inv can't be None, since this function is called internally
                return R_inv.dot(A.dot(R))

    def _warp_forward(self, image, affine_inv):
        r"""
        Applies this transformation in the forward direction to the given image
        using tri-linear interpolation

        Parameters
        ----------
        image : array, shape (R, C) or (S, R, C)
            the 2D (dim=2) or 3D (dim=3) image to be warped
        affine_inv : array, shape(dim+1, dim+1)
            the linear transformation bringing points in physical-space coordinates
            to voxel coordinates in image 

        Returns
        -------
        warped : array, shape (R', C') or (S', R', C')
            the warped image, where S', R', C' are the dimensions of the forward
            displacement field
        """
        if image.dtype is not floating:
            image = image.astype(floating)

        affines = compute_warping_affines(T_inv=affine_inv, 
                                          R=self.affine_forward,
                                          R_inv=self.affine_forward_inv, 
                                          A=self.affine_pre, 
                                          B=self.affine_post)

        #Apply the warping
        if self.dim == 3:
            warped = vfu.warp_volume(image,
                                     self.forward,
                                     affines[0],
                                     affines[1],
                                     affines[2])
        else:
            warped = vfu.warp_image(image,
                                    self.forward,
                                    affines[0],
                                    affines[1],
                                    affines[2])
        return np.array(warped)

    def _warp_backward(self, image, affine_inv):
        r"""
        Applies this transformation in the backward direction to the given
        image using tri-linear interpolation

        Parameters
        ----------
        image : array, shape (R, C) or (S, R, C)
            the 2D or 3D image to be warped
        affine_inv : array, shape(dim+1, dim+1)
            the linear transformation bringing points in physical-space coordinates
            to voxel coordinates in image 

        Returns
        -------
        warped : array, shape (R', C') or (S', R', C')
            the warped image, where S', R', C' are the dimensions of the backward
            displacement field
        """
        if image.dtype is not floating:
            image = image.astype(floating)

        affines = compute_warping_affines(T_inv=affine_inv, 
                                          R=self.affine_backward,
                                          R_inv=self.affine_backward_inv, 
                                          A=self.affine_post_inv, 
                                          B=self.affine_pre_inv)

        # full_premult = np.eye(self.dim+1,dtype=np.float32)
        # full_postmult = np.eye(self.dim+1, dtype=np.float32)
        # print '###premult###:', full_premult
        # print '###postmult###:', full_postmult
        #print '### affines ###:',affines[0], affines[1], affines[2]
        

        if self.dim == 3:
            warped = vfu.warp_volume(image,
                                     self.backward,
                                     affines[0],
                                     affines[1],
                                     affines[2])
        else:
            warped = vfu.warp_image(image,
                                    self.backward,
                                    affines[0],
                                    affines[1],
                                    affines[2])
        return np.array(warped)

    def _warp_forward_nn(self, image):
        r"""
        Applies this transformation in the forward direction to the given image
        using nearest-neighbor interpolation

        Parameters
        ----------
        image : array, shape (R, C) or (S, R, C)
            the 2D or 3D image to be warped

        Returns
        -------
        warped : array, shape (R', C') or (S', R', C')
            the warped image, where S', R', C' are the dimensions of the forward
            displacement field
        """
        if image.dtype is np.float64 and floating is not np.float64:
            image = image.astype(floating)
        full_premult = self._get_full_premultiplier_matrix(
            self.affine_forward_inv, self.affine_pre, self.affine_forward)
        if self.dim == 3:
            warped = vfu.warp_volume_nn(image,
                                        self.forward,
                                        self.affine_pre,
                                        self.affine_post)
        else:
            warped = vfu.warp_image_nn(image,
                                       self.forward,
                                       self.affine_pre,
                                       self.affine_post)
        return np.array(warped)

    def _warp_backward_nn(self, image):
        r"""
        Applies this transformation in the backward direction to the given
        image using nearest-neighbor interpolation

        Parameters
        ----------
        image : array, shape (R, C) or (S, R, C)
            the 2D or 3D image to be warped

        Returns
        -------
        warped : array, shape (R', C') or (S', R', C')
            the warped image, where S', R', C' are the dimensions of the backward
            displacement field
        """
        if image.dtype is np.float64 and floating is not np.float64:
            image = image.astype(floating)
        full_premult = self._get_full_premultiplier_matrix(
            self.affine_backward_inv, self.affine_post_inv, self,affine_backward)
        if self.dim == 3:
            warped = vfu.warp_volume_nn(image,
                                        self.backward,
                                        full_premult,
                                        self.affine_pre_inv)
        else:
            warped = vfu.warp_image_nn(image,
                                       self.backward,
                                       full_premult,
                                       self.affine_pre_inv)
        return np.array(warped)

    def transform(self, image, affine_inv, interpolation):
        r"""
        Transforms the given image under this transformation (in the forward 
        direction, i.e. from target space to the reference space) using the
        specified interpolation method.

        Parameters
        ----------
        image : array, shape (R, C) or (S, R, C)
            the image to be transformed
        interpolation : string
            interpolation method to be used for warping, either 'tri' for 
            tri-linear interpolation or 'nn' for nearest-neighbor

        Returns
        -------
        warped : array, shape (R', C') or (S', R', C')
            the warped image, where S', R', C' are the dimensions of the forward
            displacement field
        """
        if interpolation == 'tri':
            return self._warp_forward(image, affine_inv)
        elif interpolation == 'nn':
            return self._warp_forward_nn(image, affine_inv)
        else:
            return None


    def transform_inverse(self, image, affine_inv, interpolation):
        r"""
        Transforms the given image under this transformation (in the backward 
        direction, i.e. from reference space to the target space) using the 
        specified interpolation method.

        Parameters
        ----------
        image : array, shape (R, C) or (S, R, C)
            the image to be transformed
        interpolation : string
            interpolation method to be used for warping, either 'tri' for 
            tri-linear interpolation or 'nn' for nearest-neighbor

        Returns
        -------
        warped : array, shape (R', C') or (S', R', C')
            the warped image, where S', R', C' are the dimensions of the backward
            displacement field
        """
        if interpolation == 'tri':
            return self._warp_backward(image, affine_inv)
        elif interpolation == 'nn':
            return self._warp_backward_nn(image, affine_inv)
        else:
            return None


    def scale_affines(self, factor):
        r"""
        Scales the pre- and post-multiplication affine matrices to be used
        with a scaled domain. It updates the inverses as well.

        Parameters
        ----------
        factor : float
            the scale factor to be applied to the affine matrices
        """
        if self.affine_pre != None:
            self.affine_pre = scale_affine(self.affine_pre, factor)
            self.affine_pre_inv = np.array(linalg.inv(self.affine_pre), dtype = floating)
        if self.affine_post != None:
            self.affine_post = scale_affine(self.affine_post, factor)
            self.affine_post_inv = np.array(linalg.inv(self.affine_post), dtype = floating)

    def upsample(self, new_domain_forward, new_domain_backward):
        r"""
        Upsamples the displacement fields and scales the affine
        pre- and post-multiplication affine matrices by a factor of 2. The
        final outcome is that this transformation can be used in an upsampled
        domain.

        Parameters
        ----------
        new_domain_forward : array, shape (2,) or (3,)
            the shape of the intended upsampled forward displacement field 
            (see notes)
        new_domain_backward : array, shape (2,) or (3,)
            the shape of the intended upsampled backward displacement field 
            (see notes)

        Notes
        -----
        The reason we need to receive the intended domain sizes as parameters
        (and not simply double their size) is because the current sizes may be 
        the result of down-sampling an original displacement field and the user 
        may need to upsample the transformation to go back to the original 
        domain. This way we can register arbitrary image/volume shapes instead
        of only powers of 2. 
        """
        if self.dim == 2:
            if self.forward != None:
                self.forward = 2 * np.array(
                    vfu.upsample_displacement_field(
                        self.forward,
                        np.array(new_domain_forward).astype(np.int32)))
            if self.backward != None:
                self.backward = 2 * np.array(
                    vfu.upsample_displacement_field(
                        self.backward,
                        np.array(new_domain_backward).astype(np.int32)))
        else:
            if self.forward != None:
                sh_fwd = np.array(new_domain_forward, dtype=np.int32)
                sh_bwd = np.array(new_domain_backward, dtype=np.int32)
                self.forward = 2 * np.array(
                    vfu.upsample_displacement_field_3d(
                        self.forward,
                        sh_fwd))
            if self.backward != None:
                self.backward = 2 * np.array(
                    vfu.upsample_displacement_field_3d(
                        self.backward,
                        sh_bwd))
        self.scale_affines(2.0)

    def expand_fields(self, target_scaling_forward, target_shape_forward,
                            target_scaling_backward, target_shape_backward):
        expand_factors_forward = (target_scaling_forward / self.scalings_forward).astype(floating)
        expand_factors_backward = (target_scaling_backward / self.scalings_backward).astype(floating)
        if self.dim == 2:
            expanded_forward = vfu.expand_displacement_field_2d(self.forward, expand_factors_forward, target_shape_forward)
            expanded_backward = vfu.expand_displacement_field_2d(self.backward, expand_factors_backward, target_shape_backward)
        else:
            expanded_forward = vfu.expand_displacement_field_3d(self.forward, expand_factors_forward, target_shape_forward)
            expanded_backward = vfu.expand_displacement_field_3d(self.backward, expand_factors_backward, target_shape_backward)
        ext_fwd = np.ones(self.dim+1)
        ext_fwd[:self.dim] = expand_factors_forward[...]
        ext_bwd = np.ones(self.dim+1)
        ext_bwd[:self.dim] = expand_factors_backward[...]
        expanded_affine_forward = np.diag(ext_fwd).dot(self.affine_forward)
        expanded_affine_backward = np.diag(ext_bwd).dot(self.affine_backward)

        self.set_forward(expanded_forward, expanded_affine_forward, target_scaling_forward)
        self.set_backward(expanded_backward, expanded_affine_backward, target_scaling_backward)

    def compute_inversion_error(self):
        r"""
        Computes the inversion error of the displacement fields

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
        Currently, we only measure the error of the non-linear components of the
        transformation. Assuming the two affine matrices are invertible, the
        error should, in theory, be the same (although in practice, the 
        boundaries may be problematic).
        """
        premult_index = None
        if self.affine_backward_inv is None:
            premult_index = self.affine_forward
        elif self.affine_forward is None:
            premult_index = affine_backward_inv
        else:
            premult_index = self.affine_backward_inv.dot(self.affine_forward)
        premult_disp = self.affine_backward_inv
        if self.dim == 2:
            residual, stats = vfu.compose_vector_fields_2d(self.forward,
                                                           self.backward,
                                                           premult_index,
                                                           premult_disp,
                                                           1.0)
        else:
            residual, stats = vfu.compose_vector_fields_3d(self.forward,
                                                           self.backward,
                                                           premult_index,
                                                           premult_disp,
                                                           1.0)
        return residual, stats

    def compose(self, apply_first):
        r"""
        Computes the composition G(F(.)) where G is this transformation and
        F is the transformation given as parameter. Since we are using the
        model F(x) = B*phi1(A*x) and G(x) = D*phi2(C*x), the final composition
        is of the form  H(x) = D*phi2(C*B*phi1(A*x)). In the comments below, 
        R_1 and R_2 are the transformations bringing voxel coordinates of phi1 and
        phi2 to physical space, respectively

        Parameters
        ----------
        apply_first : DiffeomorphicMap object
            the diffeomorphic map to be composed with this transformation

        Returns
        -------
        composition : DiffeomorphicMap object
            the composition of this Diffeomorphic map and the given map
        """
        #Compute the discretization matrices to be used in the forward
        #composition

        #the first transformation has discretization T1 = A^{-1}R_1 
        #the second has discretization inverse T2inv = R_2^{-1}.dot(C.dot(B)), 
        #so the index premultiplication is 
        #Compute the product C.dot(B)
        C_dot_B = mult_aff(self.affine_pre, apply_first.affine_post)
        
        Ainv = apply_first.affine_pre_inv
        R1 = apply_first.affine_forward
        R2inv = self.affine_forward_inv
        T1 = mult_aff(Ainv, R1)
        T2inv = mult_aff(R2inv, C_dot_B)

        #finaly, the index premultiplication is T2inv.dot(T1) and the
        #displacement premultiplication is just T2inv
        f_premult_index = mult_aff(T2inv, T1)
        f_premult_disp = T2inv

        #Compute the discretization matrices to be used in the backward
        #composition (we apply inverse of self first, then inverse of apply_first)

        #Compute the product B^{-1}.dot(C^{-1})
        Binv_dot_Cinv = mult_aff(apply_first.affine_post_inv, self.affine_pre_inv)

        Ainv = self.affine_post
        R1 = self.affine_backward
        R2inv = apply_first.affine_backward_inv
        T1 = mult_aff(Ainv, R1)
        T2_inv = mult_aff(R2inv, Binv_dot_Cinv)

        #finaly, the index premultiplication is T2inv.dot(T1) and the
        #displacement premultiplication is just T2inv
        b_premult_index = mult_aff(T2inv, T1)
        b_premult_disp = T2inv
        
        if self.dim == 2:
            forward, stats = vfu.compose_vector_fields_2d(apply_first.forward, 
                                                          self.forward,
                                                          f_premult_index,
                                                          f_premult_disp,
                                                          1.0)
            backward, stats = vfu.compose_vector_fields_2d(self.backward,
                                                           apply_first.backward,
                                                           b_premult_index,
                                                           b_premult_disp,
                                                           1.0)
        else:
            forward, stats = vfu.compose_vector_fields_3d(apply_first.forward, 
                                                          self.forward,
                                                          f_premult_index,
                                                          f_premult_disp,
                                                          1.0)
            backward, stats = vfu.compose_vector_fields_3d(self.backward,
                                                           apply_first.backward,
                                                           b_premult_index,
                                                           b_premult_disp,
                                                           1.0)
        composition = DiffeomorphicMap(self.dim,
                                       forward,
                                       backward,
                                       None,
                                       None,
                                       None,
                                       self.scalings_backward,
                                       apply_first.affine_forward,
                                       self.affine_backward)
        return composition

    def inverse(self):
        r"""
        Returns the inverse of this Diffeomorphic Map. Warning: the matrices
        and displacement fields are not copied
        """
        inv = DiffeomorphicMap(self.dim, self.backward, self.forward,
                               self.affine_post_inv, self.affine_pre_inv)
        return inv

    def consolidate(self):
        r"""
        Eliminates the affine transformations from the representation of this
        transformation by appending/prepending them to the deformation fields,
        so that the Diffeomorphic Map is represented as a single deformation field.
        """
        if self.dim == 2:
            vfu.prepend_affine_to_displacement_field_2d(
                self.forward, self.affine_pre)
            vfu.append_affine_to_displacement_field_2d(
                self.forward, self.affine_post)
            vfu.prepend_affine_to_displacement_field_2d(
                self.backward, self.affine_post_inv)
            vfu.append_affine_to_displacement_field_2d(
                self.backward, self.affine_pre_inv)
        else:
            vfu.prepend_affine_to_displacement_field_3d(
                self.forward, self.affine_pre)
            vfu.append_affine_to_displacement_field_3d(
                self.forward, self.affine_post)
            vfu.prepend_affine_to_displacement_field_3d(
                self.backward, self.affine_post_inv)
            vfu.append_affine_to_displacement_field_3d(
                self.backward, self.affine_pre_inv)
        self.affine_post = None
        self.affine_pre = None
        self.affine_post_inv = None
        self.affine_pre_inv = None


class DiffeomorphicRegistration(object):

    def __init__(self,
                 metric=None,
                 update_function=None):
        r""" Diffeomorphic Registration

        This abstract class defines the interface to be implemented by any
        optimization algorithm for diffeomorphic Registration.

        Parameters
        ----------
        metric : SimilarityMetric object
            the object measuring the similarity of the two images. The registration 
            algorithm will minimize (or maximize) the provided similarity.
        update_function : function
            the function to be applied to perform a small deformation to a 
            displacement field (the small deformation is given as a deformation 
            field as well). An update function may for example compute the composition
            of the two displacement fields or the sum of them, etc.
        """
        self.metric = metric
        self.dim = metric.dim
        if update_function is None:
            self.update = compose_displacements
        else:
            self.update = update_function

    def set_static_image(self, static, static_affine):
        r"""
        Establishes the static image to be used by this registration optimizer

        Parameters
        ----------
        static : array, shape (R, C) or (S, R, C)
            the static image, consisting of R rows and C columns (and S slices,
            if 3D)
        """
        if static is None:
            return
        self.static = static.astype(floating)
        self.static_affine = static_affine
        if not static_affine is None:
            self.static_affine_inv = np.linalg.inv(static_affine)
        else:
            self.static_affine_inv = None
        self.static_direction, self.static_scalings = \
            get_direction_and_scalings(static_affine, self.dim)

    def set_moving_image(self, moving, moving_affine):
        r"""
        Establishes the moving image to be used by this registration optimizer.

        Parameters
        ----------
        static : array, shape (R, C) or (S, R, C)
            the static image, consisting of R rows and C columns (and S slices,
            if 3D)
        """
        if moving is None:
            return
        self.moving = moving.astype(floating)
        self.moving_affine = moving_affine
        if not moving_affine is None:
            self.moving_affine_inv = np.linalg.inv(moving_affine)
        else:
            self.moving_affine_inv = None
        self.moving_direction, self.moving_scalings = \
            get_direction_and_scalings(moving_affine, self.dim)


    # def _set_affine_init(self, affine_init):
    #     r"""
    #     --Deprecated--
    #     Establishes the affine transformation the diffeomorphic registration
    #     starts from. Initializes the appropriate Diffeomorphic transformation
    #     objects from the given affine transformation

    #     Parameters
    #     ----------
    #     affine_init : array, shape (3, 3) or (4, 4)
    #         the initial affine transformation "roughly" aligning the moving
    #         image towards the static
    #     """
    #     inv_affine_init = None
    #     if affine_init != None:
    #         inv_affine_init = np.array(np.linalg.inv(affine_init), dtype = floating)
    #     self.forward_model = DiffeomorphicMap(self.dim, None, None, None, None)
    #     self.backward_model = DiffeomorphicMap(self.dim, None, None, 
    #                                            inv_affine_init, None)

    # def set_static_affine(self, static_affine):
    #     r"""
    #     Establishes the affine transformation mapping index coordinates of the static
    #     image to physical coordinates. This transformation affects the forward transformation
    #     model directly

    #     Parameters
    #     ----------
    #     static_affine : array, shape (3, 3) or (4, 4)
    #         the affine transformation mapping index coordinates of the static
    #         image to physical coordinates
            
    #     """
    #     inv_static_affine = None
    #     if static_affine != None:
    #         inv_static_affine = np.array(np.linalg.inv(static_affine), dtype = floating)
    #         static_affine = static_affine.astype(floating)
    #     self.forward_model = DiffeomorphicMap(self.dim, None, None, static_affine, None)
    #     print 'Static:',static_affine

    # def set_moving_affine(self, moving_affine):
    #     r"""
    #     Establishes the affine transformation mapping index coordinates of the moving
    #     image to physical coordinates. This transformation affects the backward transformation
    #     model directly

    #     Parameters
    #     ----------
    #     moving_affine : array, shape (3, 3) or (4, 4)
    #         the affine transformation mapping index coordinates of the moving
    #         image to physical coordinates
            
    #     """
    #     inv_moving_affine = None
    #     if moving_affine != None:
    #         inv_moving_affine = np.array(np.linalg.inv(moving_affine), dtype = floating)
    #         moving_affine = moving_affine.astype(floating)
    #     self.backward_model = DiffeomorphicMap(self.dim, None, None, moving_affine, None)
    #     print 'Moving:',moving_affine

    # def set_initial_reference_affine(self, initial_reference_affine):
    #     r"""
    #     The reference grid defines where the static and moving images will be sampled on.
    #     Typically, it will be the static image's grid. The initial_reference_affine matrix
    #     transforms index coordinates in the reference grid to physical space.
    #     """
    #     self.initial_reference_affine = initial_reference_affine

    def set_opt_iter(self, opt_iter):
        r"""
        Establishes the maximum number of iterations to be performed at each
        level of the Gaussian pyramid, similar to ANTS.

        Parameters
        ----------
        opt_iter : list
            the number of iterations at each level of the Gaussian pyramid.
            opt_iter[0] corresponds to the finest level, opt_iter[n-1] the
            coarcest, where n is the length of the list
        """
        self.levels = len(opt_iter) if opt_iter else 0
        self.opt_iter = opt_iter

    @abc.abstractmethod
    def optimize(self):
        r"""
        This is the main function each especialized class derived from this must
        implement. Upon completion, the deformation field must be available from
        the forward transformation model.
        """
        pass

    def get_forward(self):
        r"""
        Returns the forward model's forward deformation field
        """
        return self.forward_model.forward

    def get_backward(self):
        r"""
        Returns the forward model's backward (inverse) deformation field
        """
        return self.forward_model.backward


class SymmetricDiffeomorphicRegistration(DiffeomorphicRegistration):
    def __init__(self,
                 metric=None,
                 opt_iter = [25, 100, 100],
                 opt_tol = 1e-4,
                 inv_iter = 40,
                 inv_tol = 1e-3,
                 call_back = None,
                 update_function=None):
        r""" Symmetric Diffeomorphic Registration (SyN) Algorithm
        Performs the multi-resolution optimization algorithm for non-linear
        registration using a given similarity metric and update rule (this
        scheme was inspider on the ANTS package).

        Parameters
        ----------
        metric : SimilarityMetric object
            the metric to be optimized
        opt_iter : list of int
            the number of iterations at each level of the Gaussian Pyramid (the
            length of the list defines the number of pyramid levels to be 
            used)
        opt_tol : float
            the optimization will stop when the estimated derivative of the
            energy profile w.r.t. time falls below this threshold
        inv_iter : int
            the number of iterations to be performed by the displacement field 
            inversion algorithm
        inv_tol : float
            the displacement field inversion algorithm will stop iterating
            when the inversion error falls below this threshold
        call_back : function(SymmetricDiffeomorphicRegistration)
            a function receiving a SymmetricDiffeomorphicRegistration object 
            to be called after each iteration (this optimizer will call this
            function passing self as parameter)
        update_function : function
            the function to be applied to update the displacement field after
            each iteration. By default, it will use the displacement field
            composition
        """
        super(SymmetricDiffeomorphicRegistration, self).__init__(
                metric, update_function)
        self.set_opt_iter(opt_iter)
        self.opt_tol = opt_tol
        self.inv_tol = inv_tol
        self.inv_iter = inv_iter
        self.call_back = call_back
        self.energy_window = 12
        self.energy_list = []
        self.full_energy_profile = []
        self.verbosity = 1

    def _connect_functions(self):
        r"""
        Assigns the appropriate functions to be called for displacement field
        inversion, Gaussian pyramid, and affine/dense deformation composition
        according to the dimension of the input images e.g. 2D or 3D.
        """
        if self.dim == 2:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point_2d
            self.generate_pyramid = pyramid_gaussian_2D
            self.append_affine = vfu.append_affine_to_displacement_field_2d
            self.prepend_affine = vfu.prepend_affine_to_displacement_field_2d
        else:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point_3d
            self.generate_pyramid = pyramid_gaussian_3D
            self.append_affine = vfu.append_affine_to_displacement_field_3d
            self.prepend_affine = vfu.prepend_affine_to_displacement_field_3d

    def _check_ready(self):
        r"""
        Verifies that the configuration of the optimizer and input data are
        consistent and the optimizer is ready to run
        """
        ready = True
        if self.static == None:
            ready = False
            print('Error: static image not set.')
        elif self.dim != len(self.static.shape):
            ready = False
            print('Error: inconsistent dimensions. Last dimension update: %d.'
                  'static image dimension: %d.' % (self.dim,
                                                  len(self.static.shape)))
        if self.moving == None:
            ready = False
            print('Error: Moving image not set.')
        elif self.dim != len(self.moving.shape):
            ready = False
            print('Error: inconsistent dimensions. Last dimension update: %d.'
                  'Moving image dimension: %d.' % (self.dim,
                                                   len(self.moving.shape)))
        if self.metric == None:
            ready = False
            print('Error: Similarity metric not set.')
        if self.update == None:
            ready = False
            print('Error: Update rule not set.')
        if self.opt_iter == None:
            ready = False
            print('Error: Maximum number of iterations per level not set.')
        return ready

    def _init_optimizer(self):
        r"""
        Computes the Gaussian Pyramid of the input images and allocates
        the required memory for the transformation models at the coarcest
        scale.
        """
        ready = self._check_ready()
        self._connect_functions()
        if not ready:
            print 'Not ready'
            return False
        #build the scale space of the input images
        self.moving_ss = [(filtered, size, spacing, scaling, affine) 
            for (filtered, size, spacing, scaling, affine) in 
                scale_space(self.moving, self.levels - 1, self.moving_affine)]
        self.static_ss = [(filtered, size, spacing, scaling, affine) 
            for (filtered, size, spacing, scaling, affine) in 
                scale_space(self.static, self.levels - 1, self.static_affine)]

        print 'Moving scale space:'
        for scale_info in self.moving_ss:
            print scale_info[1], scale_info[2], scale_info[4]
        print 'Static scale space:'
        for scale_info in self.static_ss:
            print scale_info[1], scale_info[2], scale_info[4]

        #Create the displacement fields of the forward diffeomorphic transformation
        #at the coarcest resolution
        starting_forward = np.zeros(
            shape=tuple(self.static_ss[self.levels - 1][1]) + (self.dim,),
            dtype=floating)
        starting_forward_inv = np.zeros(
            shape=tuple(self.static_ss[self.levels - 1][1]) + (self.dim,),
            dtype=floating)
        self.forward_model.set_forward(starting_forward, 
                                       self.static_ss[self.levels - 1][4], 
                                       self.static_ss[self.levels - 1][2])
        self.forward_model.set_backward(starting_forward_inv, 
                                       self.static_ss[self.levels - 1][4], 
                                       self.static_ss[self.levels - 1][2])
        
        #Create the displacement fields of the backward diffeomorphic transformation
        #at the coarcest resolution
        starting_backward = np.zeros(
            shape=tuple(self.moving_ss[self.levels - 1][1]) + (self.dim,),
            dtype=floating)
        starting_backward_inverse = np.zeros(
            shape=tuple(self.static_ss[self.levels - 1][1]) + (self.dim,),
            dtype=floating)
        self.backward_model.set_forward(starting_backward, 
                                        self.moving_ss[self.levels - 1][4], 
                                        self.moving_ss[self.levels - 1][2])
        self.backward_model.set_backward(starting_backward_inverse,
                                         self.static_ss[self.levels - 1][4], 
                                         self.static_ss[self.levels - 1][2])

    def _end_optimizer(self):
        r"""
        Frees the resources allocated during initialization
        """
        del self.moving_ss
        del self.static_ss

    def _iterate(self):
        r"""
        Performs one symmetric iteration:
            1.Compute forward
            2.Compute backward
            3.Update forward
            4.Update backward
            5.Compute inverses
            6.Invert the inverses to improve invertibility
        """
        #Warp the input images (smoothed to the current scale) to the common (reference) space
        wmoving = self.backward_model.transform_inverse(self.current_moving, self.moving_affine_inv, 'tri')
        wstatic = self.forward_model.transform_inverse(self.current_static, self.static_affine_inv, 'tri')
        # plt.figure()
        # plt.subplot(1,2,1)
        # plt.imshow(wmoving, cmap = plt.cm.gray)
        # plt.subplot(1,2,2)
        # plt.imshow(wstatic, cmap = plt.cm.gray)

        
        #Pass both images to the metric
        self.metric.set_moving_image(wmoving)
        self.metric.use_moving_image_dynamics(
            self.current_moving, self.backward_model.inverse())
        self.metric.set_static_image(wstatic)
        self.metric.use_static_image_dynamics(
            self.current_static, self.forward_model.inverse())

        #Initialize the metric for a new iteration
        self.metric.initialize_iteration()

        #Allocate memory to keep the update fields (direct and inverse for 
        #both transformations of the mid-point SyN algorithm)
        ff_shape = np.array(self.forward_model.forward.shape).astype(np.int32)
        fb_shape = np.array(self.forward_model.backward.shape).astype(np.int32)
        bf_shape = np.array(self.backward_model.forward.shape).astype(np.int32)
        bb_shape = np.array(
            self.backward_model.backward.shape).astype(np.int32)

        #Free some memory (useful when usinf double precision)
        del self.forward_model.backward
        del self.backward_model.backward

        #Compute the forward step (to be used to update the forward transform)
        #Note that fw_step's sampling is the same as the current forward model's 
        fw_step = np.array(self.metric.compute_forward())
        nrm = np.sqrt(np.sum((fw_step/self.current_static_spacing)**2, -1)).max()
        fw_step*=(0.25/nrm)
        
        self.forward_model.forward, md_forward = self.update(
            self.forward_model.forward, fw_step, 
            self.forward_model.affine_forward_inv)
        del fw_step

        #Keep track of the forward energy
        fw_energy = self.metric.get_energy()

        #Compose the backward step (to be used to update the backward transform)
        #Note that bw_step's sampling is the same as the current backward model's 
        bw_step = np.array(self.metric.compute_backward())
        nrm = np.sqrt(np.sum((bw_step/self.current_moving_spacing)**2, -1)).max()
        bw_step*=(0.25/nrm)

        self.backward_model.forward, md_backward = self.update(
            self.backward_model.forward, bw_step, 
            self.backward_model.affine_forward_inv)
        del bw_step

        #Keep track of the energy
        bw_energy = self.metric.get_energy()
        der = '-'
        n_iter = len(self.energy_list)
        if len(self.energy_list) >= self.energy_window:
            der = self._get_energy_derivative()
        if self.verbosity > 1:
            print(
                '%d:\t%0.6f\t%0.6f\t%0.6f\t%s' % (n_iter, fw_energy, bw_energy,
                                                  fw_energy + bw_energy, der))
        self.energy_list.append(fw_energy + bw_energy)

        #Free resources no longer needed to compute the forward and backward steps
        self.metric.free_iteration()

        #Invert the current reformation fields
        inv_iter = self.inv_iter
        inv_tol = self.inv_tol
        self.forward_model.backward = np.array(
            self.invert_vector_field(
                self.forward_model.forward,
                self.forward_model.affine_forward,
                self.forward_model.affine_forward_inv,
                fb_shape,
                self.forward_model.affine_backward,
                self.forward_model.affine_backward_inv,
                inv_iter, inv_tol, None))
        self.backward_model.backward = np.array(
            self.invert_vector_field(
                self.backward_model.forward,
                self.backward_model.affine_forward,
                self.backward_model.affine_forward_inv,
                bb_shape,
                self.backward_model.affine_backward,
                self.backward_model.affine_backward_inv,
                inv_iter, inv_tol, None))

        #Invert the inverses
        self.forward_model.forward = np.array(
            self.invert_vector_field(
                self.forward_model.backward,
                self.forward_model.affine_backward,
                self.forward_model.affine_backward_inv,
                ff_shape,
                self.forward_model.affine_forward,
                self.forward_model.affine_forward_inv,
                inv_iter, inv_tol, self.forward_model.forward))
        self.backward_model.forward = np.array(
            self.invert_vector_field(
                self.backward_model.backward,
                self.backward_model.affine_backward,
                self.backward_model.affine_backward_inv,
                bf_shape,
                self.backward_model.affine_forward,
                self.backward_model.affine_forward_inv,
                inv_iter, inv_tol, self.backward_model.forward))

        #We finished the iteration, report using the provided callback
        if self.call_back is not None:
            self.call_back(self)
        return 1 if der == '-' else der

    def _approximate_derivative_direct(self, x, y):
        r"""
        Directly computes the derivative of the least-squares-fit quadratic
        function estimated from (x[...],y[...]) pairs.

        Parameters
        ----------
        x : array, shape(n,)
            increasing array representing the x-coordinates of the points to be fit
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
        beta = np.linalg.solve(XX,b)
        x0 = 0.5 * len(x)
        y0 = 2.0 * beta[0] * (x0) + beta[1]
        return y0

    def _get_energy_derivative(self):
        r"""
        Returns the derivative of the estimated energy as a function of "time"
        (iterations) at the last iteration
        """
        n_iter = len(self.energy_list)
        if n_iter < self.energy_window:
            print 'Error: attempting to fit the energy profile with less points (', n_iter, ') than required (energy_window=', self.energy_window, ')'
            return 1
        x = range(self.energy_window)
        y = self.energy_list[(n_iter - self.energy_window):n_iter]
        ss = sum(y)
        if(ss > 0):
            ss *= -1
        y = [v / ss for v in y]
        der = self._approximate_derivative_direct(x,y)
        return der
    
    def _optimize(self):
        r"""
        The main multi-scale symmetric optimization algorithm
        """
        self.full_energy_profile = []
        for level in range(self.levels - 1, -1, -1):
            if self.verbosity > 0:
                print('Optimizing level %d'%(level,))
                
            #self.moving_ss[level][0] = [(filtered, size, spacing, scaling) ...
            self.current_static = self.static_ss[level][0]
            self.current_moving = self.moving_ss[level][0]

            self.current_static_affine = self.static_ss[level][4]
            self.current_moving_affine = self.moving_ss[level][4]

            self.current_static_spacing = self.static_ss[level][2]
            self.current_moving_spacing = self.moving_ss[level][2]
            
            self.metric.use_original_static_image(self.current_static)
            self.metric.use_original_moving_image(self.current_moving)
            
            self.metric.set_levels_below(self.levels - level)
            self.metric.set_levels_above(level)
            if self.verbosity > 0:
                print '***************Before***************'
                print 'fw scalings:',self.forward_model.scalings_forward, self.forward_model.scalings_backward
                print 'fw affines:',self.forward_model.affine_forward, self.forward_model.affine_backward, 
                print 'bw scalings:',self.backward_model.scalings_forward, self.backward_model.scalings_backward
                print 'bw affines:',self.backward_model.affine_forward, self.backward_model.affine_backward


            if level < self.levels - 1:
                self.forward_model.expand_fields(self.static_ss[level][2], 
                                                 self.static_ss[level][1],
                                                 self.static_ss[level][2], 
                                                 self.static_ss[level][1])
                self.backward_model.expand_fields(self.moving_ss[level][2],
                                                  self.moving_ss[level][1],
                                                  self.static_ss[level][2], 
                                                  self.static_ss[level][1])
            if self.verbosity > 0:
                print '***************After***************'
                print 'fw scalings:',self.forward_model.scalings_forward, self.forward_model.scalings_backward
                print 'fw affines:',self.forward_model.affine_forward, self.forward_model.affine_backward, 
                print 'bw scalings:',self.backward_model.scalings_forward, self.backward_model.scalings_backward
                print 'bw affines:',self.backward_model.affine_forward, self.backward_model.affine_backward
                print '***********************************'

            niter = 0
            self.full_energy_profile.extend(self.energy_list)
            self.energy_list = []
            derivative = 1
            while ((niter < self.opt_iter[level]) and (self.opt_tol < derivative)):
                niter += 1
                derivative = self._iterate()
            plt.figure()
            plt.subplot(1,2,1)
            wmoving = self.backward_model.transform_inverse(self.moving, self.moving_affine, 'tri')
            plt.imshow(wmoving, cmap = plt.cm.gray)
            plt.subplot(1,2,2)
            wstatic = self.forward_model.transform_inverse(self.static, self.static_affine, 'tri')
            plt.imshow(wstatic, cmap = plt.cm.gray)
        # Reporting mean and std in stats[1] and stats[2]
        residual, stats = self.forward_model.compute_inversion_error()
        if self.verbosity > 0:
            print('Forward Residual error: %0.6f (%0.6f)'
                  % (stats[1], stats[2]))
        residual, stats = self.backward_model.compute_inversion_error()
        if self.verbosity > 0:
            print('Backward Residual error :%0.6f (%0.6f)'
                  % (stats[1], stats[2]))

        # Compose the two partial transformations
        self.forward_model = self.backward_model.inverse().compose(
            self.forward_model)

        # Put affines inside the deformation field
        self.forward_model.consolidate()
        del self.backward_model
        
        # Report mean and std for the composed deformation field
        residual, stats = self.forward_model.compute_inversion_error()
        if self.verbosity > 0:
            print('Final residual error: %0.6f (%0.6f)'
                  % (stats[1], stats[2]))

    def initialize_transforms(self, static_affine, moving_affine, pre_align):
        print "Moving scalings:", self.moving_scalings
        print "Static scalings:", self.static_scalings
        print "Moving dir:", self.moving_direction
        print "Static dir:", self.static_direction

        pre_align = None
        inv_pre_align = None
        if not pre_align is None:
            inv_pre_align = np.array(np.linalg.inv(pre_align), dtype = floating)
            pre_align = pre_align

        #The forward model transforms the static image to the reference space, which is the 
        #same as the static domain, so the discretization scalings and affines are the same
        self.forward_model = DiffeomorphicMap(self.dim, None, None, None, None)

        #The backward model transforms moving points to reference points, so the discretization
        #of the forward direction of the backward model corresponds to the moving image zooms 
        #and affine, while the discretization of the backward direction of the backward model
        #corresponds to the static zooms and affine 
        self.backward_model = DiffeomorphicMap(self.dim, None, None, inv_pre_align, None)

    def optimize(self, static, moving, static_affine=None, moving_affine=None, pre_align=None):
        r"""
        Starts the optimnization

        Parameters
        ----------
        static : array, shape (R, C) or (S, R, C)
            the static (reference) image
        moving : array, shape (R, C) or (S, R, C)
            the moving (target) image to be warped towards static
        static_affine: array, shape (3, 3) or (4, 4)
            the affine transformation bringing the static image to physical
            space
        moving_affine: array, shape (3, 3) or (4, 4)
            the affine transformation bringing the moving image to physical
            space
        
        Returns
        -------
        forward_model : DiffeomorphicMap object
            the diffeomorphic map that brings the moving image towards the
            static one in the forward direction (i.e. by calling 
            forward_model.transform) and the static image towards the
            moving one in the backward direction (i.e. by calling 
            forward_model.transform_inverse). 

        """
        print 'pre_align:', pre_align
        self.set_static_image(static, static_affine)
        self.set_moving_image(moving, moving_affine)

        self.initialize_transforms(static_affine, moving_affine, pre_align)

        self._init_optimizer()
        self._optimize()
        self._end_optimizer()
        return self.forward_model
