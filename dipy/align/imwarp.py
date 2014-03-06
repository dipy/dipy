import numpy as np
import numpy.linalg as linalg
import abc
import vector_fields as vfu
import registration_common as rcommon
from dipy.align import floating


def compose_displacements(new_displacement, current_displacement):
    r"""
    Interpolates current displacement at the locations defined by 
    new_displacement. Equivalently, computes the composition C of the given 
    displacement fields as C(x) = B(A(x)), where A is new_displacement and B is 
    currentDisplacement

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
    dim = len(new_displacement.shape) - 1
    mse = np.sqrt(np.sum((current_displacement ** 2), -1)).mean()
    if dim == 2:
        updated, stats = vfu.compose_vector_fields(new_displacement,
                                                   current_displacement)
    else:
        updated, stats = vfu.compose_vector_fields_3d(new_displacement,
                                                      current_displacement)
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


class DiffeomorphicMap(object):

    def __init__(self,
                 dim,
                 forward=None,
                 backward=None,
                 affine_pre=None,
                 affine_post=None):
        r""" Diffeomorphic Map

        Defines the mapping between two spaces: "reference" and "target".
        The transformations modeled are of the form B*phi(A*x), with inverse
        given by A^{-1}*phi^{-1}(B^{-1}(x)) where A and B are affine matrices 
        and phi is a deformation field.
        Internally, the individual terms of the transformation can be accessed
        through:
        A : self.affine_pre
        A^{-1} : self.affine_pre_inv
        B : self.affine_post
        B^{-1} : self.affine_post_inv
        phi : self.forward
        phi^{-1} : self.backward

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
        """
        self.dim = dim
        self.set_forward(forward)
        self.set_backward(backward)
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

    def set_forward(self, forward):
        r"""
        Establishes the forward displacement field

        Notes
        -----
        This assignment does not compute the inverse of the provided displacement
        field. The user must update the backward displacement field accordingly
        """
        self.forward = forward

    def set_backward(self, backward):
        r"""
        Establishes the backward displacement field

        Notes
        -----
        This assignment does not compute the inverse of the provided displacement
        field. The user must update the backward displacement field accordingly
        """
        self.backward = backward

    def _warp_forward(self, image):
        r"""
        Applies this transformation in the forward direction to the given image
        using tri-linear interpolation

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
        if self.dim == 3:
            warped = vfu.warp_volume(image,
                                     self.forward,
                                     self.affine_pre,
                                     self.affine_post)
        else:
            warped = vfu.warp_image(image,
                                    self.forward,
                                    self.affine_pre,
                                    self.affine_post)
        return np.array(warped)

    def _warp_backward(self, image):
        r"""
        Applies this transformation in the backward direction to the given
        image using tri-linear interpolation

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
        if self.dim == 3:
            warped = vfu.warp_volume(image,
                                     self.backward,
                                     self.affine_post_inv,
                                     self.affine_pre_inv)
        else:
            warped = vfu.warp_image(image,
                                    self.backward,
                                    self.affine_post_inv,
                                    self.affine_pre_inv)
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
        if self.dim == 3:
            warped = vfu.warp_volume_nn(image,
                                        self.backward,
                                        self.affine_post_inv,
                                        self.affine_pre_inv)
        else:
            warped = vfu.warp_image_nn(image,
                                       self.backward,
                                       self.affine_post_inv,
                                       self.affine_pre_inv)
        return np.array(warped)

    def transform(self, image, interpolation):
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
            return self._warp_forward(image)
        elif interpolation == 'nn':
            return self._warp_forward_nn(image)
        else:
            return None


    def transform_inverse(self, image, interpolation):
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
            return self._warp_backward(image)
        elif interpolation == 'nn':
            return self._warp_backward_nn(image)
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
        if self.dim == 2:
            residual, stats = vfu.compose_vector_fields(self.forward,
                                                        self.backward)
        else:
            residual, stats = vfu.compose_vector_fields_3d(self.forward,
                                                           self.backward)
        return residual, stats

    def compose(self, apply_first):
        r"""
        Computes the composition G(F(.)) where G is this transformation and
        F is the transformation given as parameter

        Parameters
        ----------
        apply_first : DiffeomorphicMap object
            the diffeomorphic map to be composed with this transformation

        Returns
        -------
        composition : DiffeomorphicMap object
            the composition of this Diffeomorphic map and the given map
        """
        B = apply_first.affine_post
        C = self.affine_pre
        if B == None:
            affine_prod = C
        elif C == None:
            affine_prod = B
        else:
            affine_prod = C.dot(B)
        if affine_prod != None:
            affine_prod_inv = np.array(linalg.inv(affine_prod), order='C', dtype = floating)
        else:
            affine_prod_inv = None
        if self.dim == 2:
            forward = apply_first.forward.copy()
            vfu.append_affine_to_displacement_field_2d(forward, affine_prod)
            forward, stats = vfu.compose_vector_fields(forward,
                                                       self.forward)
            backward = self.backward.copy()
            vfu.append_affine_to_displacement_field_2d(
                backward, affine_prod_inv)
            backward, stats = vfu.compose_vector_fields(backward,
                                                        apply_first.backward)
        else:
            forward = apply_first.forward.copy()
            vfu.append_affine_to_displacement_field_3d(forward, affine_prod)
            forward, stats = vfu.compose_vector_fields_3d(forward,
                                                          self.forward)
            backward = self.backward.copy()
            vfu.append_affine_to_displacement_field_3d(
                backward, affine_prod_inv)
            backward, stats = vfu.compose_vector_fields_3d(backward,
                                                           apply_first.backward)
        composition = DiffeomorphicMap(self.dim,
                                       forward,
                                       backward,
                                       apply_first.affine_pre,
                                       self.affine_post)
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
                 dim=3,
                 static=None,
                 moving=None,
                 affine_init=None,                 
                 update_function=None):
        r""" Diffeomorphic Registration

        This abstract class defines the interface to be implemented by any
        optimization algorithm for diffeomorphic Registration.

        Parameters
        ----------
        metric : SimilarityMetric object
            the object measuring the similarity of the two images. The registration 
            algorithm will minimize (or maximize) the provided similarity.
        dim : int (either 2 or 3)
            the dimension of the diffeomorphism domain. Default 3.
        static : array, shape (R, C) or (S, R, C)
            the static (reference) image
        moving : array, shape (R, C) or (S, R, C)
            the moving (target) image to be warped towards static
        affine_init : array, shape (3, 3) or (4, 4)
            the initial affine transformation aligning the moving towards the 
            reference image
        update_function : function
            the function to be applied to perform a small deformation to a 
            displacement field (the small deformation is given as a deformation 
            field as well). An update function may for example compute the composition
            of the two displacement fields or the sum of them, etc.
        """
        self.dim = dim
        self.set_static_image(static)
        self.set_moving_image(moving)
        self.set_affine_init(affine_init)
        self.metric = metric
        self.update = update_function

    def set_static_image(self, static):
        r"""
        Establishes the static image to be used by this registration optimizer

        Parameters
        ----------
        static : array, shape (R, C) or (S, R, C)
            the static image, consisting of R rows and C columns (and S slices,
            if 3D)
        """
        self.static = static

    def set_moving_image(self, moving):
        r"""
        Establishes the moving image to be used by this registration optimizer.

        Parameters
        ----------
        static : array, shape (R, C) or (S, R, C)
            the static image, consisting of R rows and C columns (and S slices,
            if 3D)
        """
        self.moving = moving

    def set_affine_init(self, affine_init):
        r"""
        Establishes the affine transformation the diffeomorphic registration
        starts from. Initializes the appropriate Diffeomorphic transformation
        objects from the given affine transformation

        Parameters
        ----------
        affine_init : array, shape (3, 3) or (4, 4)
            the initial affine transformation "roughly" aligning the moving
            image towards the static
        """
        inv_affine_init = None
        if affine_init != None:
            inv_affine_init = np.array(np.linalg.inv(affine_init), dtype = floating)
        self.forward_model = DiffeomorphicMap(self.dim, None, None, None, None)
        self.backward_model = DiffeomorphicMap(self.dim, None, None, 
                                               inv_affine_init, None)

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
                 dim=3,
                 static=None,
                 moving=None,
                 affine_init=None,
                 update_function=None,
                 opt_iter = [25, 100, 100],
                 opt_tol = 1e-4,
                 inv_iter = 20,
                 inv_tol = 1e-3,
                 report_status = False):
        r""" Symmetric Diffeomorphic Registration (SyN) Algorithm
        Performs the multi-resolution optimization algorithm for non-linear
        registration using a given similarity metric and update rule (this
        scheme was inspider on the ANTS package).

        Parameters
        ----------
        metric : SimilarityMetric object
            the metric to be optimized
        dim : int (either 2 or 3)
            the dimension of the image domain
        static : array, shape (R, C) or (S, R, C)
            the static image (this also defines the reference)
        """
        super(SymmetricDiffeomorphicRegistration, self).__init__(
                metric, dim, static, moving, affine_init, update_function)
        self.set_opt_iter(opt_iter)
        self.opt_tol = opt_tol
        self.inv_tol = inv_tol
        self.inv_iter = inv_iter
        self.report_status = report_status
        self.energy_window = 12
        self.energy_list = []
        self.full_energy_profile = []

    def _connect_functions(self):
        r"""
        Assigns the appropriate functions to be called for displacement field
        inversion, Gaussian pyramid, and affine/dense deformation composition
        according to the dimension of the input images e.g. 2D or 3D.
        """
        if self.dim == 2:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point
            self.generate_pyramid = rcommon.pyramid_gaussian_2D
            self.append_affine = vfu.append_affine_to_displacement_field_2d
            self.prepend_affine = vfu.prepend_affine_to_displacement_field_2d
        else:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point_3d
            self.generate_pyramid = rcommon.pyramid_gaussian_3D
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
        self.moving_pyramid = [img for img
                               in self.generate_pyramid(self.moving,
                                                        self.levels - 1)]
        self.static_pyramid = [img for img
                              in self.generate_pyramid(self.static,
                                                       self.levels - 1)]
        starting_forward = np.zeros(
            shape=self.static_pyramid[self.levels - 1].shape + (self.dim,),
            dtype=floating)
        starting_forward_inv = np.zeros(
            shape=self.static_pyramid[self.levels - 1].shape + (self.dim,),
            dtype=floating)
        self.forward_model.scale_affines(0.5 ** (self.levels - 1))
        self.forward_model.set_forward(starting_forward)
        self.forward_model.set_backward(starting_forward_inv)
        starting_backward = np.zeros(
            shape=self.moving_pyramid[self.levels - 1].shape + (self.dim,),
            dtype=floating)
        starting_backward_inverse = np.zeros(
            shape=self.static_pyramid[self.levels - 1].shape + (self.dim,),
            dtype=floating)
        self.backward_model.scale_affines(0.5 ** (self.levels - 1))
        self.backward_model.set_forward(starting_backward)
        self.backward_model.set_backward(starting_backward_inverse)

    def _end_optimizer(self):
        r"""
        Frees the resources allocated during initialization
        """
        del self.moving_pyramid
        del self.static_pyramid

    def _iterate(self, call_back=None):
        r"""
        Performs one symmetric iteration:
            1.Compute forward
            2.Compute backward
            3.Update forward
            4.Update backward
            5.Compute inverses
            6.Invert the inverses to improve invertibility

        Parameters
        ----------
        call_back : function
            a function to be called after each iteration
        """
        wmoving = self.backward_model.transform_inverse(self.current_moving, 'tri')
        wstatic = self.forward_model.transform_inverse(self.current_static, 'tri')
        
        self.metric.set_moving_image(wmoving)
        self.metric.use_moving_image_dynamics(
            self.current_moving, self.backward_model.inverse())
        self.metric.set_static_image(wstatic)
        self.metric.use_static_image_dynamics(
            self.current_static, self.forward_model.inverse())
        self.metric.initialize_iteration()
        ff_shape = np.array(self.forward_model.forward.shape).astype(np.int32)
        fb_shape = np.array(self.forward_model.backward.shape).astype(np.int32)
        bf_shape = np.array(self.backward_model.forward.shape).astype(np.int32)
        bb_shape = np.array(
            self.backward_model.backward.shape).astype(np.int32)
        del self.forward_model.backward
        del self.backward_model.backward
        fw_step = np.array(self.metric.compute_forward())
        self.forward_model.forward, md_forward = self.update(
            self.forward_model.forward, fw_step)
        del fw_step
        try:
            fw_energy = self.metric.energy
        except NameError:
            pass
        bw_step = np.array(self.metric.compute_backward())
        self.backward_model.forward, md_backward = self.update(
            self.backward_model.forward, bw_step)
        del bw_step
        try:
            bw_energy = self.metric.energy
        except NameError:
            pass
        der = '-'
        try:
            n_iter = len(self.energy_list)
            if len(self.energy_list) >= self.energy_window:
                der = self._get_energy_derivative()
            print(
                '%d:\t%0.6f\t%0.6f\t%0.6f\t%s' % (n_iter, fw_energy, bw_energy,
                                                  fw_energy + bw_energy, der))
            self.energy_list.append(fw_energy + bw_energy)
        except NameError:
            pass
        self.metric.free_iteration()
        inv_iter = self.inv_iter
        inv_tol = self.inv_tol
        self.forward_model.backward = np.array(
            self.invert_vector_field(
                self.forward_model.forward, fb_shape, inv_iter, inv_tol, None))
        self.backward_model.backward = np.array(
            self.invert_vector_field(
                self.backward_model.forward, bb_shape, inv_iter, inv_tol, None))
        self.forward_model.forward = np.array(
            self.invert_vector_field(
                self.forward_model.backward, ff_shape, inv_iter, inv_tol,
                self.forward_model.forward))
        self.backward_model.forward = np.array(
            self.invert_vector_field(
                self.backward_model.backward, bf_shape, inv_iter, inv_tol,
                self.backward_model.forward))
        if call_back is not None:
            call_back()
        return 1 if der == '-' else der

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
        spline = interpolate.UnivariateSpline(x, y, s=1e6, k=2)
        derivative = spline.derivative()
        der = derivative(0.5 * self.energy_window)
        return der
    
    def _optimize(self):
        r"""
        The main multi-scale symmetric optimization algorithm
        """
        self._init_optimizer()
        self.full_energy_profile = []
        for level in range(self.levels - 1, -1, -1):
            
            self.current_static = self.static_pyramid[level]
            self.current_moving = self.moving_pyramid[level]
            
            self.metric.use_original_static_image(self.current_static)
            self.metric.use_original_static_image(self.current_moving)
            
            self.metric.set_levels_below(self.levels - level)
            self.metric.set_levels_above(level)

            if level < self.levels - 1:
                self.forward_model.upsample(self.current_static.shape,
                                            self.current_static.shape)
                self.backward_model.upsample(self.current_moving.shape,
                                             self.current_static.shape)

            niter = 0
            self.full_energy_profile.extend(self.energy_list)
            self.energy_list = []
            derivative = 1
            while ((niter < self.opt_iter[level]) and (self.opt_tol < derivative)):
                niter += 1
                derivative = self._iterate()

        # Reporting mean and std in stats[1] and stats[2]
        residual, stats = self.forward_model.compute_inversion_error()
        print('Forward Residual error (Symmetric diffeomorphism):%0.6f (%0.6f)'
              % (stats[1], stats[2]))
        residual, stats = self.backward_model.compute_inversion_error()
        print('Backward Residual error (Symmetric diffeomorphism):%0.6f (%0.6f)'
              % (stats[1], stats[2]))

        # Compose the two partial transformations
        self.forward_model = self.backward_model.inverse().compose(
            self.forward_model)

        # Put affines inside the deformation field
        self.forward_model.consolidate()
        del self.backward_model
        
        # Report mean and std for the composed deformation field
        residual, stats = self.forward_model.compute_inversion_error()
        print('Residual error (Symmetric diffeomorphism):%0.6f (%0.6f)'
              % (stats[1], stats[2]))
        self._end_optimizer()

    def optimize(self, static=None, moving=None, affine_init=None):
        if static is not None:
            self.set_static_image(static)
        if moving is not None:
            self.set_moving_image(moving)
        if affine_init is not None:
            self.set_affine_init(affine_init)
        self._optimize()
        return self.forward_model


