import numpy as np
import scipy as sp
import numpy.linalg as linalg
import abc
import vector_fields as vfu
from dipy.align import floating
import nibabel as nib
import matplotlib.pyplot as plt


def mult_aff(A, B):
    if A is None:
        return B
    elif B is None:
        return A
    return A.dot(B)


def inv_aff(A):
    if A is None:
        return None
    return np.linalg.inv(A)


def as_type(obj, t):
    if obj is None:
        return None
    return obj.astype(t)


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

class ScaleSpace(object):
    def __init__(self, image, num_levels, input_affine, input_spacing, sigma_factor=0.2):
        r""" ScaleSpace
        Implements the Scale Space representation of an image
        """
        self.dim = len(image.shape)
        self.num_levels = num_levels
        input_size = np.array(image.shape)
        #normalize input image to [0,1]
        img = (image - image.min())/(image.max() - image.min())

        #The properties are saved in separate lists        
        self.images = []
        self.domain_shapes = []
        self.spacings = []
        self.scalings = []
        self.affines = []
        self.affine_invs = []
        self.sigmas = []

        #insert input image properties at the first level of the scale space
        self.images.append(as_type(img, floating))
        self.domain_shapes.append(as_type(input_size, np.int32))
        self.spacings.append(input_spacing)
        self.scalings.append(np.ones(self.dim))
        self.affines.append(input_affine)
        if input_affine is not None:
            self.affine_invs.append(np.linalg.inv(input_affine))
        else:
            self.affine_invs.append(None)
        self.sigmas.append(np.zeros(self.dim))

        #compute the rest of the levels
        min_spacing = np.min(input_spacing)
        for i in range(1, num_levels):
            scaling_factor = 2**i
            scaling = np.ndarray((self.dim+1,))
            #Note: the minimum below is present in ANTS to prevent the scaling from
            #being too large (making the subsampled image to be too small),
            #this makes the subsampled image at least 32 voxels at each direction
            #it is risky to make this decision based on image size, though
            #(we need to investigate more the effect of this)
            
            #scaling = np.minimum(scaling_factor * min_spacing / input_spacing, input_size / 32)

            scaling = scaling_factor * min_spacing / input_spacing
            output_spacing = input_spacing * scaling
            extended = np.append(scaling, [1])
            if not input_affine is None:
                affine = input_affine.dot(np.diag(extended))
            else:
                affine = np.diag(extended)
            output_size = input_size * (input_spacing / output_spacing) + 0.5
            output_size = output_size.astype(np.int32)
            sigmas = sigma_factor * (output_spacing / input_spacing - 1.0)

            #filter along each direction with the appropriate sigma
            filtered = sp.ndimage.filters.gaussian_filter(image, sigmas)
            filtered = (filtered - filtered.min())/(filtered.max() - filtered.min())

            #Add current level to the scale space
            self.images.append(as_type(filtered, floating))
            self.domain_shapes.append(output_size)
            self.spacings.append(output_spacing)
            self.scalings.append(scaling)
            self.affines.append(affine)
            self.affine_invs.append(np.linalg.inv(affine))
            self.sigmas.append(sigmas)

    def get_expand_factors(self, from_level, to_level):
        factors = np.array(self.spacings[to_level]) / \
                  np.array(self.spacings[from_level])
        return factors

    def print_level(self, level):
        print 'Domain shape:', self.get_domain_shape(level)
        print 'Spacing:', self.get_spacing(level)
        print 'Scaling:', self.get_scaling(level)
        print 'Affine:', self.get_affine(level)
        print 'Sigmas:', self.get_sigmas(level)

    def get_image(self, level):
        if 0 <= level < self.num_levels:
            return self.images[level]
        return None

    def get_domain_shape(self, level):
        if 0 <= level < self.num_levels:
            return self.domain_shapes[level]
        return None

    def get_spacing(self, level):
        if 0 <= level < self.num_levels:
            return self.spacings[level]
        return None

    def get_scaling(self, level):
        if 0 <= level < self.num_levels:
            return self.scalings[level]
        return None

    def get_affine(self, level):
        if 0 <= level < self.num_levels:
            return self.affines[level]
        return None

    def get_affine_inv(self, level):
        if 0 <= level < self.num_levels:
            return self.affine_invs[level]
        return None

    def get_sigmas(self, level):
        if 0 <= level < self.num_levels:
            return self.sigmas[level]
        return None        


def compose_displacements(new_displacement, current_displacement, affine_inv,
    time_scaling):
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
    #Define the premultiplication matrices to be used in the composition:
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
                                                      time_scaling)
    else:
        updated, stats = vfu.compose_vector_fields_3d(new_displacement,
                                                      current_displacement,
                                                      premult_index, 
                                                      premult_disp,
                                                      time_scaling)
    return np.array(updated), np.array(mse)


class DiffeomorphicMap(object):
    def __init__(self, 
                 dim,
                 domain_shape=None,
                 domain_affine=None,
                 input_shape=None,
                 input_affine=None,
                 input_prealign=None):
        r""" DiffeomorphicMap

        Implements a diffeomorphic transformation on the physical space. The 
        deformation fields share the same discretization of shape domain_shape
        and voxel-to-space matrix domain_affine. The input coordinates (in the 
        physical coordinates) are first aligned using input_prealign, and then 
        displaced using the corresponding vector field interpolated at the aligned
        coordinates (reference space).
        """

        self.dim = dim
        self.domain_shape = np.asarray(domain_shape, dtype = np.int32)
        self.domain_affine = domain_affine
        self.input_shape = np.asarray(input_shape, dtype = np.int32)
        self.input_affine = input_affine
        self.input_prealign = input_prealign

        self.domain_affine_inv = None if domain_affine is None else np.linalg.inv(domain_affine)
        self.input_affine_inv = None if input_affine is None else np.linalg.inv(input_affine)
        self.input_prealign_inv = None if input_prealign is None else np.linalg.inv(input_prealign)

        self.is_inverse = False

    def get_forward_field(self):
        if self.is_inverse:
            return self.backward
        else:
            return self.forward

    def get_backward_field(self):
        if self.is_inverse:
            return self.forward
        else:
            return self.backward

    def allocate(self):
        self.forward = np.zeros(tuple(self.domain_shape)+(self.dim,), dtype = floating)
        self.backward = np.zeros(tuple(self.domain_shape)+(self.dim,), dtype = floating)

    def _warp_forward(self, image, interpolation='tri', world_to_image=None, 
                      sampling_shape=None, sampling_affine=None):
        r"""
        Deforms the input image under this diffeomorphic map in the forward direction.
        Since the mapping is defined in the physical space, the user must specify 
        the sampling grid shape and its voxel-to-space mapping. By default, 
        the samplig grid will be self.input_shape (exception raised if it's None), 
        with default voxel-to-space mapping given by self.input_affine (identity, 
        if None). If world_to_image is None, self.domain_affine_inv is used (identity,
        if it's None as well).

        The forward warping with pre-aligning P is give by the interpolation:
        I[W*backward[Dinv*P*S*i] + W*P*S*i] where i is an index in the sampling 
        domain, S is the sampling affine, P is the pre-aligning matrix, Dinv is 
        the inverse of domain affine (Dinv maps world points to voxels in the 
        displacement field discretization) and W is the world-to-image mapping. 
        """
        if world_to_image is None:
            world_to_image = self.domain_affine_inv
        if sampling_shape is None:
            if self.input_shape is None:
                raise Exception('DiffeomorphicMap::_warp_forward','Sampling shape is None')
            sampling_shape = self.input_shape
        if sampling_affine is None:
            sampling_affine = self.input_affine
        W = world_to_image
        Dinv = self.domain_affine_inv
        P = self.input_prealign 
        S = sampling_affine
        affine_idx_in = mult_aff(Dinv, mult_aff(P, S))
        affine_idx_out = mult_aff(W, mult_aff(P, S))
        affine_disp = W

        affine_idx_in = as_type(affine_idx_in, floating)
        affine_idx_out = as_type(affine_idx_out, floating)
        affine_disp = as_type(affine_disp, floating)

        if image.dtype is np.dtype('float64') and floating is np.float32:
            image = image.astype(floating)
        elif image.dtype is np.dtype('int64'):
            image = image.astype(np.int32)

        if self.dim == 2:
            if interpolation == 'tri':
                warped = vfu.warp_image(image, self.forward,
                                        affine_idx_in,
                                        affine_idx_out,
                                        affine_disp,
                                        sampling_shape)
            else:
                warped = vfu.warp_image_nn(image, self.forward,
                                           affine_idx_in,
                                           affine_idx_out,
                                           affine_disp,
                                           sampling_shape)
        else:
            if interpolation == 'tri':
                warped = vfu.warp_volume(image, self.forward,
                                affine_idx_in,
                                affine_idx_out,
                                affine_disp,
                                sampling_shape)
            else:
                warped = vfu.warp_volume_nn(image, self.forward,
                                  affine_idx_in,
                                  affine_idx_out,
                                  affine_disp,
                                  sampling_shape)
        return warped

    def _warp_backward(self, image, interpolation='tri', world_to_image=None, 
                       sampling_shape=None, sampling_affine=None):
        r"""
        Deforms the input image under this diffeomorphic map in the backward direction.
        Since the mapping is defined in the physical space, the user must specify 
        the sampling grid shape and its voxel-to-space mapping. By default, 
        the samplig grid will be self.domain_shape (exception raised if it's None), 
        with default voxel-to-space mapping given by self.domain_affine (identity, 
        if None). If world_to_image is None, self.input_affine_inv is used (identity,
        if it's None as well).

        The backward warping with post-aligning Pinv is give by the interpolation:
        J[W*Pinv*backward[Dinv*S*i] + W*Pinv*S*i] where i is an index in the sampling domain,
        S is the sampling affine, Pinv is the post-aligning matrix, Dinv is the
        inverse of domain affine (Dinv maps world points to voxels in the 
        displacement field discretization) and W is the world-to-image mapping. 
        """

        if world_to_image is None:
            world_to_image = self.input_affine_inv
        if sampling_shape is None:
            if self.domain_shape is None:
                raise Exception('DiffeomorphicMap::_warp_backward','Sampling shape is None')
            sampling_shape = self.domain_shape
        if sampling_affine is None:
            sampling_affine = self.domain_affine

        W = world_to_image
        Dinv = self.domain_affine_inv
        Pinv = self.input_prealign_inv
        S = sampling_affine
        affine_idx_in = mult_aff(Dinv, S)
        affine_idx_out = mult_aff(W, mult_aff(Pinv, S))
        affine_disp = mult_aff(W, Pinv)

        affine_idx_in = as_type(affine_idx_in, floating)
        affine_idx_out = as_type(affine_idx_out, floating)
        affine_disp = as_type(affine_disp, floating)

        if image.dtype is np.dtype('float64') and floating is np.float32:
            image = image.astype(floating)
        elif image.dtype is np.dtype('int64'):
            image = image.astype(np.int32)

        if self.dim == 2:
            if interpolation == 'tri':
                warped = vfu.warp_image(image, self.backward,
                                        affine_idx_in,
                                        affine_idx_out,
                                        affine_disp,
                                        sampling_shape)
            else:
                warped = vfu.warp_image_nn(image, self.backward,
                                           affine_idx_in,
                                           affine_idx_out,
                                           affine_disp,
                                           sampling_shape)
        else:
            if interpolation == 'tri':
                warped = vfu.warp_volume(image, self.backward,
                                         affine_idx_in,
                                         affine_idx_out,
                                         affine_disp,
                                         sampling_shape)
            else:
                warped = vfu.warp_volume_nn(image, self.backward,
                                            affine_idx_in,
                                            affine_idx_out,
                                            affine_disp,
                                            sampling_shape)
        return warped

    def transform(self, image, interpolation='tri', world_to_image=None, 
                  sampling_shape=None, sampling_affine=None):
        if self.is_inverse:
            warped = self._warp_backward(image, interpolation, world_to_image, 
                                       sampling_shape, sampling_affine)
        else:
            warped = self._warp_forward(image, interpolation, world_to_image, 
                                       sampling_shape, sampling_affine)
        return np.asarray(warped)

    def transform_inverse(self, image, interpolation='tri', world_to_image=None, 
                          sampling_shape=None, sampling_affine=None):
        if self.is_inverse:
            warped = self._warp_forward(image, interpolation, world_to_image, 
                                        sampling_shape, sampling_affine)
        else:
            warped = self._warp_backward(image, interpolation, world_to_image, 
                                       sampling_shape, sampling_affine)
        return np.asarray(warped)

    def inverse(self):
        inv = DiffeomorphicMap(self.dim,
                               self.domain_shape,
                               self.domain_affine,
                               self.input_shape,
                               self.input_affine,
                               self.input_prealign)
        inv.forward = self.forward
        inv.backward = self.backward
        inv.is_inverse = True
        return inv

    def expand_fields(self, expand_factors, new_shape):
        if self.dim == 2:
            expanded_forward = vfu.expand_displacement_field_2d(self.forward, 
                expand_factors.astype(floating), new_shape)
            expanded_backward = vfu.expand_displacement_field_2d(self.backward, 
                expand_factors.astype(floating), new_shape)
        else:
            expanded_forward = vfu.expand_displacement_field_3d(self.forward, 
                expand_factors.astype(floating), new_shape)
            expanded_backward = vfu.expand_displacement_field_3d(self.backward,
                expand_factors.astype(floating), new_shape)
        expand_factors = np.append(expand_factors, [1])
        expanded_affine = mult_aff(self.domain_affine, np.diag(expand_factors))
        expanded_affine_inv = np.linalg.inv(expanded_affine)
        self.forward = expanded_forward
        self.backward = expanded_backward
        self.domain_shape = new_shape
        self.domain_affine = expanded_affine
        self.domain_affine_inv = expanded_affine_inv

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
        premult_disp = None 
        if self.domain_affine_inv is not None:
            premult_disp = self.domain_affine_inv.astype(floating)

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

    def shallow_copy(self):
        new_map = DiffeomorphicMap(self.dim,
                                   self.domain_shape,
                                   self.domain_affine,
                                   self.input_shape,
                                   self.input_affine,
                                   self.input_prealign)
        new_map.forward = self.forward
        new_map.backward = self.backward
        new_map.is_inverse = self.is_inverse
        return new_map

    def warp_endomorphism(self, phi):
        r"""
        Creates a new DiffeomorphicMap C with the same properties as self and
        composes its displacement fields with phi's corresponding fields. 
        The resulting diffeomorphism is of the form C(x) = phi(self(x)) with
        inverse C^{-1}(y) = self^{-1}(phi^{-1}(y)). We assume that phi is an 
        endomorphism with the same discretization and domain affines as self 
        to ensure that the composition inherits self's properties. 
        """
        #Compose the forward deformation fields
        d1 = self.get_forward_field()
        d2 = phi.get_forward_field()
        d1_inv = self.get_backward_field()
        d2_inv = phi.get_backward_field()

        premult_disp = as_type(self.domain_affine_inv, floating)

        if self.dim == 2:
            forward, stats = vfu.compose_vector_fields_2d(d1, d2, None, premult_disp, 1.0)
            backward, stats, = vfu.compose_vector_fields_2d(d2_inv, d1_inv, None, premult_disp, 1.0)
        else:
            forward, stats = vfu.compose_vector_fields_3d(d1, d2, None, premult_disp, 1.0)
            backward, stats, = vfu.compose_vector_fields_3d(d2_inv, d1_inv, None, premult_disp, 1.0)
        composition = self.shallow_copy()
        composition.forward = forward
        composition.backward = backward
        return composition


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


def renormalize_image(image):
    m=np.min(image)
    M=np.max(image)
    if(M-m<1e-8):
        return image
    return 127.0*(image-m)/(M-m)

def overlay_images(L, R, ltitle='Left', rtitle='Right', fname=None):
    sh=L.shape

    colorImage=np.zeros(shape=(sh[0], sh[1], 3), dtype=np.int8)
    ll=renormalize_image(L).astype(np.int8)
    rr=renormalize_image(R).astype(np.int8)
    colorImage[...,0]=ll*(ll>ll[0,0])
    colorImage[...,1]=rr*(rr>rr[0,0])

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(ll, cmap = plt.cm.gray)
    plt.title(ltitle)
    plt.subplot(1,3,2)
    plt.imshow(colorImage)
    plt.title('Overlay')
    plt.subplot(1,3,3)
    plt.imshow(rr, cmap = plt.cm.gray)
    plt.title(rtitle)
    if fname is not None:
        from time import sleep
        sleep(1)
        plt.savefig(fname, bbox_inches='tight')


class SymmetricDiffeomorphicRegistration(DiffeomorphicRegistration):
    def __init__(self,
                 metric=None,
                 opt_iter = [25, 100, 100],
                 step_length = 0.25,
                 ss_sigma_factor = 0.5,
                 opt_tol = 1e-4,
                 inv_iter = 20,
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
        step_length : float
            the length of the maximum displacement vector of the update 
            displacement field at each iteration
        ss_sigma_factor : float
            parameter of the scale-space smoothing kernel. For example, the 
            std. dev. of the kernel will be factor*(2^i) in the isotropic case
            where i=0,1,..,n_scales is the scale
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
        print 'step_length:',step_length
        print 'ss_sigma_factor:', ss_sigma_factor
        print 'opt_tol:', opt_tol
        print 'inv_iter:', inv_iter
        print 'inv_tol:', inv_tol


        self.set_opt_iter(opt_iter)
        self.step_length = step_length
        self.ss_sigma_factor = ss_sigma_factor
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
            self.append_affine = vfu.append_affine_to_displacement_field_2d
            self.prepend_affine = vfu.prepend_affine_to_displacement_field_2d
        else:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point_3d
            self.append_affine = vfu.append_affine_to_displacement_field_3d
            self.prepend_affine = vfu.prepend_affine_to_displacement_field_3d

    def _check_ready(self):
        r"""
        Verifies that the configuration of the optimizer and input data are
        consistent and the optimizer is ready to run
        """
        ready = True
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

    def _init_optimizer(self, static, moving, static_affine, moving_affine, prealign):
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
        #Extract information from the affine matrices to create the scale space
        static_direction, static_spacing = get_direction_and_scalings(static_affine, self.dim)
        moving_direction, moving_spacing = get_direction_and_scalings(moving_affine, self.dim)

        #Build the scale space of the input images
        self.moving_ss = ScaleSpace(moving, self.levels, moving_affine, moving_spacing, self.ss_sigma_factor)
        self.static_ss = ScaleSpace(static, self.levels, static_affine, static_spacing, self.ss_sigma_factor)

        if self.verbosity>1:
            print('Moving scale space:')
            for level in range(self.levels):
                self.moving_ss.print_level(level)

            print('Static scale space:')
            for level in range(self.levels):
                self.static_ss.print_level(level)
        
        #Get the coarcest level's properties from the static image. These
        #properties will be taken as the reference discretization.
        domain_shape = self.static_ss.get_domain_shape(self.levels-1)
        domain_affine = self.static_ss.get_affine(self.levels-1)

        #Create the forward diffeomorphic transformation at the coarcest resolution
        #The forward model transforms points from the static image to points on
        #the reference (which is the static as well). So the input properties 
        #are taken from the static image. Since its the same as the reference,
        #we don't need to prealign.
        input_shape = static.shape
        input_affine = static_affine
        input_prealign = None
        self.forward_model = DiffeomorphicMap(self.dim,
                                              domain_shape,
                                              domain_affine,
                                              input_shape,
                                              input_affine,
                                              input_prealign)
        self.forward_model.allocate()

        #Create the backward diffeomorphic transformation at the coarcest resolution
        #The backward model transforms points from the moving image to points on
        #the reference (which is the static). So the input properties 
        #are taken from the moving image, and we need to pre-align points on the
        #moving physical space to the reference physical space by applying the
        #inverse of prealign. This is dome this way to make it clear for the
        #user: the pre-align matrix is usually obtained by doing affine registration
        #of the moving image towards the static image, which results in a matrix 
        #transforming points in the static physical space to points in the moving 
        #physical space
        prealign_inv = None if prealign is None else np.linalg.inv(prealign)
        input_shape = moving.shape
        input_affine = moving_affine
        input_prealign = prealign_inv
        self.backward_model = DiffeomorphicMap(self.dim,
                                               domain_shape,
                                               domain_affine,
                                               input_shape,
                                               input_affine,
                                               input_prealign)
        self.backward_model.allocate()

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
        #Acquire current resolution information from scale spaces
        current_moving = self.moving_ss.get_image(self.current_level)
        current_static = self.static_ss.get_image(self.current_level)

        current_domain_shape = self.static_ss.get_domain_shape(self.current_level)
        current_domain_affine = self.static_ss.get_affine(self.current_level)
        current_domain_affine_inv = self.static_ss.get_affine_inv(self.current_level)
        current_domain_spacing = self.static_ss.get_spacing(self.current_level)
        
        current_domain_affine = as_type(current_domain_affine, floating)
        current_domain_affine_inv = as_type(current_domain_affine_inv, floating)
            
        #Warp the input images (smoothed to the current scale) to the common (reference) space
        wstatic = self.forward_model.transform_inverse(current_static, 'tri')
        wmoving = self.backward_model.transform_inverse(current_moving, 'tri')
        
        if self.verbosity > 10:
            if self.dim == 2:
                overlay_images(wmoving, wstatic, 'Moving', 'Static')
            else:
                overlay_images(wmoving[:,wmoving.shape[1]//2,:], 
                               wstatic[:,wstatic.shape[1]//2,:], 
                               'Moving', 'Static')
        
        #Pass both images to the metric
        self.metric.set_moving_image(wmoving)
        self.metric.use_moving_image_dynamics(
            current_moving, self.backward_model.inverse())
        self.metric.set_static_image(wstatic)
        self.metric.use_static_image_dynamics(
            current_static, self.forward_model.inverse())

        #Initialize the metric for a new iteration
        self.metric.initialize_iteration()

        #Free some memory (useful when using double precision)
        del self.forward_model.backward
        del self.backward_model.backward

        #Compute the forward step (to be used to update the forward transform) 
        fw_step = np.array(self.metric.compute_forward())

        #Normalize the forward step
        nrm = np.sqrt(np.sum((fw_step/current_domain_spacing)**2, -1)).max()
        if nrm>0:
            fw_step /= nrm
        
        #Add to current total field
        self.forward_model.forward, md_forward = self.update(
            self.forward_model.forward, fw_step, 
            current_domain_affine_inv, self.step_length)
        del fw_step

        #Keep track of the forward energy
        fw_energy = self.metric.get_energy()

        #Compose the backward step (to be used to update the backward transform)
        bw_step = np.array(self.metric.compute_backward())
        
        #Normalize the backward step
        nrm = np.sqrt(np.sum((bw_step/current_domain_spacing)**2, -1)).max()
        if nrm>0:
            bw_step /= nrm

        #Add to current total field
        self.backward_model.forward, md_backward = self.update(
            self.backward_model.forward, bw_step, 
            current_domain_affine_inv, self.step_length)
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

        #Invert the forward model's forward field
        self.forward_model.backward = np.array(
            self.invert_vector_field(
                self.forward_model.forward,
                current_domain_affine_inv,
                current_domain_spacing,
                self.inv_iter, self.inv_tol, None))

        #Invert the backward model's forward field
        self.backward_model.backward = np.array(
            self.invert_vector_field(
                self.backward_model.forward,
                current_domain_affine_inv,
                current_domain_spacing,
                self.inv_iter, self.inv_tol, None))

        #Invert the forward model's backward field
        self.forward_model.forward = np.array(
            self.invert_vector_field(
                self.forward_model.backward,
                current_domain_affine_inv,
                current_domain_spacing,
                self.inv_iter, self.inv_tol, self.forward_model.forward))

        #Invert the backward model's backward field
        self.backward_model.forward = np.array(
            self.invert_vector_field(
                self.backward_model.backward,
                current_domain_affine_inv,
                current_domain_spacing,
                self.inv_iter, self.inv_tol, self.backward_model.forward))

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
        print 'Verbosity:', self.verbosity
        self.full_energy_profile = []
        for level in range(self.levels - 1, -1, -1):
            if self.verbosity > 0:
                print('Optimizing level %d'%(level,))

            self.current_level = level
            
            self.metric.use_original_static_image(self.static_ss.get_image(level))
            self.metric.use_original_moving_image(self.moving_ss.get_image(level))
            
            self.metric.set_levels_below(self.levels - level)
            self.metric.set_levels_above(level)

            if level < self.levels - 1:
                expand_factors = self.static_ss.get_expand_factors(level+1, level) 
                new_shape = self.static_ss.get_domain_shape(level)
                self.forward_model.expand_fields(expand_factors, new_shape)
                self.backward_model.expand_fields(expand_factors, new_shape)

            niter = 0
            self.full_energy_profile.extend(self.energy_list)
            self.energy_list = []
            derivative = 1
            while ((niter < self.opt_iter[level]) and (self.opt_tol < derivative)):
                niter += 1
                derivative = self._iterate()
            if self.verbosity>10:
                wmoving = self.backward_model.transform_inverse(
                        self.moving_ss.get_image(self.current_level), 'tri')
                wstatic = self.forward_model.transform_inverse(
                        self.static_ss.get_image(self.current_level), 'tri')
                if self.dim == 2:
                    overlay_images(wmoving, wstatic, 'Moving', 'Static')
                else:
                    overlay_images(wmoving[:,wmoving.shape[1]//2,:], 
                                   wstatic[:,wstatic.shape[1]//2,:], 
                                   'Moving', 'Static')

        # Reporting mean and std in stats[1] and stats[2]
        residual, stats = self.forward_model.compute_inversion_error()
        if self.verbosity > 0:
            print('Forward Residual error: %0.6f (%0.6f)'
                  % (stats[1], stats[2]))
        residual, stats = self.backward_model.compute_inversion_error()
        if self.verbosity > 0:
            print('Backward Residual error :%0.6f (%0.6f)'
                  % (stats[1], stats[2]))

        #Compose the two partial transformations
        self.forward_model = self.backward_model.warp_endomorphism(
                                    self.forward_model.inverse()).inverse()
                
        # Report mean and std for the composed deformation field
        residual, stats = self.forward_model.compute_inversion_error()
        if self.verbosity > 0:
            print('Final residual error: %0.6f (%0.6f)' % (stats[1], stats[2]))

    def optimize(self, static, moving, static_affine=None, moving_affine=None, prealign=None):
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
        print "Pre-align:",prealign
        self._init_optimizer(static, moving, static_affine, moving_affine, prealign)
        self._optimize()
        self._end_optimizer()
        return self.forward_model

