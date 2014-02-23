'''
Definition of the TransformationModel class, which is the composition of
an affine pre-aligning transformation followed by a nonlinear transformation
followed by an affine post-multiplication.
'''
import numpy as np
import VectorFieldUtils as vfu
import numpy.linalg as linalg

def scale_affine(affine, factor):
    r'''
    Multiplies the translation part of the affine transformation by a factor
    to be used with upsampled/downsampled images (if the affine transformation)
    corresponds to an Image I and we need to apply the corresponding
    transformation to a downsampled version J of I, then the affine matrix
    is the same as for I but the translation is scaled.
    '''
    scaled_affine = affine.copy()
    domain_dimension = affine.shape[1]-1
    scaled_affine[:domain_dimension, domain_dimension] *= factor
    return scaled_affine

class TransformationModel(object):
    '''
    This class maps points between two spaces: "reference space" and "target
    space"
    Forward: maps target to reference, y=affine_post*forward(affine_pre*x)
    Backward: maps reference to target,
    x = affine_pre^{-1}*backward(affine_post^{-1}*y)
    '''
    def __init__(self,
                 forward = None,
                 backward = None,
                 affine_pre = None,
                 affine_post = None):
        self.dim = None
        self.set_forward(forward)
        self.set_backward(backward)
        self.set_affine_pre(affine_pre)
        self.set_affine_post(affine_post)

    def set_affine_pre(self, affine_pre):
        r'''
        Establishes the pre-multiplication affine matrix of this
        transformation, computes its inverse and adjusts the dimension of
        the transformation's domain accordingly
        '''
        if affine_pre != None:
            self.dim = affine_pre.shape[1]-1
            self.affine_pre_inv = linalg.inv(affine_pre).copy(order='C')
        else:
            self.affine_pre_inv = None
        self.affine_pre = affine_pre

    def set_affine_post(self, affine_post):
        r'''
        Establishes the post-multiplication affine matrix of this
        transformation, computes its inverse and adjusts the dimension of
        the transformation's domain accordingly
        '''
        if affine_post != None:
            self.dim = affine_post.shape[1]-1
            self.affine_post_inv = linalg.inv(affine_post).copy(order='C')
        else:
            self.affine_post_inv = None
        self.affine_post = affine_post

    def set_forward(self, forward):
        r'''
        Establishes the forward non-linear displacement field and adjusts
        the dimension of the transformation's domain accordingly
        '''
        if forward != None:
            self.dim = len(forward.shape)-1
        self.forward = forward

    def set_backward(self, backward):
        r'''
        Establishes the backward non-linear displacement field and adjusts
        the dimension of the transformation's domain accordingly
        '''
        if backward != None:
            self.dim = len(backward.shape)-1
        self.backward = backward

    def warp_forward(self, image):
        r'''
        Applies this transformation in the forward direction to the given image
        using tri-linear interpolation
        '''
        if len(image.shape) == 3:
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

    def warp_backward(self, image):
        r'''
        Applies this transformation in the backward direction to the given
        image using tri-linear interpolation
        '''
        if len(image.shape) == 3:
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

    def warp_forward_nn(self, image):
        r'''
        Applies this transformation in the forward direction to the given image
        using nearest-neighbor interpolation
        '''
        if len(image.shape) == 3:
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

    def warp_backward_nn(self, image):
        r'''
        Applies this transformation in the backward direction to the given
        image using nearest-neighbor interpolation
        '''
        if len(image.shape) == 3:
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

    def scale_affines(self, factor):
        r'''
        Scales the pre- and post-multiplication affine matrices to be used
        with a scaled domain. It updates the inverses as well.
        '''
        if self.affine_pre != None:
            self.affine_pre = scale_affine(self.affine_pre, factor)
            self.affine_pre_inv = linalg.inv(self.affine_pre).copy(order='C')
        if self.affine_post != None:
            self.affine_post = scale_affine(self.affine_post, factor)
            self.affine_post_inv = linalg.inv(self.affine_post).copy(order='C')

    def upsample(self, new_domain_forward, new_domain_backward):
        r'''
        Upsamples the displacement fields and scales the affine
        pre- and post-multiplication affine matrices by a factor of 2. The
        final outcome is that this transformation can be used in an upsampled
        domain.
        '''
        if self.dim == 2:
            if self.forward != None:
                self.forward = 2*np.array(
                    vfu.upsample_displacement_field(
                        self.forward,
                        np.array(new_domain_forward).astype(np.int32)))
            if self.backward != None:
                self.backward = 2*np.array(
                    vfu.upsample_displacement_field(
                        self.backward,
                        np.array(new_domain_backward).astype(np.int32)))
        else:
            if self.forward != None:
                self.forward = 2*np.array(
                    vfu.upsample_displacement_field_3d(
                        self.forward,
                        np.array(new_domain_forward).astype(np.int32)))
            if self.backward != None:
                self.backward = 2*np.array(
                    vfu.upsample_displacement_field_3d(
                        self.backward,
                        np.array(new_domain_backward).astype(np.int32)))
        self.scale_affines(2.0)


    def compute_inversion_error(self):
        r'''
        Returns the inversion error of the displacement fields
        TO-DO: the inversion error should take into account the affine
        transformations as well.
        '''
        if self.dim == 2:
            residual, stats = vfu.compose_vector_fields(self.forward,
                                                       self.backward)
        else:
            residual, stats = vfu.compose_vector_fields_3d(self.forward,
                                                         self.backward)
        return residual, stats

    def compose(self, applyFirst):
        r'''
        Computes the composition G(F(.)) where G is this transformation and
        F is the transformation given as parameter
        '''
        B=applyFirst.affine_post
        C=self.affine_pre
        if B==None:
            affine_prod=C
        elif C==None:
            affine_prod=B
        else:
            affine_prod=C.dot(B)
        if affine_prod!=None:
            affine_prod_inv=linalg.inv(affine_prod).copy(order='C')
        else:
            affine_prod_inv=None
        if self.dim == 2:
            forward=applyFirst.forward.copy()
            vfu.append_affine_to_displacement_field_2d(forward, affine_prod)
            forward, stats = vfu.compose_vector_fields(forward,
                                                       self.forward)
            backward=self.backward.copy()
            vfu.append_affine_to_displacement_field_2d(backward, affine_prod_inv)
            backward, stats = vfu.compose_vector_fields(backward, 
                                                        applyFirst.backward)
        else:
            forward=applyFirst.forward.copy()
            vfu.append_affine_to_displacement_field_3d(forward, affine_prod)
            forward, stats = vfu.compose_vector_fields_3d(forward,
                                                         self.forward)
            backward=self.backward.copy()
            vfu.append_affine_to_displacement_field_3d(backward, affine_prod_inv)
            backward, stats = vfu.compose_vector_fields_3d(backward, 
                                                          applyFirst.backward)
        composition=TransformationModel(forward, 
                                        backward, 
                                        applyFirst.affine_pre, 
                                        self.affine_post)
        return composition

    def inverse(self):
        r'''
        Return the inverse of this transformation model. Warning: the matrices 
        and displacement fields are not copied
        '''
        inv=TransformationModel(self.backward, self.forward, 
                                self.affine_post_inv, self.affine_pre_inv)
        return inv

    def consolidate(self):
        r'''
        Eliminates the affine transformations from the representation of this
        transformation by appending/prepending them to the deformation fields.
        '''
        if self.dim == 2:
            vfu.prepend_affine_to_displacement_field_2d(self.forward, self.affine_pre)
            vfu.append_affine_to_displacement_field_2d(self.forward, self.affine_post)
            vfu.prepend_affine_to_displacement_field_2d(self.backward, self.affine_post_inv)
            vfu.append_affine_to_displacement_field_2d(self.backward, self.affine_pre_inv)
        else:
            vfu.prepend_affine_to_displacement_field_3d(self.forward, self.affine_pre)
            vfu.append_affine_to_displacement_field_3d(self.forward, self.affine_post)
            vfu.prepend_affine_to_displacement_field_3d(self.backward, self.affine_post_inv)
            vfu.append_affine_to_displacement_field_3d(self.backward, self.affine_pre_inv)
        self.affine_post = None
        self.affine_pre = None
        self.affine_post_inv = None
        self.affine_pre_inv = None
