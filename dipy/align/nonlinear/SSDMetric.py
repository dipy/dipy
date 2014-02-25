r'''
Similarity metric defined by the sum of squared differences
'''
import numpy as np
from scipy import gradient, ndimage
import SSDFunctions as ssd
import VectorFieldUtils as vfu
from SimilarityMetric import SimilarityMetric
import RegistrationCommon as rcommon
import matplotlib.pyplot as plt

class SSDMetric(SimilarityMetric):
    r'''
    Similarity metric for (monomodal) nonlinear image registration defined by
    the sum of squared differences (SSD).
    '''
    GAUSS_SEIDEL_STEP = 0
    DEMONS_STEP = 1
    def get_default_parameters(self):
        return {'lambda':1.0, 'max_inner_iter':5, 'scale':1,
                'max_step_length':0.25, 'sigma_diff':3.0, 'step_type':0}

    def __init__(self, dim, parameters):
        super(SSDMetric, self).__init__(dim, parameters)
        self.step_type = self.parameters['step_type']
        self.levels_below = 0

    def initialize_iteration(self):
        r'''
        Precomputes the gradient of the input images to be used in the
        computation of the forward and backward steps.
        '''
        self.gradient_moving = np.empty(
            shape = (self.moving_image.shape)+(self.dim,), dtype = np.float64)
        i = 0
        for grad in gradient(self.moving_image):
            self.gradient_moving[..., i] = grad
            i += 1
        i = 0
        self.gradient_fixed = np.empty(
            shape = (self.fixed_image.shape)+(self.dim,), dtype = np.float64)
        for grad in gradient(self.fixed_image):
            self.gradient_fixed[..., i] = grad
            i += 1

    def compute_forward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the moving image towards the fixed image
        '''
        if self.step_type == SSDMetric.GAUSS_SEIDEL_STEP:
            return self.compute_gauss_seidel_step(True)
        elif self.step_type == SSDMetric.DEMONS_STEP:
            return self.compute_demons_step(True)
        return None

    def compute_backward(self):
        r'''
        Computes the update displacement field to be used for registration of
        the fixed image towards the moving image
        '''
        if self.step_type == SSDMetric.GAUSS_SEIDEL_STEP:
            return self.compute_gauss_seidel_step(False)
        elif self.step_type == SSDMetric.DEMONS_STEP:
            return self.compute_demons_step(False)
        return None

    def compute_gauss_seidel_step(self, forward_step = True):
        r'''
        Minimizes the linearized energy function defined by the sum of squared
        differences of corresponding pixels of the input images with respect
        to the displacement field.
        '''
        max_inner_iter = self.parameters['max_inner_iter']
        lambda_param = self.parameters['lambda']
        max_step_length = self.parameters['max_step_length']
        if forward_step:
            shape = self.fixed_image.shape
        else:
            shape = self.moving_image.shape
        if forward_step:
            delta_field = self.fixed_image-self.moving_image
        else:
            delta_field = self.moving_image - self.fixed_image
        #gradient = self.gradient_moving+self.gradient_fixed
        gradient = self.gradient_moving
        displacement = np.zeros(shape = (shape)+(self.dim,), dtype = np.float64)
        if self.dim == 2:
            self.energy = v_cycle_2d(self.levels_below, max_inner_iter, 
                                    delta_field, None, gradient, None, 
                                    lambda_param, displacement)
        else:
            self.energy = v_cycle_3d(self.levels_below, max_inner_iter,
                                    delta_field, None, gradient, None, 
                                    lambda_param, displacement)
        max_norm = np.sqrt(np.sum(displacement**2, -1)).max()
        if max_norm > max_step_length:
            displacement *= max_step_length/max_norm
        return displacement

    def compute_demons_step(self, forward_step = True):
        r'''
        Computes the demons step proposed by Vercauteren et al.[1] for the SSD
        metric.
        [1] Tom Vercauteren, Xavier Pennec, Aymeric Perchant, Nicholas Ayache,
            "Diffeomorphic Demons: Efficient Non-parametric Image Registration",
            Neuroimage 2009
        '''
        sigma_diff = self.parameters['sigma_diff']
        max_step_length = self.parameters['max_step_length']
        scale = self.parameters['scale']
        if forward_step:
            delta_field = self.fixed_image-self.moving_image
        else:
            delta_field = self.moving_image - self.fixed_image
        gradient = self.gradient_moving+self.gradient_fixed
        if self.dim == 2:
            forward = ssd.compute_demons_step2D(delta_field, gradient,
                                               max_step_length, scale)
            forward[..., 0] = ndimage.filters.gaussian_filter(forward[..., 0],
                                                                sigma_diff)
            forward[..., 1] = ndimage.filters.gaussian_filter(forward[..., 1],
                                                                sigma_diff)
        else:
            forward = ssd.compute_demons_step3D(delta_field, gradient,
                                               max_step_length, scale)
            forward[..., 0] = ndimage.filters.gaussian_filter(forward[..., 0],
                                                                sigma_diff)
            forward[..., 1] = ndimage.filters.gaussian_filter(forward[..., 1],
                                                                sigma_diff)
            forward[..., 2] = ndimage.filters.gaussian_filter(forward[..., 2],
                                                                sigma_diff)
        return forward

    def get_energy(self):
        return NotImplemented

    def use_original_fixed_image(self, originalfixed_image):
        r'''
        SSDMetric does not take advantage of the original fixed image, just pass
        '''
        pass

    def use_original_moving_image(self, original_moving_image):
        r'''
        SSDMetric does not take advantage of the original moving image just pass
        '''
        pass

    def use_fixed_image_dynamics(self, originalfixed_image, transformation):
        r'''
        SSDMetric does not take advantage of the image dynamics, just pass
        '''
        pass

    def use_moving_image_dynamics(self, original_moving_image, transformation):
        r'''
        SSDMetric does not take advantage of the image dynamics, just pass
        '''
        pass

    def report_status(self):
        plt.figure()
        rcommon.overlayImages(self.moving_image, self.fixed_image, False)

    def get_metric_name(self):
        return "SSDMetric"

    def free_iteration(self):
        pass
#######################Multigrid algorithms for SSD-like metrics#############

printEnergy = False
def single_cycle_2d(n, k, delta_field, sigma_field, gradient_field,
                    lambda_param, displacement, depth = 0):
    r'''
    One-pass multi-resolution Gauss-Seidel solver: solves the SSD-like linear
    system starting at the coarcer resolution and then refining at the finer
    resolution.
    '''
    if n == 0:
        for i in range(k):
            error = ssd.iterate_residual_displacement_field_SSD2D(delta_field,
                                                                 sigma_field,
                                                                 gradient_field,
                                                                 None,
                                                                 lambda_param,
                                                                 displacement)
            if printEnergy and depth == 0:
                energy = ssd.compute_energy_SSD2D(delta_field,
                                                 sigma_field,
                                                 gradient_field,
                                                 lambda_param,
                                                 displacement)
                print 'Energy after top-level iter', i+1, ' [unique]:', energy
        return error
    #solve at coarcer grid
    subsigma_field = None
    if sigma_field != None:
        subsigma_field = vfu.downsample_scalar_field2D(sigma_field)
    subdelta_field = vfu.downsample_scalar_field2D(delta_field)
    subgradient_field = np.array(
        vfu.downsample_displacement_field2D(gradient_field))
    shape = np.array(displacement.shape).astype(np.int32)
    sub_displacement = np.zeros(
        shape = ((shape[0]+1)//2, (shape[1]+1)//2, 2 ), dtype = np.float64)
    sublambda_param = lambda_param*0.25
    single_cycle_2d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
                  sublambda_param, sub_displacement, depth+1)
    displacement += np.array(
        vfu.upsample_displacement_field(sub_displacement, shape))
    if printEnergy and depth == 0:
        energy = ssd.compute_energy_SSD2D(delta_field, sigma_field, 
                                         gradient_field, lambda_param, 
                                         displacement)
        print 'Energy after low-res iteration:', energy
    #post-smoothing
    for i in range(k):
        error = ssd.iterate_residual_displacement_field_SSD2D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             None,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD2D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [unique]:', energy
    try:
        energy
    except NameError:
        energy = ssd.compute_energy_SSD2D(delta_field, sigma_field, 
                                         gradient_field, lambda_param, 
                                         displacement)
    return energy

def v_cycle_2d(n, k, delta_field, sigma_field, gradient_field, target,
             lambda_param, displacement, depth = 0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by first
    filtering (GS-iterate) the current level, then solves for the residual
    at a coarcer resolution andfinally refines the solution at the current
    resolution. This scheme corresponds to the V-cycle proposed by Bruhn and
    Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
            combining highest accuracy with real-time performance",
            10th IEEE International Conference on Computer Vision, 2005.
            ICCV 2005.
    '''
    #presmoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_SSD2D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD2D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [unique]:', energy
    if n == 0:
        try:
            energy
        except NameError:
            energy = ssd.compute_energy_SSD2D(delta_field, sigma_field, 
                                         gradient_field, lambda_param, 
                                         displacement)
        return energy
    #solve at coarcer grid
    residual = None
    residual = ssd.compute_residual_displacement_field_SSD2D(delta_field,
                                                            sigma_field,
                                                            gradient_field,
                                                            target,
                                                            lambda_param,
                                                            displacement,
                                                            residual)
    sub_residual = np.array(vfu.downsample_displacement_field2D(residual))
    del residual
    subsigma_field = None
    if sigma_field != None:
        subsigma_field = vfu.downsample_scalar_field2D(sigma_field)
    subdelta_field = vfu.downsample_scalar_field2D(delta_field)
    subgradient_field = np.array(
        vfu.downsample_displacement_field2D(gradient_field))
    shape = np.array(displacement.shape).astype(np.int32)
    #sub_displacement = np.array(vfu.downsample_displacement_field(displacement))
    sub_displacement = np.zeros(shape = ((shape[0]+1)//2, (shape[1]+1)//2, 2 ),
                               dtype = np.float64)
    sublambda_param = lambda_param*0.25
    v_cycle_2d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
             sub_residual, sublambda_param, sub_displacement, depth+1)
    displacement += np.array(
        vfu.upsample_displacement_field(sub_displacement, shape))
    if printEnergy and depth == 0:
        energy = ssd.compute_energy_SSD2D(delta_field, sigma_field, 
                                         gradient_field, lambda_param, 
                                         displacement)
        print 'Energy after low-res iteration:', energy
    #post-smoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_SSD2D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD2D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [unique]:', energy
    try:
        energy
    except NameError:
        energy = ssd.compute_energy_SSD2D(delta_field, sigma_field, 
                                         gradient_field, lambda_param, 
                                         displacement)
    return energy

def w_cycle_2d(n, k, delta_field, sigma_field, gradient_field, target,
             lambda_param, displacement, depth = 0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by performing
    two v-cycles at each resolution, which corresponds to the w-cycle scheme
    proposed by Bruhn and Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
            combining highest accuracy with real-time performance",
            10th IEEE International Conference on Computer Vision, 2005.
            ICCV 2005.
    '''
    #presmoothing
    for i in range(k):
        error = ssd.iterate_residual_displacement_field_SSD2D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD2D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [first]:', energy
    if n == 0:
        return error
    residual = ssd.compute_residual_displacement_field_SSD2D(delta_field,
                                                            sigma_field,
                                                            gradient_field,
                                                            target,
                                                            lambda_param,
                                                            displacement,
                                                            None)
    sub_residual = np.array(vfu.downsample_displacement_field2D(residual))
    del residual
    #solve at coarcer grid
    subsigma_field = None
    if sigma_field != None:
        subsigma_field = vfu.downsample_scalar_field2D(sigma_field)
    subdelta_field = vfu.downsample_scalar_field2D(delta_field)
    subgradient_field = np.array(
        vfu.downsample_displacement_field2D(gradient_field))
    shape = np.array(displacement.shape).astype(np.int32)
    #sub_displacement = np.array(vfu.downsample_displacement_field(displacement))
    sub_displacement = np.zeros(shape = ((shape[0]+1)//2, (shape[1]+1)//2, 2 ),
                               dtype = np.float64)
    sublambda_param = lambda_param*0.25
    w_cycle_2d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
             sub_residual, sublambda_param, sub_displacement, depth+1)
    displacement += np.array(
        vfu.upsample_displacement_field(sub_displacement, shape))
    if printEnergy and depth == 0:
        energy = ssd.compute_energy_SSD2D(delta_field, sigma_field, 
                                         gradient_field, lambda_param, 
                                         displacement)
        print 'Energy after low-res iteration[first]:', energy
    #post-smoothing (second smoothing)
    for i in range(k):
        error = ssd.iterate_residual_displacement_field_SSD2D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD2D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [second]:', energy
    residual = ssd.compute_residual_displacement_field_SSD2D(delta_field,
                                                            sigma_field,
                                                            gradient_field,
                                                            target,
                                                            lambda_param,
                                                            displacement,
                                                            None)
    sub_residual = np.array(vfu.downsample_displacement_field2D(residual))
    del residual
    #sub_displacement = np.array(vfu.downsample_displacement_field(displacement))
    sub_displacement = np.zeros(shape = ((shape[0]+1)//2, (shape[1]+1)//2, 2 ),
                               dtype = np.float64)
    w_cycle_2d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
             sub_residual, sublambda_param, sub_displacement, depth+1)
    displacement += np.array(
        vfu.upsample_displacement_field(sub_displacement, shape))
    if printEnergy and depth == 0:
        energy = ssd.compute_energy_SSD2D(delta_field, sigma_field, 
                                         gradient_field, lambda_param,
                                         displacement)
        print 'Energy after low-res iteration[second]:', energy
    for i in range(k):
        error = ssd.iterate_residual_displacement_field_SSD2D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD2D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [third]:', energy
    try:
        energy
    except NameError:
        energy = ssd.compute_energy_SSD2D(delta_field, sigma_field,
                                         gradient_field, lambda_param, 
                                         displacement)
    return energy

def single_cycle_3d(n, k, delta_field, sigma_field, gradient_field,
                    lambda_param, displacement, depth = 0):
    r'''
    One-pass multi-resolution Gauss-Seidel solver: solves the SSD-like linear
    system starting at the coarcer resolution and then refining at the finer
    resolution.
    '''
    if n == 0:
        for i in range(k):
            error = ssd.iterate_residual_displacement_field_SSD3D(delta_field,
                                                                 sigma_field,
                                                                 gradient_field,
                                                                 None,
                                                                 lambda_param,
                                                                 displacement)
            if printEnergy and depth == 0:
                energy = ssd.compute_energy_SSD3D(delta_field,
                                                 sigma_field,
                                                 gradient_field,
                                                 lambda_param,
                                                 displacement)
                print 'Energy after top-level iter', i+1, ' [unique]:', energy
        return error
    #solve at coarcer grid
    subsigma_field = None
    if sigma_field != None:
        subsigma_field = vfu.downsample_scalar_field3D(sigma_field)
    subdelta_field = vfu.downsample_scalar_field3D(delta_field)
    subgradient_field = np.array(
        vfu.downsample_displacement_field3D(gradient_field))
    shape = np.array(displacement.shape).astype(np.int32)
    sub_displacement = np.zeros(
        shape = ((shape[0]+1)//2, (shape[1]+1)//2, (shape[2]+1)//2, 3 ),
        dtype = np.float64)
    sublambda_param = lambda_param*0.25
    single_cycle_3d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
                  sublambda_param, sub_displacement, depth+1)
    displacement += np.array(
        vfu.upsample_displacement_field3D(sub_displacement, shape))
    if printEnergy and depth == 0:
        energy = ssd.compute_energy_SSD3D(delta_field, sigma_field,
                                         gradient_field, lambda_param,
                                         displacement)
        print 'Energy after low-res iteration:', energy
    #post-smoothing
    for i in range(k):
        error = ssd.iterate_residual_displacement_field_SSD3D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             None,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD3D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [unique]:', energy
    try:
        energy
    except NameError:
        energy = ssd.compute_energy_SSD3D(delta_field, sigma_field,
                                         gradient_field, lambda_param,
                                         displacement)
    return energy

def v_cycle_3d(n, k, delta_field, sigma_field, gradient_field, target,
             lambda_param, displacement, depth = 0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by first
    filtering (GS-iterate) the current level, then solves for the residual
    at a coarcer resolution andfinally refines the solution at the current 
    resolution. This scheme corresponds to the V-cycle proposed by Bruhn and 
    Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
        combining highest accuracy with real-time performance",
        10th IEEE International Conference on Computer Vision, 2005.
        ICCV 2005.
    '''
    #presmoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_SSD3D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD3D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [unique]:', energy
    if n == 0:
        try:
            energy
        except NameError:
            energy = ssd.compute_energy_SSD3D(delta_field, sigma_field,
                                         gradient_field, lambda_param,
                                         displacement)
        return energy
    #solve at coarcer grid
    residual = ssd.compute_residual_displacement_field_SSD3D(delta_field,
                                                            sigma_field,
                                                            gradient_field,
                                                            target,
                                                            lambda_param,
                                                            displacement,
                                                            None)
    sub_residual = np.array(vfu.downsample_displacement_field3D(residual))
    del residual
    subsigma_field = None
    if sigma_field != None:
        subsigma_field = vfu.downsample_scalar_field3D(sigma_field)
    subdelta_field = vfu.downsample_scalar_field3D(delta_field)
    subgradient_field = np.array(
        vfu.downsample_displacement_field3D(gradient_field))
    shape = np.array(displacement.shape).astype(np.int32)
    sub_displacement = np.zeros(
        shape = ((shape[0]+1)//2, (shape[1]+1)//2, (shape[2]+1)//2, 3 ),
        dtype = np.float64)
    sublambda_param = lambda_param*0.25
    v_cycle_3d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
             sub_residual, sublambda_param, sub_displacement, depth+1)
    del subdelta_field
    del subsigma_field
    del subgradient_field
    del sub_residual
    vfu.accumulate_upsample_displacement_field3D(sub_displacement, displacement)
    del sub_displacement
    if printEnergy and depth == 0:
        energy = ssd.compute_energy_SSD3D(delta_field, sigma_field, 
                                         gradient_field, lambda_param,
                                         displacement)
        print 'Energy after low-res iteration:', energy
    #post-smoothing
    for i in range(k):
        ssd.iterate_residual_displacement_field_SSD3D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD3D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [unique]:', energy
    try:
        energy
    except NameError:
        energy = ssd.compute_energy_SSD3D(delta_field, sigma_field,
                                         gradient_field, lambda_param,
                                         displacement)
    return energy

def w_cycle_3d(n, k, delta_field, sigma_field, gradient_field, target,
             lambda_param, displacement, depth = 0):
    r'''
    Multi-resolution Gauss-Seidel solver: solves the linear system by performing
    two v-cycles at each resolution, which corresponds to the w-cycle scheme
    proposed by Bruhn and Weickert[1].
    [1] Andres Bruhn and Joachim Weickert, "Towards ultimate motion estimation:
        combining highest accuracy with real-time performance",
        10th IEEE International Conference on Computer Vision, 2005. ICCV 2005.
    '''
    #presmoothing
    for i in range(k):
        error = ssd.iterate_residual_displacement_field_SSD3D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD3D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [first]:', energy
    if n == 0:
        return error
    residual = ssd.compute_residual_displacement_field_SSD3D(delta_field,
                                                            sigma_field,
                                                            gradient_field,
                                                            target,
                                                            lambda_param,
                                                            displacement,
                                                            None)
    sub_residual = np.array(vfu.downsample_displacement_field3D(residual))
    del residual
    #solve at coarcer grid
    subsigma_field = None
    if sigma_field != None:
        subsigma_field = vfu.downsample_scalar_field3D(sigma_field)
    subdelta_field = vfu.downsample_scalar_field3D(delta_field)
    subgradient_field = np.array(
        vfu.downsample_displacement_field3D(gradient_field))
    shape = np.array(displacement.shape).astype(np.int32)
    sub_displacement = np.zeros(
        shape = ((shape[0]+1)//2, (shape[1]+1)//2, (shape[2]+1)//2, 3 ),
        dtype = np.float64)
    sublambda_param = lambda_param*0.25
    w_cycle_3d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
             sub_residual, sublambda_param, sub_displacement, depth+1)
    displacement += np.array(
        vfu.upsample_displacement_field3D(sub_displacement, shape))
    if printEnergy and depth == 0:
        energy = ssd.compute_energy_SSD3D(delta_field, sigma_field,
                                         gradient_field, lambda_param, 
                                         displacement)
        print 'Energy after low-res iteration[first]:', energy
    #post-smoothing (second smoothing)
    for i in range(k):
        error = ssd.iterate_residual_displacement_field_SSD3D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD3D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [second]:', energy
    residual = ssd.compute_residual_displacement_field_SSD3D(delta_field,
                                                            sigma_field,
                                                            gradient_field,
                                                            target,
                                                            lambda_param,
                                                            displacement,
                                                            None)
    sub_residual = np.array(vfu.downsample_displacement_field3D(residual))
    del residual
    sub_displacement[...] = 0
    w_cycle_3d(n-1, k, subdelta_field, subsigma_field, subgradient_field,
             sub_residual, sublambda_param, sub_displacement, depth+1)
    displacement += np.array(
        vfu.upsample_displacement_field3D(sub_displacement, shape))
    if printEnergy and depth == 0:
        energy = ssd.compute_energy_SSD3D(delta_field, sigma_field,
                                         gradient_field, lambda_param,
                                         displacement)
        print 'Energy after low-res iteration[second]:', energy
    for i in range(k):
        error = ssd.iterate_residual_displacement_field_SSD3D(delta_field,
                                                             sigma_field,
                                                             gradient_field,
                                                             target,
                                                             lambda_param,
                                                             displacement)
        if printEnergy and depth == 0:
            energy = ssd.compute_energy_SSD3D(delta_field,
                                             sigma_field,
                                             gradient_field,
                                             lambda_param,
                                             displacement)
            print 'Energy after top-level iter', i+1, ' [third]:', energy
    try:
        energy
    except NameError:
        energy = ssd.compute_energy_SSD3D(delta_field, sigma_field,
                                         gradient_field, lambda_param, 
                                         displacement)
    return energy
