'''
Especialization of the registration optimizer to perform asymmetric
(unidirectional) registration
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import RegistrationCommon as rcommon
import VectorFieldUtils as vfu
import UpdateRule
from SSDMetric import SSDMetric
from EMMetric import EMMetric
from RegistrationOptimizer import RegistrationOptimizer

class AsymmetricRegistrationOptimizer(RegistrationOptimizer):
    r'''
    Performs the multi-resolution optimization algorithm for non-linear
    registration using a given similarity metric and update rule (this
    scheme was inspider on the ANTS package).
    '''
    def get_default_parameters(self):
        return {'max_iter':[25, 50, 100], 'inversion_iter':20,
                'inversion_tolerance':1e-3, 'tolerance':1e-6,
                'report_status':True}

    def __init__(self,
                 fixed = None,
                 moving = None, affine_fixed = None,
                 affine_moving = None,
                 similarity_metric = None,
                 update_rule = None,
                 parameters = None):
        super(AsymmetricRegistrationOptimizer, self).__init__(
            fixed, moving, affine_fixed, affine_moving, similarity_metric,
            update_rule, parameters)
        print('Warning: AsymmetricRegistrationOptimizer has not been'
              'maintained in a long time, it may not work properly.'
              'Use SymmetricRegistrationOptimizer instead')
        self.set_max_iter(self.parameters['max_iter'])
        self.tolerance = self.parameters['tolerance']
        self.inversion_tolerance = self.parameters['inversion_tolerance']
        self.inversion_iter = self.parameters['inversion_iter']
        self.report_status = self.parameters['report_status']
        self.energy_list = []

    def __connect_functions(self):
        r'''
        Assigns the appropriate functions to be called for displacement field
        inversion and Gaussian pyramid according to the dimension of the
        input images
        '''
        if self.dim == 2:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point
            self.generate_pyramid = rcommon.pyramid_gaussian_2D
        else:
            self.invert_vector_field = vfu.invert_vector_field_fixed_point_3d
            self.generate_pyramid = rcommon.pyramid_gaussian_3D

    def __check_ready(self):
        r'''
        Verifies that the configuration of the optimizer and input data are
        consistent and the optimizer is ready to run
        '''
        ready  =  True
        if self.fixed  ==  None:
            ready  =  False
            print('Error: Fixed image not set.')
        elif self.dim !=  len(self.fixed.shape):
            ready  =  False
            print('Error: inconsistent dimensions. Last dimension update: %d.'
                  'Fixed image dimension: %d.'%(self.dim,
                                                len(self.fixed.shape)))
        if self.moving  ==  None:
            ready  =  False
            print('Error: Moving image not set.')
        elif self.dim !=  len(self.moving.shape):
            ready  =  False
            print('Error: inconsistent dimensions. Last dimension update: %d.'
                  'Moving image dimension: %d.'%(self.dim,
                                                 len(self.moving.shape)))
        if self.similarity_metric  ==  None:
            ready  =  False
            print('Error: Similarity metric not set.')
        if self.update_rule  ==  None:
            ready  =  False
            print('Error: Update rule not set.')
        if self.max_iter  ==  None:
            ready  =  False
            print('Error: Maximum number of iterations per level not set.')
        return ready

    def __init_optimizer(self):
        r'''
        Computes the Gaussian Pyramid of the input images and allocates
        the required memory for the transformation models at the coarcest
        scale.
        '''
        ready = self.__check_ready()
        self.__connect_functions()
        if not ready:
            print 'Not ready'
            return False
        self.moving_pyramid  =  [img for img
                               in self.generate_pyramid(self.moving,
                                                       self.levels-1)]
        self.fixed_pyramid  =  [img for img
                              in self.generate_pyramid(self.fixed,
                                                      self.levels-1)]
        starting_forward = np.zeros(
            shape = self.fixed_pyramid[self.levels-1].shape+(self.dim,),
            dtype = np.float64)
        starting_forward_inv = np.zeros(
            shape = self.fixed_pyramid[self.levels-1].shape+(self.dim,),
            dtype = np.float64)
        self.forward_model.scale_affines(0.5**(self.levels-1))
        self.forward_model.set_forward(starting_forward)
        self.forward_model.set_backward(starting_forward_inv)

    def __end_optimizer(self):
        r'''
        Frees the resources allocated during initialization
        '''
        del self.moving_pyramid
        del self.fixed_pyramid

    def __iterate(self, show_images = False):
        r'''
        Performs one unidirectional iteration
        '''
        wmoving = self.forward_model.warp_forward(self.current_moving)
        self.similarity_metric.set_moving_image(wmoving)
        self.similarity_metric.use_moving_image_dynamics(
            self.current_moving, self.forward_model)
        self.similarity_metric.set_fixed_image(self.current_fixed)
        self.similarity_metric.use_fixed_image_dynamics(
            self.current_fixed, None)
        self.similarity_metric.initialize_iteration()
        fw_step = np.array(self.similarity_metric.compute_forward())
        self.forward_model.forward, mean_diff = self.update_rule.update(
            fw_step, self.forward_model.forward)
        try:
            fw_energy = self.similarity_metric.energy
        except NameError:
            pass
        try:
            n_iter = len(self.energy_list)
            der = '-'
            if len(self.energy_list)>=3:
                der = self.__get_energy_derivative()
            print('%d:\t%0.6f\t%s'%(n_iter , fw_energy, der))
            self.energy_list.append(fw_energy)
        except NameError:
            pass
        if show_images:
            self.similarity_metric.report_status()
        return mean_diff

    def __optimize(self):
        r'''
        The main multi-scale optimization algorithm for unidirectional
        registration
        '''
        self.__init_optimizer()
        for level in range(self.levels-1, -1, -1):
            print 'Processing level', level
            self.current_fixed = self.fixed_pyramid[level]
            self.current_moving = self.moving_pyramid[level]
            self.similarity_metric.use_original_fixed_image(
                self.fixed_pyramid[level])
            self.similarity_metric.use_original_moving_image(
                self.moving_pyramid[level])
            if level < self.levels-1:
                self.forward_model.upsample(self.current_fixed.shape,
                                            self.current_moving.shape)
            niter = 0
            self.energy_list = []
            while (niter<self.max_iter[level]):
                niter += 1
                self.__iterate()
            if self.report_status:
                self.__report_status(level)
        self.forward_model.backward = self.invert_vector_field(
                self.forward_model.forward, None, self.inversion_iter, 
                self.inversion_tolerance, None)
        self.__end_optimizer()

    def __get_energy_derivative(self):
        r'''
        Returns the derivative of the estimated energy as a function of "time"
        (iterations) at the last iteration
        '''
        n_iter = len(self.energy_list)
        poly_der = np.poly1d(
            np.polyfit(range(n_iter), self.energy_list, 2)).deriv()
        der = poly_der(n_iter-1.5)
        return der

    def __report_status(self, level):
        r'''
        Shows the current overlaid images on the reference space
        '''
        wmoving = self.forward_model.warp_forward(self.current_moving)
        self.similarity_metric.set_moving_image(wmoving)
        self.similarity_metric.use_moving_image_dynamics(
            self.current_moving, self.forward_model)
        self.similarity_metric.set_fixed_image(self.current_fixed)
        self.similarity_metric.use_fixed_image_dynamics(
            self.current_fixed, None)
        self.similarity_metric.initialize_iteration()
        self.similarity_metric.report_status()

    def optimize(self):
        print 'Outer iter:', self.max_iter
        print 'Metric:', self.similarity_metric.get_metric_name()
        print 'Metric parameters:\n', self.similarity_metric.parameters
        self.__optimize()

def test_optimizer_monomodal_2d():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration
    '''
    fname_moving = 'data/circle.png'
    fname_fixed = 'data/C.png'
    nib_moving = plt.imread(fname_moving)
    nib_fixed = plt.imread(fname_fixed)
    moving = nib_moving[:, :, 0].astype(np.float64)
    fixed = nib_fixed[:, :, 1].astype(np.float64)
    moving = np.copy(moving, order = 'C')
    fixed = np.copy(fixed, order = 'C')
    moving = (moving-moving.min())/(moving.max() - moving.min())
    fixed = (fixed-fixed.min())/(fixed.max() - fixed.min())
    ################Configure and run the Optimizer#####################
    max_iter = [i for i in [25, 100, 100, 100]]
    similarity_metric = SSDMetric({'lambda':5.0,
                                   'max_inner_iter':50,
                                   'step_type':SSDMetric.GAUSS_SEIDEL_STEP})
    update_rule = UpdateRule.Composition()
    optimizer_parameters = {
        'max_iter':max_iter,
        'inversion_iter':40,
        'inversion_tolerance':1e-3,
        'report_status':True}
    registration_optimizer = AsymmetricRegistrationOptimizer(
        fixed, moving, None, None,
        similarity_metric, update_rule, optimizer_parameters)
    registration_optimizer.optimize()
    #######################show results#################################
    displacement = registration_optimizer.get_forward()
    direct_inverse = registration_optimizer.get_backward()
    moving_to_fixed = np.array(vfu.warp_image(moving, displacement))
    fixed_to_moving = np.array(vfu.warp_image(fixed, direct_inverse))
    rcommon.overlayImages(moving_to_fixed, fixed, True)
    rcommon.overlayImages(fixed_to_moving, moving, True)
    direct_residual, stats = vfu.compose_vector_fields(displacement,
                                                      direct_inverse)
    direct_residual = np.array(direct_residual)
    rcommon.plotDiffeomorphism(displacement, direct_inverse, direct_residual,
                               'inv-direct', 7)

def test_optimizer_multimodal_2d(lambda_param):
    r'''
    Registers one of the mid-slices (axial, coronal or sagital) of each input
    volume (the volumes are expected to be from diferent modalities and
    should already be affine-registered, for example Brainweb t1 vs t2)
    '''
    fname_moving = 'data/t2/IBSR_t2template_to_01.nii.gz'
    fname_fixed = 'data/t1/IBSR_template_to_01.nii.gz'
#    fname_moving = 'data/circle.png'
#    fname_fixed = 'data/C.png'
    nifti = True
    if nifti:
        nib_moving  =  nib.load(fname_moving)
        nib_fixed  =  nib.load(fname_fixed)
        moving = nib_moving.get_data().squeeze().astype(np.float64)
        fixed = nib_fixed.get_data().squeeze().astype(np.float64)
        moving = np.copy(moving, order = 'C')
        fixed = np.copy(fixed, order = 'C')
        moving_shape = moving.shape
        fixed_shape = fixed.shape
        moving = moving[:, moving_shape[1]//2, :].copy()
        fixed = fixed[:, fixed_shape[1]//2, :].copy()
#        moving = histeq(moving)
#        fixed = histeq(fixed)
        moving = (moving-moving.min())/(moving.max()-moving.min())
        fixed = (fixed-fixed.min())/(fixed.max()-fixed.min())
    else:
        nib_moving = plt.imread(fname_moving)
        nib_fixed = plt.imread(fname_fixed)
        moving = nib_moving[:, :, 0].astype(np.float64)
        fixed = nib_fixed[:, :, 1].astype(np.float64)
        moving = np.copy(moving, order = 'C')
        fixed = np.copy(fixed, order = 'C')
        moving = (moving-moving.min())/(moving.max() - moving.min())
        fixed = (fixed-fixed.min())/(fixed.max() - fixed.min())
    #max_iter = [i for i in [25,50,100,100]]
    max_iter = [i for i in [25, 50, 100]]
    similarity_metric = EMMetric({'symmetric':True,
                               'lambda':lambda_param,
                               'step_type':SSDMetric.GAUSS_SEIDEL_STEP,
                               'q_levels':256,
                               'max_inner_iter':40,
                               'use_double_gradient':True,
                               'max_step_length':0.25})
    optimizer_parameters = {
        'max_iter':max_iter,
        'inversion_iter':40,
        'inversion_tolerance':1e-3,
        'report_status':True}
    update_rule = UpdateRule.Composition()
    print 'Generating synthetic field...'
    #----apply synthetic deformation field to fixed image
    ground_truth = rcommon.createDeformationField2D_type2(fixed.shape[0],
                                                          fixed.shape[1], 8)
    warped_fixed = rcommon.warpImage(fixed, ground_truth)
    print 'Registering T2 (template) to deformed T1 (template)...'
    plt.figure()
    rcommon.overlayImages(warped_fixed, moving, False)
    registration_optimizer = AsymmetricRegistrationOptimizer(
        warped_fixed, moving, None, None, similarity_metric,
        update_rule, optimizer_parameters)
    registration_optimizer.optimize()
    #######################show results#################################
    displacement = registration_optimizer.get_forward()
    direct_inverse = registration_optimizer.get_backward()
    moving_to_fixed = np.array(vfu.warp_image(moving, displacement))
    fixed_to_moving = np.array(vfu.warp_image(warped_fixed, direct_inverse))
    rcommon.overlayImages(moving_to_fixed, fixed_to_moving, True)
    direct_residual, stats = vfu.compose_vector_fields(displacement,
                                                      direct_inverse)
    direct_residual = np.array(direct_residual)
    rcommon.plotDiffeomorphism(displacement, direct_inverse, direct_residual,
                               'inv-direct', 7)

    residual = ((displacement-ground_truth))**2
    mean_error = np.sqrt(residual.sum(2)*(warped_fixed>0)).mean()
    stdev_error = np.sqrt(residual.sum(2)*(warped_fixed>0)).std()
    print 'Mean displacement error: ', mean_error, '(', stdev_error, ')'

if __name__ == '__main__':
    TIC = time.time()
    test_optimizer_multimodal_2d(50)
    TOC = time.time()
    print('Registration time: %f sec' % (TOC - TIC))
