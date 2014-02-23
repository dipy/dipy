'''
Especialization of the registration optimizer to perform symmetric registration
'''
import time
import numpy as np
import matplotlib.pyplot as plt
import RegistrationCommon as rcommon
import VectorFieldUtils as vfu
import UpdateRule
from TransformationModel import TransformationModel
from SSDMetric import SSDMetric
from RegistrationOptimizer import RegistrationOptimizer
from scipy import interpolate

class SymmetricRegistrationOptimizer(RegistrationOptimizer):
    r'''
    Performs the multi-resolution optimization algorithm for non-linear
    registration using a given similarity metric and update rule (this
    scheme was inspider on the ANTS package).
    '''
    def get_default_parameters(self):
        return {'max_iter':[25, 50, 100], 'inversion_iter':20,
                'inversion_tolerance':1e-3, 'tolerance':1e-4,
                'report_status':True}

    def __init__(self,
                 fixed = None,
                 moving = None,
                 affine_fixed = None,
                 affine_moving = None,
                 similarity_metric = None,
                 update_rule = None,
                 parameters = None):
        super(SymmetricRegistrationOptimizer, self).__init__(
            fixed, moving, affine_fixed, affine_moving, similarity_metric,
            update_rule, parameters)
        self.set_max_iter(self.parameters['max_iter'])
        self.tolerance = self.parameters['tolerance']
        self.inversion_tolerance = self.parameters['inversion_tolerance']
        self.inversion_iter = self.parameters['inversion_iter']
        self.report_status = self.parameters['report_status']
        self.energy_window = 12

    def __connect_functions(self):
        r'''
        Assigns the appropriate functions to be called for displacement field
        inversion, Gaussian pyramid, and affine/dense deformation composition 
        according to the dimension of the input images
        '''
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

    def __check_ready(self):
        r'''
        Verifies that the configuration of the optimizer and input data are
        consistent and the optimizer is ready to run
        '''
        ready = True
        if self.fixed == None:
            ready = False
            print('Error: Fixed image not set.')
        elif self.dim != len(self.fixed.shape):
            ready = False
            print('Error: inconsistent dimensions. Last dimension update: %d.'
                  'Fixed image dimension: %d.'%(self.dim,
                                                len(self.fixed.shape)))
        if self.moving == None:
            ready = False
            print('Error: Moving image not set.')
        elif self.dim != len(self.moving.shape):
            ready = False
            print('Error: inconsistent dimensions. Last dimension update: %d.'
                  'Moving image dimension: %d.'%(self.dim,
                                                 len(self.moving.shape)))
        if self.similarity_metric == None:
            ready = False
            print('Error: Similarity metric not set.')
        if self.update_rule == None:
            ready = False
            print('Error: Update rule not set.')
        if self.max_iter == None:
            ready = False
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
        self.moving_pyramid = [img for img
                               in self.generate_pyramid(self.moving,
                                                        self.levels-1)]
        self.fixed_pyramid = [img for img
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
        starting_backward = np.zeros(
            shape = self.moving_pyramid[self.levels-1].shape+(self.dim,),
            dtype = np.float64)
        starting_backward_inverse = np.zeros(
            shape = self.fixed_pyramid[self.levels-1].shape+(self.dim,),
            dtype = np.float64)
        self.backward_model.scale_affines(0.5**(self.levels-1))
        self.backward_model.set_forward(starting_backward)
        self.backward_model.set_backward(starting_backward_inverse)

    def __end_optimizer(self):
        r'''
        Frees the resources allocated during initialization
        '''
        del self.moving_pyramid
        del self.fixed_pyramid

    def __iterate(self, show_images = False):
        r'''
        Performs one symmetric iteration:
            1.Compute forward
            2.Compute backward
            3.Update forward
            4.Update backward
            5.Compute inverses
            6.Invert the inverses to improve invertibility
        '''
        #tic = time.time()
        wmoving = self.backward_model.warp_backward(self.current_moving)
        wfixed = self.forward_model.warp_backward(self.current_fixed)
        self.similarity_metric.set_moving_image(wmoving)
        self.similarity_metric.use_moving_image_dynamics(
            self.current_moving, self.backward_model.inverse())
        self.similarity_metric.set_fixed_image(wfixed)
        self.similarity_metric.use_fixed_image_dynamics(
            self.current_fixed, self.forward_model.inverse())
        self.similarity_metric.initialize_iteration()
        ff_shape = np.array(self.forward_model.forward.shape).astype(np.int32)
        fb_shape = np.array(self.forward_model.backward.shape).astype(np.int32)
        bf_shape = np.array(self.backward_model.forward.shape).astype(np.int32)
        bb_shape = np.array(self.backward_model.backward.shape).astype(np.int32)
        del self.forward_model.backward
        del self.backward_model.backward
        fw_step = np.array(self.similarity_metric.compute_forward())
        self.forward_model.forward, md_forward = self.update_rule.update(
            self.forward_model.forward, fw_step)
        del fw_step
        try:
            fw_energy = self.similarity_metric.energy
        except NameError:
            pass
        bw_step = np.array(self.similarity_metric.compute_backward())
        self.backward_model.forward, md_backward = self.update_rule.update(
            self.backward_model.forward, bw_step)
        del bw_step
        try:
            bw_energy = self.similarity_metric.energy
        except NameError:
            pass
        der = '-'
        try:
            n_iter = len(self.energy_list)
            if len(self.energy_list)>=self.energy_window:
                der = self.__get_energy_derivative()
            print('%d:\t%0.6f\t%0.6f\t%0.6f\t%s'%(n_iter , fw_energy, bw_energy,
                fw_energy + bw_energy, der))
            self.energy_list.append(fw_energy+bw_energy)
        except NameError:
            pass
        self.similarity_metric.free_iteration()
        inv_iter = self.inversion_iter
        inv_tol = self.inversion_tolerance
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
        if show_images:
            self.similarity_metric.report_status()
        #toc = time.time()
        #print('Iter time: %f sec' % (toc - tic))
        return 1 if der=='-' else der

    def __get_energy_derivative(self):
        r'''
        Returns the derivative of the estimated energy as a function of "time"
        (iterations) at the last iteration
        '''
        n_iter = len(self.energy_list)
        if n_iter<self.energy_window:
            print 'Error: attempting to fit the energy profile with less points (',n_iter,') than required (energy_window=', self.energy_window,')'
            return 1
        x=range(self.energy_window)
        y=self.energy_list[(n_iter-self.energy_window):n_iter]
        ss=sum(y)
        if(ss>0):
            ss*=-1
        y=[v/ss for v in y]
        spline = interpolate.UnivariateSpline(x, y, s = 1e6, k=2)
        derivative = spline.derivative()
        der = derivative(0.5*self.energy_window)
        return der

    def __report_status(self, level):
        r'''
        Shows the current overlaid images either on the common space or the
        reference space
        '''
        show_common_space = True
        if show_common_space:
            wmoving = self.backward_model.warp_backward(self.current_moving)
            wfixed = self.forward_model.warp_backward(self.current_fixed)
            self.similarity_metric.set_moving_image(wmoving)
            self.similarity_metric.use_moving_image_dynamics(
                self.current_moving, self.backward_model.inverse())
            self.similarity_metric.set_fixed_image(wfixed)
            self.similarity_metric.use_fixed_image_dynamics(
                self.current_fixed, self.forward_model.inverse())
            self.similarity_metric.initialize_iteration()
            self.similarity_metric.report_status()
        else:
            phi1 = self.forward_model.forward
            phi2 = self.backward_model.backward
            phi1_inv = self.forward_model.backward
            phi2_inv = self.backward_model.forward
            phi, mean_disp = self.update_rule.update(phi1, phi2)
            phi_inv, mean_disp = self.update_rule.update(phi2_inv, phi1_inv)
            composition = TransformationModel(phi, phi_inv, None, None)
            composition.scale_affines(0.5**level)
            residual, stats = composition.compute_inversion_error()
            print('Current inversion error: %0.6f (%0.6f)'%(stats[1], stats[2]))
            wmoving = composition.warp_forward(self.current_moving)
            self.similarity_metric.set_moving_image(wmoving)
            self.similarity_metric.use_moving_image_dynamics(
                self.current_moving, composition)
            self.similarity_metric.set_fixed_image(self.current_fixed)
            self.similarity_metric.use_fixed_image_dynamics(
                self.current_fixed, None)
            self.similarity_metric.initialize_iteration()
            self.similarity_metric.report_status()

    def __optimize(self):
        r'''
        The main multi-scale symmetric optimization algorithm
        '''
        self.__init_optimizer()
        for level in range(self.levels-1, -1, -1):
            print 'Processing level', level
            self.current_fixed = self.fixed_pyramid[level]
            self.current_moving = self.moving_pyramid[level]
            self.similarity_metric.use_original_fixed_image(
                self.fixed_pyramid[level])
            self.similarity_metric.use_original_fixed_image(
                self.moving_pyramid[level])
            self.similarity_metric.set_levels_below(self.levels-level)
            self.similarity_metric.set_levels_above(level)
            if level < self.levels - 1:
                self.forward_model.upsample(self.current_fixed.shape,
                                           self.current_fixed.shape)
                self.backward_model.upsample(self.current_moving.shape,
                                            self.current_fixed.shape)
            niter = 0
            self.energy_list = []
            derivative = 1
            while ((niter < self.max_iter[level]) and (self.tolerance<derivative)):
                niter += 1
                derivative = self.__iterate()
            if self.report_status:
                self.__report_status(level)
        residual, stats = self.forward_model.compute_inversion_error()
        print('Forward Residual error (Symmetric diffeomorphism):%0.6f (%0.6f)'
              %(stats[1], stats[2]))
        residual, stats = self.backward_model.compute_inversion_error()
        print('Backward Residual error (Symmetric diffeomorphism):%0.6f (%0.6f)'
              %(stats[1], stats[2]))
        #Compose the two partial transformations
        self.forward_model=self.backward_model.inverse().compose(self.forward_model)
        self.forward_model.consolidate()
        del self.backward_model
        residual, stats = self.forward_model.compute_inversion_error()
        print('Residual error (Symmetric diffeomorphism):%0.6f (%0.6f)'
              %(stats[1], stats[2]))
        self.__end_optimizer()

    def optimize(self):
        print 'Optimizer parameters:\n', self.parameters
        print 'Metric:', self.similarity_metric.get_metric_name()
        print 'Metric parameters:\n', self.similarity_metric.parameters
        self.__optimize()

def test_optimizer_monomodal_2d():
    r'''
    Classical Circle-To-C experiment for 2D Monomodal registration
    '''
    fname_moving = 'data/circle.png'
    fname_fixed = 'data/C.png'
    moving = plt.imread(fname_moving)
    fixed = plt.imread(fname_fixed)
    moving = moving[:, :, 0].astype(np.float64)
    fixed = fixed[:, :, 0].astype(np.float64)
    moving = np.copy(moving, order = 'C')
    fixed = np.copy(fixed, order = 'C')
    moving = (moving-moving.min())/(moving.max() - moving.min())
    fixed = (fixed-fixed.min())/(fixed.max() - fixed.min())
    ################Configure and run the Optimizer#####################
    max_iter = [i for i in [20, 100, 100, 100]]
    similarity_metric = SSDMetric(2, {'symmetric':True,
                                'lambda':4.0,
                                'stepType':SSDMetric.GAUSS_SEIDEL_STEP})
    optimizer_parameters = {
        'max_iter':max_iter,
        'inversion_iter':40,
        'inversion_tolerance':1e-3,
        'report_status':True}
    update_rule = UpdateRule.Composition()
    registration_optimizer = SymmetricRegistrationOptimizer(fixed, moving,
                                                         None, None,
                                                         similarity_metric,
                                                         update_rule, optimizer_parameters)
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

if __name__ == '__main__':
    start_time = time.time()
    test_optimizer_monomodal_2d()
    end_time = time.time()
    print('Registration time: %f sec' % (end_time - start_time))

