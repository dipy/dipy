'''
This abstract class defines the contract that must be fulfilled by especialized
registration optimizers
'''
import numpy as np
from TransformationModel import TransformationModel
import abc

class RegistrationOptimizer(object):
    r'''
    This abstract class defines the interface to be implemented by any
    optimization algorithm for nonlinear Registration
    '''
    @abc.abstractmethod
    def get_default_parameters(self):
        r'''
        Derived classes must return a dictionary containing its parameter names
        and default values
        '''

    def __init__(self,
                 fixed = None,
                 moving = None,
                 affine_fixed = None,
                 affine_moving = None,
                 similarity_metric = None,
                 update_rule = None,
                 parameters = None):
        default_parameters = self.get_default_parameters()
        if parameters != None:
            for key, val in parameters.iteritems():
                if key in default_parameters:
                    default_parameters[key] = val
                else:
                    print "Warning: parameter '", key, "' unknown. Ignored."
        if affine_fixed != None:
            print('Warning: an affine_fixed matrix was given as argument.'
                  'This functionality has not been implemented yet.')
        self.parameters = default_parameters
        inv_affine_moving = None
        if affine_moving != None:
            inv_affine_moving = np.linalg.inv(affine_moving).copy(order = 'C')
        self.dim = 0
        self.set_fixed_image(fixed)
        self.forward_model = TransformationModel(None, None, None, None)
        self.set_moving_image(moving)
        self.backward_model = TransformationModel(None, None, inv_affine_moving,
                                                None)
        self.similarity_metric = similarity_metric
        self.update_rule = update_rule
        self.energy_list = None

    def set_fixed_image(self, fixed):
        r'''
        Establishes the fixed image to be used by this registration optimizer.
        Updates the domain dimension information accordingly
        '''
        if fixed != None:
            self.dim = len(fixed.shape)
        self.fixed = fixed

    def set_moving_image(self, moving):
        r'''
        Establishes the moving image to be used by this registration optimizer.
        Updates the domain dimension information accordingly
        '''
        if moving != None:
            self.dim = len(moving.shape)
        self.moving = moving

    def set_max_iter(self, max_iter):
        r'''
        Establishes the maximum number of iterations to be performed at each
        level of the Gaussian pyramid, similar to ANTS
        '''
        self.levels = len(max_iter) if max_iter else 0
        self.max_iter = max_iter

    @abc.abstractmethod
    def optimize(self):
        r'''
        This is the main function each especialized class derived from this must
        implement. Upon completion, the deformation field must be available from
        the forward transformation model.
        '''

    def get_forward(self):
        r'''
        Returns the forward model's forward deformation field
        '''
        return self.forward_model.forward

    def get_backward(self):
        r'''
        Returns the forward model's backward (inverse) deformation field
        '''
        return self.forward_model.backward
