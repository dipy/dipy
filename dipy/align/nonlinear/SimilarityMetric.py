'''
Defines the contract that must be fulfilled by the especialized similarity 
metrics to be used with a RegistrationOptimizer
'''
import abc
class SimilarityMetric(object):
    '''
    A similarity metric is in charge of keeping track of the numerical value
    of the similarity (or distance) between the two given images. It also
    computes the update field for the forward and inverse
    displacement fields to be used in a gradient-based optimization algorithm.
    Note that this metric does not depend on any transformation (affine or
    non-linear), so it assumes the fixed and reference images are already warped
    '''
    __metaclass__ = abc.ABCMeta
    def __init__(self, dim, parameters):
        self.dim = dim
        default_parameters = self.get_default_parameters()
        for key, val in parameters.iteritems():
            if key in default_parameters:
                default_parameters[key] = val
            else:
                print "Warning: parameter '", key, "' unknown. Ignored."
        self.parameters = default_parameters
        self.set_fixed_image(None)
        self.set_moving_image(None)
        self.levels_above = 0
        self.levels_below = 0
        self.symmetric = False

    def set_levels_below(self, levels):
        r'''
        Informs this metric the number of pyramid levels below the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly
        '''
        self.levels_below = levels

    def set_levels_above(self, levels):
        r'''
        Informs this metric the number of pyramid levels above the current one.
        The metric may change its behavior (e.g. number of inner iterations)
        accordingly
        '''
        self.levels_above = levels

    def set_fixed_image(self, fixed_image):
        '''
        Sets the fixed image. Verifies that the image dimension is consistent
        with this metric.
        '''
        new_dim = len(fixed_image.shape) if fixed_image != None else self.dim
        if new_dim!=self.dim:
            raise AttributeError('Unexpected fixed_image dimension: '+str(new_dim))
        self.fixed_image = fixed_image

    @abc.abstractmethod
    def get_metric_name(self):
        '''
        Must return the name of the metric that specializes this generic metric
        '''
        pass

    @abc.abstractmethod
    def use_fixed_image_dynamics(self,
                              original_fixed_image,
                              transformation):
        '''
        This methods provides the metric a chance to compute any useful
        information from knowing how the current fixed image was generated
        (as the transformation of an original fixed image). This method is
        called by the optimizer just after it sets the fixed image.
        Transformation will be an instance of TransformationModel or None if
        the originalMovingImage equals self.moving_image.
        '''

    @abc.abstractmethod
    def use_original_fixed_image(self, original_fixed_image):
        '''
        This methods provides the metric a chance to compute any useful
        information from the original moving image (to be used along with the
        sequence of movingImages during optimization, for example the binary
        mask delimiting the object of interest can be computed from the original
        image only and then warp this binary mask instead of thresholding
        at each iteration, which might cause artifacts due to interpolation)
        '''

    def set_moving_image(self, moving_image):
        '''
        Sets the moving image. Verifies that the image dimension is consistent
        with this metric.
        '''
        new_dim = len(moving_image.shape) if moving_image != None else self.dim
        if new_dim!=self.dim:
            raise AttributeError('Unexpected fixed_image dimension: '+str(new_dim))
        self.moving_image = moving_image

    @abc.abstractmethod
    def use_original_moving_image(self, original_moving_image):
        '''
        This methods provides the metric a chance to compute any useful
        information from the original moving image (to be used along with the
        sequence of movingImages during optimization, for example the binary
        mask delimiting the object of interest can be computed from the original
        image only and then warp this binary mask instead of thresholding
        at each iteration, which might cause artifacts due to interpolation)
        '''

    @abc.abstractmethod
    def use_moving_image_dynamics(self,
                               original_moving_image,
                               transformation):
        '''
        This methods provides the metric a chance to compute any useful
        information from knowing how the current fixed image was generated
        (as the transformation of an original fixed image). This method is
        called by the optimizer just after it sets the fixed image.
        Transformation will be an instance of TransformationModel or None if
        the originalMovingImage equals self.moving_image.
        '''

    @abc.abstractmethod
    def initialize_iteration(self):
        '''
        This method will be called before any computeUpdate or computeInverse
        call, this gives the chance to the Metric to precompute any useful
        information for speeding up the update computations. This initialization
        was needed in ANTS because the updates are called once per voxel. In
        Python this is unpractical, though.
        '''

    @abc.abstractmethod
    def free_iteration(self):
        '''
        This method is called by the RegistrationOptimizer after the required
        iterations have been computed (forward and/or backward) so that the
        SimilarityMetric can safely delete any data it computed as part of the
        initialization
        '''

    @abc.abstractmethod
    def compute_forward(self):
        '''
        Must return the forward update field for a gradient-based optimization
        algorithm
        '''

    @abc.abstractmethod
    def compute_backward(self):
        '''
        Must return the inverse update field for a gradient-based optimization
        algorithm
        '''

    @abc.abstractmethod
    def get_energy(self):
        '''
        Must return the numeric value of the similarity between the given fixed
        and moving images
        '''

    @abc.abstractmethod
    def get_default_parameters(self):
        r'''
        Derived classes must return a dictionary containing its parameter names
        and default values
        '''

    @abc.abstractmethod
    def report_status(self):
        '''
        This function is called mostly for debugging purposes. The metric
        can for example show the overlaid images or print some statistics
        '''
