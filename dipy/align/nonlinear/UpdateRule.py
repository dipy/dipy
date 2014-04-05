'''
This file contains the abstract UpdateRule which is in charge of updating 
a displacement field with the new computed step. There are three main
different update rules: additive, compositive and compositive with previous
projection to the diffeomorphism space via displacement field exponentiation
'''
import abc
import numpy as np
import VectorFieldUtils as vfu
class UpdateRule(object):
    r'''
    The abstract class defining the contract to be fulfilled by especialized 
    update rules.
    '''
    __metaclass__   =   abc.ABCMeta
    def __init__(self):
        pass

    @abc.abstractmethod
    def update(self, new_displacement, current_displacement):
        '''
        Must return the updated displacement field and the mean norm of the 
        difference between the displacements before and after the update
        '''

class Addition(UpdateRule):
    r'''
    Additive rule (simply adds the current displacement field with the new
    step)
    '''
    def __init__(self):
        pass

    @staticmethod
    def update(new_displacement, current_displacement):
        mean_norm = np.sqrt(np.sum(new_displacement**2, -1)).mean()
        updated = current_displacement+new_displacement
        return updated, mean_norm

class Composition(UpdateRule):
    r'''
    Compositive update rule, composes the two displacement fields using
    trilinear interpolation
    '''
    def __init__(self):
        pass

    @staticmethod
    def update(new_displacement, current_displacement):
        dim = len(new_displacement.shape)-1
        mse = np.sqrt(np.sum((current_displacement**2), -1)).mean()
        if dim == 2:
            updated, stats = vfu.compose_vector_fields(new_displacement, 
                                                       current_displacement)
        else:
            updated, stats = vfu.compose_vector_fields_3d(new_displacement, 
                                                          current_displacement)
        return np.array(updated), np.array(mse)

#class ProjectedComposition(UpdateRule):
#    r'''
#    Compositive update rule, composes the two displacement fields using
#    trilinear interpolation. Before composition, it applies the displacement
#    field exponentiation to the new step.
#    '''
#    def __init__(self):
#        pass
#
#    @staticmethod
#    def update(new_displacement, current_displacement):
#        expd, invexpd = vfu.vector_field_exponential(new_displacement, True)
#        updated, stats = vfu.compose_vector_fields(expd, current_displacement)
#        return updated, stats[0]
