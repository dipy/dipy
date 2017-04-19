from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.testing as npt
import math

from dipy.data import get_sphere, get_data
from dipy.reconst.dti import quantize_evecs


def _normalize(evec):
    """ Routine for normalizing a vector in place """
    n = math.sqrt(sum([x*x for x in evec]))
    i = 0
    while i < len(evec):
        evec[i] = evec[i] / n
        i += 1

def random_evecs(shape):
    """ Random evec with a given shape """
    np.random.seed(7)
    evecs = np.random.rand(np.prod(shape)*3,3)
    for evec in evecs:
        _normalize(evec)
    evecs = np.reshape(evecs, shape+(3,3))
    return evecs

def test_quantize_evecs():
    test_shape = (50,60,70)
    
    sphere = get_sphere('symmetric724')
    
    # Test for zero case
    zerovecs = np.zeros(test_shape+(3,3))
    peak_indices = quantize_evecs(zerovecs, sphere.vertices)
    npt.assert_equal(np.zeros(test_shape),peak_indices)
    
    # Test for I case
    eyevecs = np.tile(np.identity(3), (np.prod(test_shape),1))
    eyevecs = np.reshape(eyevecs, test_shape+(3,3))
    peak_indices = quantize_evecs(eyevecs, sphere.vertices)
    npt.assert_equal(360*np.ones(test_shape),peak_indices)
    
    # Test an artificial evecs dataset
    evecs = random_evecs(test_shape)
    peak_indices = quantize_evecs(evecs, sphere.vertices)
    npt.assert_equal(159,peak_indices[0,0,0])
    
    # Test parallel processing
    peak_indices = quantize_evecs(zerovecs, sphere.vertices, nbr_processes=-1)
    npt.assert_equal(np.zeros(test_shape),peak_indices)
    
    peak_indices = quantize_evecs(eyevecs, sphere.vertices, nbr_processes=-1)
    npt.assert_equal(360*np.ones(test_shape),peak_indices)
    
    peak_indices = quantize_evecs(evecs, sphere.vertices, nbr_processes=-1)
    npt.assert_equal(159,peak_indices[0,0,0])  
    
    # Test v
    peak_indices = quantize_evecs(eyevecs, sphere.vertices, v=1)
    npt.assert_equal(np.zeros(test_shape),peak_indices)
    
    peak_indices = quantize_evecs(eyevecs, sphere.vertices, v=2)
    npt.assert_equal(358*np.ones(test_shape),peak_indices)

 
if __name__ == '__main__':
    test_quantize_evecs()
