"""Testing weights methods"""

import random
import warnings

import numpy as np
from numpy.testing import (
    assert_almost_equal,
    assert_array_almost_equal,
    assert_array_equal,
    assert_raises,
)

from dipy.reconst.weights_method import (
    simple_cutoff,
    two_eyes_cutoff,
    weights_method_wls_m_est,
    weights_method_nlls_m_est,
)


def test_cutoffs():
    # does nothing, just trying to set up my tests properly    
    pass


def test_weights():
    # does nothing, just trying to set up my tests properly    
    pass
