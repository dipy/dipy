"""
Test the stats module's utility functions

"""
from __future__ import division, print_function, absolute_import

import numpy as np
import numpy.testing as npt
import dipy.stats.utils as stu

def test_coeff_of_determination():
    """

    Yup

    """

    model = np.random.randn(10,10,10,150)
    data = np.copy(model)
    # If the model predicts the data perfectly, the COD is all 100s:
    cod = stu.coeff_of_determination(data, model)
    npt.assert_array_equal(100 * np.ones(data.shape[:3]), cod)
