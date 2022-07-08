import pytest
from packaging.version import Version

from dipy.core.gradients import gradient_table
from dipy.data import get_fnames
from dipy.io.image import load_nifti
from dipy.io.gradients import read_bvals_bvecs
from dipy.nn.synb0 import Synb0
from dipy.utils.optpkg import optional_package
import numpy as np
from numpy.testing import assert_almost_equal, assert_raises, assert_equal

tf, have_tf, _ = optional_package('tensorflow')
tfa, have_tfa, _ = optional_package('tensorflow_addons')

if have_tf and have_tfa:
    if Version(tf.__version__) < Version('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
@pytest.mark.skipif(not have_tfa, reason='Requires TensorFlow_addons')
def test_default_weights():
    input_arr = np.zeros((1, 80, 96, 80, 2)).astype(float)

    target_arr = np.load(get_fnames('synb0_test_output'))

    synb0_model = Synb0()
    synb0_model.fetch_default_weights()
    input_arr, target_arr = synb0_model.get_test_data()
    results_arr = synb0_model.__predict(input_arr)
    assert_almost_equal(results_arr, target_arr)