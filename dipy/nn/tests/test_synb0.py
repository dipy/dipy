import pytest

from dipy.data import get_fnames
from dipy.utils.optpkg import optional_package
import numpy as np
from numpy.testing import assert_almost_equal

tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')

if have_tf:
    from dipy.nn.synb0 import Synb0


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_weights():
    file_names = get_fnames('synb0_test_data')
    input_arr1 = np.load(file_names[0])['b0'][0]
    input_arr2 = np.load(file_names[0])['T1'][0]
    target_arr = np.load(file_names[1])['arr_0'][0]

    synb0_model = Synb0()
    synb0_model.fetch_default_weights(0)
    results_arr = synb0_model.predict(input_arr1, input_arr2, average=False)
    assert_almost_equal(results_arr, target_arr, decimal=1)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_weights_batch():
    file_names = get_fnames('synb0_test_data')
    input_arr1 = np.load(file_names[0])['b0']
    input_arr2 = np.load(file_names[0])['T1']
    target_arr = np.load(file_names[1])['arr_0']
    synb0_model = Synb0()
    synb0_model.fetch_default_weights(0)
    results_arr = synb0_model.predict(input_arr1, input_arr2,
                                      batch_size=2, average=False)
    assert_almost_equal(results_arr, target_arr, decimal=1)
