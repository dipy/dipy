import pytest

import numpy as np
import numpy.testing as npt

from dipy.data import get_fnames
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')

if have_tf:
    from dipy.nn.evac import EVACPlus


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_weights():
    file_path = get_fnames('evac_test_data')
    input_arr = np.load(file_path)['input'][0]
    output_arr = np.load(file_path)['output'][0]

    evac_model = EVACPlus()
    results_arr = evac_model.predict(input_arr, np.eye(4), return_prob=True)
    npt.assert_almost_equal(results_arr, output_arr, decimal=4)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_weights_batch():
    file_path = get_fnames('evac_test_data')
    input_arr = np.load(file_path)['input']
    output_arr = np.load(file_path)['output']
    input_arr = [T1 for T1 in input_arr]

    evac_model = EVACPlus()
    fake_affine = np.array([np.eye(4), np.eye(4)])
    fake_voxsize = np.ones((2, 3))
    results_arr = evac_model.predict(input_arr, fake_affine, fake_voxsize, batch_size=2, return_prob=True)
    npt.assert_almost_equal(results_arr, output_arr, decimal=4)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_T1_error():
    T1 = np.ones((3, 32, 32, 32))
    evac_model = EVACPlus()
    fake_affine = np.array([np.eye(4), np.eye(4), np.eye(4)])
    npt.assert_raises(ValueError, evac_model.predict, T1, fake_affine)