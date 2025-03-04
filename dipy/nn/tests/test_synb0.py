import importlib
import os
import sys
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from dipy.data import get_fnames
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
original_backend = os.environ.get("DIPY_NN_BACKEND")
synb0_mod = None


def setup_module(module):
    """Set up environment variable for all tests in this module."""
    global synb0_mod
    os.environ["DIPY_NN_BACKEND"] = "tensorflow"
    with warnings.catch_warnings():
        msg = ".*uses TensorFlow.*install PyTorch.*"
        warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
        dipy_nn = importlib.reload(sys.modules["dipy.nn"])
    synb0_mod = dipy_nn.synb0


def teardown_module(module):
    """Restore the original environment variable after all tests in this module."""
    global synb0_mod
    if original_backend is not None:
        os.environ["DIPY_NN_BACKEND"] = original_backend
    else:
        del os.environ["DIPY_NN_BACKEND"]
    synb0_mod = None


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_weights():
    file_names = get_fnames(name="synb0_test_data")
    input_arr1 = np.load(file_names[0])["b0"][0]
    input_arr2 = np.load(file_names[0])["T1"][0]
    target_arr = np.load(file_names[1])["arr_0"][0]

    synb0_model = synb0_mod.Synb0()
    synb0_model.fetch_default_weights(0)
    results_arr = synb0_model.predict(input_arr1, input_arr2, average=False)
    assert_almost_equal(results_arr, target_arr, decimal=1)


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_weights_batch():
    file_names = get_fnames(name="synb0_test_data")
    input_arr1 = np.load(file_names[0])["b0"]
    input_arr2 = np.load(file_names[0])["T1"]
    target_arr = np.load(file_names[1])["arr_0"]
    synb0_model = synb0_mod.Synb0()
    synb0_model.fetch_default_weights(0)
    results_arr = synb0_model.predict(
        input_arr1, input_arr2, batch_size=2, average=False
    )
    assert_almost_equal(results_arr, target_arr, decimal=1)
