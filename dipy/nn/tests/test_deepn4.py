import importlib
import os
import sys
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from dipy.data import get_fnames
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.0.0")
original_backend = os.environ.get("DIPY_NN_BACKEND")
deepn4_mod = None


def setup_module(module):
    """Set up environment variable for all tests in this module."""
    global deepn4_mod
    os.environ["DIPY_NN_BACKEND"] = "tensorflow"
    with warnings.catch_warnings():
        msg = ".*uses TensorFlow.*install PyTorch.*"
        warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
        dipy_nn = importlib.reload(sys.modules["dipy.nn"])
    deepn4_mod = dipy_nn.deepn4


def teardown_module(module):
    """Restore the original environment variable after all tests in this module."""
    global deepn4_mod
    if original_backend is not None:
        os.environ["DIPY_NN_BACKEND"] = original_backend
    else:
        del os.environ["DIPY_NN_BACKEND"]
    deepn4_mod = None


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_weights():
    file_names = get_fnames(name="deepn4_test_data")
    input_arr = np.load(file_names[0])["img"]
    input_affine_arr = np.load(file_names[0])["affine"]
    target_arr = np.load(file_names[1])["corr"]

    deepn4_model = deepn4_mod.DeepN4()
    deepn4_model.fetch_default_weights()
    results_arr = deepn4_model.predict(input_arr, input_affine_arr)
    assert_almost_equal(results_arr, target_arr, decimal=1)


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_weights_batch():
    file_names = get_fnames(name="deepn4_test_data")
    input_arr = np.load(file_names[0])["img"]
    input_affine_arr = np.load(file_names[0])["affine"]
    target_arr = np.load(file_names[1])["corr"]

    deepn4_model = deepn4_mod.DeepN4()
    deepn4_model.fetch_default_weights()
    results_arr = deepn4_model.predict(input_arr, input_affine_arr)
    assert_almost_equal(results_arr, target_arr, decimal=1)
