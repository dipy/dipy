import importlib
import os
import sys
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.testing import assert_percent_almost_equal
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
original_backend = os.environ.get("DIPY_NN_BACKEND")
synthseg = None


def setup_module(module):
    """Set up environment variable for all tests in this module."""
    global synthseg
    os.environ["DIPY_NN_BACKEND"] = "torch"
    with warnings.catch_warnings():
        msg = ".*uses TensorFlow.*install PyTorch.*"
        warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
        dipy_nn = importlib.reload(sys.modules["dipy.nn"])
    synthseg = dipy_nn.synthseg


def teardown_module(module):
    """Restore the original environment variable after all tests in this module."""
    global synthseg
    if original_backend is not None:
        os.environ["DIPY_NN_BACKEND"] = original_backend
    else:
        del os.environ["DIPY_NN_BACKEND"]
    synthseg = None


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
def test_default_weights():
    file_path = get_fnames(name="synthseg_test_data")
    input_arr = np.load(file_path)["input"][0]
    output_arr = np.load(file_path)["output"][0]

    synthseg_model = synthseg.SynthSeg()
    results_arr, _, _ = synthseg_model.predict(input_arr, np.eye(4), return_prob=True)
    results_arr = results_arr[..., 5]
    assert_percent_almost_equal(results_arr, output_arr, decimal=4, percent=0.99)


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
def test_default_weights_batch():
    file_path = get_fnames(name="synthseg_test_data")
    data = np.load(file_path)
    # Use a single image passed as a list to exercise the batch code path
    # while keeping peak memory equivalent to the single-image test.
    input_arr = [data["input"][0]]
    output_arr = data["output"][0]

    synthseg_model = synthseg.SynthSeg()
    fake_affine = np.array([np.eye(4)])
    results_arr, _, _ = synthseg_model.predict(
        input_arr, fake_affine, batch_size=1, return_prob=True
    )
    results_arr = results_arr[0, ..., 5]
    assert_percent_almost_equal(results_arr, output_arr, decimal=4, percent=0.99)


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
def test_default_weights_labels():
    file_path = get_fnames(name="synthseg_test_data")
    input_arr = np.load(file_path)["input"][0]

    synthseg_model = synthseg.SynthSeg()
    labels, label_dict, _ = synthseg_model.predict(input_arr, np.eye(4))

    npt.assert_equal(labels.shape, input_arr.shape)
    npt.assert_equal(labels.dtype, np.int32)
    assert isinstance(label_dict, dict)


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
def test_default_weights_return_masks():
    file_path = get_fnames(name="synthseg_test_data")
    input_arr = np.load(file_path)["input"][0]

    synthseg_model = synthseg.SynthSeg()
    with pytest.warns(UserWarning, match="The returned brain masks"):
        labels, label_dict, masks = synthseg_model.predict(
            input_arr, np.eye(4), return_masks=True
        )

    npt.assert_equal(labels.shape, input_arr.shape)
    npt.assert_equal(masks.shape, input_arr.shape)
    npt.assert_equal(labels.dtype, np.int32)
    npt.assert_equal(masks.dtype, np.int32)
    assert np.isin(masks, [0, 1]).all()
    assert isinstance(label_dict, dict)


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
def test_T1_error():
    T1 = np.ones((3, 32, 32, 32))
    synthseg_model = synthseg.SynthSeg()
    fake_affine = np.array([np.eye(4), np.eye(4), np.eye(4)])
    npt.assert_raises(ValueError, synthseg_model.predict, T1, fake_affine)
