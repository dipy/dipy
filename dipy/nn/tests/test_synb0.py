import importlib
import os
import sys
import warnings

import numpy as np
from numpy.testing import assert_almost_equal
import pytest

from dipy.data import get_fnames
from dipy.nn.torch.synb0 import Synb0 as TorchSynb0
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
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


def _reload_synb0_backend(backend):
    os.environ["DIPY_NN_BACKEND"] = backend
    with warnings.catch_warnings():
        msg = ".*uses TensorFlow.*install PyTorch.*"
        warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
        dipy_nn = importlib.reload(sys.modules["dipy.nn"])
    return dipy_nn.synb0


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


@pytest.mark.skipif(
    not (have_tf and have_torch), reason="Requires TensorFlow and PyTorch"
)
def test_torch_tf_default_weight_parity():
    file_names = get_fnames(name="synb0_test_data")
    input_arr1 = np.load(file_names[0])["b0"][0]
    input_arr2 = np.load(file_names[0])["T1"][0]

    try:
        tf_synb0_mod = _reload_synb0_backend("tensorflow")
        tf_model = tf_synb0_mod.Synb0()
        tf_model.fetch_default_weights(0)
        tf_results = tf_model.predict(input_arr1, input_arr2, average=False)

        torch_synb0_mod = _reload_synb0_backend("torch")
        torch_model = torch_synb0_mod.Synb0()
        torch_model.fetch_default_weights(0)
        torch_results = torch_model.predict(input_arr1, input_arr2, average=False)
    finally:
        _reload_synb0_backend("tensorflow")

    assert_almost_equal(torch_results, tf_results, decimal=1)


@pytest.mark.skipif(not have_torch, reason="Requires PyTorch")
def test_torch_prediction_fold_average_and_partial_batch(monkeypatch):
    """The five folds and final partial batch contribute to every output volume."""
    model = object.__new__(TorchSynb0)
    calls = []

    def fetch_weights(index):
        model.fold = index

    def predict_batch(images):
        calls.append((model.fold, len(images)))
        return np.full(
            (len(images), 1, 80, 96, 80),
            -0.5 + 0.1 * model.fold,
            dtype=np.float32,
        )

    monkeypatch.setattr(model, "fetch_default_weights", fetch_weights)
    monkeypatch.setattr(model, "_predict", predict_batch)
    b0 = np.broadcast_to(
        np.array([10, 20, 30], dtype=np.float32)[:, None, None, None],
        (3, 77, 91, 77),
    )
    t1 = np.zeros_like(b0)

    result = model.predict(b0, t1, batch_size=2, average=True)

    assert result.shape == b0.shape
    assert calls == [(fold, size) for fold in range(5) for size in (2, 1)]
    assert_almost_equal(result[:, 0, 0, 0], [3.5, 7.0, 10.5], decimal=5)

    calls.clear()
    result = model.predict(b0, t1, batch_size=2, average=False)
    assert calls == [(4, 2), (4, 1)]
    assert_almost_equal(result[:, 0, 0, 0], [4.5, 9.0, 13.5], decimal=5)
