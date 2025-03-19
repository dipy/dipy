import importlib
import sys
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
have_nn = have_tf or have_torch
BACKENDS = [
    backend
    for backend, available in [("tensorflow", have_tf), ("torch", have_torch)]
    if available
]


@pytest.mark.skipif(not have_nn, reason="Requires TensorFlow or Torch")
def test_default_weights(monkeypatch):
    file_path = get_fnames(name="evac_test_data")
    input_arr = np.load(file_path)["input"][0]
    output_arr = np.load(file_path)["output"][0]

    for backend in BACKENDS:
        monkeypatch.setenv("DIPY_NN_BACKEND", backend)
        with warnings.catch_warnings():
            msg = ".*uses TensorFlow.*install PyTorch.*"
            warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
            dipy_nn = importlib.reload(sys.modules["dipy.nn"])
            evac = dipy_nn.evac

        evac_model = evac.EVACPlus()
        results_arr = evac_model.predict(input_arr, np.eye(4), return_prob=True)
        npt.assert_almost_equal(results_arr, output_arr, decimal=2)


@pytest.mark.skipif(not have_nn, reason="Requires TensorFlow or Torch")
def test_default_weights_batch(monkeypatch):
    file_path = get_fnames(name="evac_test_data")
    input_arr = np.load(file_path)["input"]
    output_arr = np.load(file_path)["output"]
    input_arr = list(input_arr)

    for backend in BACKENDS:
        print(backend)
        monkeypatch.setenv("DIPY_NN_BACKEND", backend)
        with warnings.catch_warnings():
            msg = ".*uses TensorFlow.*install PyTorch.*"
            warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
            dipy_nn = importlib.reload(sys.modules["dipy.nn"])
            evac = dipy_nn.evac

        evac_model = evac.EVACPlus()
        fake_affine = np.array([np.eye(4), np.eye(4)])
        fake_voxsize = np.ones((2, 3))
        results_arr = evac_model.predict(
            input_arr, fake_affine, voxsize=fake_voxsize, batch_size=2, return_prob=True
        )
        npt.assert_almost_equal(results_arr, output_arr, decimal=2)


@pytest.mark.skipif(not have_nn, reason="Requires TensorFlow or Torch")
def test_T1_error(monkeypatch):
    for backend in BACKENDS:
        print(backend)
        monkeypatch.setenv("DIPY_NN_BACKEND", backend)
        with warnings.catch_warnings():
            msg = ".*uses TensorFlow.*install PyTorch.*"
            warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
            dipy_nn = importlib.reload(sys.modules["dipy.nn"])
            evac = dipy_nn.evac

        T1 = np.ones((3, 32, 32, 32))
        evac_model = evac.EVACPlus()
        fake_affine = np.array([np.eye(4), np.eye(4), np.eye(4)])
        npt.assert_raises(ValueError, evac_model.predict, T1, fake_affine)
