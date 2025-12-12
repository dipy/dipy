import importlib
import sys
import warnings

import numpy as np
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


def assert_percent_almost_equal(a, b, decimal=7, percent=0.99):
    a = np.asarray(a)
    b = np.asarray(b)
    tol = 1.5 * 10 ** (-decimal)

    diff = np.abs(a - b)
    ok = diff <= tol

    fraction = np.mean(ok)

    if fraction < percent:
        raise AssertionError(
            f"Only {fraction*100:.2f}% of elements match within {decimal} decimals; "
            f"required {percent*100:.2f}%"
        )


@pytest.mark.skipif(not have_nn, reason="Requires TensorFlow or Torch")
def test_default_weights(monkeypatch):
    file_names = get_fnames(name="deepn4_test_data")
    data = np.load(file_names)
    input_arr = data["input"]
    input_affine_arr = np.eye(4)
    target_arr = data["output"]

    for backend in BACKENDS:
        monkeypatch.setenv("DIPY_NN_BACKEND", backend)
        with warnings.catch_warnings():
            msg = ".*uses TensorFlow.*install PyTorch.*"
            warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
            dipy_nn = importlib.reload(sys.modules["dipy.nn"])
            deepn4_mod = dipy_nn.deepn4

        deepn4_model = deepn4_mod.DeepN4()
        deepn4_model.fetch_default_weights()
        results_arr = deepn4_model.predict(input_arr, input_affine_arr)
        assert_percent_almost_equal(results_arr, target_arr, decimal=3, percent=0.95)
