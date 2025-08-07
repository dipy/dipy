import importlib
import os
import sys
import warnings

import pytest

from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
sklearn, have_sklearn, _ = optional_package("sklearn.model_selection")
original_backend = os.environ.get("DIPY_NN_BACKEND")
cnnden_mod = None


def setup_module(module):
    """Set up environment variable for all tests in this module."""
    global cnnden_mod
    os.environ["DIPY_NN_BACKEND"] = "tensorflow"
    with warnings.catch_warnings():
        msg = ".*uses TensorFlow.*install PyTorch.*"
        warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
        dipy_nn = importlib.reload(sys.modules["dipy.nn"])
        cnnden_mod = dipy_nn.cnn_1d_denoising


def teardown_module(module):
    """Restore the original environment variable after all tests in this module."""
    global cnnden_mod
    if original_backend is not None:
        os.environ["DIPY_NN_BACKEND"] = original_backend
    else:
        del os.environ["DIPY_NN_BACKEND"]
    cnnden_mod = None


@pytest.mark.skipif(
    not have_tf or not have_sklearn, reason="Requires TensorFlow and scikit-learn"
)
@set_random_number_generator()
def test_default_Cnn1DDenoiser_sequential(rng=None):
    # Create dummy data
    normal_img = rng.random((10, 10, 10, 30))
    nos_img = normal_img + rng.normal(loc=0.0, scale=0.1, size=normal_img.shape)
    x = rng.random((10, 10, 10, 30))
    # Test 1D denoiser
    model = cnnden_mod.Cnn1DDenoiser(30)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    epochs = 1
    hist = model.fit(nos_img, normal_img, epochs=epochs)
    data = model.predict(x)
    model.evaluate(nos_img, normal_img, verbose=2)
    _ = hist.history["accuracy"][0]
    assert data.shape == x.shape


@pytest.mark.skipif(
    not have_tf or not have_sklearn, reason="Requires TensorFlow and scikit-learn"
)
@set_random_number_generator()
def test_default_Cnn1DDenoiser_flow(pytestconfig, rng):
    # Create dummy data
    normal_img = rng.random((10, 10, 10, 30))
    nos_img = normal_img + rng.normal(loc=0.0, scale=0.1, size=normal_img.shape)
    x = rng.random((10, 10, 10, 30))
    # Test 1D denoiser with flow API
    model = cnnden_mod.Cnn1DDenoiser(30)
    if pytestconfig.getoption("verbose") > 0:
        model.summary()
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])
    epochs = 1
    hist = model.fit(nos_img, normal_img, epochs=epochs)
    _ = model.predict(x)
    model.evaluate(nos_img, normal_img, verbose=2)
    accuracy = hist.history["accuracy"][0]
    assert accuracy > 0
