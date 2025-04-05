import importlib
import os
import sys
import warnings

from numpy.testing import assert_, assert_equal
import pytest

from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
original_backend = os.environ.get("DIPY_NN_BACKEND")
model_mod = None


def setup_module(module):
    """Set up environment variable for all tests in this module."""
    global model_mod
    os.environ["DIPY_NN_BACKEND"] = "tensorflow"
    with warnings.catch_warnings():
        msg = ".*uses TensorFlow.*install PyTorch.*"
        warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
        dipy_nn = importlib.reload(sys.modules["dipy.nn"])
    model_mod = dipy_nn.model


def teardown_module(module):
    """Restore the original environment variable after all tests in this module."""
    global model_mod
    if original_backend is not None:
        os.environ["DIPY_NN_BACKEND"] = original_backend
    else:
        del os.environ["DIPY_NN_BACKEND"]
    model_mod = None


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_mnist_sequential():
    mnist = tf.keras.datasets.mnist

    epochs = 5

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(28, 28)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    hist = model.fit(x_train, y_train, epochs=epochs)
    model.evaluate(x_test, y_test, verbose=2)
    accuracy = hist.history["accuracy"][0]
    assert_(accuracy > 0.9)


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_mnist_slp():
    mnist = tf.keras.datasets.mnist
    epochs = 5

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    slp = model_mod.SingleLayerPerceptron(input_shape=(28, 28))
    hist = slp.fit(x_train, y_train, epochs=epochs)
    slp.evaluate(x_test, y_test, verbose=2)
    x_test_prob = slp.predict(x_test)

    accuracy = hist.history["accuracy"][0]
    assert_(slp.accuracy > 0.9)
    assert_(slp.loss < 0.4)
    assert_equal(slp.accuracy, accuracy)
    assert_equal(x_test_prob.shape, (10000, 10))


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_mnist_mlp():
    mnist = tf.keras.datasets.mnist
    epochs = 5

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    mlp = model_mod.MultipleLayerPercepton(input_shape=(28, 28), num_hidden=[128, 128])
    hist = mlp.fit(x_train, y_train, epochs=epochs)
    mlp.evaluate(x_test, y_test, verbose=2)
    x_test_prob = mlp.predict(x_test)

    accuracy = hist.history["accuracy"][0]
    assert_(mlp.accuracy > 0.8)
    assert_(mlp.loss < 0.4)
    assert_equal(mlp.accuracy, accuracy)
    assert_equal(x_test_prob.shape, (10000, 10))
