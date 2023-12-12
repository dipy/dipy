import pytest
from numpy.testing import assert_equal, assert_

from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')

if have_tf:
    from dipy.nn.model import SingleLayerPerceptron, MultipleLayerPercepton


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_mnist_sequential():

    mnist = tf.keras.datasets.mnist

    epochs = 5

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    hist = model.fit(x_train, y_train, epochs=epochs)
    model.evaluate(x_test, y_test, verbose=2)
    accuracy = hist.history['accuracy'][0]
    assert_(accuracy > 0.9)


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_mnist_slp():

    mnist = tf.keras.datasets.mnist
    epochs = 5

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    slp = SingleLayerPerceptron(input_shape=(28, 28))
    hist = slp.fit(x_train, y_train, epochs=epochs)
    slp.evaluate(x_test, y_test, verbose=2)
    x_test_prob = slp.predict(x_test)

    accuracy = hist.history['accuracy'][0]
    assert_(slp.accuracy > 0.9)
    assert_(slp.loss < 0.4)
    assert_equal(slp.accuracy, accuracy)
    assert_equal(x_test_prob.shape, (10000, 10))


@pytest.mark.skipif(not have_tf, reason='Requires TensorFlow')
def test_default_mnist_mlp():
    mnist = tf.keras.datasets.mnist
    epochs = 5

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    mlp = MultipleLayerPercepton(input_shape=(28, 28), num_hidden=[128, 128])
    hist = mlp.fit(x_train, y_train, epochs=epochs)
    mlp.evaluate(x_test, y_test, verbose=2)
    x_test_prob = mlp.predict(x_test)

    accuracy = hist.history['accuracy'][0]
    assert_(mlp.accuracy > 0.8)
    assert_(mlp.loss < 0.4)
    assert_equal(mlp.accuracy, accuracy)
    assert_equal(x_test_prob.shape, (10000, 10))
