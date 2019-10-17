import pytest
from distutils.version import LooseVersion
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_, assert_raises)

from dipy.utils.optpkg import optional_package

tf , have_tf, _ = optional_package('tensorflow')

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')


@pytest.mark.skipif(not have_tf)
def test_default_mnist_sequential():

    mnist = tf.keras.datasets.mnist

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

    model.fit(x_train, y_train, epochs=5)

    model.evaluate(x_test,  y_test, verbose=2)


if __name__ == "__main__":
    test_default_mnist_sequential()
