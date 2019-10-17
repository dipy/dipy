from distutils.version import LooseVersion
from numpy.testing import (assert_array_almost_equal, assert_almost_equal,
                           assert_, assert_raises)

from dipy.utils.optpkg import optional_package

tf , have_tf, _ = optional_package('tensorflow')

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')

    from tensorflow.keras import Model, layers


class SingleLayerPerceptron(object):

    def __init__(self, input_shape=(28, 28),
                 num_hidden=128, act_hidden='relu',
                 dropout=0.2,
                 num_out=10, act_out='softmax'):
        """ Single Layer Perceptron with Dropout

        Parameters
        ----------
        input_shape : tuple
        num_hidden : int
        act_hidden : string
        dropout : float
        num_out : 10
        act_out : string
        """
        self.accuracy = None
        self.loss = None

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(num_hidden, activation=act_hidden),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_out, activation=act_out)
            ])

        model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

        self.model = model

    def fit(self, x_train, y_train, epochs=5):
        hist = self.model.fit(x_train, y_train, epochs=epochs)
        self.accuracy = hist.history['accuracy'][0]
        self.loss = hist.history['loss'][0]
        return hist

    def evaluate(self, x_test, y_test, verbose=2):
        return self.model.evaluate(x_test,  y_test, verbose=verbose)

    def predict(self, x_test):
        return self.model.predict(x_test)


