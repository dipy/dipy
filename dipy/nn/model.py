from distutils.version import LooseVersion

from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package('tensorflow')

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')


class SingleLayerPerceptron(object):

    def __init__(self, input_shape=(28, 28),
                 num_hidden=128, act_hidden='relu',
                 dropout=0.2,
                 num_out=10, act_out='softmax',
                 optimizer='adam',
                 loss='sparse_categorical_crossentropy'):
        """ Single Layer Perceptron with Dropout

        Parameters
        ----------
        input_shape : tuple
            Shape of data to be trained
        num_hidden : int
            Number of nodes in hidden layer
        act_hidden : string
            Activation function used in hidden layer
        dropout : float
            Dropout ratio
        num_out : 10
            Number of nodes in output layer
        act_out : string
            Activation function used in output layer
        optimizer :  string
            Select optimizer. Default adam.
        loss : string
            Select loss function for measuring accuracy.
            Default sparse_categorical_crossentropy.
        """
        self.accuracy = None
        self.loss = None

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(num_hidden, activation=act_hidden),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_out, activation=act_out)
            ])

        model.compile(optimizer=optimizer,
                      loss=loss,
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
