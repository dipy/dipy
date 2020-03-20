from distutils.version import LooseVersion

from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package('tensorflow')

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')


class MultipleLayerPercepton(object):

    def __init__(self,input_shape=(28,28),
                        num_hidden=[128],
                        act_hidden='relu',
                        dropout = 0.2
                        num_out=10,
                        act_out='softmax',
                        loss='sparse_categorical_crossentropy',
                        optimizer='adam'):


    """ Multiple Layer Perceptron with Dropout

            Parameters
            ----------
            input_shape : tuple
                Shape of data to be trained
            num_hidden : list
                List of number of nodes in hidden layers
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


        self.input_shape = input_shape
        self.num_hidden = num_hidden
        self.act_hidden = act_hidden
        self.dropout = dropout
        self.num_out = num_out
        self.act_out = act_out
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = None



        #model building

        inp = tf.keras.layers.Input(input_shape=self.input_shape)
        x = tf.keras.layers.Flatten()(inp)

        for i in range(len(self.num_hidden)):
            x = tf.keras.layers.Dense(self.num_hidden[i])(x)

        x = tf.keras.layers.Dropout(self.dropout)(x)
        out = tf.keras.layers.Dense(self.num_out, activation=self.act_out)(x)

        self.model = tf.keras.layers.Model(inputs=inp, outputs=out)



        #compiling the model
        self.model.compile(optimizer=self.optimizer,
                              loss=self.loss,
                              metrics=['accuracy'])


    def summary(self):
        return self.model.summary()


    def fit(self, x_train, y_train, epochs=5):
        hist = self.model.fit(x_train, y_train, epochs=epochs)
        self.accuracy = hist.history['accuracy'][0]
        self.loss = hist.history['loss'][0]
        return hist

    def evaluate(self, x_test, y_test, verbose=2):
        return self.model.evaluate(x_test,  y_test, verbose=verbose)

    def predict(self, x_test):
        return self.model.predict(x_test)
