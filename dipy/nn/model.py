from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')


class SingleLayerPerceptron:

    def __init__(self, input_shape=(28, 28),
                 num_hidden=128, act_hidden='relu',
                 dropout=0.2,
                 num_out=10, act_out='softmax',
                 optimizer='adam',
                 loss='sparse_categorical_crossentropy'):
        """Single Layer Perceptron with Dropout.

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

    def summary(self):
        """Get the summary of the model.

        The summary is textual and includes information about:
        The layers and their order in the model.
        The output shape of each layer.

        Returns
        -------
        summary : NoneType
            the summary of the model

        """
        return self.model.summary()

    def fit(self, x_train, y_train, epochs=5):
        """Train the model on train dataset.

        The fit method will train the model for a fixed
        number of epochs (iterations) on a dataset.

        Parameters
        ----------
        x_train : ndarray
            the x_train is the train dataset
        y_train : ndarray shape=(BatchSize,)
            the y_train is the labels of the train dataset
        epochs : int (Default = 5)
            the number of epochs

        Returns
        -------
        hist : object
            A History object. Its History.history attribute is a record of
            training loss values and metrics values at successive epochs

        """
        hist = self.model.fit(x_train, y_train, epochs=epochs)
        self.accuracy = hist.history['accuracy'][0]
        self.loss = hist.history['loss'][0]
        return hist

    def evaluate(self, x_test, y_test, verbose=2):
        """Evaluate the model on test dataset.

        The evaluate method will evaluate the model on a test
        dataset.

        Parameters
        ----------
        x_test : ndarray
            the x_test is the test dataset
        y_test : ndarray shape=(BatchSize,)
            the y_test is the labels of the test dataset
        verbose : int (Default = 2)
            By setting verbose 0, 1 or 2 you just say how do you want to
            'see' the training progress for each epoch.

        Returns
        -------
        evaluate : List
            return list of loss value and accuracy value on test dataset

        """
        return self.model.evaluate(x_test,  y_test, verbose=verbose)

    def predict(self, x_test):
        """Predict the output from input samples.

        The predict method will generates output predictions
        for the input samples.

        Parameters
        ----------
        x_train : ndarray
            the x_test is the test dataset or input samples

        Returns
        -------
        predict : ndarray shape(TestSize,OutputSize)
            Numpy array(s) of predictions.

        """
        return self.model.predict(x_test)


class MultipleLayerPercepton:

    def __init__(self, input_shape=(28, 28),
                 num_hidden=(128, ),
                 act_hidden='relu',
                 dropout=0.2,
                 num_out=10,
                 act_out='softmax',
                 loss='sparse_categorical_crossentropy',
                 optimizer='adam'):
        """Multiple Layer Perceptron with Dropout.

        Parameters
        ----------
        input_shape : tuple
            Shape of data to be trained
        num_hidden : array-like
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

        # model building

        inp = tf.keras.layers.Input(shape=self.input_shape)
        x = tf.keras.layers.Flatten()(inp)

        for i in range(len(self.num_hidden)):
            x = tf.keras.layers.Dense(self.num_hidden[i])(x)

        x = tf.keras.layers.Dropout(self.dropout)(x)
        out = tf.keras.layers.Dense(self.num_out, activation=self.act_out)(x)

        self.model = tf.keras.models.Model(inputs=inp, outputs=out)

        # compiling the model
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])

    def summary(self):
        """Get the summary of the model.

        The summary is textual and includes information about:
        The layers and their order in the model.
        The output shape of each layer.

        Returns
        -------
        summary :  NoneType
            the summary of the model

        """
        return self.model.summary()

    def fit(self, x_train, y_train, epochs=5):
        """Train the model on train dataset.

        The fit method will train the model for a fixed
        number of epochs (iterations) on a dataset.

        Parameters
        ----------
        x_train : ndarray
            the x_train is the train dataset
        y_train : ndarray shape=(BatchSize,)
            the y_train is the labels of the train dataset
        epochs : int (Default = 5)
            the number of epochs

        Returns
        -------
        hist : object
            A History object. Its History.history attribute is a record of
            training loss values and metrics values at successive epochs

        """
        hist = self.model.fit(x_train, y_train, epochs=epochs)
        self.accuracy = hist.history['accuracy'][0]
        self.loss = hist.history['loss'][0]
        return hist

    def evaluate(self, x_test, y_test, verbose=2):
        """Evaluate the model on test dataset.

        The evaluate method will evaluate the model on a test
        dataset.

        Parameters
        ----------
        x_test : ndarray
            the x_test is the test dataset
        y_test : ndarray shape=(BatchSize,)
            the y_test is the labels of the test dataset
        verbose : int (Default = 2)
            By setting verbose 0, 1 or 2 you just say how do you want to
            'see' the training progress for each epoch.

        Returns
        -------
        evaluate : List
            return list of loss value and accuracy value on test dataset

        """
        return self.model.evaluate(x_test,  y_test, verbose=verbose)

    def predict(self, x_test):
        """Predict the output from input samples.

        The predict method will generates output predictions
        for the input samples.

        Parameters
        ----------
        x_train : ndarray
            the x_test is the test dataset or input samples

        Returns
        -------
        predict : ndarray shape(TestSize,OutputSize)
            Numpy array(s) of predictions.

        """
        return self.model.predict(x_test)
