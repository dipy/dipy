"""
Title : Denoising diffusion weighted imaging data using CNN
===========================================================

Obtaining tissue microstructure measurements from diffusion weighted imaging
(DWI) with multiple, high b-values is crucial. However, the high noise levels
present in these images can adversely affect the accuracy of the
microstructural measurements. In this context, we suggest a straightforward
denoising technique that can be applied to any DWI dataset as long as a
low-noise, single-subject dataset is obtained using the same DWI sequence.

We created a simple 1D-CNN model with five layers, based on the 1D CNN for
denoising speech. The model consists of two convolutional layers followed by
max-pooling layers, and a dense layer. The first convolutional layer has
16 one-dimensional filters of size 16, and the second layer has 32 filters of
size 8. ReLu activation function is applied to both convolutional layers.
The max-pooling layer has a kernel size of 2 and a stride of 2.
The dense layer maps the features extracted from the noisy image to the
low-noise reference image.

Reference
---------
Cheng H, Vinci-Booher S, Wang J, Caron B, Wen Q, Newman S, et al.
(2022) Denoising diffusion weighted imaging data using convolutional neural
networks.
PLoS ONE 17(9): e0274396. https://doi.org/10.1371/journal.pone.0274396

"""

from dipy.utils.optpkg import optional_package
import numpy as np
tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')
if have_tf:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv1D, Activation

sklearn, have_sklearn, _ = optional_package('sklearn.model_selection')


class Cnn1DDenoiser:
    def __init__(self, sig_length, optimizer='adam', loss='mean_squared_error',
                 metrics=('accuracy',), loss_weights=None):
        """Initialize the CNN 1D denoiser with the given parameters.

        Parameters
        ----------
        sig_length : int
            Length of the DWI signal.
        optimizer : str, optional
            Name of the optimization algorithm to use. Options: 'adam', 'sgd',
            'rmsprop', 'adagrad', 'adadelta'.
        loss : str, optional
            Name of the loss function to use. Available options are
            'mean_squared_error', 'mean_absolute_error',
            'mean_absolute_percentage_error', 'mean_squared_logarithmic_error',
            'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh',
            'categorical_crossentropy', 'sparse_categorical_crossentropy',
            'binary_crossentropy', 'kullback_leibler_divergence', 'poisson',
            'cosine_similarity'.
            Suggested to go with 'mean_squared_error'.
        metrics : tuple of str or function, optional
            List of metrics to be evaluated by the model during training and
            testing. Available options are 'accuracy', 'binary_accuracy',
            'categorical_accuracy', 'top_k_categorical_accuracy',
            'sparse_categorical_accuracy', 'sparse_top_k_categorical_accuracy',
            and any custom function.
        loss_weights : float or dict, optional
            Scalar coefficients to weight the loss contributions of different
            model outputs. Can be a single float value or a dictionary mapping
            output names to scalar coefficients.

        """
        if not have_tf:
            raise ImportError('TensorFlow is not available. Please install '
                              'TensorFlow 2+.')
        if not have_sklearn:
            raise ImportError('scikit-learn is not available. Please install '
                              'scikit-learn.')
        input_layer = Input(shape=(sig_length, 1))
        x = Conv1D(filters=16, kernel_size=16, kernel_initializer='Orthogonal',
                   padding='same', name='Conv1')(input_layer)
        x = Activation('relu', name='ReLU1')(x)
        max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2,
                                                   padding='valid')
        pool1 = max_pool_1d(x)
        x = Conv1D(filters=32, kernel_size=8, padding='same',
                   name='Conv2')(pool1)
        x = Activation('relu', name='ReLU2')(x)
        pool2 = max_pool_1d(x)
        pool2_flat = tf.keras.layers.Flatten()(pool2)
        logits = tf.keras.layers.Dense(units=sig_length,
                                       activation='relu')(pool2_flat)
        model = Model(inputs=input_layer, outputs=logits)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                      loss_weights=loss_weights)
        self.model = model

    def compile(self, optimizer='adam', loss=None, metrics=None,
                loss_weights=None):
        """Configure the model for training.

        Parameters
        ----------
        optimizer : str or optimizer object, optional
            Name of optimizer or optimizer object.
        loss : str or objective function, optional
            Name of objective function or objective function itself.
            If 'None', the model will be compiled without any loss function
            and can only be used to predict output.
        metrics : list of metrics, optional
            List of metrics to be evaluated by the model during training
            and testing.
        loss_weights : list or dict, optional
            Optional list or dictionary specifying scalar coefficients(floats)
            to weight the loss contributions of different model outputs.
            The loss value that will be minimized by the model will then be
            the weighted sum of all individual losses. If a list, it is
            expected to have a 1:1 mapping to the model's outputs. If a dict,
            it is expected to map output names (strings) to scalar
            coefficients.

        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                           loss_weights=loss_weights)

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

    def train_test_split(self, x, y, test_size=None, train_size=None,
                         random_state=None, shuffle=True, stratify=None):
        """Split the input data into random train and test subsets.

        Parameters
        ----------
        x: numpy array
           input data.
        y: numpy array
           target data.
        test_size: float or int, optional
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
            If train_size is also None, it will be set to 0.25.
        train_size: float or int, optional
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the train split.
            If int, represents the absolute number of train samples.
            If None, the value is automatically set to the complement of the
            test size.
        random_state: int, RandomState instance or None, optional
            Controls the shuffling applied to the data before applying
            the split. Pass an int for reproducible output across multiple
            function calls. See Glossary.
        shuffle: bool, optional
            Whether or not to shuffle the data before splitting.
            If shuffle=False then stratify must be None.
        stratify: array-like, optional
            If not None, data is split in a stratified fashion,
            using this as the class labels. Read more in the User Guide.

        Returns
        -------
        Tuple of four numpy arrays: x_train, x_test, y_train, y_test.
        """
        sz = x.shape
        if len(sz) == 4:
            x = np.reshape(x, (sz[0]*sz[1]*sz[2], sz[3]))
        sz = y.shape
        if len(sz) == 4:
            y = np.reshape(y, (sz[0]*sz[1]*sz[2], sz[3]))
        return sklearn.train_test_split(
            x, y, test_size=test_size, train_size=train_size,
            random_state=random_state, shuffle=shuffle, stratify=stratify)

    def fit(self, x, y, batch_size=None, epochs=1, verbose=1,
            callbacks=None, validation_split=0.0, validation_data=None,
            shuffle=True, initial_epoch=0, steps_per_epoch=None,
            validation_steps=None, validation_batch_size=None,
            validation_freq=1, max_queue_size=10, workers=1,
            use_multiprocessing=False):
        """Train the model on train dataset.

        The fit method will train the model for a fixed number of epochs
        (iterations) on a dataset. If given data is  4D it will convert
        it into 1D.

        Parameters
        ----------
        x : ndarray
            The input data, as an ndarray.
        y : ndarray
            The target data, as an ndarray.
        batch_size : int or None, optional
            Number of samples per batch of computation.
        epochs : int, optional
            The number of epochs.
        verbose : 'auto', 0, 1, or 2, optional
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line
            per epoch.
        callbacks : list of keras.callbacks.Callback instances, optional
            List of callbacks to apply during training.
        validation_split : float between 0 and 1, optional
            Fraction of the training data to be used as validation data.
        validation_data : tuple (x_val, y_val) or None, optional
            Data on which to evaluate the loss and any model metrics at
            the end of each epoch.
        shuffle : boolean, optional
            This argument is ignored when x is a generator or an object of
            tf.data.Dataset.
        initial_epoch : int, optional
            Epoch at which to start training.
        steps_per_epoch : int or None, optional
            Total number of steps (batches of samples) before declaring one
            epoch finished and starting the next epoch.
        validation_batch_size : int or None, optional
            Number of samples per validation batch.
        validation_steps : int or None, optional
            Only relevant if validation_data is provided and is a
            tf.data dataset.
        validation_freq : int or list/tuple/set, optional
            Only relevant if validation data is provided. If an integer,
            specifies how many training epochs to run before a new validation
            run is performed. If a list, tuple, or set, specifies the epochs
            on which to run validation.
        max_queue_size : int, optional
            Used for generator or keras.utils.Sequence input only.
        workers : integer, optional
            Used for generator or keras.utils.Sequence input only.
        use_multiprocessing : boolean, optional
            Used for generator or keras.utils.Sequence input only.

        Returns
        -------
        hist : object
            A History object. Its History.history attribute is a record of
            training loss values and metrics values at successive epochs.

        """
        sz = x.shape
        if len(sz) == 4:
            x = np.reshape(x, (sz[0]*sz[1]*sz[2], sz[3]))
        sz = y.shape
        if len(sz) == 4:
            y = np.reshape(y, (sz[0]*sz[1]*sz[2], sz[3]))
        return self.model.fit(x=x, y=y, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              callbacks=callbacks,
                              validation_split=validation_split,
                              validation_data=validation_data, shuffle=shuffle,
                              initial_epoch=initial_epoch,
                              steps_per_epoch=steps_per_epoch,
                              validation_steps=validation_steps,
                              validation_batch_size=validation_batch_size,
                              validation_freq=validation_freq,
                              max_queue_size=max_queue_size, workers=workers,
                              use_multiprocessing=use_multiprocessing)

    def evaluate(self, x, y, batch_size=None, verbose=1,
                 steps=None, callbacks=None, max_queue_size=10, workers=1,
                 use_multiprocessing=False, return_dict=False):
        """Evaluate the model on a test dataset.

        Parameters
        ----------
        x : ndarray
            Test dataset (high-noise data). If 4D, it will be converted to 1D.
        y : ndarray
            Labels of the test dataset (low-noise data). If 4D, it will be
            converted to 1D.
        batch_size : int, optional
            Number of samples per gradient update.
        verbose : int, optional
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line
            per epoch.
        steps : int, optional
            Total number of steps (batches of samples) before declaring the
            evaluation round finished.
        callbacks : list, optional
            List of callbacks to apply during evaluation.
        max_queue_size : int, optional
            Maximum size for the generator queue.
        workers : int, optional
            Maximum number of processes to spin up when using process-based
            threading.
        use_multiprocessing : bool, optional
            If `True`, use process-based threading.
        return_dict : bool, optional
            If `True`, loss and metric results are returned as a dictionary.

        Returns
        -------
        List or dict
            If `return_dict` is `False`, returns a list of [loss, metrics]
            values on the test dataset. If `return_dict` is `True`, returns
            a dictionary of metric names and their corresponding values.

        """
        sz = x.shape
        if len(sz) == 4:
            x = np.reshape(x, (sz[0]*sz[1]*sz[2], sz[3]))
        sz = y.shape
        if len(sz) == 4:
            y = np.reshape(y, (sz[0]*sz[1]*sz[2], sz[3]))
        return self.model.evaluate(x=x, y=y, batch_size=batch_size,
                                   verbose=verbose, steps=steps,
                                   callbacks=callbacks,
                                   max_queue_size=max_queue_size,
                                   workers=workers,
                                   use_multiprocessing=use_multiprocessing,
                                   return_dict=return_dict)

    def predict(self, x, batch_size=None, verbose=0,
                steps=None, callbacks=None, max_queue_size=10, workers=1,
                use_multiprocessing=False):
        """Generate predictions for input samples.

        Parameters
        ----------
        x : ndarray
            Input samples.
        batch_size : int, optional
            Number of samples per batch.
        verbose : int, optional
            Verbosity mode.
        steps : int, optional
            Total number of steps (batches of samples) before declaring the
            prediction round finished.
        callbacks : list, optional
            List of Keras callbacks to apply during prediction.
        max_queue_size : int, optional
            Maximum size for the generator queue.
        workers : int, optional
            Maximum number of processes to spin up when using process-based
            threading.
        use_multiprocessing : bool, optional
            If `True`, use process-based threading. If `False`, use
            thread-based threading.

        Returns
        -------
        ndarray
            Numpy array of predictions.

        """
        sz = x.shape
        x = np.reshape(x, (sz[0]*sz[1]*sz[2], sz[3]))
        predicted_output = self.model.predict(
            x=x, batch_size=batch_size, verbose=verbose, steps=steps,
            callbacks=callbacks, max_queue_size=max_queue_size,
            workers=workers, use_multiprocessing=use_multiprocessing)
        predicted_output = np.float32(np.reshape(predicted_output,
                                                 (sz[0], sz[1], sz[2], sz[3])))
        return predicted_output

    def save_weights(self, filepath, overwrite=True):
        """Save the weights of the model to HDF5 file format.

        Parameters
        ----------
        filepath : str
            The path where the weights should be saved.
        overwrite : bool,optional
            If `True`, overwrites the file if it already exists. If `False`,
            raises an error if the file already exists.

        """
        self.model.save_weights(filepath=filepath, overwrite=overwrite,
                                save_format=None)

    def load_weights(self, filepath):
        """Load the model weights from the specified file path.

        Parameters
        ----------
        filepath : str
            The file path from which to load the weights.

        """
        self.model.load_weights(filepath)