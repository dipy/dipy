import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split
'''
Title : Denoising diffusion weighted imaging data using CNN
=====
Obtaining tissue microstructure measurements from diffusion weighted imaging (DWI) with multiple,
high b-values is crucial. However, the high noise levels present in these images can adversely affect
the accuracy of the microstructural measurements. In this context, we suggest a straightforward denoising technique that
can be applied to any DWI dataset as long as a low-noise, single-subject dataset is obtained using the same DWI sequence.
We created a simple 1D-CNN model with five layers, based on the 1D CNN for denoising speech.
The model consists of two convolutional layers followed by max-pooling layers, and a dense layer.
The first convolutional layer has 16 one-dimensional filters of size 16, and the second layer has 32 filters of size 8.
ReLu activation function is applied to both convolutional layers. The max-pooling layer has a kernel
size of 2 and a stride of 2. 
The dense layer maps the features extracted from the noisy image to the low-noise reference image.

Reference:
---------
Denoising diffusion weighted imaging data using convolutional neural networks
Hu Cheng, Sophia Vinci-Booher, Jian Wang, Bradley Caron, Qiuting Wen, Sharlene Newman, Franco Pestilli
'''
class Cnn1DDenoiser:
    def __init__(self, sig_length, optimizer='adam', loss='mean_squared_error', metrics=('accuracy',),
         loss_weights=None): 
        '''
        Parameters
        ----------
        sig_length : Int
            Length of DWI signal 
        optimizer : str, optional  
            Adam (Default) optimization algorithm name
        loss : str, optional ,mean_squared_error(Default)[fixed]
            Select loss function for measuring accuracy.
            Computes the mean of squares of errors between labels and predictions
        metrics : Funtion name 
            Accuracy (Default) ,judge the performance of your model
        optimizer : str, optional
            Select optimizer. Default adam.
        loss : floats
            Optional list or dictionary specifying scalar coefficients to weight.    
        '''
    # Layer 1 - Convolutional Layer(16*1*16) + ReLU activation
    #           Pooling
    # Layer 2 - Convolutional Layer(8*16*32) + ReLU activation
    #           Pooling
    #           Flatten
    #           Dense 
        input_layer = Input(shape=(sig_length, 1))
        x = Conv1D(filters=16, kernel_size=16, kernel_initializer='Orthogonal',
                padding='same', name=f'Conv1')(input_layer)
        x = Activation('relu', name=f'ReLU1')(x)
        max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')
        pool1 = max_pool_1d(x)
        x = Conv1D(filters=32, kernel_size=8, padding='same', name=f'Conv2')(pool1)
        x = Activation('relu', name=f'ReLU2')(x)
        pool2 = max_pool_1d(x)
        pool2_flat = tf.keras.layers.Flatten()(pool2)
        logits = tf.keras.layers.Dense(units=sig_length, activation='relu')(pool2_flat)
        model = Model(inputs=input_layer, outputs=logits)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics,loss_weights=loss_weights)
        self.model = model

    def compile(self, optimizer='adam', loss=None, metrics=None, loss_weights=None):
        """
        Configures the model for training.

        Parameters:
        -----------
        optimizer - String (name of optimizer) or optimizer instance. Defaults to 'adam'.
        loss - String (name of objective function) or objective function. If 'None', the
                model will be compiled without any loss function and can only be used to
                predict output. Defaults to `None`.
        metrics - List of metrics to be evaluated by the model during training and testing.
        loss_weights - Optional list or dictionary specifying scalar coefficients (Python
                        floats) to weight the loss contributions of different model outputs.
                        The loss value that will be minimized by the model will then be the
                        weighted sum of all individual losses. If a list, it is expected to
                        have a 1:1 mapping to the model's outputs. If a dict, it is expected
                        to map output names (strings) to scalar coefficients. Defaults to `None`.
        """
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, loss_weights=loss_weights)

    def summary(self):
        """
        Get the summary of the model.

        The summary is textual and includes information about:
        The layers and their order in the model.
        The output shape of each layer.

        Returns
        -------
        summary : NoneType
            the summary of the model
        """
        return self.model.summary()
    

    def train_test_split(self, x=None, y=None, test_size=None,
        train_size=None, random_state=None, shuffle=True, stratify=None):
        """
        Splits the input data into random train and test subsets.

        Parameters:
        -----------
        x: numpy array, input data.
        y: numpy array, target data.
        test_size: float or int, optional (default=None)
            If float, should be between 0.0 and 1.0 and 
            represent the proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.
            If None, the value is set to the complement of the train size.
            If train_size is also None, it will be set to 0.25.
        train_size: float or int, optional (default=None)
            If float, should be between 0.0 and 1.0 and
            represent the proportion of the dataset to include in the train split.
            If int, represents the absolute number of train samples.
            If None, the value is automatically set to the complement of the test size.
        random_state: int, RandomState instance or None, optional (default=None)
            Controls the shuffling applied to the data before applying the split.
            Pass an int for reproducible output across multiple function calls. See Glossary.
        shuffle: bool, optional (default=True)
            Whether or not to shuffle the data before splitting.
            If shuffle=False then stratify must be None.
        stratify: array-like, optional (default=None)
            If not None, data is split in a stratified fashion,
            using this as the class labels. Read more in the User Guide.

        Returns:
        Tuple of four numpy arrays: x_train, x_test, y_train, y_test.
        """
        sz = x.shape
        if len(sz) == 4:
            x = np.reshape(x, (sz[0]*sz[1]*sz[2], sz[3]))
        sz = y.shape
        if len(sz) == 4:
            y = np.reshape(y, (sz[0]*sz[1]*sz[2], sz[3]))
        return train_test_split(x, y, test_size=test_size, train_size=train_size, 
                                random_state=random_state, shuffle=shuffle,
                                stratify=stratify)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
        callbacks=None, validation_split=0.0, validation_data=None,
        shuffle=True, initial_epoch=0, steps_per_epoch=None,
        validation_steps=None, validation_batch_size=None,
        validation_freq=1, max_queue_size=10, workers=1,
        use_multiprocessing=False):
        """
        Train the model on train dataset.

        The fit method will train the model for a fixed
        number of epochs (iterations) on a dataset. If given data 
        is  4D it will convert it into 1D.

        Parameters
        ----------
        x_train         : ndarray
            The x_train is the train dataset(noise-free data)
        y_train         : ndarray shape=(BatchSize,)
            The y_train is the labels of the train dataset(noise-added data)
        epochs          : Int (Default = 5)
            The number of epochs
        verbose         :'auto', 0, 1, or 2. 
            Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch
        callbacks       : List of keras.callbacks.Callback instances
            List of callbacks to apply during training.
        validation_split: Float between 0 and 1
            Fraction of the training data to be used as validation data.
        validation_data	: 1D data
            Data on which to evaluate the loss and any model metrics at the end of each epoch
        shuffle : Boolean 
           This argument is ignored when x is a generator or an object of tf.data.Dataset.
        initial_epoch   :	Integer. 
           Epoch at which to start training
        steps_per_epoch :	Integer or None.
           Total number of steps (batches of samples) before declaring one epoch
           finished and starting the next epoch.
        validation_batch_size :Integer or None. 
           Number of samples per validation batch.
        validation_steps  :	Int
           Only relevant if validation_data is provided and is a tf.data dataset.
        validation_freq	  :
           Only relevant if validation data is provided.
        max_queue_size	  :Integer 
           Used for generator or keras.utils.Sequence input only.
        workers	Integer  : Int
           Used for generator or keras.utils.Sequence input only.
        use_multiprocessing	 : Boolean
           Used for generator or keras.utils.Sequence input only.

        Returns
        -------
        hist : object
            A History object. Its History.history attribute is a record of
            training loss values and metrics values at successive epochs
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

    def evaluate(self, x=None, y=None, batch_size=None, verbose=1,
                steps=None, callbacks=None, max_queue_size=10, workers=1,
                use_multiprocessing=False, return_dict=False):
        """
        Evaluate the model on test dataset.

        The evaluate method will evaluate the model on a test
        dataset. If data is 4D it is converted into 1D.

        Parameters
        ----------
        x_test : ndarray
            The x_test is the test dataset (high-noise data).
        y_test : ndarray shape=(BatchSize,)
            The y_test is the labels of the test dataset (low-noise data) 
        verbose : Int (Default = 2)
            By setting verbose 0, 1 or 2 you just say how do you want to
            'see' the training progress for each epoch.

        Returns
        -------
        evaluate : List
            Return list of loss value and accuracy value on test dataset.
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
        """
        Predict the output from input samples.

        The predict method will generate output predictions
        for the input samples.

        Parameters
        ----------
        x_train : ndarray
            The x_test is the test dataset or input samples.

        Returns
        -------
        predict : ndarray shape(TestSize,OutputSize)
            Numpy array(s) of predictions.          
        """        
        sz = x.shape
        x = np.reshape(x, (sz[0]*sz[1]*sz[2], sz[3]))
        predicted_img = self.model.predict(x=x, batch_size=batch_size,
                                  verbose=verbose, steps=steps,
                                  callbacks=callbacks,
                                  max_queue_size=max_queue_size,
                                  workers=workers,
                                  use_multiprocessing=use_multiprocessing)
        x_img = np.float32( np.reshape(predicted_img, (sz[0],sz[1],sz[2],sz[3])) )
        img_denoised = x_img
        return img_denoised

    def save_weights(self, filepath, overwrite=True):
        '''
        Saves the weights to HDF5. 

        Parameters
        ----------
        Filepath : str, optional
            File path.
        Overwrite: Boolean
            True or False.
        '''
        self.model.save_weights(filepath=filepath, overwrite=overwrite,
                                save_format=None)

    def load_weights(self, filepath):
        '''
        Loads the weights

        Parameters
        ----------
        Filepath : str, optional
            File path.
        '''
        self.model.load_weights(filepath)