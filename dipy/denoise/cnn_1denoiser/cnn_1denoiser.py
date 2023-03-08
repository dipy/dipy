import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Activation
from tensorflow.keras.models import *
from tensorflow.keras.layers import Input
#from patchify import patchify, unpatchify
from sklearn.model_selection import train_test_split
from datetime import datetime
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)






'''

we constructed a simple 1D-CNN model that has five layers, including two convolutional layers, each
followed by a max-pooling layer, and a dense layer (Figure 1). The first convolutional layer was given an
input of the ‘noisy’ high-noise image and consisted of 16 one-dimensional filtering kernels of size 16. The
second convolutional layer consisted of 32 one-dimensional filtering kernels of size 8. The ReLu
activation function was used in both convolutional layers. There was 1 max-pooling layer that had a kernel
size of 2 with stride 2. In the dense layer, the extracted features of the high-noise image were mapped to
the low-noise reference image.




Contributed by :
Hu Chenga,b Sophia Vinci-Boohera, Jian Wangc, Bradley Carona, Qiuting Wend, Sharlene Newmane,Franco Pestillif

Ref:Denoising diffusion weighted imaging data using convolutional neural networks( https://www.biorxiv.org/content/10.1101/2022.01.17.476708v1.abstract )
Ref Code : https://github.com/huchengMRI/DWI-SoS-denoising




'''





class cnn_1denoiser(object):
    def __init__(self,sigLength,optimizer='adam', loss='mean_squared_error', metrics=['accuracy'], loss_weights=None):
        
        
        '''
        Parameters
        ----------
        sigLength : Int
            Shape of data(Time) 
        optimizer : String  
            Adam (Default) optimization algorithm name
        loss : string ,mean_squared_error(Default)[fixed]
            Select loss function for measuring accuracy.Computes the mean of squares of errors between labels and predictions
        metrics : Funtion name 
            accuracy (Default) ,judge the performance of your model
        optimizer : String
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
    


        input_layer = Input(shape=(sigLength, 1))
        x = Conv1D(filters=16, kernel_size=16, kernel_initializer='Orthogonal',
                padding='same', name=f'Conv1')(input_layer)
        x = Activation('relu', name=f'ReLU1')(x)
        max_pool_1d = tf.keras.layers.MaxPooling1D(pool_size=2, strides=2, padding='valid')
        pool1 = max_pool_1d(x)
        x = Conv1D(filters=32, kernel_size=8, padding='same', name=f'Conv2')(pool1)
        x = Activation('relu', name=f'ReLU2')(x)
        pool2 = max_pool_1d(x)
        pool2_flat = tf.keras.layers.Flatten()(pool2)
        logits = tf.keras.layers.Dense(units=sigLength, activation='relu')(pool2_flat)
        model = Model(inputs=input_layer, outputs=logits)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics,loss_weights=loss_weights)
        self.model = model

    def compile(self, optimizer='adam', loss=None, metrics=None,loss_weights=None):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics,
                           loss_weights=loss_weights)

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
    

    def train_test_split(self,x=None,y=None):
        sz = x.shape
        x = np.reshape(x, (sz[0]*sz[1]*sz[2], sz[3]))
        sz = y.shape
        y = np.reshape(y, (sz[0]*sz[1]*sz[2], sz[3]))

        return train_test_split(x,y)

    def fit(self, x=None, y=None, batch_size=None, epochs=1, verbose=1,
        callbacks=None, validation_split=0.0, validation_data=None,
        shuffle=True, initial_epoch=0, steps_per_epoch=None,
        validation_steps=None, validation_batch_size=None,
        validation_freq=1, max_queue_size=10, workers=1,
        use_multiprocessing=False):

        """
        Train the model on train dataset.

        The fit method will train the model for a fixed
        number of epochs (iterations) on a dataset.If given data 
        is  4D it will convert it into 1D

        Parameters
        ----------
        x_train         : ndarray
            the x_train is the train dataset
        y_train         : ndarray shape=(BatchSize,)
            the y_train is the labels of the train dataset
        epochs          : int (Default = 5)
            the number of epochs
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
        max_queue_size	  :Integer. 
           Used for generator or keras.utils.Sequence input only.
        workers	Integer  : Int
           Used for generator or keras.utils.Sequence input only.
        use_multiprocessing	Boolean :
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
        dataset.If Data is 4D it is converted into 1D.

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
        save_weights()  saves the weights to HDF5 

        Parameters
        ----------
        filepath : String
            File path
        overwrite: Boolean
            True or False

        '''
        
        self.model.save_weights(filepath=filepath, overwrite=overwrite,
                                save_format=None)

    def load_weights(self, filepath):

        '''
        loads the weights

        Parameters
        ----------
        filepath : String
            File path

        '''
        self.model.load_weights(filepath)

        

    
