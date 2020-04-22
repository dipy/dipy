"""
This script is intended for the model object
of Null Space Harmonization Network.

The model was re-trained for usage with different basis function ('mrtrix') set
as per the proposed model from the paper:

[1] Nath, Vishwesh, Prasanna Parvathaneni, Colin B. Hansen, Allison E. Hainline,
Camilo Bermudez, Samuel Remedios, Justin A. Blaber et al.
"Inter-scanner harmonization of high angular resolution DW-MRI using
null space deep learning." In International Conference on Medical Image Computing
and Computer-Assisted Intervention, pp. 193-201. Springer, Cham, 2018.

"""

from distutils.version import LooseVersion

from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package('tensorflow')

if have_tf:
    if LooseVersion(tf.__version__) < LooseVersion('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Activation, concatenate
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import RMSprop

#from tf.keras.layers import Input, Dense, Dropout, merge, concatenate, Convolution3D, Flatten


def identity_loss(y_true, y_pred):
    # loss_1 = K.cast(loss_1, dtype='float64')

    # Extract aux1, aux2 and main o/p
    loss_1 = tf.mean(tf.square(y_pred[:, :66] - y_pred[:, 66:132]))
    loss_2 = tf.mean(tf.square(y_pred[:, 132:] - y_true))
    loss = loss_1 + loss_2
    # loss_1 = K.mean(y_true - y_pred)
    # return K.mean(y_pred - 0 * y_true)
    return loss

def calc_acc(y_true, y_pred):
    # Normalize each vector
    # TODO Need a get around here for Harmonic Coefficients of varying orders.
    # TODO Also needs a generic ACC function which can be used commonly for Null space and ResDNN
    y_true = y_true[0:45]
    y_pred = y_pred[0:45]

    comp_true = tf.math.conj(y_true)
    norm_true = y_true / tf.sqrt(tf.reduce_sum(tf.multiply(y_true, comp_true)))

    comp_pred = tf.math.conj(y_pred)
    norm_pred = y_pred / tf.sqrt(tf.reduce_sum(tf.multiply(y_pred, comp_pred)))

    comp_p2 = tf.math.conj(norm_pred)
    acc = tf.math.real(tf.reduce_sum(tf.multiply(norm_true, comp_p2)))
    return acc

class Histo_NSDN(object):

    def __init__(self, input_shape=(45,),
                 num_hidden=45, act_hidden='relu',
                 num_out=45, act_out='linear',
                 optimizer='rmsprop',
                 loss='mse'):
        """ Single Layer Perceptron with Dropout
        Parameters
        ----------
        input_shape : tuple
            Shape of data to be trained
        num_hidden : int
            Number of nodes in hidden layers 2 and 4
        act_hidden : string
            Activation function used in hidden layer
        num_out : 45
            Number of nodes in output layer
        act_out : string
            Activation function used in output layer
        optimizer :  string
            Select optimizer. Default rmsprop.
        loss : string
            Select loss function for measuring accuracy.
            Default mse.
        """
        self.accuracy = None
        self.loss = None

        input_dim = 45
        # Three inputs
        input_a = Input(shape=(input_dim,))
        input_b = Input(shape=(input_dim,))
        main_ip = Input(shape=(input_dim,))

        # Create the base net structure
        base_network = self.create_cnn_network(input_dim)

        # Feed the inputs to the base network
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        processed_main = base_network(main_ip)

        # Combine the pairwise voxels with the histo voxel.
        # distance = Lambda(euclidean_distance)([processed_a, processed_b, processed_main])
        concat_layer = concatenate([processed_a, processed_b, processed_main])
        model = Model(input=[input_a, input_b, main_ip], output=concat_layer)

        opt_func = RMSprop(lr=0.0001)
        model.compile(loss=identity_loss, optimizer=opt_func, metrics=[calc_acc])

        print(model.summary())
        self.model = model
        #input_dims = self.input_shape
        #skip_neurons = self.num_hidden
        #inputs = Input(shape=input_shape)

        # ResDNN Network Flow
        #x1 = Dense(400, activation='relu')(inputs)
        #x2 = Dense(num_hidden, activation='relu')(x1)
        #x3 = Dense(200, activation='relu')(x2)
        #x4 = Dense(num_hidden, activation='linear')(x3)
        #res_add = Add()([x2, x4])
        #x5 = Dense(200, activation='relu')(res_add)
        #x6 = Dense(num_hidden)(x5)

        #model = Model(inputs=inputs, outputs=x6)

        #opt_func = RMSprop(lr=0.0001)
        '''
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=input_shape),
            tf.keras.layers.Dense(num_hidden, activation=act_hidden),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_out, activation=act_out)
            ])
        '''

        #model.compile(optimizer=opt_func,
        #              loss=loss,
        #              metrics=[calc_acc])

        #self.model = model
    #TODO Consider making the base network as a residual one as for ResDNN Histology
    def base_network(self, input_dims):
        seq = Sequential()

        seq.add(Dense(45, input_shape=(input_dims,)))
        # seq.add(BatchNormalization())
        seq.add(Dense(400))
        seq.add(Activation("relu"))
        # seq.add(BatchNormalization())
        seq.add(Dense(66))
        seq.add(Activation("relu"))
        # seq.add(BatchNormalization())
        seq.add(Dense(200))
        # seq.add(BatchNormalization())
        # Output Layer.
        seq.add(Dense(45))
        print(seq.summary())
        return seq

    def fit(self, x_train, y_train, epochs=5):
        hist = self.model.fit(x_train, y_train, epochs=epochs, batch_size=10000, validation_split=0.2)
        self.accuracy = hist.history['calc_acc'][0]
        self.loss = hist.history['loss'][0]
        return hist

    def evaluate(self, x_test, y_test, verbose=2):
        return self.model.evaluate(x_test,  y_test, verbose=verbose)

    def load_model_weights(self, weights_path):
        self.model.load_weights(weights_path)
        return self.model

    def predict(self, x_test):
        return self.model.predict(x_test)