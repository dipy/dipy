#!/usr/bin/python
"""
Class and helper functions for fitting the Synb0 model.
"""


from packaging.version import Version
import logging
from dipy.data import get_fnames
from dipy.testing.decorators import doctest_skip_parser
from dipy.utils.optpkg import optional_package
import numpy as np

tf, have_tf, _ = optional_package('tensorflow')
tfa, have_tfa, _ = optional_package('tensorflow_addons')
if have_tf and have_tfa:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import MaxPool3D, Conv3DTranspose
    from tensorflow.keras.layers import Conv3D, LeakyReLU
    from tensorflow.keras.layers import Concatenate, Layer
    from tensorflow_addons.layers import InstanceNormalization
    if Version(tf.__version__) < Version('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')
else:
    class Model:
        pass

    class Layer:
        pass

logging.basicConfig()
logger = logging.getLogger('synb0')


def set_logger_level(log_level):
    """ Change the logger of the Synb0 to one on the following:
    DEBUG, INFO, WARNING, CRITICAL, ERROR

    Parameters
    ----------
    log_level : str
        Log level for the Synb0 only
    """
    logger.setLevel(level=log_level)


def normalize(image, min_v=None, max_v=None, new_min=-1, new_max=1):
    r"""
    normalization function

    Parameters
    ----------
    image : np.ndarray
    min_v : int or float (optional)
        minimum value range for normalization
        intensities below min_v will be clipped
        if None it is set to min value of image
        Default : None
    max_v : int or float (optional)
        maximum value range for normalization
        intensities above max_v will be clipped
        if None it is set to max value of image
        Default : None
    new_min : int or float (optional)
        new minimum value after normalization
        Default : 0
    new_max : int or float (optional)
        new maximum value after normalization
        Default : 1

    Returns
    -------
    np.ndarray
        Normalized image from range new_min to new_max
    """
    if min_v is None:
        min_v = np.min(image)
    if max_v is None:
        max_v = np.max(image)
    return np.interp(image, (min_v, max_v), (new_min, new_max))

def unnormalize(image, norm_min, norm_max, min_v, max_v):
    r"""
    unnormalization function

    Parameters
    ----------
    image : np.ndarray
    norm_min : int or float
        minimum value of normalized image
    norm_max : int or float
        maximum value of normalized image
    min_v : int or float
        minimum value of unnormalized image
    max_v : int or float
        maximum value of unnormalized image

    Returns
    -------
    np.ndarray
        unnormalized image from range min_v to max_v
    """
    return (image-norm_min)/(norm_max-norm_min)*(max_v-min_v) + min_v

class EncoderBlock(Layer):
    def __init__(self, out_channels, kernel_size, strides, padding):
        super(EncoderBlock, self).__init__()
        self.conv3d = Conv3D(out_channels,
                             kernel_size,
                             strides=strides,
                             padding=padding,
                             use_bias=False)
        self.instnorm = InstanceNormalization()
        self.activation = LeakyReLU(0.01)

    def call(self, input):
        x = self.conv3d(input)
        x = self.instnorm(x)
        x = self.activation(x)

        return x

class DecoderBlock(Layer):
    def __init__(self, out_channels, kernel_size, strides, padding):
        super(DecoderBlock, self).__init__()
        self.conv3d = Conv3DTranspose(out_channels,
                                      kernel_size,
                                      strides=strides,
                                      padding=padding,
                                      use_bias=False)
        self.instnorm = InstanceNormalization()
        self.activation = LeakyReLU(0.01)

    def call(self, input):
        x = self.conv3d(input)
        x = self.instnorm(x)
        x = self.activation(x)

        return x

def UNet3D(input_shape):
    inputs = tf.keras.Input(input_shape)
    # Encode
    x = EncoderBlock(32, kernel_size=3,
                     strides=1, padding='same')(inputs)
    syn0 = EncoderBlock(64, kernel_size=3,
                        strides=1, padding='same')(x)
    
    x = MaxPool3D()(syn0)
    x = EncoderBlock(64, kernel_size=3,
                     strides=1, padding='same')(x)
    syn1 = EncoderBlock(128, kernel_size=3,
                        strides=1, padding='same')(x)

    x = MaxPool3D()(syn1)
    x = EncoderBlock(128, kernel_size=3,
                     strides=1, padding='same')(x)
    syn2 = EncoderBlock(256, kernel_size=3,
                        strides=1, padding='same')(x)

    x = MaxPool3D()(syn2)
    x = EncoderBlock(256, kernel_size=3,
                     strides=1, padding='same')(x)
    x = EncoderBlock(512, kernel_size=3,
                     strides=1, padding='same')(x)

    # Last layer without relu
    x = Conv3D(512, kernel_size=1,
               strides=1, padding='same')(x)
    
    x = DecoderBlock(512, kernel_size=2,
                     strides=2, padding='valid')(x)

    x = Concatenate()([x, syn2])

    x = DecoderBlock(256, kernel_size=3,
                     strides=1, padding='same')(x)
    x = DecoderBlock(256, kernel_size=3,
                     strides=1, padding='same')(x)
    x = DecoderBlock(256, kernel_size=2,
                     strides=2, padding='valid')(x)
    
    x = Concatenate()([x, syn1])

    x = DecoderBlock(128, kernel_size=3,
                     strides=1, padding='same')(x)
    x = DecoderBlock(128, kernel_size=3,
                     strides=1, padding='same')(x)
    x = DecoderBlock(128, kernel_size=2,
                     strides=2, padding='valid')(x)

    x = Concatenate()([x, syn0])

    x = DecoderBlock(64, kernel_size=3,
                     strides=1, padding='same')(x)
    x = DecoderBlock(64, kernel_size=3,
                     strides=1, padding='same')(x)

    x = DecoderBlock(1, kernel_size=1,
                     strides=1, padding='valid')(x)

    # Last layer without relu
    out = Conv3DTranspose(1, kernel_size=1,
                          strides=1, padding='valid')(x)

    return Model(inputs, out)


class Synb0:
    """
    This class is intended for the Synb0 model.
    The model is the deep learning part of the Synb0-Disco
    pipeline, thus stand-alone usage is not
    recommended.
    """

    @doctest_skip_parser
    def __init__(self, verbose=False):
        r"""
        The model was pre-trained for usage on pre-processed images
        following the synb0-disco pipeline.
        One can load their own weights using load_model_weights.

        This model is designed to take as input
        a b0 image and a T1 weighted image.

        It was designed to predict a b-inf image.

        Parameters
        ----------
        verbose : bool (optional)
            Whether to show information about the processing.
            Default: False

        References
        ----------
        ..  [1] Schilling, K. G., Blaber, J., Huo, Y., Newton, A.,
            Hansen, C., Nath, V., ... & Landman, B. A. (2019).
            Synthesized b0 for diffusion distortion correction (Synb0-DisCo).
            Magnetic resonance imaging, 64, 62-70.
        ..  [2] Schilling, K. G., Blaber, J., Hansen, C., Cai, L.,
            Rogers, B., Anderson, A. W., ... & Landman, B. A. (2020).
            Distortion correction of diffusion weighted MRI without reverse
            phase-encoding scans or field-maps.
            PloS one, 15(7), e0236418.
        """

        if not have_tf:
            raise tf()
        if not have_tfa:
            raise tfa()

        log_level = 'INFO' if verbose else 'CRITICAL'
        set_logger_level(log_level)

        # Synb0 network load

        self.model = UNet3D(input_shape=(80, 80, 96, 2))


    def fetch_default_weights(self, idx):
        r"""
        Load the model pre-training weights to use for the fitting.
        While the user can load different weights, the function
        is mainly intended for the class function 'predict'.

        Parameters
        ----------
        idx : int
            The idx of the default weights. It can be from 0~4.
        """
        fetch_model_weights_path = get_fnames('synb0_default_weights')
        print('fetched ' + fetch_model_weights_path[idx])
        self.load_model_weights(fetch_model_weights_path[idx])

    def load_model_weights(self, weights_path):
        r"""
        Load the custom pre-training weights to use for the fitting.

        Parameters
        ----------
        weights_path : str
            Path to the file containing the weights (hdf5, saved by tensorflow)
        """
        try:
            self.model.load_weights(weights_path)
        except ValueError:
            raise ValueError('Expected input for the provided model weights \
                             do not match the declared model')

    def __predict(self, x_test):
        r"""
        Internal prediction function

        Parameters
        ----------
        x_test : np.ndarray (batch, 80, 80, 96, 2)
            Image should match the required shape of the model.

        Returns
        -------
        np.ndarray (...) or (batch, ...)
            Reconstructed b-inf image(s)
        """

        return self.model.predict(x_test)

    def predict(self, b0, T1, batch_size=None, average=True):
        r"""
        Wrapper function to facilitate prediction of larger dataset.
        The function will pad the data to meet the required shape of image.
        Note that the b0 and T1 image should have the same shape

        Parameters
        ----------
        b0 : np.ndarray (batch, 77, 91, 77) or (77, 91, 77)
            For a single image, input should be a 3D array. If multiple images,
            there should also be a batch dimension.

        T1 : np.ndarray (batch, 77, 91, 77) or (77, 91, 77)
            For a single image, input should be a 3D array. If multiple images,
            there should also be a batch dimension.

        batch_size : int
            Number of images per prediction pass. Only available if data
            is provided with a batch dimension.
            Consider lowering it if you get an out of memory error.
            Increase it if you want it to be faster and have a lot of data.
            If None, batch_size will be set to 1 if the provided image
            has a batch dimension.
            Default is None

        average : bool
            Whether the function follows the Synb0-Disco pipeline and
            averages the prediction of 5 different models.
            If False, it uses the loaded weights for prediction.
            Default is True.
        Returns
        -------
        pred_output : np.ndarray (...) or (batch, ...)
            Reconstructed b-inf image(s)

        """
        # Check if shape is as intended
        if all([b0.shape[1:] != (77, 91, 77), b0.shape != (77, 91, 77)]) or \
                b0.shape != T1.shape:
            raise ValueError('Expected shape (batch, 77, 91, 77) or \
                             (77, 91, 77) for both inputs')

        dim = len(b0.shape)

        # Add batch dimension if not provided
        if dim == 3:
            T1 = np.expand_dims(T1, 0)
            b0 = np.expand_dims(b0, 0)
        shape = b0.shape

        # Pad the data to match the model's input shape
        T1 = np.pad(T1, ((0, 0), (2, 1), (3, 2), (2, 1)), 'constant')
        b0 = np.pad(b0, ((0, 0), (2, 1), (3, 2), (2, 1)), 'constant')

        # Normalize the data.
        p99 = np.percentile(b0, 99, axis=(1, 2, 3))
        for i in range(shape[0]):
            T1[i] = normalize(T1[i], 0, 150, -1, 1)
            b0[i] = normalize(b0[i], 0, p99[i], -1, 1)

        if dim == 3:
            if batch_size is not None:
                logger.warning('Batch size specified, but not used',
                               'due to the input not having \
                               a batch dimension')
            batch_size = 1

        # Prediction stage
        if average:
            mean_pred = np.zeros(shape+(5,), dtype=np.float32)
            for i in range(5):
                self.fetch_default_weights(i)
                temp = np.stack([b0, T1], -1)
                input_data = np.moveaxis(temp, 3, 1).astype(np.float32)
                prediction = np.zeros((shape[0], 80, 80, 96, 1),
                                      dtype=np.float32)
                for batch_idx in range(batch_size, shape[0]+1, batch_size):
                    temp_input = input_data[batch_idx-batch_size:batch_idx]
                    temp_pred = self.__predict(temp_input)
                    prediction[batch_idx-batch_size:batch_idx] = temp_pred
                remainder = np.mod(shape[0], batch_size)
                if remainder != 0:
                    temp_pred = self.__predict(input_data[-remainder:])
                    prediction[-remainder:] = temp_pred
                for j in range(shape[0]):
                    temp_pred = unnormalize(prediction[j], -1, 1, 0, p99[j])
                    prediction[j] = temp_pred

                prediction = prediction[:, 2:-1, 2:-1, 3:-2, 0]
                prediction = np.moveaxis(prediction, 1, -1)

                mean_pred[..., i] = prediction
            prediction = np.mean(mean_pred, axis=-1)
        else:
            temp = np.stack([b0, T1], -1)
            input_data = np.moveaxis(temp, 3, 1).astype(np.float32)
            prediction = np.zeros((shape[0], 80, 80, 96, 1),
                                  dtype=np.float32)
            for batch_idx in range(batch_size, shape[0]+1, batch_size):
                temp_input = input_data[batch_idx-batch_size:batch_idx]
                temp_pred = self.__predict(temp_input)
                prediction[:batch_idx] = temp_pred
            remainder = np.mod(shape[0], batch_size)
            if remainder != 0:
                temp_pred = self.__predict(input_data[-remainder:])
                prediction[-remainder:] = temp_pred
            for j in range(shape[0]):
                prediction[j] = unnormalize(prediction[j], -1, 1, 0, p99[j])

            prediction = prediction[:, 2:-1, 2:-1, 3:-2, 0]
            prediction = np.moveaxis(prediction, 1, -1)

        if dim == 3:
            prediction = prediction[0]

        return prediction
