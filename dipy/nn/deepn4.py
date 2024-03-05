#!/usr/bin/python
"""
Class and helper functions for fitting the DeepN4 model.
"""
import logging
import numpy as np
from scipy.ndimage import gaussian_filter
from dipy.data import get_fnames
from dipy.testing.decorators import doctest_skip_parser
from dipy.utils.optpkg import optional_package
from dipy.nn.utils import set_logger_level
from dipy.nn.utils import transform_img, recover_img, normalize

tf, have_tf, _ = optional_package('tensorflow')
if have_tf:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import MaxPool3D, Conv3DTranspose
    from tensorflow.keras.layers import Conv3D, LeakyReLU
    from tensorflow.keras.layers import Concatenate, Layer
    from tensorflow.keras.layers import GroupNormalization
else:
    class Model:
        pass

    class Layer:
        pass
    logging.warning('This model requires Tensorflow.\
                    Please install these packages using \
                    pip. If using mac, please refer to this \
                    link for installation. \
                    https://github.com/apple/tensorflow_macos')


logging.basicConfig()
logger = logging.getLogger('deepn4')


class EncoderBlock(Layer):
    def __init__(self, out_channels, kernel_size, strides, padding):
        super(EncoderBlock, self).__init__()
        self.conv3d = Conv3D(out_channels,
                             kernel_size,
                             strides=strides,
                             padding=padding,
                             use_bias=False)
        self.instnorm = GroupNormalization(groups=-1, axis=-1,
                                           center=False, scale=False)
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
        self.instnorm = GroupNormalization(groups=-1, axis=-1,
                                           center=False, scale=False)
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


class DeepN4:
    """
    This class is intended for the DeepN4 model.
    """

    @doctest_skip_parser
    def __init__(self, verbose=False):
        r"""

        To obtain the pre-trained model, use fetch_default_weights() like:
        >>> deepn4_model = DeepN4() # skip if not have_tf
        >>> deepn4_model.fetch_default_weights() # skip if not have_tf

        This model is designed to take as input file T1 signal and predict
        bias field. Effectively, this model is mimicking bias correction.

        Parameters
        ----------
        verbose : bool (optional)
            Whether to show information about the processing.
            Default: False

        References
        ----------
        Kanakaraj, P., Yao, T., Cai, L. Y., Lee, H. H., Newlin, N. R.,
        Kim, M. E., & Moyer, D. (2023). DeepN4: Learning N4ITK Bias Field
        Correction for T1-weighted Images.
        """

        if not have_tf:
            raise tf()

        log_level = 'INFO' if verbose else 'CRITICAL'
        set_logger_level(log_level, logger)

        # Synb0 network load

        self.model = UNet3D(input_shape=(128, 128, 128, 1))

    def fetch_default_weights(self):
        r"""
        Load the model pre-training weights to use for the fitting.
        """
        fetch_model_weights_path = get_fnames('deepn4_default_weights')
        self.load_model_weights(fetch_model_weights_path)

    def load_model_weights(self, weights_path):
        r"""
        Load the custom pre-training weights to use for the fitting.
        get_fnames('deepn4_default_weights').

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
        Predict bias field from input T1 signal

        Parameters
        ----------
        x_test : np.ndarray (128, 128, 128, 1)
            Image should match the required shape of the model.

        Returns
        -------
        np.ndarray (128, 128, 128)
            Predicted bias field
        """

        return self.model.predict(x_test)

    def pad(self, img, sz):

        tmp = np.zeros((sz, sz, sz))

        diff = int((sz - img.shape[0]) / 2)
        lx = max(diff, 0)
        lX = min(img.shape[0] + diff, sz)

        diff = (img.shape[0] - sz) / 2
        rx = max(int(np.floor(diff)), 0)
        rX = min(img.shape[0] - int(np.ceil(diff)), img.shape[0])

        diff = int((sz - img.shape[1]) / 2)
        ly = max(diff, 0)
        lY = min(img.shape[1] + diff, sz)

        diff = (img.shape[1] - sz) / 2
        ry = max(int(np.floor(diff)), 0)
        rY = min(img.shape[1] - int(np.ceil(diff)), img.shape[1])

        diff = int((sz - img.shape[2]) / 2)
        lz = max(diff, 0)
        lZ = min(img.shape[2] + diff, sz)

        diff = (img.shape[2] - sz) / 2
        rz = max(int(np.floor(diff)), 0)
        rZ = min(img.shape[2] - int(np.ceil(diff)), img.shape[2])

        tmp[lx:lX, ly:lY, lz:lZ] = img[rx:rX, ry:rY, rz:rZ]

        return tmp, [lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ]

    def load_resample(self, subj):

        input_data, [lx, lX, ly, lY, lz, lZ, rx, rX,
                     ry, rY, rz, rZ] = self.pad(subj, 128)
        in_max = np.percentile(input_data[np.nonzero(input_data)], 99.99)
        input_data = normalize(input_data, 0, in_max, 0, 1)
        input_data = np.squeeze(input_data)
        input_vols = np.zeros((1, 128, 128, 128, 1))
        input_vols[0, :, :, :, 0] = input_data

        return (tf.convert_to_tensor(input_vols, dtype=tf.float32),
                lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ, in_max)

    def predict(self, img, img_affine):
        """ Wrapper function to facilitate prediction of larger dataset.
        The function will mask, normalize, split, predict and 're-assemble'
        the data as a volume.

        Parameters
        ----------
        input_file : string
            Path to the T1 scan

        Returns
        -------
        final_corrected : np.ndarray (x, y, z)
            Predicted bias corrected image.
            The volume has matching shape to the input data

        """
        # Preprocess input data (resample, normalize, and pad)
        resampled_T1, affine2, ori_shape = transform_img(img, img_affine)
        (in_features, lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ,
         in_max) = self.load_resample(resampled_T1)

        # Run the model to get the bias field
        logfield = self.__predict(in_features)
        field = np.exp(logfield)
        field = field.squeeze()

        # Postprocess predicted field (reshape - unpad, smooth the field,
        # upsample)
        final_field = np.zeros([resampled_T1.shape[0],
                                resampled_T1.shape[1],
                                resampled_T1.shape[2]])
        final_field[rx:rX, ry:rY, rz:rZ] = field[lx:lX, ly:lY, lz:lZ]
        final_fields = gaussian_filter(final_field, sigma=3)
        upsample_final_field = recover_img(
            final_fields, affine2, ori_shape, np.shape(final_fields))

        # Correct the image
        THRESHOLD = 0.5
        below_threshold_mask = (np.abs(upsample_final_field) < THRESHOLD)
        with np.errstate(divide='ignore', invalid='ignore'):
            final_corrected = np.where(
                below_threshold_mask, 0, img / upsample_final_field)

        return final_corrected
