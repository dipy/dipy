#!/usr/bin/python
"""
Class and helper functions for fitting the EVAC+ model.
"""
import logging
import numpy as np

from dipy.data import get_fnames
from dipy.testing.decorators import doctest_skip_parser
from dipy.utils.optpkg import optional_package
from dipy.align.reslice import reslice
from dipy.nn.utils import normalize, set_logger_level, transform_img
from dipy.nn.utils import recover_img, correct_minor_errors

tf, have_tf, _ = optional_package('tensorflow', min_version='2.0.0')
if have_tf:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Softmax, Conv3DTranspose
    from tensorflow.keras.layers import Conv3D, LayerNormalization, ReLU
    from tensorflow.keras.layers import Concatenate, Layer, Dropout, Add
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
logger = logging.getLogger('EVAC+')

def prepare_img(image):
    r"""
    Function to prepare image for model input
    Specific to EVAC+

    Parameters
    ----------
    image : np.ndarray
        Input image

    Returns
    -------
    input_data : dict
    """
    input1 = np.moveaxis(image, -1, 0)
    input1 = np.expand_dims(input1, -1)

    input2, _ = reslice(image, np.eye(4),
                        (1, 1, 1), (2, 2, 2))
    input2 = np.moveaxis(input2, -1, 0)
    input2 = np.expand_dims(input2, -1)

    input3, _ = reslice(image, np.eye(4),
                        (1, 1, 1), (4, 4, 4))
    input3 = np.moveaxis(input3, -1, 0)
    input3 = np.expand_dims(input3, -1)

    input4, _ = reslice(image, np.eye(4),
                        (1, 1, 1), (8, 8, 8))
    input4 = np.moveaxis(input4, -1, 0)
    input4 = np.expand_dims(input4, -1)

    input5, _ = reslice(image, np.eye(4),
                        (1, 1, 1), (16, 16, 16))
    input5 = np.moveaxis(input5, -1, 0)
    input5 = np.expand_dims(input5, -1)

    input_data = {"input_1":input1,
                  "input_2":input2,
                  "input_3":input3,
                  "input_4":input4,
                  "input_5":input5}

    return input_data

class Block(Layer):
    def __init__(self, out_channels, kernel_size, strides,
                 padding, drop_r, n_layers, layer_type='down'):
        super(Block, self).__init__()
        self.layer_list = []
        self.layer_list2 = []
        self.n_layers = n_layers
        for _ in range(n_layers):
            self.layer_list.append(Conv3D(out_channels,
                                          kernel_size,
                                          strides=strides,
                                          padding=padding))
            self.layer_list.append(Dropout(drop_r))
            self.layer_list.append(LayerNormalization())
            self.layer_list.append(ReLU())
        if layer_type=='down':
            self.layer_list2.append(Conv3D(1, 2, strides=2, padding='same'))
            self.layer_list2.append(ReLU())
        elif layer_type=='up':
            self.layer_list2.append(Conv3DTranspose(1, 2, strides=2, padding='same'))
            self.layer_list2.append(ReLU())

        self.channel_sum = ChannelSum()
        self.add = Add()

    def call(self, input, passed):
        x = input
        for layer in self.layer_list:
            x = layer(x)

        x = self.channel_sum(x)
        fwd = self.add([x, passed])
        x = fwd

        for layer in self.layer_list2:
            x = layer(x)

        return fwd, x

class ChannelSum(Layer):
    def __init__(self):
        super(ChannelSum, self).__init__()

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=-1, keepdims=True)

def init_model(model_scale=16):
    r"""
    Function to create model for EVAC+

    Parameters
    ----------
    model_scale : int, optional
        The scale of the model
        Should match the saved weights from fetcher
        Default is 16

    Returns
    -------
    model : tf.keras.Model
    """
    inputs = tf.keras.Input(shape=(128, 128, 128, 1))
    raw_input_2 = tf.keras.Input(shape=(64, 64, 64, 1))
    raw_input_3 = tf.keras.Input(shape=(32, 32, 32, 1))
    raw_input_4 = tf.keras.Input(shape=(16, 16, 16, 1))
    raw_input_5 = tf.keras.Input(shape=(8, 8, 8, 1))
    # Encode
    fwd1, x = Block(model_scale, kernel_size=5,
                   strides=1, padding='same',
                   drop_r=0.2, n_layers=1)(inputs, inputs)

    x = Concatenate()([x, raw_input_2])

    fwd2, x = Block(model_scale*2, kernel_size=5,
                    strides=1, padding='same',
                    drop_r=0.5, n_layers=2)(x, x)

    x = Concatenate()([x, raw_input_3])

    fwd3, x = Block(model_scale*4, kernel_size=5,
                    strides=1, padding='same',
                    drop_r=0.5, n_layers=3)(x, x)

    x = Concatenate()([x, raw_input_4])

    fwd4, x = Block(model_scale*8, kernel_size=5,
                    strides=1, padding='same',
                    drop_r=0.5, n_layers=3)(x, x)

    x = Concatenate()([x, raw_input_5])

    _, up = Block(model_scale*16, kernel_size=5,
                  strides=1, padding='same',
                  drop_r=0.5, n_layers=3,
                  layer_type='up')(x, x)

    x = Concatenate()([fwd4, up])

    _, up = Block(model_scale*8, kernel_size=5,
                  strides=1, padding='same',
                  drop_r=0.5, n_layers=3,
                  layer_type='up')(x, up)

    x = Concatenate()([fwd3, up])

    _, up = Block(model_scale*4, kernel_size=5,
                  strides=1, padding='same',
                  drop_r=0.5, n_layers=3,
                  layer_type='up')(x, up)

    x = Concatenate()([fwd2, up])

    _, up = Block(model_scale*2, kernel_size=5,
                  strides=1, padding='same',
                  drop_r=0.5, n_layers=2,
                  layer_type='up')(x, up)

    x = Concatenate()([fwd1, up])

    _, pred = Block(model_scale, kernel_size=5,
                    strides=1, padding='same',
                    drop_r=0.5, n_layers=1,
                    layer_type='none')(x, up)

    pred = Conv3D(2, 1, padding='same')(pred)
    output = Softmax(axis=-1)(pred)

    model = Model({"input_1":inputs,
                   "input_2":raw_input_2,
                   "input_3":raw_input_3,
                   "input_4":raw_input_4,
                   "input_5":raw_input_5},
                  output[..., 0])
    return model


class EVACPlus:
    """
    This class is intended for the EVAC+ model.
    """

    @doctest_skip_parser
    def __init__(self, verbose=False):
        r"""
        The model was pre-trained for usage on
        brain extraction of T1 images.

        This model is designed to take as input
        a T1 weighted image.

        Parameters
        ----------
        verbose : bool (optional)
            Whether to show information about the processing.
            Default: False

        References
        ----------
        ..  [1] Park, J.S., Fadnavis, S., & Garyfallidis, E. (2022).
                EVAC+: Multi-scale V-net with Deep Feature
                CRF Layers for Brain Extraction.
        """

        if not have_tf:
            raise tf()

        log_level = 'INFO' if verbose else 'CRITICAL'
        set_logger_level(log_level, logger)

        # EVAC+ network load

        self.model = init_model()
        self.fetch_default_weights()

    def fetch_default_weights(self):
        r"""
        Load the model pre-training weights to use for the fitting.
        While the user can load different weights, the function
        is mainly intended for the class function 'predict'.
        """
        fetch_model_weights_path = get_fnames('evac_default_weights')
        print('fetched ' + fetch_model_weights_path)
        self.load_model_weights(fetch_model_weights_path)

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
        x_test : np.ndarray (batch, 128, 128, 128, 1)
            Image should match the required shape of the model.

        Returns
        -------
        np.ndarray (batch, ...)
            Predicted brain mask
        """

        return self.model.predict(x_test)

    def predict(self, T1, affine,
                voxsize=(1, 1, 1), batch_size=None,
                return_affine=False, return_prob=False,
                largest_area=True):
        r"""
        Wrapper function to facilitate prediction of larger dataset.

        Parameters
        ----------
        T1 : np.ndarray or list of np.ndarrys
            For a single image, input should be a 3D array.
            If multiple images, it should be a a list or tuple.

        affine : np.ndarray (4, 4) or (batch, 4, 4)
            or list of np.ndarrays with len of batch
            Affine matrix for the T1 image. Should have
            batch dimension if T1 has one.

        voxsize : np.ndarray or list or tuple, optional
            (3,) or (batch, 3)
            voxel size of the T1 image.
            Default is (1, 1, 1)

        batch_size : int, optional
            Number of images per prediction pass. Only available if data
            is provided with a batch dimension.
            Consider lowering it if you get an out of memory error.
            Increase it if you want it to be faster and have a lot of data.
            If None, batch_size will be set to 1 if the provided image
            has a batch dimension.
            Default is None

        return_affine : bool, optional
            Whether to return the affine matrix. Useful if the input was a
            file path.
            Default is False

        return_prob : bool, optional
            Whether to return the probability map instead of a
            binary mask. Useful for testing.
            Default is False

        largest_area : bool, optional
            Whether to exclude only the largest background/foreground.
            Useful for solving minor errors.
            Default is True

        Returns
        -------
        pred_output : np.ndarray (...) or (batch, ...)
            Predicted brain mask

        affine : np.ndarray (...) or (batch, ...)
            affine matrix of mask
            only if return_affine is True

        """

        voxsize = np.array(voxsize)
        affine = np.array(affine)

        if isinstance(T1, (list, tuple)):
            dim = 4
            T1 = np.array(T1)
        elif len(T1.shape) == 3:
            dim = 3
            if batch_size is not None:
                logger.warning('Batch size specified, but not used',
                               'due to the input not having \
                                a batch dimension')
            batch_size = 1

            T1 = np.expand_dims(T1, 0)
            affine = np.expand_dims(affine, 0)
            voxsize = np.expand_dims(voxsize, 0)
        else:
            raise ValueError("T1 data should be a np.ndarray of dimension 3 "
                             "or a list/tuple of it")

        input_data = np.zeros((128, 128, 128, len(T1)))
        rev_affine = np.zeros((len(T1), 4, 4))
        ori_shapes = np.zeros((len(T1), 3)).astype(int)

        # Normalize the data.
        n_T1 = np.zeros(T1.shape)
        for i, T1_img in enumerate(T1):
            n_T1[i] = normalize(T1_img, new_min=0, new_max=1)
            t_img, t_affine, ori_shape = transform_img(n_T1[i],
                                                       affine[i],
                                                       voxsize[i])
            input_data[..., i] = t_img
            rev_affine[i] = t_affine
            ori_shapes[i] = ori_shape

        # Prediction stage
        prediction = np.zeros((len(T1), 128, 128, 128),
                              dtype=np.float32)
        for batch_idx in range(batch_size, len(T1)+1, batch_size):
            batch = input_data[..., batch_idx-batch_size:batch_idx]
            temp_input = prepare_img(batch)
            temp_pred = self.__predict(temp_input)
            prediction[:batch_idx] = temp_pred
        remainder = np.mod(len(T1), batch_size)
        if remainder != 0:
            temp_input = prepare_img(input_data[..., -remainder:])
            temp_pred = self.__predict(temp_input)
            prediction[-remainder:] = temp_pred

        output_mask = []
        for i, T1_img in enumerate(T1):
            output = recover_img(prediction[i],
                                 rev_affine[i],
                                 voxsize=voxsize[i],
                                 ori_shape=ori_shapes[i],
                                 image_shape=T1_img.shape)
            if not return_prob:
                output = np.where(output >= 0.5, 1, 0)
                if largest_area:
                    output = correct_minor_errors(output)
            output_mask.append(output)

        if dim == 3:
            output_mask = output_mask[0]
            affine = affine[0]

        output_mask = np.array(output_mask)
        affine = np.array(affine)
        if return_affine:
            return output_mask, affine
        else:
            return output_mask
