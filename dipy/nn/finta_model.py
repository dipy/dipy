from math import sqrt
import numpy as np
import nibabel as nib
import tensorflow as tf
import keras
from keras import layers, Layer, Model, initializers
from tractoencoder_gsoc.utils import pre_pad
from tractoencoder_gsoc.utils import dict_kernel_size_flatten_encoder_shape


class Encoder(Layer):
    def __init__(self, latent_space_dims=32, kernel_size=3, **kwargs):
        """
        Encoder block of the AutoEncoder.
        Encodes the input data into a latent
        space representation.

        Parameters
        ----------
        latent_space_dims : int, optional
            Number of dimensions of the latent
            space where the data will be encoded.
            Default: 32
        kernel_size : int, optional
            Length of the 1D kernel used in the
            convolutional layers.
            Default: 3
        """
        super(Encoder, self).__init__(**kwargs)

        # Parameter Initialization
        self.latent_space_dims = latent_space_dims
        self.kernel_size = kernel_size

        # Weight and bias initializers for Conv1D layers (matching PyTorch initialization)
        # Link: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html (Variables section)
        # Weights
        self.k_conv1d_weights_initializer = sqrt(1 / (3 * self.kernel_size))
        self.conv1d_weights_initializer = initializers.RandomUniform(minval=-self.k_conv1d_weights_initializer,
                                                                     maxval=self.k_conv1d_weights_initializer,
                                                                     seed=2208)
        # Biases
        self.k_conv1d_biases_initializer = self.k_conv1d_weights_initializer
        self.conv1d_biases_initializer = self.conv1d_weights_initializer

        self.encod_conv1 = pre_pad(
            layers.Conv1D(32, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv1",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv2 = pre_pad(
            layers.Conv1D(64, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv2",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv3 = pre_pad(
            layers.Conv1D(128, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv3",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv4 = pre_pad(
            layers.Conv1D(256, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv4",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv5 = pre_pad(
            layers.Conv1D(512, self.kernel_size, strides=2, padding='valid',
                          name="encoder_conv5",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )
        self.encod_conv6 = pre_pad(
            layers.Conv1D(1024, self.kernel_size, strides=1, padding='valid',
                          name="encoder_conv6",
                          kernel_initializer=self.conv1d_weights_initializer,
                          bias_initializer=self.conv1d_biases_initializer)
        )

        self.flatten = layers.Flatten(name='flatten')

        # For Dense layers
        # Link: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html (Variables section)
        # Weights
        self.k_dense_weights_initializer = sqrt(1 / dict_kernel_size_flatten_encoder_shape[self.kernel_size])
        self.dense_weights_initializer = initializers.RandomUniform(minval=-self.k_dense_weights_initializer,
                                                                    maxval=self.k_dense_weights_initializer,
                                                                    seed=2208)
        # Biases
        self.k_dense_biases_initializer = self.k_dense_weights_initializer
        self.dense_biases_initializer = self.dense_weights_initializer

        self.fc1 = layers.Dense(self.latent_space_dims, name='fc1',
                                kernel_initializer=self.dense_weights_initializer,
                                bias_initializer=self.dense_biases_initializer)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_space_dims": keras.saving.serialize_keras_object(self.latent_space_dims),
            "kernel_size": keras.saving.serialize_keras_object(self.kernel_size)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        latent_space_dims = keras.saving.deserialize_keras_object(config.pop('latent_space_dims'))
        kernel_size = keras.saving.deserialize_keras_object(config.pop('kernel_size'))
        return cls(latent_space_dims, kernel_size, **config)

    def call(self, input_data):
        """
        Run the input data through the Encoder.
        Encode the input_data into a latent
        space representation.

        Parameters
        ----------
        input_data : tf.Tensor or np.ndarray
            Data to run through the Encoder.
            Should be of the shape of a standard
            streamline with 256 points:
            (n_samples, 256, 3).

        Returns
        -------
        tf.Tensor
            Encoded input in the latent space,
            with the shape
            (n_samples, latent_space_dims).
        """
        x = input_data

        h1 = tf.nn.relu(self.encod_conv1(x))
        h2 = tf.nn.relu(self.encod_conv2(h1))
        h3 = tf.nn.relu(self.encod_conv3(h2))
        h4 = tf.nn.relu(self.encod_conv4(h3))
        h5 = tf.nn.relu(self.encod_conv5(h4))
        h6 = self.encod_conv6(h5)

        self.encoder_out_size = h6.shape[1:]

        # Flatten
        # First transpose the tensor to match the PyTorch implementation so the flattening is equal
        h7 = tf.transpose(h6, perm=[0, 2, 1])
        h7 = self.flatten(h7)
        fc1 = self.fc1(h7)

        return fc1

class Decoder(Layer):
    def __init__(self, encoder_out_size, kernel_size=3, **kwargs):
        """
        Decoder block of the AutoEncoder. Decodes
        the latent space representation to the original
        data space. The encoder_out_size parameter is
        used to reshape the output of the fully connected
        layer to match the size of the convolutional
        output of the Encoder block.

        Parameters
        ----------
        encoder_out_size : tuple
            Size of the convolutional output of the
            Encoder block.
        kernel_size : int, optional
            Length of the 1D kernel used in the
            convolutional layers.
            Default: 3
        """
        super(Decoder, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.encoder_out_size = encoder_out_size

        self.fc2 = layers.Dense(8192, name="fc2")
        self.decod_conv1 = pre_pad(
            layers.Conv1D(512, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv1")
        )
        self.upsampl1 = layers.UpSampling1D(size=2, name="upsampling1")
        self.decod_conv2 = pre_pad(
            layers.Conv1D(256, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv2")
        )
        self.upsampl2 = layers.UpSampling1D(size=2, name="upsampling2")
        self.decod_conv3 = pre_pad(
            layers.Conv1D(128, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv3")
        )
        self.upsampl3 = layers.UpSampling1D(size=2, name="upsampling3")
        self.decod_conv4 = pre_pad(
            layers.Conv1D(64, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv4")
        )
        self.upsampl4 = layers.UpSampling1D(size=2, name="upsampling4")
        self.decod_conv5 = pre_pad(
            layers.Conv1D(32, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv5")
        )
        self.upsampl5 = layers.UpSampling1D(size=2, name="upsampling5")
        self.decod_conv6 = pre_pad(
            layers.Conv1D(3, self.kernel_size, strides=1, padding='valid',
                          name="decoder_conv6")
        )

    def get_config(self):
        base_config = super().get_config()
        config = {
            "encoder_out_size": keras.saving.serialize_keras_object(self.encoder_out_size),
            "kernel_size": keras.saving.serialize_keras_object(self.kernel_size)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        encoder_out_size = keras.saving.deserialize_keras_object(config.pop('encoder_out_size'))
        kernel_size = keras.saving.deserialize_keras_object(config.pop('kernel_size'))
        return cls(encoder_out_size, kernel_size, **config)

    def call(self, input_data):
        """
        Run the input data through the Decoder.
        Decode the latent space representation
        to the original data space.

        Parameters
        ----------
        input_data : tf.Tensor or np.ndarray
            Data to run through the decoder.
            Should be of the shape of the latent
            space: (n_samples, latent_space_dims).

        Returns
        -------
        tf.Tensor
            Decoded input in the original
            data space, with the shape
            (n_samples, 256, 3).
        """
        z = input_data
        fc = self.fc2(z)

        # Reshape to match encoder output size
        fc_reshape = tf.reshape(fc, (-1, self.encoder_out_size[0],
                                     self.encoder_out_size[1]))

        h1 = tf.nn.relu(self.decod_conv1(fc_reshape))
        h2 = self.upsampl1(h1)
        h3 = tf.nn.relu(self.decod_conv2(h2))
        h4 = self.upsampl2(h3)
        h5 = tf.nn.relu(self.decod_conv3(h4))
        h6 = self.upsampl3(h5)
        h7 = tf.nn.relu(self.decod_conv4(h6))
        h8 = self.upsampl4(h7)
        h9 = tf.nn.relu(self.decod_conv5(h8))
        h10 = self.upsampl5(h9)
        h11 = self.decod_conv6(h10)

        return h11


def init_model(latent_space_dims=32, kernel_size=3):
    r"""
    Initialize the model with the given latent
    space dimensions and kernel size.
    Instantiates the Encoder, the Decoder, and
    returns the model comprising both blocks.

    Parameters
    ----------
    latent_space_dims : int, optional
        Number of dimensions of the latent
        space where the data will be encoded
        Default: 32
    kernel_size : int, optional
        Length of the 1D kernel used in the
        convolutional layers of the Encoder
        and the Decoder
        Default: 3

    Returns
    -------
    keras.Model
        AutoEncoder model comprising an
        Encoder and a Decoder.
    """
    input_data = keras.Input(shape=(256, 3), name='input_streamline')

    # encode
    encoder = Encoder(latent_space_dims=latent_space_dims,
                      kernel_size=kernel_size)
    encoded = encoder(input_data)

    # decode
    decoder = Decoder(encoder.encoder_out_size,
                      kernel_size=kernel_size)
    decoded = decoder(encoded)
    output_data = decoded

    # Instantiate model and name it
    model = Model(input_data, output_data)
    model.name = 'IncrFeatStridedConvFCUpsampReflectPadAE'
    return model


class IncrFeatStridedConvFCUpsampReflectPadAE():
    """
    Strided convolution-upsampling-based
    AutoEncoder using reflection-padding and
    increasing feature maps in decoder.
    Architecture based on Jon Haitz Legarreta Gorroño's
    PyTorch implementation, from his FINTA [1] work.
    [1] https://doi.org/10.1016/j.media.2021.102126
    """

    def __init__(self, latent_space_dims=32, kernel_size=3):

        # Parameter Initialization
        self.kernel_size = kernel_size
        self.latent_space_dims = latent_space_dims
        self.input = keras.Input(shape=(256, 3), name='input_streamline')

        self.model = init_model(latent_space_dims=self.latent_space_dims,
                                kernel_size=self.kernel_size)

    def __call__(self, x):
        r"""
        Run the input streamlines 'x' through
        the model (encoder -> decoder)

        Parameters
        ----------
        x : tf.Tensor, np.ndarray
            Tensor of shape (n_samples, 256, 3)
            containing the input streamlines.

        Returns
        -------
        tf.Tensor
            Reconstruction of input streamlines
            after passing through the model.
        """
        return self.model(x)

    def compile(self, **kwargs):
        r"""
        Wrapper of the built in compile method
        of the model, which configures the model
        for training. Sets the optimizer weight
        decay to the same value as in the PyTorch
        implementation (FINTA [1]).
        [1] https://doi.org/10.1016/j.media.2021.102126
        """
        kwargs['optimizer'].weight_decay = 0.13
        self.model.compile(**kwargs)

    def summary(self, **kwargs):
        r"""
        Get the summary of the model. The summary
        is textual and includes information about:
        - The layers and their order in the model.
        - The output shape of each layer.
        """
        return self.model.summary(**kwargs)

    def fit(self, *args, **kwargs,):
        r"""
        Wrapper of the built in fit method of the
        model, which trains the model for a fixed
        number of epochs. In case of inputing
        streamlines as input, converts them to
        numpy arrays.
        """
        if isinstance(kwargs['x'], nib.streamlines.ArraySequence):
            kwargs['x'] = np.array(kwargs['x'])
        if isinstance(kwargs['y'], nib.streamlines.ArraySequence):
            kwargs['y'] = np.array(kwargs['y'])
        return self.model.fit(*args, **kwargs)

    def save_weights(self, *args, **kwargs):
        """
        Wrapper of the built in method to save
        the weights of the model in the specified
        filepath, which must end
        in '*.weights.h5'.
        """
        self.model.save_weights(*args, **kwargs)

    def save(self, *args, **kwargs):
        """
        Wrapper of the built in method to save
        the model together with its weights in
        the specified filepath, which must end
        in '*.keras'.
        """
        self.model.save(*args, **kwargs)
