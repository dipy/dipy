import logging
from math import sqrt

import nibabel as nib
import numpy as np

from dipy.nn.utils import dict_kernel_encoder_shape, pre_pad
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.0.0")
if have_tf:
    import keras
    from keras import Model, layers
    from keras.initializers import RandomUniform
    from keras.layers import Layer
    from keras.saving import deserialize_keras_object, serialize_keras_object
    import tensorflow as tf
else:
    have_tf = False

    class Model:
        pass

    class Layer:
        pass

    logging.warning(
        "This model requires Tensorflow. Please install these packages "
        "using pip. If using mac, please refer to this link for "
        " installation https://github.com/apple/tensorflow_macos."
    )

logging.basicConfig()


class Encoder(Layer):
    def __init__(self, latent_space_dims=32, kernel_size=3, **kwargs):
        """Encoder block of the AutoEncoder.

        Encodes the input data into a latent space representation.
        It is composed of 6 1D convolutional layers with strides of 2 and 1 fully
        connected layer that takes the convolutional output and outputs the latent space
        representation. The architecture has been tested majorly with a kernel size of 3
        and 32 latent space dimensions, but it can be modified to fit other needs.
        The detailed architecture is summarized as follows:
        - Input: [n_samples, 256, 3]
        - 1D Conv (32, kernel_size, strides=2) -> ReLU
        - 1D Conv (64, kernel_size, strides=2) -> ReLU
        - 1D Conv (128, kernel_size, strides=2) -> ReLU
        - 1D Conv (256, kernel_size, strides=2) -> ReLU
        - 1D Conv (512, kernel_size, strides=2) -> ReLU
        - 1D Conv (1024, kernel_size, strides=1) -> ReLU
        - Flatten Operation
        - Dense Layer (latent_space_dims) -> No activation function
        - Output: [n_samples, latent_space_dims]
        The architecture is based on :footcite:t:`Legarreta_2021`.

        Parameters
        ----------
        latent_space_dims : int, optional
            Space where the data will be encoded.
        kernel_size : int, optional
            Length of the 1D kernel used in the convolutional layers.

        References
        ----------
        .. [1] Legarreta JH, Petit L, Rheault F, Theaud G, Lemaire C, Descoteaux M, et
        al. Filtering in tractography using autoencoders (FINTA). Med Image Anal. 2021
        Aug;72:102126.
        :footbibliography::
        """

        super(Encoder, self).__init__(**kwargs)

        if kernel_size not in [1, 2, 3, 4, 5]:
            raise ValueError("Kernel size must be between 1 and 5.")
        # Parameter Initialization
        self.latent_space_dims = latent_space_dims
        self.kernel_size = kernel_size

        # Weight and bias initializers for Conv1D layers
        # (matching PyTorch initialization)
        # Link: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        # (Variables section)
        # Weights
        self.k_conv1d_weight_init = sqrt(1 / (3 * self.kernel_size))
        self.conv1d_weight_init = RandomUniform(
            minval=-self.k_conv1d_weight_init,
            maxval=self.k_conv1d_weight_init,
            seed=2208,
        )
        # Biases
        self.k_conv1d_bias_init = self.k_conv1d_weight_init
        self.conv1d_bias_init = self.conv1d_weight_init

        self.encod_conv1 = pre_pad(
            layers.Conv1D(
                32,
                self.kernel_size,
                strides=2,
                padding="valid",
                name="encoder_conv1",
                kernel_initializer=self.conv1d_weight_init,
                bias_initializer=self.conv1d_bias_init,
            )
        )
        self.encod_conv2 = pre_pad(
            layers.Conv1D(
                64,
                self.kernel_size,
                strides=2,
                padding="valid",
                name="encoder_conv2",
                kernel_initializer=self.conv1d_weight_init,
                bias_initializer=self.conv1d_bias_init,
            )
        )
        self.encod_conv3 = pre_pad(
            layers.Conv1D(
                128,
                self.kernel_size,
                strides=2,
                padding="valid",
                name="encoder_conv3",
                kernel_initializer=self.conv1d_weight_init,
                bias_initializer=self.conv1d_bias_init,
            )
        )
        self.encod_conv4 = pre_pad(
            layers.Conv1D(
                256,
                self.kernel_size,
                strides=2,
                padding="valid",
                name="encoder_conv4",
                kernel_initializer=self.conv1d_weight_init,
                bias_initializer=self.conv1d_bias_init,
            )
        )
        self.encod_conv5 = pre_pad(
            layers.Conv1D(
                512,
                self.kernel_size,
                strides=2,
                padding="valid",
                name="encoder_conv5",
                kernel_initializer=self.conv1d_weight_init,
                bias_initializer=self.conv1d_bias_init,
            )
        )
        self.encod_conv6 = pre_pad(
            layers.Conv1D(
                1024,
                self.kernel_size,
                strides=1,
                padding="valid",
                name="encoder_conv6",
                kernel_initializer=self.conv1d_weight_init,
                bias_initializer=self.conv1d_bias_init,
            )
        )

        self.flatten = layers.Flatten(name="flatten")

        # For Dense layers
        # Link: https://pytorch.org/docs/stable/generated/torch.nn.Linear.html
        # (Variables section)
        # Weights
        self.k_dense_weight_init = sqrt(1 / dict_kernel_encoder_shape[self.kernel_size])
        self.dense_weight_init = RandomUniform(
            minval=-self.k_dense_weight_init, maxval=self.k_dense_weight_init, seed=2208
        )

        # Biases
        self.k_dense_bias_init = self.k_dense_weight_init
        self.dense_bias_init = self.dense_weight_init

        self.fc1 = layers.Dense(
            self.latent_space_dims,
            name="fc1",
            kernel_initializer=self.dense_weight_init,
            bias_initializer=self.dense_bias_init,
        )

    def get_config(self):
        """Serialize the custom objects of the layer, retrieve the configuration of a
        layer or model.

        Necessary for saving and loading the model, included due to the use of
        ``ReflectionPadding1D`` custom Layer inside ``pre_pad``.

        Returns
        -------
        dict
            Configuration of the layer
        """

        base_config = super().get_config()
        config = {
            "latent_space_dims": serialize_keras_object(self.latent_space_dims),
            "kernel_size": serialize_keras_object(self.kernel_size),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Deserialize the configuration of the layer when custom objects are found.

        Recreate a model instance from its config. Used because of the custom Layer
        ``ReflectionPadding1D``.

        Parameters
        ----------
        config : dict
            Configuration of the model

        Returns
        -------
        keras.layers.Layer
            Layer instance recreated from the retrieved configuration
        """

        latent_space_dims = deserialize_keras_object(config.pop("latent_space_dims"))
        kernel_size = deserialize_keras_object(config.pop("kernel_size"))
        return cls(latent_space_dims, kernel_size, **config)

    def call(self, input_data):
        """Run the input data through the Encoder.

        Encode the input_data into a latent space representation.

        Parameters
        ----------
        input_data : tf.Tensor or np.ndarray
            Data to run through the Encoder. Should be of the shape of a standard
            streamline with 256 points: (n_samples, 256, 3).

        Returns
        -------
        tf.Tensor
            Encoded input in the latent space, with the shape
            (n_samples, latent_space_dims).
        """

        # Check if input_data has 256 points
        if input_data.shape[1] != 256:
            raise ValueError("Input streamlines in Encoder must have 256 points.")
        x = input_data

        h1 = tf.nn.relu(self.encod_conv1(x))
        h2 = tf.nn.relu(self.encod_conv2(h1))
        h3 = tf.nn.relu(self.encod_conv3(h2))
        h4 = tf.nn.relu(self.encod_conv4(h3))
        h5 = tf.nn.relu(self.encod_conv5(h4))
        h6 = self.encod_conv6(h5)

        self.encoder_out_size = h6.shape[1:]

        # Flatten
        # Transpose tensor to match PyTorch implementation so the flattening is equal
        h7 = tf.transpose(h6, perm=[0, 2, 1])
        h7 = self.flatten(h7)
        fc1 = self.fc1(h7)

        return fc1


class Decoder(Layer):
    def __init__(self, encoder_out_size, kernel_size=3, **kwargs):
        """Decoder block of the AutoEncoder.

        Decode the latent space representation to the original data space. The
        encoder_out_size parameter is used to reshape the output of the fully connected
        layer to match the size of the convolutional output of the Encoder block.
        Encodes the input data into a latent space representation.
        It is composed of 6 1D convolutional layers with strides of 1. The architecture
        has been tested majorly with a kernel size of 3 and an encoder output size of
        [8, 1024], corresponding to an Encoder of 32 latent space dimensions, but it can
        be modified to fit other needs.
        The detailed architecture is summarized as follows:
        - Input: [n_samples, latent_space_dims]
        - Dense Layer (8192) -> Reshape to match encoder convolutional output size
        - 1D Conv (512, kernel_size, strides=1) -> ReLU -> Upsampling (2)
        - 1D Conv (256, kernel_size, strides=1) -> ReLU -> Upsampling (2)
        - 1D Conv (128, kernel_size, strides=1) -> ReLU -> Upsampling (2)
        - 1D Conv (64, kernel_size, strides=1) -> ReLU -> Upsampling (2)
        - 1D Conv (32, kernel_size, strides=1) -> ReLU -> Upsampling (2)
        - 1D Conv (3, kernel_size, strides=1)
        - Output: [n_samples, 256, 3]
        The architecture is based on :footcite:t:`Legarreta_2021`.

        Parameters
        ----------
        encoder_out_size : tuple
            Size of the convolutional output of the Encoder block.
        kernel_size : int, optional
            Length of the 1D kernel used in the convolutional layers.

        References
        ----------
        .. [1] Legarreta JH, Petit L, Rheault F, Theaud G, Lemaire C, Descoteaux M, et
        al. Filtering in tractography using autoencoders (FINTA). Med Image Anal. 2021
        Aug;72:102126.
        :footbibliography::
        """

        super(Decoder, self).__init__(**kwargs)

        if kernel_size not in [1, 2, 3, 4, 5]:
            raise ValueError("Kernel size must be between 1 and 5.")

        self.kernel_size = kernel_size
        self.encoder_out_size = encoder_out_size

        self.fc2 = layers.Dense(8192, name="fc2")
        self.decod_conv1 = pre_pad(
            layers.Conv1D(
                512, self.kernel_size, strides=1, padding="valid", name="decoder_conv1"
            )
        )
        self.upsampl1 = layers.UpSampling1D(size=2, name="upsampling1")
        self.decod_conv2 = pre_pad(
            layers.Conv1D(
                256, self.kernel_size, strides=1, padding="valid", name="decoder_conv2"
            )
        )
        self.upsampl2 = layers.UpSampling1D(size=2, name="upsampling2")
        self.decod_conv3 = pre_pad(
            layers.Conv1D(
                128, self.kernel_size, strides=1, padding="valid", name="decoder_conv3"
            )
        )
        self.upsampl3 = layers.UpSampling1D(size=2, name="upsampling3")
        self.decod_conv4 = pre_pad(
            layers.Conv1D(
                64, self.kernel_size, strides=1, padding="valid", name="decoder_conv4"
            )
        )
        self.upsampl4 = layers.UpSampling1D(size=2, name="upsampling4")
        self.decod_conv5 = pre_pad(
            layers.Conv1D(
                32, self.kernel_size, strides=1, padding="valid", name="decoder_conv5"
            )
        )
        self.upsampl5 = layers.UpSampling1D(size=2, name="upsampling5")
        self.decod_conv6 = pre_pad(
            layers.Conv1D(
                3, self.kernel_size, strides=1, padding="valid", name="decoder_conv6"
            )
        )

    def get_config(self):
        """Serialize the custom objects of the layer.

        Necessary for saving and loading the model, included due to the use of
        ``pre_pad``, which is a custom Layer.

        Returns
        -------
        dict
            dictionary with the configuration of the layer
        """
        base_config = super().get_config()
        config = {
            "encoder_out_size": serialize_keras_object(self.encoder_out_size),
            "kernel_size": serialize_keras_object(self.kernel_size),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Deserialize the configuration of the layer when custom objects are found

        Recreates a model instance from its config. Used because of the custom Layer
        ``ReflectionPadding1D``.

        Parameters
        ----------
        config : dict
            Configuration of the model

        Returns
        -------
        keras.layers.Layer
            Layer instance recreated from the retrieved configuration
        """

        encoder_out_size = deserialize_keras_object(config.pop("encoder_out_size"))
        kernel_size = deserialize_keras_object(config.pop("kernel_size"))
        return cls(encoder_out_size, kernel_size, **config)

    def call(self, input_data):
        """Run the input data through the Decoder.

        Decode the latent space representation to the original data space.

        Parameters
        ----------
        input_data : tf.Tensor or np.ndarray
            Data to run through the decoder. Should be of the shape of the latent space:
            (n_samples, latent_space_dims).

        Returns
        -------
        tf.Tensor
            Decoded input in the original data space, with the shape
            (n_samples, 256, 3).
        """
        z = input_data
        fc = self.fc2(z)

        # Reshape to match encoder output size
        fc_reshape = tf.reshape(
            fc, (-1, self.encoder_out_size[0], self.encoder_out_size[1])
        )

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
    """Initialize the model with the given latent space dimensions and kernel size.

    Instantiate the Encoder, the Decoder, and return the model comprising both blocks.

    Parameters
    ----------
    latent_space_dims : int, optional
        Number of dimensions of the latent space where the data will be encoded.
    kernel_size : int, optional
        Length of the 1D kernel used in the convolutional layers of the Encoder and the
        Decoder.

    Returns
    -------
    keras.Model
        Encoder model
    keras.Model
        Decoder model
    tuple
        Size of the convolutional output of the Encoder block.
    """

    input_data = keras.Input(shape=(256, 3), name="input_streamline")

    # encode
    encoder = Encoder(latent_space_dims=latent_space_dims, kernel_size=kernel_size)
    encoded = encoder(input_data)
    # Instantiate encoder model
    model_encoder = Model(input_data, encoded, name="Encoder")

    # decode
    decoder = Decoder(encoder.encoder_out_size, kernel_size=kernel_size)
    decoded = decoder(encoded)

    # Instantiate decoder model
    model_decoder = Model(encoded, decoded, name="Decoder")

    # Instantiate model and name it
    return model_encoder, model_decoder, encoder.encoder_out_size


class IncrFeatStridedConvFCUpsampReflectPadAE(Model):
    def __init__(self, latent_space_dims=32, kernel_size=3, **kwargs):
        """Strided convolution-upsampling-based AutoEncoder using reflection-padding and
        increasing feature maps in decoder.

        Unsupervised AutoEncoder class based on the PyTorch implementation from FINTA
        :footcite:t:`Legarreta_2021`. The model is composed of an Encoder and a
        Decoder, included in the ``Encoder`` and ``Decoder`` classes.

        Parameters
        ----------
        latent_space_dims : int, optional
            Number of dimensions of the latent space where the data will be encoded.
        kernel_size : int, optional
            Length of the 1D kernel used in the convolutional layers of the Encoder and
            the Decoder.

        References
        ----------
        .. [1] Legarreta JH, Petit L, Rheault F, Theaud G, Lemaire C, Descoteaux M, et
        al. Filtering in tractography using autoencoders (FINTA). Med Image Anal. 2021
        Aug;72:102126.
        :footbibliography::
        """
        super(IncrFeatStridedConvFCUpsampReflectPadAE, self).__init__(**kwargs)

        # Parameter Initialization
        self.kernel_size = kernel_size
        self.latent_space_dims = latent_space_dims

        model = init_model(
            latent_space_dims=self.latent_space_dims, kernel_size=self.kernel_size
        )
        self.encoder = model[0]
        self.decoder = model[1]
        self.encoder_out_size = model[2]

        # Model name is: "incr_feat_strided_conv_fc_upsamp_reflect_pad_ae"

    def get_config(self):
        """Serialize the custom objects of the layer.

        Necessary for saving and loading the model, included due to the use of the
        ``Encoder`` and ``Decoder`` layers, which are custom Layer objects.

        Returns
        -------
        dict
            dictionary with the configuration of the layer
        """

        base_config = super().get_config()
        config = {
            "latent_space_dims": serialize_keras_object(self.latent_space_dims),
            "kernel_size": serialize_keras_object(self.kernel_size),
            "encoder_out_size": serialize_keras_object(self.encoder_out_size),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls, config):
        """Deserialize the configuration of the Model when custom objects are found.

        Recreate a model instance from its config. Used because of the custom
        ``Encoder`` and ``Decoder`` layers.

        Parameters
        ----------
        config : dict
            Configuration of the model

        Returns
        -------
        keras.Model
            Model instance recreated from the retrieved configuration
        """
        latent_space_dims = deserialize_keras_object(config.pop("latent_space_dims"))
        # encoder_out_size = deserialize_keras_object(config.pop("encoder_out_size"))
        kernel_size = deserialize_keras_object(config.pop("kernel_size"))
        return cls(latent_space_dims, kernel_size, **config)

    def call(self, x):
        """Run the input streamlines ``x`` through the model (encoder -> decoder)

        Parameters
        ----------
        x : tf.Tensor, np.ndarray
            Tensor of shape (n_samples, 256, 3) containing the input streamlines.

        Returns
        -------
        tf.Tensor
            Reconstruction of input streamlines after passing through the model.
        """

        # Check that the input streamlines have 256 points
        if not x.shape[1] == 256:
            raise ValueError(
                "Input streamlines to AutoEncoder must have " "256 points."
            )

        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded

    def compile(self, **kwargs):
        """Wrapper of the built in compile method of the model

        Configure the model for training. Sets the optimizer weight decay to the same
        value as in the PyTorch implementation :footcite:t:`Legarreta_2021`.
        """
        if "optimizer" in kwargs:
            if hasattr(kwargs["optimizer"], "weight_decay"):
                kwargs["optimizer"].weight_decay = 0.13
            else:
                print("Optimizer does not have a weight_decay attribute. Ignoring...")

        # Call the compile method of the superclass with the modified kwargs
        super().compile(**kwargs)

    def fit(self, *args, **kwargs):
        """Wrapper of the built in fit method of the model

        Trains the model for a fixed number of epochs. In case of inputing streamlines
        as input, converts them to numpy arrays.
        """

        # Check that the input streamlines have 256 points
        if not kwargs["x"].shape[1] == 256 or not kwargs["y"].shape[1] == 256:
            raise ValueError("Training streamlines must have 256 points.")

        # Convert streamlines to numpy arrays if nibabel streamlines are passed
        if isinstance(kwargs["x"], nib.streamlines.ArraySequence):
            kwargs["x"] = np.array(kwargs["x"])
        if isinstance(kwargs["y"], nib.streamlines.ArraySequence):
            kwargs["y"] = np.array(kwargs["y"])

        return super().fit(*args, **kwargs)
