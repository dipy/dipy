import os
import tempfile

import numpy as np
import numpy.testing as npt
import pytest

import dipy.nn.utils as utils
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.0.0")

if have_tf:
    from keras import Sequential

    from dipy.nn.finta_model import (
        Decoder,
        Encoder,
        IncrFeatStridedConvFCUpsampReflectPadAE as AE,
    )

else:
    have_tf = False


class DummyData:
    def __init__(self, latent_space_dims=32):
        """Generate three streamlines, aligned with the x, y, and z axes. Each
        streamline has 256 points. Generate a latent vector of the input size.

        Parameters
        ----------
        latent_space_dims : int, optional
            Number of dimensions of the latent space vector
        """

        streamlines = []
        latent_vectors = []
        for i in range(3):
            # Streamlines aligned with the x, y, and z axes
            streamline = np.zeros((256, 3))
            streamline[:, i] = np.arange(256)
            streamline = streamline[np.newaxis, ...]
            streamlines.append(streamline)

            # Arbitrary latent vectors
            latent_vector = (i + 1) * np.arange(latent_space_dims)
            latent_vector = latent_vector[np.newaxis, ...]
            latent_vectors.append(latent_vector)

        self.streamlines = np.concatenate(streamlines, axis=0)
        self.latent_vectors = np.concatenate(latent_vectors, axis=0)


# Test the Encoder on its own as a usable block
@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_encoder_parameter_initialization():
    test_encoder = Encoder(latent_space_dims=10, kernel_size=3)

    # Check initialization parameters
    assert test_encoder.latent_space_dims == 10
    assert test_encoder.kernel_size == 3
    k_weight_init = np.sqrt(1 / (3 * test_encoder.kernel_size))
    npt.assert_almost_equal(test_encoder.k_conv1d_weight_init, k_weight_init, decimal=5)


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_encoder_decoder_wrong_kernel_size():
    with pytest.raises(ValueError) as excinfo:
        Encoder(latent_space_dims=10, kernel_size=6)

    assert str(excinfo.value) == "Kernel size must be between 1 and 5."
    assert excinfo.type == ValueError

    with pytest.raises(ValueError) as excinfo:
        Decoder(encoder_out_size=[8, 1024], kernel_size=6)

    assert str(excinfo.value) == "Kernel size must be between 1 and 5."
    assert excinfo.type == ValueError


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_encoder_ae_wrong_streamline_size_call():
    bad_streamlines = np.zeros((10, 128, 3))

    # Instantiate the Encoder and the AE
    test_encoder = Encoder(latent_space_dims=10, kernel_size=3)
    test_ae = AE(latent_space_dims=10, kernel_size=3)

    # Encoder
    with pytest.raises(ValueError) as excinfo:
        test_encoder(bad_streamlines)

    assert "Input streamlines in Encoder must have 256 points." in str(excinfo.value)
    assert excinfo.type == ValueError

    # AE
    with pytest.raises(ValueError) as excinfo:
        test_ae(bad_streamlines)

    exc_str = str(excinfo.value)
    assert "Input streamlines to AutoEncoder must have 256 points." in exc_str
    assert excinfo.type == ValueError


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_encoder_call_correct():
    latent_space_dims = 10
    test_encoder = Encoder(latent_space_dims=latent_space_dims)
    data = DummyData().streamlines
    output = test_encoder(data)

    # Check output shape
    assert output.shape == (data.shape[0], latent_space_dims)

    # Check encoder_out_size
    assert test_encoder.encoder_out_size == [8, 1024]

    # Check that the encoder is built
    assert test_encoder.built is True


# Test the Decoder on its own as a usable block
@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_decoder_parameter_initialization():
    test_decoder = Decoder(encoder_out_size=[4, 1024])

    # Check initialization parameters
    assert test_decoder.kernel_size == 3
    assert test_decoder.encoder_out_size == [4, 1024]


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_decoder_call_correct():
    test_decoder = Decoder(encoder_out_size=[8, 1024])
    data = DummyData(latent_space_dims=10).latent_vectors
    output = test_decoder(data)

    # Check output shape
    assert output.shape == (data.shape[0], 256, 3)

    # Check that the decoder is built
    assert test_decoder.built is True


# Test the full AutoEncoder


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_ae_calling_output_shape():
    latent_space_dims = 32
    kernel_size = 3
    test_ae = AE(latent_space_dims=latent_space_dims, kernel_size=kernel_size)

    data = DummyData().streamlines
    output = test_ae(data)

    # Check output shape
    assert output.shape == (data.shape[0], 256, 3)

    # Check that the autoencoder is built
    assert test_ae.name == "incr_feat_strided_conv_fc_upsamp_reflect_pad_ae"

    # Check that the autoencoder is built
    assert test_ae.built is True


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_ae_fit_wrong_streamline_size():
    bad_streamlines = np.zeros((10, 128, 3))

    test_ae = AE(latent_space_dims=10, kernel_size=3)

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.MeanSquaredError()
    test_ae.compile(optimizer=optimizer, loss=loss)
    # Fit the model
    input_train_data = tf.convert_to_tensor(bad_streamlines)

    with pytest.raises(ValueError) as excinfo:
        test_ae.fit(x=input_train_data, y=input_train_data, batch_size=1, epochs=3)

    assert excinfo.type == ValueError
    assert "Training streamlines must have 256 points." in str(excinfo.value)


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_ae_train_weight_saving_and_loading():
    latent_space_dims = 32
    kernel_size = 3
    test_ae = AE(latent_space_dims=latent_space_dims, kernel_size=kernel_size)

    # Train the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)
    loss = tf.keras.losses.MeanSquaredError()

    # Compile the model
    test_ae.compile(optimizer=optimizer, loss=loss)

    # Fit the model
    input_train_data = DummyData().streamlines
    input_train_data = tf.convert_to_tensor(input_train_data)
    test_ae.fit(x=input_train_data, y=input_train_data, batch_size=1, epochs=3)

    # Run the training data through the trained model
    output_after_train = test_ae(input_train_data)

    # Save the model weights
    path_weights = tempfile.NamedTemporaryFile().name + ".weights.h5"
    test_ae.save_weights(path_weights)
    # Save the model with the weights
    path_model = tempfile.NamedTemporaryFile().name + ".keras"
    test_ae.save(path_model)

    # Load the model
    loaded_ae = tf.keras.models.load_model(path_model, compile=False)

    # Check that the loaded model is the same as the original in parameters
    # We use inclusion because the loaded one has some extra characters
    assert "incr_feat_strided_conv_fc_upsamp_reflect_pad_ae" in loaded_ae.name
    assert loaded_ae.latent_space_dims == test_ae.latent_space_dims
    assert loaded_ae.kernel_size == test_ae.kernel_size

    # Check that the loaded model outputs the same as the original
    output_after_load = loaded_ae(input_train_data)
    npt.assert_allclose(output_after_train, output_after_load)

    # Cleanup the weights and the model
    os.remove(path_weights)
    os.remove(path_model)


# Test reflection padding layer
@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_pre_pad():
    # Test passing a layer, check that the output is a keras.Sequential
    test_layer = tf.keras.layers.Conv1D(32, 3)
    padded_layer = utils.pre_pad(test_layer)

    assert isinstance(padded_layer, Sequential)
    # Pass an input to the padded_layer to cover the call method
    input_data = np.random.rand(10, 256, 3)
    padded_layer(input_data)

    # Test passing a thing that is not a layer
    with pytest.raises(ValueError) as excinfo:
        utils.pre_pad("not_a_layer")

    expected_error_str = (
        "Input must be an instance of keras.layers.Layer."
        " Pass a layer to apply ReflectionPadding1D to."
    )
    assert str(excinfo.value) == expected_error_str
    assert excinfo.type == ValueError
