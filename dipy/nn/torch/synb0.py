#!/usr/bin/python
"""
Class and helper functions for fitting the Synb0 model using PyTorch.
"""

import numpy as np

from dipy.data import get_fnames
from dipy.nn.utils import normalize, set_logger_level, unnormalize
from dipy.testing.decorators import doctest_skip_parser, warning_for_keywords
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
if have_torch:
    from torch.nn import (
        Conv3d,
        ConvTranspose3d,
        InstanceNorm3d,
        LeakyReLU,
        MaxPool3d,
        Module,
        Sequential,
    )
else:

    class Module:
        pass

    logger.warning(
        "This model requires PyTorch. Please install PyTorch using pip or conda."
    )


class UNet3D(Module):
    def __init__(self, n_in, n_out):
        super(UNet3D, self).__init__()
        # Encoder
        self.ec0 = self.encoder_block(n_in, 32, kernel_size=3, stride=1, padding=1)
        self.ec1 = self.encoder_block(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool0 = MaxPool3d(2)
        self.ec2 = self.encoder_block(64, 64, kernel_size=3, stride=1, padding=1)
        self.ec3 = self.encoder_block(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool3d(2)
        self.ec4 = self.encoder_block(128, 128, kernel_size=3, stride=1, padding=1)
        self.ec5 = self.encoder_block(128, 256, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool3d(2)
        self.ec6 = self.encoder_block(256, 256, kernel_size=3, stride=1, padding=1)
        self.ec7 = self.encoder_block(256, 512, kernel_size=3, stride=1, padding=1)
        self.el = Conv3d(512, 512, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.dc9 = self.decoder_block(512, 512, kernel_size=2, stride=2, padding=0)
        self.dc8 = self.decoder_block(
            512 + 256, 256, kernel_size=3, stride=1, padding=1
        )
        self.dc7 = self.decoder_block(256, 256, kernel_size=3, stride=1, padding=1)
        self.dc6 = self.decoder_block(256, 256, kernel_size=2, stride=2, padding=0)
        self.dc5 = self.decoder_block(
            256 + 128, 128, kernel_size=3, stride=1, padding=1
        )
        self.dc4 = self.decoder_block(128, 128, kernel_size=3, stride=1, padding=1)
        self.dc3 = self.decoder_block(128, 128, kernel_size=2, stride=2, padding=0)
        self.dc2 = self.decoder_block(128 + 64, 64, kernel_size=3, stride=1, padding=1)
        self.dc1 = self.decoder_block(64, 64, kernel_size=3, stride=1, padding=1)
        self.dc0 = self.decoder_block(64, n_out, kernel_size=1, stride=1, padding=0)
        self.dl = ConvTranspose3d(n_out, n_out, kernel_size=1, stride=1, padding=0)

    def encoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = Sequential(
            Conv3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            InstanceNorm3d(out_channels),
            LeakyReLU(),
        )
        return layer

    def decoder_block(self, in_channels, out_channels, kernel_size, stride, padding):
        layer = Sequential(
            ConvTranspose3d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            InstanceNorm3d(out_channels),
            LeakyReLU(),
        )
        return layer

    def forward(self, x):
        # Encode
        e0 = self.ec0(x)
        syn0 = self.ec1(e0)
        del e0

        e1 = self.pool0(syn0)
        e2 = self.ec2(e1)
        syn1 = self.ec3(e2)
        del e1, e2

        e3 = self.pool1(syn1)
        e4 = self.ec4(e3)
        syn2 = self.ec5(e4)
        del e3, e4

        e5 = self.pool2(syn2)
        e6 = self.ec6(e5)
        e7 = self.ec7(e6)

        # Last layer without relu
        el = self.el(e7)
        del e5, e6, e7

        # Decode
        d9 = torch.cat((self.dc9(el), syn2), 1)
        del el, syn2

        d8 = self.dc8(d9)
        d7 = self.dc7(d8)
        del d9, d8

        d6 = torch.cat((self.dc6(d7), syn1), 1)
        del d7, syn1

        d5 = self.dc5(d6)
        d4 = self.dc4(d5)
        del d6, d5

        d3 = torch.cat((self.dc3(d4), syn0), 1)
        del d4, syn0

        d2 = self.dc2(d3)
        d1 = self.dc1(d2)
        del d3, d2

        d0 = self.dc0(d1)
        del d1

        # Last layer without relu
        out = self.dl(d0)

        return out


class Synb0:
    """
    PyTorch implementation of the Synb0 model.

    Synb0 :footcite:p:`Schilling2019`, :footcite:p:`Schilling2020` uses a neural
    network to synthesize a b0 volume for distortion correction in DWI images.

    The model is the deep learning part of the Synb0-Disco pipeline, thus
    stand-alone usage is not recommended.

    References
    ----------
    .. footbibliography::
    """

    @doctest_skip_parser
    @warning_for_keywords()
    def __init__(self, *, verbose=False):
        r"""
        Initialize the Synb0 model.

        The model was pre-trained for usage on pre-processed images
        following the synb0-disco pipeline.
        One can load their own weights using load_model_weights.

        This model is designed to take as input a b0 image and a T1 weighted
        image.

        It was designed to predict a b-inf image.

        Parameters
        ----------
        verbose : bool, optional
            Whether to show information about the processing.
            Default is False.
        """
        if not have_torch:
            raise ImportError(
                "PyTorch is required for this model. "
                "Please install it using: pip install torch"
            )

        log_level = "INFO" if verbose else "CRITICAL"
        set_logger_level(log_level, logger)

        # Initialize the model
        self.model = UNet3D(n_in=2, n_out=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

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
        fetch_model_weights_path = get_fnames(name="synb0_default_torch_weights")
        logger.info(f"fetched {fetch_model_weights_path[idx]}")
        self.load_model_weights(fetch_model_weights_path[idx])

    def load_model_weights(self, weights_path):
        r"""
        Load the custom pre-training weights to use for the fitting.

        Parameters
        ----------
        weights_path : str
            Path to the file containing the weights (.pth file saved by PyTorch)
        """
        try:
            state_dict = torch.load(
                weights_path, map_location=self.device, weights_only=True
            )
            self.model.load_state_dict(state_dict)
            self.model.eval()
        except Exception as e:
            raise ValueError(
                "Expected input for the provided model weights "
                "do not match the declared model"
            ) from e

    def __predict(self, x_test):
        r"""
        Internal prediction function.

        Parameters
        ----------
        x_test : torch.Tensor (batch, 2, 80, 96, 80)
            Image should match the required shape of the model.

        Returns
        -------
        np.ndarray (batch, 1, 80, 96, 80)
            Reconstructed b-inf image(s)
        """
        with torch.no_grad():
            x_test = torch.from_numpy(x_test).float().to(self.device)
            prediction = self.model(x_test)
            return prediction.cpu().numpy()

    @warning_for_keywords()
    def predict(self, b0, T1, *, batch_size=None, average=True):
        r"""
        Wrapper function to facilitate prediction of larger dataset.

        The function will pad the data to meet the required shape of image.
        Note that the b0 and T1 image should have the same shape.

        Parameters
        ----------
        b0 : np.ndarray (batch, 77, 91, 77) or (77, 91, 77)
            For a single image, input should be a 3D array. If multiple images,
            there should also be a batch dimension.

        T1 : np.ndarray (batch, 77, 91, 77) or (77, 91, 77)
            For a single image, input should be a 3D array. If multiple images,
            there should also be a batch dimension.

        batch_size : int, optional
            Number of images per prediction pass. Only available if data
            is provided with a batch dimension.
            Consider lowering it if you get an out of memory error.
            Increase it if you want it to be faster and have a lot of data.
            If None, batch_size will be set to 1 if the provided image
            has a batch dimension.
            Default is None.

        average : bool, optional
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
        if (
            all([b0.shape[1:] != (77, 91, 77), b0.shape != (77, 91, 77)])
            or b0.shape != T1.shape
        ):
            raise ValueError(
                "Expected shape (batch, 77, 91, 77) or (77, 91, 77) for both inputs"
            )

        dim = len(b0.shape)

        # Add batch dimension if not provided
        if dim == 3:
            T1 = np.expand_dims(T1, 0)
            b0 = np.expand_dims(b0, 0)
        shape = b0.shape

        if batch_size is None:
            batch_size = 1

        # Pad the data to match the model's input shape
        # From (77, 91, 77) to (80, 96, 80)
        T1 = np.pad(T1, ((0, 0), (2, 1), (3, 2), (2, 1)), "constant")
        b0 = np.pad(b0, ((0, 0), (2, 1), (3, 2), (2, 1)), "constant")

        # Normalize the data
        p99 = np.percentile(b0, 99, axis=(1, 2, 3))
        for i in range(shape[0]):
            T1[i] = normalize(T1[i], min_v=0, max_v=150, new_min=-1, new_max=1)
            b0[i] = normalize(b0[i], min_v=0, max_v=p99[i], new_min=-1, new_max=1)

        if dim == 3 and batch_size != 1:
            logger.warning(
                "Batch size specified, but not used "
                "due to the input not having a batch dimension"
            )
            batch_size = 1

        # Prediction stage
        if average:
            mean_pred = np.zeros(shape + (5,), dtype=np.float32)
            for i in range(5):
                self.fetch_default_weights(i)
                # Stack and prepare input: (batch, 80, 96, 80, 2)
                # -> (batch, 2, 80, 96, 80)
                temp = np.stack([b0, T1], axis=-1)
                input_data = np.moveaxis(temp, -1, 1).astype(np.float32)

                prediction = np.zeros((shape[0], 1, 80, 96, 80), dtype=np.float32)
                for batch_idx in range(batch_size, shape[0] + 1, batch_size):
                    temp_input = input_data[batch_idx - batch_size : batch_idx]
                    temp_pred = self.__predict(temp_input)
                    prediction[batch_idx - batch_size : batch_idx] = temp_pred

                remainder = np.mod(shape[0], batch_size)
                if remainder != 0:
                    temp_pred = self.__predict(input_data[-remainder:])
                    prediction[-remainder:] = temp_pred

                # Unnormalize
                for j in range(shape[0]):
                    temp_pred = unnormalize(prediction[j], -1, 1, 0, p99[j])
                    prediction[j] = temp_pred

                # Remove padding: (batch, 1, 80, 96, 80) -> (batch, 77, 91, 77)
                prediction = prediction[:, 0, 2:-1, 3:-2, 2:-1]

                mean_pred[..., i] = prediction

            prediction = np.mean(mean_pred, axis=-1)
        else:
            # Stack and prepare input
            temp = np.stack([b0, T1], axis=-1)
            input_data = np.moveaxis(temp, -1, 1).astype(np.float32)

            prediction = np.zeros((shape[0], 1, 80, 96, 80), dtype=np.float32)
            for batch_idx in range(batch_size, shape[0] + 1, batch_size):
                temp_input = input_data[batch_idx - batch_size : batch_idx]
                temp_pred = self.__predict(temp_input)
                prediction[batch_idx - batch_size : batch_idx] = temp_pred

            remainder = np.mod(shape[0], batch_size)
            if remainder != 0:
                temp_pred = self.__predict(input_data[-remainder:])
                prediction[-remainder:] = temp_pred

            # Unnormalize
            for j in range(shape[0]):
                prediction[j] = unnormalize(prediction[j], -1, 1, 0, p99[j])

            # Remove padding
            prediction = prediction[:, 0, 2:-1, 3:-2, 2:-1]

        if dim == 3:
            prediction = prediction[0]

        return prediction
