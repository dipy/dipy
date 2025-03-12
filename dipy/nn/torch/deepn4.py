#!/usr/bin/python
"""Class and helper functions for fitting the DeepN4 model."""

import logging

import numpy as np
from scipy.ndimage import gaussian_filter

from dipy.data import get_fnames
from dipy.nn.utils import normalize, recover_img, set_logger_level, transform_img
from dipy.testing.decorators import doctest_skip_parser
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

    logging.warning(
        "This model requires Pytorch.\
                    Please install these packages using \
                    pip."
    )

logging.basicConfig()
logger = logging.getLogger("deepn4")


class UNet3D(Module):
    def __init__(self, n_in, n_out):
        super(UNet3D, self).__init__()
        # Encoder
        c = 32
        self.ec0 = self.encoder_block(n_in, c, kernel_size=3, stride=1, padding=1)
        self.ec1 = self.encoder_block(c, c * 2, kernel_size=3, stride=1, padding=1)
        self.pool0 = MaxPool3d(2)
        self.ec2 = self.encoder_block(c * 2, c * 2, kernel_size=3, stride=1, padding=1)
        self.ec3 = self.encoder_block(c * 2, c * 4, kernel_size=3, stride=1, padding=1)
        self.pool1 = MaxPool3d(2)
        self.ec4 = self.encoder_block(c * 4, c * 4, kernel_size=3, stride=1, padding=1)
        self.ec5 = self.encoder_block(c * 4, c * 8, kernel_size=3, stride=1, padding=1)
        self.pool2 = MaxPool3d(2)
        self.ec6 = self.encoder_block(c * 8, c * 8, kernel_size=3, stride=1, padding=1)
        self.ec7 = self.encoder_block(c * 8, c * 16, kernel_size=3, stride=1, padding=1)
        self.el = Conv3d(c * 16, c * 16, kernel_size=1, stride=1, padding=0)

        # Decoder
        self.dc9 = self.decoder_block(
            c * 16, c * 16, kernel_size=2, stride=2, padding=0
        )
        self.dc8 = self.decoder_block(
            c * 16 + c * 8, c * 8, kernel_size=3, stride=1, padding=1
        )
        self.dc7 = self.decoder_block(c * 8, c * 8, kernel_size=3, stride=1, padding=1)
        self.dc6 = self.decoder_block(c * 8, c * 8, kernel_size=2, stride=2, padding=0)
        self.dc5 = self.decoder_block(
            c * 8 + c * 4, c * 4, kernel_size=3, stride=1, padding=1
        )
        self.dc4 = self.decoder_block(c * 4, c * 4, kernel_size=3, stride=1, padding=1)
        self.dc3 = self.decoder_block(c * 4, c * 4, kernel_size=2, stride=2, padding=0)
        self.dc2 = self.decoder_block(
            c * 4 + c * 2, c * 2, kernel_size=3, stride=1, padding=1
        )
        self.dc1 = self.decoder_block(c * 2, c * 2, kernel_size=3, stride=1, padding=1)
        self.dc0 = self.decoder_block(c * 2, n_out, kernel_size=1, stride=1, padding=0)
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
        # Encodes
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


class DeepN4:
    """This class is intended for the DeepN4 model.

    The DeepN4 model :footcite:p:`Kanakaraj2024` predicts the bias field for
    magnetic field inhomogeneity correction on T1-weighted images.

    References
    ----------
    .. footbibliography::
    """

    @doctest_skip_parser
    def __init__(self, *, verbose=False, use_cuda=False):
        """Model initialization

        To obtain the pre-trained model, use fetch_default_weights() like:
        >>> deepn4_model = DeepN4() # skip if not have_torch
        >>> deepn4_model.fetch_default_weights() # skip if not have_torch

        This model is designed to take as input file T1 signal and predict
        bias field. Effectively, this model is mimicking bias correction.

        Parameters
        ----------
        verbose : bool, optional
            Whether to show information about the processing.
        use_cuda : bool, optional
            Whether to use GPU for processing.
            If False or no CUDA is detected, CPU will be used.
        """
        if not have_torch:
            raise torch()

        log_level = "INFO" if verbose else "CRITICAL"
        set_logger_level(log_level, logger)

        # DeepN4 network load

        self.model = UNet3D(1, 1)
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)

    def fetch_default_weights(self):
        """Load the model pre-training weights to use for the fitting."""
        fetch_model_weights_path = get_fnames(name="deepn4_default_torch_weights")
        self.load_model_weights(fetch_model_weights_path)

    def load_model_weights(self, weights_path):
        """Load the custom pre-training weights to use for the fitting.

        Parameters
        ----------
        weights_path : str
            Path to the file containing the weights
        """
        try:
            self.model.load_state_dict(
                torch.load(
                    weights_path,
                    weights_only=True,
                    map_location=self.device,
                )["model_state_dict"]
            )
            self.model.eval()
        except ValueError as e:
            raise ValueError(
                "Expected input for the provided model weights \
                             do not match the declared model"
            ) from e

    def __predict(self, x_test):
        """Internal prediction function
        Predict bias field from input T1 signal

        Parameters
        ----------
        x_test : np.ndarray
            Image should match the required shape of the model.

        Returns
        -------
        np.ndarray (batch, ...)
            Predicted bias field
        """
        return self.model(x_test)[:, 0].detach().numpy()

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
        input_data, [lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ] = self.pad(
            subj, 128
        )
        in_max = np.percentile(input_data[np.nonzero(input_data)], 99.99)
        input_data = normalize(input_data, min_v=0, max_v=in_max, new_min=0, new_max=1)
        input_data = np.squeeze(input_data)
        input_vols = np.zeros((1, 1, 128, 128, 128))
        input_vols[0, 0, :, :, :] = input_data

        return (
            torch.from_numpy(input_vols).float(),
            lx,
            lX,
            ly,
            lY,
            lz,
            lZ,
            rx,
            rX,
            ry,
            rY,
            rz,
            rZ,
            in_max,
        )

    def predict(self, img, img_affine, *, voxsize=(1, 1, 1), threshold=0.5):
        """Wrapper function to facilitate prediction of larger dataset.
        The function will mask, normalize, split, predict and 're-assemble'
        the data as a volume.

        Parameters
        ----------
        img : np.ndarray
            T1 image to predict and apply bias field
        img_affine : np.ndarray (4, 4)
            Affine matrix for the T1 image
        voxsize : np.ndarray or list or tuple (3,), optional
            voxel size of the T1 image.
        threshold : float, optional
            Threshold for cleaning the final correction field

        Returns
        -------
        final_corrected : np.ndarray (x, y, z)
            Predicted bias corrected image.
            The volume has matching shape to the input data
        """
        # Preprocess input data (resample, normalize, and pad)
        resampled_T1, inv_affine, mid_shape, offset_array, scale, crop_vs, pad_vs = (
            transform_img(img, img_affine, voxsize=voxsize)
        )
        (in_features, lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ, in_max) = (
            self.load_resample(resampled_T1)
        )

        # Run the model to get the bias field
        logfield = self.__predict(in_features.to(self.device))
        field = np.exp(logfield)
        field = field.squeeze()

        # Postprocess predicted field (reshape - unpad, smooth the field,
        # upsample)
        final_field = np.zeros(
            [resampled_T1.shape[0], resampled_T1.shape[1], resampled_T1.shape[2]]
        )
        final_field[rx:rX, ry:rY, rz:rZ] = field[lx:lX, ly:lY, lz:lZ]
        final_fields = gaussian_filter(final_field, sigma=3)
        upsample_final_field = recover_img(
            final_fields,
            inv_affine,
            mid_shape,
            img.shape,
            offset_array,
            voxsize,
            scale,
            crop_vs,
            pad_vs,
        )

        # Correct the image
        below_threshold_mask = np.abs(upsample_final_field) < threshold
        with np.errstate(divide="ignore", invalid="ignore"):
            final_corrected = np.where(
                below_threshold_mask, 0, img / upsample_final_field
            )

        return final_corrected
