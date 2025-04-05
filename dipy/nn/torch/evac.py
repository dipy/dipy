#!/usr/bin/python
"""Class and helper functions for fitting the EVAC+ model."""

import logging

import numpy as np

from dipy.align.reslice import reslice
from dipy.data import get_fnames
from dipy.nn.utils import (
    normalize,
    recover_img,
    set_logger_level,
    transform_img,
)
from dipy.segment.utils import remove_holes_and_islands
from dipy.testing.decorators import doctest_skip_parser
from dipy.utils.deprecator import deprecated_params
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
if have_torch:
    from torch.nn import (
        Conv3d,
        ConvTranspose3d,
        Dropout3d,
        LayerNorm,
        Module,
        ModuleList,
        ReLU,
        Softmax,
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
logger = logging.getLogger("EVAC+")


def prepare_img(image):
    """Function to prepare image for model input
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
    input1 = np.expand_dims(input1, 1)

    input2, _ = reslice(image, np.eye(4), (1, 1, 1), (2, 2, 2))
    input2 = np.moveaxis(input2, -1, 0)
    input2 = np.expand_dims(input2, 1)

    input3, _ = reslice(image, np.eye(4), (1, 1, 1), (4, 4, 4))
    input3 = np.moveaxis(input3, -1, 0)
    input3 = np.expand_dims(input3, 1)

    input4, _ = reslice(image, np.eye(4), (1, 1, 1), (8, 8, 8))
    input4 = np.moveaxis(input4, -1, 0)
    input4 = np.expand_dims(input4, 1)

    input5, _ = reslice(image, np.eye(4), (1, 1, 1), (16, 16, 16))
    input5 = np.moveaxis(input5, -1, 0)
    input5 = np.expand_dims(input5, 1)

    input_data = [
        torch.from_numpy(input1).float(),
        torch.from_numpy(input2).float(),
        torch.from_numpy(input3).float(),
        torch.from_numpy(input4).float(),
        torch.from_numpy(input5).float(),
    ]

    return input_data


class MoveDimLayer(Module):
    def __init__(self, source_dim, dest_dim):
        super(MoveDimLayer, self).__init__()
        self.source_dim = source_dim
        self.dest_dim = dest_dim

    def forward(self, x):
        return torch.movedim(x, self.source_dim, self.dest_dim)


class ChannelSum(Module):
    def __init__(self):
        super(ChannelSum, self).__init__()

    def forward(self, inputs):
        return torch.sum(inputs, dim=1, keepdim=True)


class Add(Module):
    def __init__(self):
        super(Add, self).__init__()

    def forward(self, x, passed):
        return x + passed


class Block(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        strides,
        padding,
        drop_r,
        n_layers,
        *,
        passed_channel=1,
        layer_type="down",
    ):
        super(Block, self).__init__()
        self.n_layers = n_layers
        self.layer_list = ModuleList()
        self.layer_list2 = ModuleList()

        cur_channel = in_channels
        for _ in range(n_layers):
            self.layer_list.append(
                Conv3d(
                    cur_channel,
                    out_channels,
                    kernel_size,
                    stride=strides,
                    padding=padding,
                )
            )
            cur_channel = out_channels
            self.layer_list.append(Dropout3d(drop_r))
            self.layer_list.append(MoveDimLayer(1, -1))
            self.layer_list.append(LayerNorm(out_channels))
            self.layer_list.append(MoveDimLayer(-1, 1))
            self.layer_list.append(ReLU())

        if layer_type == "down":
            self.layer_list2.append(Conv3d(in_channels, 1, 2, stride=2, padding=0))
            self.layer_list2.append(ReLU())
        elif layer_type == "up":
            self.layer_list2.append(
                ConvTranspose3d(passed_channel, 1, 2, stride=2, padding=0)
            )
            self.layer_list2.append(ReLU())

        self.channel_sum = ChannelSum()
        self.add = Add()

    def forward(self, input, passed):
        x = input
        for layer in self.layer_list:
            x = layer(x)

        x = self.channel_sum(x)
        fwd = self.add(x, passed)
        x = fwd

        for layer in self.layer_list2:
            x = layer(x)

        return fwd, x


class Model(Module):
    def __init__(self, model_scale=16):
        super(Model, self).__init__()

        # Block structure
        self.block1 = Block(
            1, model_scale, kernel_size=5, strides=1, padding=2, drop_r=0.2, n_layers=1
        )
        self.block2 = Block(
            2,
            model_scale * 2,
            kernel_size=5,
            strides=1,
            padding=2,
            drop_r=0.5,
            n_layers=2,
        )
        self.block3 = Block(
            2,
            model_scale * 4,
            kernel_size=5,
            strides=1,
            padding=2,
            drop_r=0.5,
            n_layers=3,
        )
        self.block4 = Block(
            2,
            model_scale * 8,
            kernel_size=5,
            strides=1,
            padding=2,
            drop_r=0.5,
            n_layers=3,
        )
        self.block5 = Block(
            2,
            model_scale * 16,
            kernel_size=5,
            strides=1,
            padding=2,
            drop_r=0.5,
            n_layers=3,
            passed_channel=2,
            layer_type="up",
        )

        # Upsample/decoder blocks
        self.up_block1 = Block(
            3,
            model_scale * 8,
            kernel_size=5,
            strides=1,
            padding=2,
            drop_r=0.5,
            n_layers=3,
            passed_channel=1,
            layer_type="up",
        )
        self.up_block2 = Block(
            3,
            model_scale * 4,
            kernel_size=5,
            strides=1,
            padding=2,
            drop_r=0.5,
            n_layers=3,
            passed_channel=1,
            layer_type="up",
        )
        self.up_block3 = Block(
            3,
            model_scale * 2,
            kernel_size=5,
            strides=1,
            padding=2,
            drop_r=0.5,
            n_layers=2,
            passed_channel=1,
            layer_type="up",
        )
        self.up_block4 = Block(
            2,
            model_scale,
            kernel_size=5,
            strides=1,
            padding=2,
            drop_r=0.5,
            n_layers=1,
            passed_channel=1,
            layer_type="none",
        )

        self.conv_pred = Conv3d(1, 2, 1, padding=0)
        self.softmax = Softmax(dim=1)

    def forward(self, inputs, raw_input_2, raw_input_3, raw_input_4, raw_input_5):
        fwd1, x = self.block1(inputs, inputs)
        x = torch.cat([x, raw_input_2], dim=1)

        fwd2, x = self.block2(x, x)
        x = torch.cat([x, raw_input_3], dim=1)

        fwd3, x = self.block3(x, x)
        x = torch.cat([x, raw_input_4], dim=1)

        fwd4, x = self.block4(x, x)
        x = torch.cat([x, raw_input_5], dim=1)

        # Decoding path
        _, up = self.block5(x, x)
        x = torch.cat([fwd4, up], dim=1)

        _, up = self.up_block1(x, up)
        x = torch.cat([fwd3, up], dim=1)

        _, up = self.up_block2(x, up)
        x = torch.cat([fwd2, up], dim=1)

        _, up = self.up_block3(x, up)
        x = torch.cat([fwd1, up], dim=1)

        _, pred = self.up_block4(x, up)

        pred = self.conv_pred(pred)
        output = self.softmax(pred)

        return output


class EVACPlus:
    """This class is intended for the EVAC+ model.

    The EVAC+ model :footcite:p:`Park2024` is a deep learning neural network for
    brain extraction. It uses a V-net architecture combined with
    multi-resolution input data, an additional conditional random field (CRF)
    recurrent layer and supplementary Dice loss term for this recurrent layer.

    References
    ----------
    .. footbibliography::
    """

    @doctest_skip_parser
    def __init__(self, *, verbose=False, use_cuda=False):
        """Model initialization

        The model was pre-trained for usage on
        brain extraction of T1 images.

        This model is designed to take as input
        a T1 weighted image.

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

        # EVAC+ network load

        self.model = self.init_model()
        if use_cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.fetch_default_weights()

    def init_model(self, model_scale=16):
        return Model(model_scale)

    def fetch_default_weights(self):
        """Load the model pre-training weights to use for the fitting.
        While the user can load different weights, the function
        is mainly intended for the class function 'predict'.
        """
        fetch_model_weights_path = get_fnames(name="evac_default_torch_weights")
        self.load_model_weights(fetch_model_weights_path)

    def load_model_weights(self, weights_path):
        """Load the custom pre-training weights to use for the fitting.

        Parameters
        ----------
        weights_path : str
            Path to the file containing the weights (pth, saved by Pytorch)
        """
        try:
            self.model.load_state_dict(
                torch.load(
                    weights_path,
                    weights_only=True,
                    map_location=self.device,
                )
            )
            self.model.eval()
        except ValueError as e:
            raise ValueError(
                "Expected input for the provided model weights \
                             do not match the declared model"
            ) from e

    def __predict(self, x_test):
        """Internal prediction function

        Parameters
        ----------
        x_test : list of np.ndarray
            Image should match the required shape of the model.

        Returns
        -------
        np.ndarray (batch, ...)
            Predicted brain mask
        """
        return self.model(*x_test)[:, 0].detach().numpy()

    @deprecated_params(
        "largest_area", new_name="finalize_mask", since="1.10", until="1.12"
    )
    def predict(
        self,
        T1,
        affine,
        *,
        voxsize=(1, 1, 1),
        batch_size=None,
        return_affine=False,
        return_prob=False,
        finalize_mask=True,
    ):
        """Wrapper function to facilitate prediction of larger dataset.

        Parameters
        ----------
        T1 : np.ndarray or list of np.ndarray
            For a single image, input should be a 3D array.
            If multiple images, it should be a a list or tuple.
        affine : np.ndarray (4, 4) or (batch, 4, 4)
            or list of np.ndarrays with len of batch
            Affine matrix for the T1 image. Should have
            batch dimension if T1 has one.
        voxsize : np.ndarray or list or tuple, optional
            (3,) or (batch, 3)
            voxel size of the T1 image.
        batch_size : int, optional
            Number of images per prediction pass. Only available if data
            is provided with a batch dimension.
            Consider lowering it if you get an out of memory error.
            Increase it if you want it to be faster and have a lot of data.
            If None, batch_size will be set to 1 if the provided image
            has a batch dimension.
        return_affine : bool, optional
            Whether to return the affine matrix. Useful if the input was a
            file path.
        return_prob : bool, optional
            Whether to return the probability map instead of a
            binary mask. Useful for testing.
        finalize_mask : bool, optional
            Whether to remove potential holes or islands.
            Useful for solving minor errors.

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
                logger.warning(
                    "Batch size specified, but not used",
                    "due to the input not having \
                                a batch dimension",
                )

            T1 = np.expand_dims(T1, 0)
            affine = np.expand_dims(affine, 0)
            voxsize = np.expand_dims(voxsize, 0)
        else:
            raise ValueError(
                "T1 data should be a np.ndarray of dimension 3 or a list/tuple of it"
            )
        if batch_size is None:
            batch_size = 1

        input_data = np.zeros((128, 128, 128, len(T1)))
        affines = np.zeros((len(T1), 4, 4))
        mid_shapes = np.zeros((len(T1), 3)).astype(int)
        offset_arrays = np.zeros((len(T1), 4, 4)).astype(int)
        scales = np.zeros(len(T1))
        crop_vss = np.zeros((len(T1), 3, 2))
        pad_vss = np.zeros((len(T1), 3, 2))

        # Normalize the data.
        n_T1 = np.zeros(T1.shape)
        for i, T1_img in enumerate(T1):
            n_T1[i] = normalize(T1_img, new_min=0, new_max=1)
            t_img, t_affine, mid_shape, offset_array, scale, crop_vs, pad_vs = (
                transform_img(n_T1[i], affine[i], voxsize=voxsize[i])
            )
            input_data[..., i] = t_img
            affines[i] = t_affine
            mid_shapes[i] = mid_shape
            offset_arrays[i] = offset_array
            scales[i] = scale
            crop_vss[i] = crop_vs
            pad_vss[i] = pad_vs

        # Prediction stage
        prediction = np.zeros((len(T1), 128, 128, 128), dtype=np.float32)
        for batch_idx in range(batch_size, len(T1) + 1, batch_size):
            batch = input_data[..., batch_idx - batch_size : batch_idx]
            temp_input = prepare_img(batch)
            temp_pred = self.__predict(temp_input)
            prediction[:batch_idx] = temp_pred
        remainder = np.mod(len(T1), batch_size)
        if remainder != 0:
            temp_input = prepare_img(input_data[..., -remainder:])
            temp_pred = self.__predict(temp_input)
            prediction[-remainder:] = temp_pred

        output_mask = []
        for i in range(len(T1)):
            output = recover_img(
                prediction[i],
                affines[i],
                mid_shapes[i],
                n_T1[i].shape,
                offset_arrays[i],
                voxsize=voxsize[i],
                scale=scales[i],
                crop_vs=crop_vss[i],
                pad_vs=pad_vss[i],
            )
            if not return_prob:
                output = np.where(output >= 0.5, 1, 0)
                if finalize_mask:
                    output = remove_holes_and_islands(output, slice_wise=True)
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
