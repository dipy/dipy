"""

Note
----

This file is a pytorch adapted version from the
sources of the SynthSeg project - https://github.com/BBillot/SynthSeg.
All weights and model artchitecture are original from the SynthSeg project.
It remains licensed as the rest of SynthSeg
(Apache 2.0 license as of January 2026).

# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
#
#   See LICENSE.txt file distributed along with the SynthSeg package for the
#   copyright and license terms.
#
# ## ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ##
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from dipy.data import get_fnames
from dipy.nn.utils import (
    normalize,
    recover_img,
    set_logger_level,
    transform_img,
)
from dipy.segment.utils import remove_holes_and_islands
from dipy.testing.decorators import doctest_skip_parser
from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
if have_torch:
    from torch.nn import (
        BatchNorm3d,
        Conv3d,
        MaxPool3d,
        Module,
        ModuleList,
        Softmax,
        Upsample,
    )
    import torch.nn.functional as F
else:

    class Module:
        pass


class Conv3dELU(Module):
    """
    Mimics TensorFlow Conv3D + ELU fused behavior.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        dilation=1,
        bias=True,
    ):
        super().__init__()
        self.conv = Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            dilation=dilation,
            bias=bias,
        )

    def forward(self, x):
        x = self.conv(x)
        x = F.elu(x)
        return x


class Block(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding,
        n_layers,
        layer_type,
    ):
        super(Block, self).__init__()
        self.n_layers = n_layers
        self.layer_list = ModuleList()
        self.layer_type = layer_type

        cur_channel = in_channels
        if self.layer_type == "up":
            self.layer_list.append(Upsample(scale_factor=2))
            cur_channel = int(cur_channel * 1.5)

        for _ in range(n_layers):
            self.layer_list.append(
                Conv3dELU(
                    cur_channel,
                    out_channels,
                    kernel_size,
                    padding=padding,
                )
            )
            cur_channel = out_channels

        self.layer_list.append(BatchNorm3d(out_channels, eps=1e-3, momentum=0.01))
        if self.layer_type == "down":
            self.layer_list.append(MaxPool3d(2, stride=2))

    def forward(self, input, passed):
        x = input
        for l_idx, layer in enumerate(self.layer_list):
            x = layer(x)
            if l_idx == 0 and self.layer_type == "up":
                x = torch.cat([passed, x], dim=1)
            if l_idx == self.n_layers - 1 and self.layer_type == "down":
                skip = x

        if self.layer_type == "down":
            return x, skip
        return x


class Model(Module):
    def __init__(self, *, model_scale=24, n_levels=5, output_channels=33):
        super(Model, self).__init__()

        self.block_list = ModuleList()
        self.channels = [model_scale * (2**i) for i in range(n_levels)]
        self.channels_rev = [
            model_scale * (2 ** (n_levels - i - 1)) for i in range(n_levels)
        ]
        self.channels.insert(0, 1)
        self.channels_rev.append(output_channels)

        self.model_scale = model_scale
        self.n_levels = n_levels

        # Block structure
        for level in range(n_levels - 1):
            block = Block(
                self.channels[level],
                self.channels[level + 1],
                kernel_size=3,
                padding=1,
                n_layers=2,
                layer_type="down",
            )
            self.block_list.append(block)

        level = n_levels - 1
        block = Block(
            self.channels[level],
            self.channels[level + 1],
            kernel_size=3,
            padding=1,
            n_layers=2,
            layer_type="down2",
        )
        self.block_list.append(block)

        for level in range(n_levels - 1):
            block = Block(
                self.channels_rev[level],
                self.channels_rev[level + 1],
                kernel_size=3,
                padding=1,
                n_layers=2,
                layer_type="up",
            )
            self.block_list.append(block)

        self.conv_pred = Conv3d(model_scale, 33, 1, padding=0)
        self.softmax = Softmax(dim=1)

    def forward(self, inputs):
        skip_list = []
        x = inputs
        for block in self.block_list[: self.n_levels - 1]:
            x, skip = block(x, None)
            skip_list.append(skip)
        x = self.block_list[self.n_levels - 1](x, None)
        for idx, block in enumerate(self.block_list[self.n_levels :]):
            passed = skip_list[-(idx + 1)]
            x = block(x, passed)
        x = self.conv_pred(x)
        output = self.softmax(x)

        return output


class SynthSeg:
    """This class is intended for the SynthSeg model.

    The SynthSeg model :footcite:p:`Billot2023` is a deep learning neural network for
    brain segmentation. It uses a U-net architecture and was trained on synthetic
    images generated from label maps. The model is robust to variations in
    contrast and resolution, making it suitable for segmenting a wide range of
    brain scans.

    Note that we are not saving any PVE maps here, only the hard segmentation
    due to the size of the output probability maps.

    References
    ----------
    .. footbibliography::
    """

    @doctest_skip_parser
    def __init__(self, *, verbose=False, use_cuda=False):
        """Model initialization

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

        self.model = self.init_model()
        self.model.eval()
        if use_cuda:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                logger.warning("CUDA requested but not found, switching to CPU")
                self.device = torch.device("cpu")
        else:
            self.device = torch.device("cpu")
        self.model = self.model.to(self.device)
        self.fetch_default_weights()
        self.labels_segmentation = np.array(
            [
                0,
                2,
                3,
                4,
                5,
                7,
                8,
                10,
                11,
                12,
                13,
                14,
                15,
                16,
                17,
                18,
                24,
                26,
                28,
                41,
                42,
                43,
                44,
                46,
                47,
                49,
                50,
                51,
                52,
                53,
                54,
                58,
                60,
            ]
        )
        self.topological_classes = np.array(
            [
                0,
                4,
                4,
                4,
                4,
                5,
                5,
                6,
                7,
                8,
                9,
                1,
                2,
                3,
                10,
                11,
                0,
                12,
                13,
                14,
                14,
                14,
                14,
                15,
                15,
                16,
                17,
                18,
                19,
                20,
                21,
                22,
                23,
            ]
        )
        self.flip_indices = np.array(
            [
                0,
                19,
                20,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                11,
                12,
                13,
                29,
                30,
                16,
                31,
                32,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                9,
                10,
                14,
                15,
                17,
                18,
            ]
        )
        self.label_dict = {
            0: "background",
            2: "left cerebral white matter",
            3: "left cerebral cortex",
            4: "left lateral ventricle",
            5: "left inferior lateral ventricle",
            7: "left cerebellum white matter",
            8: "left cerebellum cortex",
            10: "left thalamus",
            11: "left caudate",
            12: "left putamen",
            13: "left pallidum",
            14: "3rd ventricle",
            15: "4th ventricle",
            16: "brain-stem",
            17: "left hippocampus",
            18: "left amygdala",
            24: "CSF",
            26: "left accumbens area",
            28: "left ventral DC",
            41: "right cerebral white matter",
            42: "right cerebral cortex",
            43: "right lateral ventricle",
            44: "right inferior lateral ventricle",
            46: "right cerebellum white matter",
            47: "right cerebellum cortex",
            49: "right thalamus",
            50: "right caudate",
            51: "right putamen",
            52: "right pallidum",
            53: "right hippocampus",
            54: "right amygdala",
            58: "right accumbens area",
            60: "right ventral DC",
        }

    def init_model(self, model_scale=24, n_levels=5, output_channels=33):
        return Model(
            model_scale=model_scale, n_levels=n_levels, output_channels=output_channels
        )

    def fetch_default_weights(self):
        """Load the model pre-training weights to use for the fitting.

        While the user can load different weights, the function
        is mainly intended for the class function 'predict'.
        """
        fetch_model_weights_path = get_fnames(name="synthseg_torch_weights")
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

    def _flip_img_indices(self, img):
        new_img = np.zeros_like(img)
        for l_idx, flip_lbl in enumerate(self.flip_indices):
            if l_idx != 0:
                new_img = np.where(img == l_idx, flip_lbl, new_img)

    def _prepare_img(self, img):
        img = np.expand_dims(img, 1)  # add channel dimension
        img_tensor = torch.tensor(img, dtype=torch.float32).to(self.device)
        return img_tensor

    def __predict(self, x_test):
        """Internal prediction function

        Parameters
        ----------
        x_test : list of np.ndarray
            Image should match the required shape of the model.

        Returns
        -------
        np.ndarray (batch, ...)
            Predicted brain labels
        """
        return self.model(x_test).detach().numpy()

    def predict(
        self,
        T1,
        affine,
        *,
        batch_size=None,
        return_prob=False,
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
        batch_size : int, optional
            Number of images per prediction pass. Only available if data
            is provided with a batch dimension.
            Consider lowering it if you get an out of memory error.
            Increase it if you want it to be faster and have a lot of data.
            If None, batch_size will be set to 1 if the provided image
            has a batch dimension.
        return_prob : bool, optional
            Whether to return the probability map instead of a
            label map. Useful for testing.

        Returns
        -------
        pred_output : np.ndarray (...) or (batch, ...)
            Predicted brain labels. If return_prob is True, it will be a
            probability map instead of a label map.

        label_dict : dict
            Dictionary mapping label indices to anatomical structure names.
            Only if return_prob is False.

        mask : np.ndarray (...) or (batch, ...)
            Predicted brain mask.
            Only if return_prob is False.
        """
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
        else:
            raise ValueError(
                "Input data should be a np.ndarray of dimension 3 or a list/tuple of it"
            )
        if batch_size is None:
            batch_size = 1

        input_data = np.zeros((len(T1), 192, 192, 192))
        params_list = []

        # Normalize the data.
        ori_shape = T1.shape[1:]
        for i, T1_img in enumerate(T1):
            t_img, params = transform_img(
                T1_img,
                affine[i],
                target_voxsize=(1.0, 1.0, 1.0),
                final_size=(192, 192, 192),
                order=3,
            )
            min_v, max_v = np.percentile(t_img, (0.5, 99.5))
            t_img = normalize(t_img, min_v=min_v, max_v=max_v, new_min=0, new_max=1)
            input_data[i] = t_img
            params_list.append(params)
        # Prediction stage
        prediction = np.zeros((len(T1), 192, 192, 192, 33), dtype=np.float32)
        for batch_idx in range(batch_size, len(T1) + 1, batch_size):
            batch = input_data[batch_idx - batch_size : batch_idx]
            temp_input = self._prepare_img(batch)
            temp_pred = self.__predict(temp_input)
            temp_pred = gaussian_filter(temp_pred, (0, 0.5, 0.5, 0.5, 0))
            prediction[:batch_idx] = np.moveaxis(temp_pred, 1, -1)
            temp_input = self._prepare_img(np.flip(batch, axis=1).copy())
            temp_pred = self.__predict(temp_input)
            temp_pred = gaussian_filter(temp_pred, (0, 0.5, 0.5, 0.5, 0))
            temp_pred = np.flip(temp_pred, axis=2)
            temp_pred = np.stack(
                [
                    temp_pred[:, self.flip_indices[i]]
                    for i in range(len(self.flip_indices))
                ],
                axis=1,
            )
            prediction[:batch_idx] += np.moveaxis(temp_pred, 1, -1)
            prediction[:batch_idx] /= 2
        remainder = np.mod(len(T1), batch_size)
        if remainder != 0:
            batch = input_data[-remainder:]
            temp_input = self._prepare_img(batch)
            temp_pred = self.__predict(temp_input)
            temp_pred = gaussian_filter(temp_pred, (0, 0.5, 0.5, 0.5, 0))
            prediction[-remainder:] = np.moveaxis(temp_pred, 1, -1)
            temp_input = self._prepare_img(np.flip(batch, axis=1).copy())
            temp_pred = self.__predict(temp_input)
            temp_pred = gaussian_filter(temp_pred, (0, 0.5, 0.5, 0.5, 0))
            temp_pred = np.flip(temp_pred, axis=1)
            temp_pred = np.stack(
                [
                    temp_pred[:, self.flip_indices[i]]
                    for i in range(len(self.flip_indices))
                ],
                axis=1,
            )
            prediction[-remainder:] += np.moveaxis(temp_pred, 1, -1)
            prediction[-remainder:] /= 2

        if return_prob:
            labels = np.zeros((len(T1),) + (192, 192, 192, 33)).astype(np.float32)
        else:
            labels = np.zeros((len(T1),) + ori_shape).astype(np.int32)
            masks = np.zeros((len(T1),) + ori_shape)
        for i in range(len(T1)):
            output = prediction[i]

            tmp_post_patch_seg = output[..., 1:]
            post_patch_seg_mask = np.sum(tmp_post_patch_seg, axis=-1) > 0.25
            post_patch_seg_mask = remove_holes_and_islands(
                post_patch_seg_mask, remove_holes=False
            ).astype(bool)
            output[..., 1:] = output[..., 1:] * np.stack(
                [post_patch_seg_mask] * 32, axis=-1
            )

            post_patch_seg_mask = output > 0.25
            for topology_class in np.unique(self.topological_classes)[1:]:
                tmp_topology_indices = np.where(
                    self.topological_classes == topology_class
                )[0]
                tmp_mask = np.any(
                    post_patch_seg_mask[..., tmp_topology_indices], axis=-1
                )
                tmp_mask = remove_holes_and_islands(
                    tmp_mask, remove_holes=False
                ).astype(bool)
                for idx in tmp_topology_indices:
                    output[..., idx] *= tmp_mask

            output /= np.sum(output, axis=-1)[..., np.newaxis]
            if return_prob:
                labels[i] = output.astype("float32")
                continue
            else:
                temp = self.labels_segmentation[
                    output.argmax(-1).astype("int32")
                ].astype("int32")
                temp = recover_img(temp, params_list[i], order=0)
                labels[i] = np.round(temp).astype(np.int32)
            masks[i] = (labels[i] > 0).astype(np.int32)

        if dim == 3:
            labels = labels[0]
            if not return_prob:
                masks = masks[0]

        if return_prob:
            return labels
        return labels, self.label_dict, masks
