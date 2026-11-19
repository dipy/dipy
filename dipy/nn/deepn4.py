#!/usr/bin/python
"""Class and helper functions for fitting the DeepN4 model.

Inference runs on the CPU with a compiled kernel that needs no deep learning
framework, or with PyTorch on a GPU.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

from dipy.data import get_fnames
from dipy.nn.utils import (
    UNET3D_DEVICES,
    load_state_dict,
    load_unet3d,
    normalize,
    pad_crop,
    recover_img,
    run_unet3d,
    set_logger_level,
    torch_gpu_device,
    transform_img,
)
from dipy.utils.deprecator import deprecated_params
from dipy.utils.logging import get_logger

# A child of the DIPY logger: ``verbose`` only quiets the model messages.
logger = get_logger(name="dipy.nn")


class DeepN4:
    """This class is intended for the DeepN4 model.

    The DeepN4 model :footcite:p:`Kanakaraj2024` predicts the bias field for
    magnetic field inhomogeneity correction on T1-weighted images.

    Inference runs on the CPU with a compiled kernel that needs no deep
    learning framework, or with PyTorch on a GPU.

    References
    ----------
    .. footbibliography::
    """

    @deprecated_params("use_cuda", since="1.13", until="1.15")
    def __init__(
        self, *, verbose=False, use_cuda=None, device="auto", num_threads=None
    ):
        """Model initialization

        The default weights are loaded at initialization. Other weights can
        be loaded with ``load_model_weights``.

        This model is designed to take as input file T1 signal and predict
        bias field. Effectively, this model is mimicking bias correction.

        Parameters
        ----------
        verbose : bool, optional
            Whether to show information about the processing.
        use_cuda : bool, optional
            .. deprecated:: 1.13.0
               Use ``device='cuda'`` instead. This parameter will be removed
               in 1.15.0.

            Same as ``device='cuda'``. It cannot be combined with
            ``device='cpu'``.
        device : {'auto', 'cpu', 'cuda'}, optional
            Where to run the network.

            - 'auto': CUDA if PyTorch finds an NVIDIA GPU, otherwise the CPU.
            - 'cpu': compiled CPU kernel, which needs no extra package.
            - 'cuda': PyTorch on CUDA, or on the Apple MPS GPU.

            If the requested GPU is not available, the CPU is used.
        num_threads : int, optional
            Number of threads to use on the CPU. If None, the value of the
            ``OMP_NUM_THREADS`` environment variable is used if set, otherwise
            all available cores. If < 0, the maximal number of threads minus
            $|num_threads + 1|$ is used (enter -1 to use as many threads as
            possible). 0 raises an error. The matrix products use the threads
            of the BLAS library.
        """
        if device not in UNET3D_DEVICES:
            raise ValueError(
                f"device should be one of {UNET3D_DEVICES}, got {device!r}"
            )
        if use_cuda and device == "cpu":
            raise ValueError("use_cuda=True conflicts with device='cpu'")
        if use_cuda:
            device = "cuda"
        log_level = "INFO" if verbose else "CRITICAL"
        set_logger_level(log_level, logger)

        self.num_threads = num_threads
        self.torch_device = torch_gpu_device(device)
        if self.torch_device is None and device == "cuda":
            logger.warning("No GPU found by PyTorch, switching to CPU")
        self.device = "cpu" if self.torch_device is None else self.torch_device.type
        logger.info(f"DeepN4 runs on {self.device}")

        self.network = None
        self.fetch_default_weights()

    def fetch_default_weights(self):
        """Load the model pre-training weights to use for the fitting."""
        fetch_model_weights_path = get_fnames(name="deepn4_default_torch_weights")
        self.load_model_weights(fetch_model_weights_path)

    def load_model_weights(self, weights_path):
        """Load the custom pre-training weights to use for the fitting.

        Parameters
        ----------
        weights_path : str or Path
            Path to the file containing the weights, saved by PyTorch. It
            holds either the state dict of the network or a checkpoint
            with the state dict under ``model_state_dict``.
        """
        try:
            state = load_state_dict(weights_path)
        except ValueError as e:
            raise ValueError(
                "Expected input for the provided model weights "
                "do not match the declared model"
            ) from e
        state = state.get("model_state_dict", state)
        self.network = load_unet3d(state, torch_device=self.torch_device)

    def pad(self, img, sz):
        """Pad or crop the image to the specified size.

        Parameters
        ----------
        img : np.ndarray
            The image to pad or crop.
        sz : int
            The target size for each dimension.

        Returns
        -------
        padded : np.ndarray
            The padded or cropped image.
        indices : list
            A list of indices used for padding and cropping.
        """
        padded, pad_vs, crop_vs = pad_crop(img, (sz,) * 3)
        padded_end = sz - pad_vs[:, 1]
        crop_end = np.array(img.shape[:3]) - crop_vs[:, 1]
        indices = [
            int(v)
            for bounds in (zip(pad_vs[:, 0], padded_end), zip(crop_vs[:, 0], crop_end))
            for pair in bounds
            for v in pair
        ]
        return padded, indices

    def load_resample(self, subj):
        """Preprocess the subject image for the model.

        This includes padding and normalization.

        Parameters
        ----------
        subj : np.ndarray
            The subject image.

        Returns
        -------
        tuple
            The processed volume, as a float32 array of shape
            (1, 1, 128, 128, 128), followed by auxiliary parameters.
        """
        input_data, [lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ] = self.pad(
            subj, 128
        )
        in_max = np.percentile(input_data[np.nonzero(input_data)], 99.99)
        input_data = normalize(input_data, min_v=0, max_v=in_max, new_min=0, new_max=1)
        input_data = np.squeeze(input_data)
        input_vols = np.zeros((1, 1, 128, 128, 128), dtype=np.float32)
        input_vols[0, 0, :, :, :] = input_data

        return (
            input_vols,
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

    def predict(self, img, img_affine, *, threshold=0.5):
        """Wrapper function to facilitate prediction of larger dataset.
        The function will mask, normalize, split, predict and 're-assemble'
        the data as a volume.

        Parameters
        ----------
        img : np.ndarray
            T1 image to predict and apply bias field
        img_affine : np.ndarray (4, 4)
            Affine matrix for the T1 image
        threshold : float, optional
            Threshold for cleaning the final correction field

        Returns
        -------
        final_corrected : np.ndarray (x, y, z)
            Predicted bias corrected image.
            The volume has matching shape to the input data
        """
        # Preprocess input data (resample, normalize, and pad)
        resampled_T1, params = transform_img(
            img,
            img_affine,
            target_voxsize=(2.0, 2.0, 2.0),
            final_size=(128, 128, 128),
            order=3,
        )
        (in_features, lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ, in_max) = (
            self.load_resample(resampled_T1)
        )

        # Run the model to get the bias field
        logfield = run_unet3d(self.network, in_features, num_threads=self.num_threads)
        field = np.exp(logfield)
        field = field.squeeze()

        # Postprocess predicted field (reshape - unpad, smooth the field,
        # upsample)
        final_field = np.zeros(
            [resampled_T1.shape[0], resampled_T1.shape[1], resampled_T1.shape[2]]
        )
        final_field[rx:rX, ry:rY, rz:rZ] = field[lx:lX, ly:lY, lz:lZ]
        final_fields = gaussian_filter(final_field, sigma=3)
        upsample_final_field = recover_img(final_fields, params)
        # Correct the image
        below_threshold_mask = np.abs(upsample_final_field) < threshold
        with np.errstate(divide="ignore", invalid="ignore"):
            final_corrected = np.where(
                below_threshold_mask, 0, img / upsample_final_field
            )

        return final_corrected
