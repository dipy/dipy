"""Class and helper functions for the Synb0 model.

Inference runs on the CPU with a compiled kernel that needs no deep learning
framework, or with PyTorch on a GPU.
"""

import numpy as np

from dipy.data import get_fnames
from dipy.nn.utils import (
    UNET3D_DEVICES,
    load_state_dict,
    load_unet3d,
    normalize,
    run_unet3d,
    set_logger_level,
    torch_gpu_device,
    unnormalize,
)
from dipy.utils.logging import get_logger

logger = get_logger(name="dipy.nn")


_SYNB0_INPUT_SHAPE = (77, 91, 77)
# The network was trained on volumes with the axes ordered (z, x, y).
_SYNB0_TO_NETWORK = (0, 1, 4, 2, 3)
_SYNB0_FROM_NETWORK = (0, 1, 3, 4, 2)
_SYNB0_PADDING = ((0, 0), (2, 1), (3, 2), (2, 1))
_SYNB0_UNPAD = (slice(None), 0, slice(2, -1), slice(3, -2), slice(2, -1))
N_FOLDS = 5


class Synb0:
    """
    This class is intended for the Synb0 model.

    Synb0 :footcite:p:`Schilling2019`, :footcite:p:`Schilling2020` uses a neural
    network to synthesize a b0 volume for distortion correction in DWI images.

    The model is the deep learning part of the Synb0-Disco pipeline, thus
    stand-alone usage is not recommended.

    Inference runs on the CPU with a compiled kernel that needs no deep
    learning framework, or with PyTorch on CUDA.

    References
    ----------
    .. footbibliography::
    """

    def __init__(self, *, verbose=False, device="auto", num_threads=None):
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
        log_level = "INFO" if verbose else "CRITICAL"
        set_logger_level(log_level, logger)

        self.num_threads = num_threads
        self.torch_device = torch_gpu_device(device)
        if self.torch_device is None and device == "cuda":
            logger.warning("No GPU found by PyTorch, switching to CPU")
        self.device = "cpu" if self.torch_device is None else self.torch_device.type
        logger.info(f"Synb0 runs on {self.device}")

        self.network = None
        self._default_states = {}

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
        if idx not in self._default_states:
            weights_path = get_fnames(name="synb0_default_torch_weights")[idx]
            logger.info(f"fetched {weights_path}")
            self._default_states[idx] = load_state_dict(weights_path)
        self.network = load_unet3d(
            self._default_states[idx], torch_device=self.torch_device
        )

    def load_model_weights(self, weights_path):
        r"""
        Load the custom pre-training weights to use for the fitting.

        Parameters
        ----------
        weights_path : str or Path
            Path to the file containing the weights (.pth file saved by PyTorch)
        """
        try:
            state = load_state_dict(weights_path)
        except ValueError as e:
            raise ValueError(
                "Expected input for the provided model weights "
                "do not match the declared model"
            ) from e
        self.network = load_unet3d(state, torch_device=self.torch_device)

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

        average : bool, optional
            Whether the function follows the Synb0-Disco pipeline and
            averages the prediction of 5 different models.
            The default weights are then used, and the loaded weights are
            kept for the next predictions with ``average=False``.
            If False, it uses the loaded weights for prediction, or the
            first default weights if none were loaded.

        Returns
        -------
        pred_output : np.ndarray (...) or (batch, ...)
            Reconstructed b-inf image(s)
        """
        # Integer inputs would truncate the normalized values written back.
        b0 = np.asarray(b0)
        T1 = np.asarray(T1)
        if not np.issubdtype(b0.dtype, np.floating):
            b0 = b0.astype(np.float32)
        if not np.issubdtype(T1.dtype, np.floating):
            T1 = T1.astype(np.float32)
        # Check if shape is as intended
        if (
            b0.shape[1:] != _SYNB0_INPUT_SHAPE and b0.shape != _SYNB0_INPUT_SHAPE
        ) or b0.shape != T1.shape:
            raise ValueError(
                f"Expected shape (batch, {_SYNB0_INPUT_SHAPE}) or "
                f"{_SYNB0_INPUT_SHAPE} for both inputs"
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
        T1 = np.pad(T1, _SYNB0_PADDING, "constant")
        b0 = np.pad(b0, _SYNB0_PADDING, "constant")

        # Normalize the data
        p99 = np.percentile(b0, 99, axis=(1, 2, 3))
        if np.any(p99 <= 0):
            raise ValueError(
                "The b0 image has no positive signal: its 99th percentile is "
                "not above 0."
            )
        for i in range(shape[0]):
            T1[i] = normalize(T1[i], min_v=0, max_v=150, new_min=-1, new_max=1)
            b0[i] = normalize(b0[i], min_v=0, max_v=p99[i], new_min=-1, new_max=1)

        if dim == 3 and batch_size != 1:
            logger.warning(
                "Batch size specified, but not used "
                "due to the input not having a batch dimension"
            )
            batch_size = 1

        input_data = np.stack((b0, T1), axis=1).astype(np.float32)
        input_data = np.ascontiguousarray(input_data.transpose(_SYNB0_TO_NETWORK))
        prediction = np.empty((shape[0], 1) + input_data.shape[2:], dtype=np.float32)
        if average:
            mean_pred = np.zeros(shape, dtype=np.float32)

        if not average and self.network is None:
            self.fetch_default_weights(0)
        loaded = self.network

        for model_idx in range(N_FOLDS if average else 1):
            if average:
                self.fetch_default_weights(model_idx)

            for start in range(0, shape[0], batch_size):
                stop = min(start + batch_size, shape[0])
                prediction[start:stop] = run_unet3d(
                    self.network,
                    input_data[start:stop],
                    num_threads=self.num_threads,
                )

            unpadded = prediction.transpose(_SYNB0_FROM_NETWORK)[_SYNB0_UNPAD]
            for j in range(shape[0]):
                unpadded[j] = unnormalize(unpadded[j], -1, 1, 0, p99[j])

            if average:
                mean_pred += unpadded

        if average:
            self.network = loaded
        prediction = mean_pred / N_FOLDS if average else unpadded

        if dim == 3:
            prediction = prediction[0]

        return prediction
