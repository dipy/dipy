"""Class and helper functions for the MindGrab brain extraction model.

Note
----

The model architecture and pre-trained weights come from the BrainChop
project (https://github.com/neuroneural/brainchop-cli, MIT license). The CPU
inference code is adapted from brainchopC (MIT license).
"""

import numpy as np
from scipy.special import expit

from dipy.data import get_fnames
from dipy.nn.meshnet import meshnet_forward_cpu
from dipy.nn.utils import (
    load_state_dict,
    recover_img,
    set_logger_level,
    transform_img,
)
from dipy.segment.utils import remove_holes_and_islands
from dipy.testing.decorators import doctest_skip_parser
from dipy.utils.logging import get_logger
from dipy.utils.optpkg import optional_package

logger = get_logger(name="dipy.nn")

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
if have_torch:
    from torch.nn import GELU, Conv3d, GroupNorm, Module, Sequential
else:

    class Module:
        pass


MODEL_SHAPE = (256, 256, 256)
DILATIONS = (16, 8, 4, 2, 1) * 5
CHANNELS = 15
N_CLASSES = 2


def ras_to_lia(volume):
    """Reorient a RAS-aligned volume to the FreeSurfer LIA orientation.

    MindGrab was trained on volumes conformed to the FreeSurfer convention,
    where the voxel axes point Left, Inferior and Anterior.

    Parameters
    ----------
    volume : np.ndarray (X, Y, Z)
        Volume whose voxel axes point Right, Anterior and Superior.

    Returns
    -------
    np.ndarray (X, Z, Y)
        Volume whose voxel axes point Left, Inferior and Anterior.
    """
    return np.flip(volume, axis=(0, 2)).transpose(0, 2, 1)


def lia_to_ras(volume):
    """Reorient a FreeSurfer LIA volume back to a RAS-aligned volume.

    Parameters
    ----------
    volume : np.ndarray (X, Z, Y)
        Volume whose voxel axes point Left, Inferior and Anterior.

    Returns
    -------
    np.ndarray (X, Y, Z)
        Volume whose voxel axes point Right, Anterior and Superior.
    """
    return np.flip(volume.transpose(0, 2, 1), axis=(0, 2))


def quantile_normalize(volume, *, qmin=0.02, qmax=0.98, eps=1e-3):
    """Rescale intensities to [0, 1] between two quantiles, then clip.

    Parameters
    ----------
    volume : np.ndarray
        Input volume.
    qmin : float, optional
        Lower quantile mapped to 0.
    qmax : float, optional
        Upper quantile mapped to 1.
    eps : float, optional
        Added to the denominator to avoid a division by zero.

    Returns
    -------
    np.ndarray
        Normalized volume, as float32.
    """
    qlow, qhigh = np.quantile(volume, (qmin, qmax))
    volume = (volume - qlow) / (qhigh - qlow + eps)
    return np.clip(volume, 0, 1).astype(np.float32)


def read_mindgrab_weights(fname):
    """Read the MindGrab weights from a PyTorch state dict, without PyTorch.

    Parameters
    ----------
    fname : str or Path
        Path to the weights, saved by ``torch.save``.

    Returns
    -------
    kernels : list of np.ndarray (15, in_channels, 3, 3, 3)
        Weights of the dilated convolutions.
    classifier : np.ndarray (2, 15)
        Weights of the final 1x1x1 convolution.
    bias : np.ndarray (2,)
        Bias of the final 1x1x1 convolution.
    """
    state = load_state_dict(fname)
    n_layers = len(DILATIONS)
    try:
        kernels = [state[f"model.{i}.0.weight"] for i in range(n_layers)]
        classifier = state[f"model.{n_layers}.weight"]
        bias = state[f"model.{n_layers}.bias"]
    except KeyError as e:
        raise ValueError(f"{fname} does not hold MindGrab weights") from e

    expected = [(CHANNELS, 1, 3, 3, 3)] + [(CHANNELS, CHANNELS, 3, 3, 3)] * (
        n_layers - 1
    )
    if (
        len(state) != n_layers + 2
        or [kernel.shape for kernel in kernels] != expected
        or classifier.shape != (N_CLASSES, CHANNELS, 1, 1, 1)
        or bias.shape != (N_CLASSES,)
    ):
        raise ValueError(f"{fname} does not hold MindGrab weights")
    return kernels, classifier[:, :, 0, 0, 0], bias


class MeshNet(Module):
    """MeshNet architecture used by MindGrab.

    A fully convolutional network made of a stack of dilated 3D convolutions,
    each followed by a parameter-free group normalization and a GELU
    activation :footcite:p:`Fedorov2017`. MindGrab repeats the dilation
    schedule 16, 8, 4, 2, 1 five times :footcite:p:`Fani2025`.

    Parameters
    ----------
    in_channels : int, optional
        Number of input channels.
    channels : int, optional
        Number of feature channels of each hidden layer.
    n_classes : int, optional
        Number of output classes.
    dilations : sequence of int, optional
        Dilation of each hidden layer.

    References
    ----------
    .. footbibliography::
    """

    def __init__(
        self,
        *,
        in_channels=1,
        channels=CHANNELS,
        n_classes=N_CLASSES,
        dilations=DILATIONS,
    ):
        super().__init__()
        layers = []
        for i, dilation in enumerate(dilations):
            layers.append(
                Sequential(
                    Conv3d(
                        in_channels if i == 0 else channels,
                        channels,
                        kernel_size=3,
                        padding=dilation,
                        dilation=dilation,
                        bias=False,
                    ),
                    GroupNorm(channels, channels, affine=False),
                    GELU(approximate="tanh"),
                )
            )
        layers.append(Conv3d(channels, n_classes, kernel_size=1))
        self.model = Sequential(*layers)

    def forward(self, x):
        """Forward pass of the MeshNet model.

        Parameters
        ----------
        x : torch.Tensor (batch, in_channels, X, Y, Z)
            Input tensor.

        Returns
        -------
        torch.Tensor (batch, n_classes, X, Y, Z)
            Class logits.
        """
        return self.model(x)


class MindGrab:
    """This class is intended for the MindGrab model.

    The MindGrab model :footcite:p:`Fani2025` is a lightweight (146k
    parameters) fully convolutional network for skull stripping. It was
    trained only on synthetic data, which makes it modality agnostic: it
    handles T1w, T2w, FLAIR, DWI, CT or PET volumes alike.

    Inference runs on the CPU with a compiled kernel that needs no deep
    learning framework. PyTorch is only needed to run it on a GPU.

    References
    ----------
    .. footbibliography::
    """

    @doctest_skip_parser
    def __init__(self, *, verbose=False, use_cuda=False, num_threads=None):
        """Model initialization

        Parameters
        ----------
        verbose : bool, optional
            Whether to show information about the processing.
        use_cuda : bool, optional
            Whether to use GPU for processing, which requires PyTorch. CUDA is
            used when available, otherwise the Apple Metal (MPS) GPU. If False
            or no GPU is detected, CPU will be used.
        num_threads : int, optional
            Number of threads to use on the CPU. If None, the value of the
            ``OMP_NUM_THREADS`` environment variable is used if set, otherwise
            all available cores. If < 0, the maximal number of threads minus
            $|num_threads + 1|$ is used (enter -1 to use as many threads as
            possible). 0 raises an error.
        """
        log_level = "INFO" if verbose else "CRITICAL"
        set_logger_level(log_level, logger)

        self.num_threads = num_threads
        self.device = None
        if use_cuda:
            if not have_torch:
                logger.warning("GPU processing requires PyTorch, switching to CPU")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("CUDA not found, using the Apple MPS GPU instead")
            else:
                logger.warning("CUDA requested but not found, switching to CPU")

        self.model = None
        self.fetch_default_weights()

    def fetch_default_weights(self):
        """Load the model pre-training weights to use for the fitting.

        While the user can load different weights, the function
        is mainly intended for the class function 'predict'.
        """
        fetch_model_weights_path = get_fnames(name="mindgrab_weights")
        self.load_model_weights(fetch_model_weights_path)

    def load_model_weights(self, weights_path):
        """Load the custom pre-training weights to use for the fitting.

        Parameters
        ----------
        weights_path : str or Path
            Path to the file containing the weights (pth, saved by Pytorch)
        """
        kernels, classifier, bias = read_mindgrab_weights(weights_path)
        self.kernels, self.classifier, self.bias = kernels, classifier, bias
        if self.device is None:
            return

        state = {f"model.{i}.0.weight": kernel for i, kernel in enumerate(kernels)}
        state[f"model.{len(kernels)}.weight"] = classifier[..., None, None, None]
        state[f"model.{len(kernels)}.bias"] = bias
        self.model = MeshNet()
        self.model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
        self.model.to(self.device).eval()

    def _predict_volume(self, volume):
        """Compute the brain probability of a conformed volume.

        Parameters
        ----------
        volume : np.ndarray (256, 256, 256)
            Normalized volume in the LIA orientation.

        Returns
        -------
        np.ndarray (256, 256, 256)
            Brain probability.
        """
        if self.model is None:
            logits = meshnet_forward_cpu(
                volume,
                self.kernels,
                DILATIONS,
                self.classifier,
                self.bias,
                num_threads=self.num_threads,
            )
        else:
            x = torch.from_numpy(np.ascontiguousarray(volume)[None, None])
            with torch.no_grad():
                logits = self.model(x.to(self.device))[0].cpu().numpy()
        return expit(logits[1] - logits[0])

    def predict(self, image, affine, *, return_prob=False, finalize_mask=True):
        """Extract the brain from one or several volumes.

        Each volume is resampled to 1 mm isotropic, reoriented to LIA and
        padded or cropped to 256x256x256 before being fed to the model. The
        output is brought back to the original image space.

        Parameters
        ----------
        image : np.ndarray or list of np.ndarray
            For a single image, input should be a 3D array.
            If multiple images, it should be a list or tuple of 3D arrays.
        affine : np.ndarray (4, 4) or (batch, 4, 4)
            or list of np.ndarrays with len of batch
            Affine matrix of the image. Should have a batch dimension if
            ``image`` has one.
        return_prob : bool, optional
            Whether to return the brain probability map instead of a
            binary mask.
        finalize_mask : bool, optional
            Whether to remove potential holes or islands.
            Useful for solving minor errors. Ignored if ``return_prob``
            is True.

        Returns
        -------
        np.ndarray (...) or (batch, ...)
            Predicted brain mask, or brain probability map if
            ``return_prob`` is True, in the original image space.
        """
        affine = np.asarray(affine)
        if isinstance(image, (list, tuple)):
            single = False
        elif image.ndim == 3:
            single = True
            image = [image]
            affine = affine[None]
        else:
            raise ValueError(
                "Input data should be a np.ndarray of dimension 3 or a list/tuple of it"
            )

        outputs = []
        for img, aff in zip(image, affine):
            t_img, params = transform_img(
                np.asarray(img, dtype=np.float32),
                aff,
                target_voxsize=(1.0, 1.0, 1.0),
                final_size=MODEL_SHAPE,
                order=3,
            )
            t_img = quantile_normalize(ras_to_lia(t_img))
            prob = lia_to_ras(self._predict_volume(t_img))
            output = np.clip(recover_img(prob, params, order=1), 0, 1)
            if not return_prob:
                output = (output >= 0.5).astype(np.uint8)
                if finalize_mask:
                    output = remove_holes_and_islands(output).astype(np.uint8)
            outputs.append(output)

        return outputs[0] if single else np.stack(outputs)
