"""Hub adapter for the SynthSeg brain segmentation model."""

from dipy.nn.hub.base import BaseHubAdapter
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")


class SynthSegAdapter(BaseHubAdapter):
    """Hub adapter for the SynthSeg model.

    Wraps dipy.nn.torch.synthseg.SynthSeg for use via dipy.nn.hub.

    Parameters
    ----------
    weights_path : str
        Path to the model weights file.
    use_cuda : bool, optional
        Whether to use GPU if available.
    """

    def __init__(self, weights_path=None, use_cuda=False):
        if not have_torch:
            raise ImportError("PyTorch is required. Install with: pip install torch")
        from dipy.nn.torch.synthseg import SynthSeg

        self._model = SynthSeg(use_cuda=use_cuda)

    def init_model(self):
        """Not used directly — SynthSeg initializes its own model."""
        pass

    def load_model_weights(self, weights_path):
        """Not used directly — handled in __init__."""
        pass

    def predict(self, T1, affine, *, batch_size=None, return_prob=False):
        """Run brain segmentation on a T1 image.

        Parameters
        ----------
        T1 : np.ndarray
            Input T1 weighted image, 3D array.
        affine : np.ndarray (4, 4)
            Affine matrix for the T1 image.
        batch_size : int, optional
            Number of images per prediction pass.
        return_prob : bool, optional
            Whether to return probability map instead of label map.

        Returns
        -------
        Same as dipy.nn.torch.synthseg.SynthSeg.predict()
        """
        return self._model.predict(
            T1, affine, batch_size=batch_size, return_prob=return_prob
        )
