"""Base class for all dipy.nn.hub model adapters."""

from abc import ABC, abstractmethod

from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")


class BaseHubAdapter(ABC):
    """Base adapter class all hub models must follow.

    Parameters
    ----------
    weights_path : str
        Path to the model weights file.
    use_cuda : bool, optional
        Whether to use GPU. Falls back to CPU if unavailable.
    """

    def __init__(self, weights_path, use_cuda=False):
        if not have_torch:
            raise ImportError(
                "PyTorch is required to use dipy.nn.hub. "
                "Install it with: pip install torch"
            )
        self.device = (
            torch.device("cuda")
            if use_cuda and torch.cuda.is_available()
            else torch.device("cpu")
        )
        self.model = self.init_model()
        self.load_model_weights(weights_path)

    @abstractmethod
    def init_model(self):
        """Return the raw PyTorch model instance."""

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Run inference on DIPY-native inputs."""

    def load_model_weights(self, weights_path):
        """Load weights from path into the model.

        Parameters
        ----------
        weights_path : str
            Path to the .pth weights file.
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
                "Model weights do not match the declared model architecture."
            ) from e
