"""PyTorch helpers to run DIPY networks on a GPU."""

import numpy as np

from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")


def gpu_device(*, mps):
    """Find a GPU that PyTorch can use.

    Parameters
    ----------
    mps : bool
        Whether to fall back on the Apple MPS GPU if there is no CUDA GPU.

    Returns
    -------
    torch.device or None
        The GPU, or None if PyTorch is missing or finds none.
    """
    if not have_torch:
        return None
    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def load_torch_model(model, state, device):
    """Load a NumPy state dict in a PyTorch network, ready for inference.

    Parameters
    ----------
    model : torch.nn.Module
        Network matching ``state``.
    state : dict
        Parameter names mapped to NumPy arrays.
    device : torch.device
        Device to move the network to.

    Returns
    -------
    torch.nn.Module
        ``model`` with the weights of ``state``, on ``device``, in evaluation
        mode.
    """
    model.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    return model.to(device).eval()


def run_torch_model(model, x):
    """Run a PyTorch network for inference on NumPy data.

    Parameters
    ----------
    model : torch.nn.Module
        Network, on the device to run it on.
    x : np.ndarray
        Network input.

    Returns
    -------
    np.ndarray
        Network output.
    """
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(device)
        return model(x).cpu().numpy()
