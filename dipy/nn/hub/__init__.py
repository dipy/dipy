"""Hub module for loading pre-trained dMRI deep learning models."""

import importlib

from dipy.utils.logging import logger
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")

REGISTRY = {
    "SynthSeg": {
        "version": "1.0",
        "task": "brain_segmentation",
        "backend": "torch",
        "paper": "https://doi.org/10.1016/j.neuroimage.2023.119923",
        "input": "T1",
        "output": "brain_labels",
        "description": "Robust brain segmentation from T1 MRI.",
    }
}


def list_models(task=None):
    """List all models available in the DIPY hub.

    Parameters
    ----------
    task : str, optional
        Filter by task e.g. 'bundle_segmentation'.
        If None, all models are returned.

    Returns
    -------
    dict
        Registry entries matching the filter.
    """
    registry = REGISTRY
    if task:
        return {k: v for k, v in registry.items() if v.get("task") == task}
    return registry


def load(name, version=None, use_cuda=False):
    """Load a pre-trained model from the DIPY hub.

    Parameters
    ----------
    name : str
        Name of the model e.g. 'SynthSeg'.
    version : str, optional
        Version string. Uses registry default if None.
    use_cuda : bool, optional
        Whether to use GPU if available.

    Returns
    -------
    BaseHubAdapter
        A model adapter with a .predict() method.
    """
    registry = REGISTRY

    if name not in registry:
        raise ValueError(
            f"Model '{name}' not found in hub registry. "
            f"Available models: {list(registry.keys())}"
        )

    if not have_torch:
        raise ImportError(
            "PyTorch is required to load hub models. "
            "Install it with: pip install torch"
        )

    logger.info(f"Loading model '{name}' from DIPY hub...")

    # Load adapter
    adapter_path = f"dipy.nn.hub.adapters.{name.lower()}"
    adapter_class_name = f"{name}Adapter"

    try:
        module = importlib.import_module(adapter_path)
        adapter_class = getattr(module, adapter_class_name)
    except (ImportError, AttributeError) as e:
        raise ImportError(
            f"Could not load adapter for '{name}'. "
            f"Expected '{adapter_path}.{adapter_class_name}'."
        ) from e

    return adapter_class(use_cuda=use_cuda)
