# init for nn aka the deep neural network module
import os
import sys

from dipy.utils.deprecator import deprecate_with_version
from dipy.utils.optpkg import optional_package


def _load_backend():
    """Dynamically load the preferred backend based on the environment variable."""
    preferred_backend = os.getenv("DIPY_NN_BACKEND", "torch").lower()
    tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
    torch, have_torch, _ = optional_package("torch", min_version="2.2.0")

    __all__ = []

    if (have_torch and (preferred_backend == "torch" or not have_tf)) or (
        not have_tf and not have_torch
    ):
        import dipy.nn.torch.deepn4 as deepn4_module
        import dipy.nn.torch.evac as evac_module
        import dipy.nn.torch.histo_resdnn as histo_resdnn_module

        sys.modules["dipy.nn.evac"] = evac_module
        sys.modules["dipy.nn.histo_resdnn"] = histo_resdnn_module
        sys.modules["dipy.nn.deepn4"] = deepn4_module

        globals().update(
            {
                "evac": evac_module,
                "histo_resdnn": histo_resdnn_module,
                "deepn4": deepn4_module,
            }
        )

        __all__ += ["evac", "histo_resdnn", "deepn4"]

    elif have_tf:
        import dipy.nn.tf.cnn_1d_denoising as cnn_1d_denoising_module
        import dipy.nn.tf.deepn4 as deepn4_module
        import dipy.nn.tf.evac as evac_module
        import dipy.nn.tf.histo_resdnn as histo_resdnn_module
        import dipy.nn.tf.model as model_module
        import dipy.nn.tf.synb0 as synb0_module

        msg = (
            "`dipy.nn.tf` module uses TensorFlow, which is deprecated in DIPY 1.10.0. "
            "Please install PyTorch to use the `dipy.nn.torch` module instead."
        )
        dec = deprecate_with_version(msg, since="1.10.0", until="1.12.0")
        dec(lambda x=None: x)()

        sys.modules["dipy.nn.evac"] = evac_module
        sys.modules["dipy.nn.histo_resdnn"] = histo_resdnn_module
        sys.modules["dipy.nn.deepn4"] = deepn4_module
        sys.modules["dipy.nn.cnn_1d_denoising"] = cnn_1d_denoising_module
        sys.modules["dipy.nn.synb0"] = synb0_module
        sys.modules["dipy.nn.model"] = model_module
        globals().update(
            {
                "cnn_1d_denoising": cnn_1d_denoising_module,
                "deepn4": deepn4_module,
                "evac": evac_module,
                "histo_resdnn": histo_resdnn_module,
                "model": model_module,
                "synb0": synb0_module,
            }
        )

        __all__ += [
            "cnn_1d_denoising",
            "deepn4",
            "evac",
            "histo_resdnn",
            "model",
            "synb0",
        ]

    else:
        print(
            "Warning: Neither TensorFlow nor PyTorch is installed. "
            "Please install one of these packages."
        )

    return __all__


__all__ = _load_backend()
