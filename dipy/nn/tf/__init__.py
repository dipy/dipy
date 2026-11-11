"""Neural network tools using TensorFlow backend.

Deprecated: the TensorFlow backend is removed in DIPY 2.0, together with the
models that only exist here. Install PyTorch to use :mod:`dipy.nn.torch`.
"""

from dipy.utils.deprecator import deprecate_with_version

_DEPRECATION = (
    "`dipy.nn.tf` module uses TensorFlow, which is deprecated in DIPY 1.10.0 "
    "and will be removed in DIPY 2.0, along with the models that have no "
    "PyTorch version (Synb0, Cnn1DDenoiser, SingleLayerPerceptron, "
    "MultipleLayerPercepton). Please install PyTorch to use the "
    "`dipy.nn.torch` module instead."
)

deprecate_with_version(_DEPRECATION, since="1.10", until="2.0")(lambda: None)()
