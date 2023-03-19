from dipy.utils.optpkg import optional_package
from packaging.version import parse

tf, have_tf, _ = optional_package("tensorflow")
layers, _, _ = optional_package("tensorflow.keras.layers")

if have_tf:
    if parse(tf.__version__) < parse('2.0.0'):
        raise ImportError('Please upgrade to TensorFlow 2+')

    from dipy.nn.cnn_1denoiser.cnn_1denoiser import Cnn1DDenoiser
