import importlib
import sys
import warnings

import numpy as np
import numpy.testing as npt
import pytest

from dipy.data import get_fnames
from dipy.nn.deepn4 import DeepN4
from dipy.testing import assert_percent_almost_equal
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
if have_tf:
    with warnings.catch_warnings():
        msg = ".*uses TensorFlow.*install PyTorch.*"
        warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
        from dipy.nn.tf.deepn4 import DeepN4 as TFDeepN4


def _test_data():
    data = np.load(get_fnames(name="deepn4_test_data"))
    return data["input"], data["output"]


def test_default_weights():
    input_arr, target_arr = _test_data()
    deepn4_model = DeepN4(device="cpu")
    npt.assert_equal(deepn4_model.device, "cpu")
    results_arr = deepn4_model.predict(input_arr, np.eye(4))
    assert_percent_almost_equal(results_arr, target_arr, decimal=2, percent=0.95)


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_weights_tf():
    input_arr, target_arr = _test_data()
    deepn4_model = TFDeepN4()
    deepn4_model.fetch_default_weights()
    results_arr = deepn4_model.predict(input_arr, np.eye(4))
    assert_percent_almost_equal(results_arr, target_arr, decimal=2, percent=0.95)


def test_wrong_inputs():
    npt.assert_raises(ValueError, DeepN4, device="webgpu")
    with pytest.warns(DeprecationWarning, match="use_cuda"):
        npt.assert_raises(ValueError, DeepN4, use_cuda=True, device="cpu")
    deepn4_model = DeepN4(device="cpu", num_threads=1)
    npt.assert_raises(
        ValueError,
        deepn4_model.load_model_weights,
        get_fnames(name="mindgrab_weights"),
    )


@set_random_number_generator()
def test_pad(rng=None):
    deepn4_model = DeepN4(device="cpu")
    img = rng.random((13, 8, 10))
    padded, indices = deepn4_model.pad(img, 10)
    npt.assert_equal(padded.shape, (10, 10, 10))
    npt.assert_equal(indices, [0, 10, 1, 9, 0, 10, 1, 11, 0, 8, 0, 10])
    lx, lX, ly, lY, lz, lZ, rx, rX, ry, rY, rz, rZ = indices
    npt.assert_array_equal(padded[lx:lX, ly:lY, lz:lZ], img[rx:rX, ry:rY, rz:rZ])
    npt.assert_equal(padded[:, 0].sum() + padded[:, 9].sum(), 0)


def test_deprecated_torch_module():
    sys.modules.pop("dipy.nn.torch.deepn4", None)
    with pytest.warns(DeprecationWarning, match="dipy.nn.torch.deepn4"):
        module = importlib.import_module("dipy.nn.torch.deepn4")
    npt.assert_(module.DeepN4 is DeepN4)
