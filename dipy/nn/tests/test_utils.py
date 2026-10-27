from collections import OrderedDict
import os
import pickle
import subprocess
import sys
import warnings
import zipfile

import numpy as np
import numpy.testing as npt
import pytest

from dipy.nn.utils import (
    get_padded_shape,
    load_state_dict,
    normalize,
    recover_img,
    transform_img,
    unnormalize,
)
from dipy.testing.decorators import set_random_number_generator
from dipy.utils.optpkg import optional_package

torch, have_torch, _ = optional_package("torch", min_version="2.2.0")


@set_random_number_generator()
def test_norm(rng=None):
    temp = rng.random((8, 8, 8)) * 10
    temp2 = normalize(temp)
    temp2 = unnormalize(temp2, -1, 1, 0, 10)
    np.testing.assert_almost_equal(temp, temp2, 1)


@set_random_number_generator()
def test_transform(rng=None):
    temp = rng.random((28, 30, 34))
    temp2, params = transform_img(
        temp,
        np.eye(4),
        target_voxsize=tuple(np.ones(3) * 2),
        final_size=(14, 15, 16),
    )
    with warnings.catch_warnings():
        scipy_affine_txfm_msg = (
            "The behavior of affine_transform with a 1-D "
            "array supplied for the matrix parameter has changed in "
            "SciPy 0.18.0."
        )
        warnings.filterwarnings(
            "ignore", message=scipy_affine_txfm_msg, category=UserWarning
        )
        temp2 = recover_img(temp2, params)
    np.testing.assert_almost_equal(np.array(temp.shape), np.array(temp2.shape))


def test_get_padded_shape():
    cases = [
        ((32, 64, 96), 32, (32, 64, 96)),
        ((33, 65, 95), 32, (64, 96, 96)),
        ((1, 31, 32), 32, (32, 32, 32)),
        ((5, 10), 4, (8, 12)),
    ]

    for shape, multiple, expected_shape in cases:
        np.testing.assert_equal(get_padded_shape(shape, multiple), expected_shape)

    for multiple in [0, -1]:
        np.testing.assert_raises(ValueError, get_padded_shape, (32, 32, 32), multiple)


def _import_warnings(module):
    """Deprecation warnings a fresh interpreter emits when importing ``module``."""
    code = (
        "import warnings\n"
        "with warnings.catch_warnings(record=True) as caught:\n"
        "    warnings.simplefilter('always', DeprecationWarning)\n"
        f"    import {module}\n"
        "for w in caught:\n"
        "    if issubclass(w.category, DeprecationWarning):\n"
        "        print(str(w.message).splitlines()[0])\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return [line for line in result.stdout.splitlines() if "dipy.nn.tf" in line]


def test_tensorflow_backend_is_deprecated():
    (message,) = _import_warnings("dipy.nn.tf.model")
    npt.assert_equal("removed in DIPY 2.0" in message, True)
    npt.assert_equal("install PyTorch" in message, True)


def test_torch_backend_is_not_deprecated():
    npt.assert_equal(_import_warnings("dipy.nn.utils"), [])


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
@set_random_number_generator()
def test_load_state_dict(tmp_path, rng=None):
    storage = torch.from_numpy(rng.random((4, 6)).astype(np.float32))
    state = OrderedDict(
        weight=storage,
        transposed=storage.T,
        sliced=storage[1:3, 2:],
        double=torch.arange(5, dtype=torch.float64),
        count=torch.tensor(7),
    )
    fname = tmp_path / "state.pth"
    torch.save(state, fname)

    loaded = load_state_dict(fname)
    npt.assert_equal(list(loaded), list(state))
    for name, tensor in state.items():
        npt.assert_equal(loaded[name].dtype, tensor.numpy().dtype)
        npt.assert_array_equal(loaded[name], tensor.numpy())


def test_load_state_dict_wrong_files(tmp_path):
    not_zip = tmp_path / "not_zip.pth"
    not_zip.write_bytes(b"not a state dict")
    npt.assert_raises(ValueError, load_state_dict, not_zip)

    no_pickle = tmp_path / "no_pickle.pth"
    with zipfile.ZipFile(no_pickle, "w") as archive:
        archive.writestr("archive/version", "3")
    npt.assert_raises(ValueError, load_state_dict, no_pickle)

    unsafe = tmp_path / "unsafe.pth"
    with zipfile.ZipFile(unsafe, "w") as archive:
        archive.writestr("archive/data.pkl", pickle.dumps(os.system))
    npt.assert_raises(ValueError, load_state_dict, unsafe)
