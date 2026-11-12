import warnings

import numpy as np
import numpy.testing as npt
from numpy.testing import assert_almost_equal
import pytest

from dipy.data import get_fnames
from dipy.nn.synb0 import N_FOLDS, Synb0, UNet3D
from dipy.nn.utils import normalize
from dipy.utils.optpkg import optional_package

tf, have_tf, _ = optional_package("tensorflow", min_version="2.18.0")
torch, have_torch, _ = optional_package("torch", min_version="2.2.0")
with warnings.catch_warnings():
    msg = ".*uses TensorFlow.*install PyTorch.*"
    warnings.filterwarnings("ignore", message=msg, category=DeprecationWarning)
    from dipy.nn.tf import synb0 as tf_synb0


def load_test_data():
    file_names = get_fnames(name="synb0_test_data")
    data = np.load(file_names[0])
    return data["b0"], data["T1"], np.load(file_names[1])["arr_0"]


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_weights():
    b0, t1, target = load_test_data()
    synb0_model = tf_synb0.Synb0()
    synb0_model.fetch_default_weights(0)
    results_arr = synb0_model.predict(b0[0], t1[0], average=False)
    assert_almost_equal(results_arr, target[0], decimal=1)


@pytest.mark.skipif(not have_tf, reason="Requires TensorFlow")
def test_default_weights_batch():
    b0, t1, target = load_test_data()
    synb0_model = tf_synb0.Synb0()
    synb0_model.fetch_default_weights(0)
    results_arr = synb0_model.predict(b0, t1, batch_size=2, average=False)
    assert_almost_equal(results_arr, target, decimal=1)


@pytest.mark.skipif(not have_torch, reason="Requires Torch")
def test_cpu_matches_torch():
    b0, t1, _ = load_test_data()
    x = np.stack(
        (
            normalize(
                np.pad(b0[0], ((2, 1), (3, 2), (2, 1))),
                min_v=0,
                max_v=np.percentile(b0[0], 99),
                new_min=-1,
                new_max=1,
            ),
            normalize(
                np.pad(t1[0], ((2, 1), (3, 2), (2, 1))),
                min_v=0,
                max_v=150,
                new_min=-1,
                new_max=1,
            ),
        )
    )[None].astype(np.float32)
    weights = get_fnames(name="synb0_default_torch_weights")
    synb0_model = Synb0(device="cpu")
    for fold in range(N_FOLDS):
        synb0_model.fetch_default_weights(fold)
        model = UNet3D(n_in=2, n_out=1)
        model.load_state_dict(
            torch.load(weights[fold], map_location="cpu", weights_only=True)
        )
        with torch.no_grad():
            expected = model.eval()(torch.from_numpy(x)).numpy()
        npt.assert_allclose(synb0_model._predict(x), expected, rtol=0, atol=1e-4)


def test_predict_matches_tensorflow():
    # The TensorFlow port normalizes with an epsilon of 1e-3 instead of the
    # 1e-5 the network was trained with, hence the tolerance.
    b0, t1, target = load_test_data()
    result = Synb0(device="cpu").predict(b0, t1, batch_size=2, average=False)
    npt.assert_allclose(result, target, rtol=0, atol=0.1)
    npt.assert_(np.mean(np.abs(result - target)) < 0.005)


def test_predict():
    b0, t1, _ = load_test_data()
    synb0_model = Synb0(device="cpu")
    npt.assert_equal(synb0_model.device, "cpu")

    single = []
    for fold in range(N_FOLDS):
        synb0_model.fetch_default_weights(fold)
        single.append(synb0_model.predict(b0[0], t1[0], average=False))
    npt.assert_equal(single[0].shape, b0.shape[1:])
    npt.assert_(np.all(np.isfinite(single[0])))
    npt.assert_(np.any(single[0] != 0))
    average = synb0_model.predict(b0[0], t1[0])
    npt.assert_allclose(average, np.mean(single, axis=0), rtol=0, atol=1e-5)

    batch = synb0_model.predict(b0, t1, batch_size=2, average=False)
    npt.assert_equal(batch.shape, b0.shape)
    npt.assert_allclose(batch[0], single[-1], rtol=0, atol=1e-5)
    npt.assert_allclose(
        batch[1],
        synb0_model.predict(b0[1], t1[1], average=False),
        rtol=0,
        atol=1e-5,
    )

    # The first default weights are used if none were loaded.
    npt.assert_array_equal(
        Synb0(device="cpu").predict(b0[0], t1[0], average=False), single[0]
    )


def test_wrong_inputs():
    npt.assert_raises(ValueError, Synb0, device="webgpu")
    synb0_model = Synb0(device="cpu")
    with pytest.raises(ValueError, match="do not match the declared model"):
        synb0_model.load_model_weights(get_fnames(name="mindgrab_weights"))

    b0 = np.zeros((77, 91, 77), dtype=np.float32)
    with pytest.raises(ValueError, match="Expected shape"):
        synb0_model.predict(b0[:-1], b0[:-1], average=False)
    with pytest.raises(ValueError, match="Expected shape"):
        synb0_model.predict(b0, np.zeros((80, 96, 80)), average=False)
