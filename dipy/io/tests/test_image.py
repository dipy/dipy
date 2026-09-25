"""Tests for the dipy.io.image module."""

import warnings

import numpy as np
import pytest

from dipy.io.image import load_nifti, load_nifti_data, save_nifti


def test_save_nifti_decfa(tmp_path):
    data = np.random.default_rng(0).random((3, 3, 3, 3))
    fname = str(tmp_path / "rgb.nii.gz")
    save_nifti(fname, data, np.eye(4), as_decfa=True)

    with pytest.warns(UserWarning, match="get_fdata"):
        out, _, loaded = load_nifti(fname, return_img=True)
    assert out.dtype == np.dtype([("R", "uint8"), ("G", "uint8"), ("B", "uint8")])
    assert out.shape == (3, 3, 3)
    assert loaded.header.get_intent("code")[0] == 1001

    with pytest.warns(UserWarning, match="structured RGB"):
        plain = load_nifti_data(fname)
    assert plain.dtype == np.uint8
    assert plain.shape == (3, 3, 3, 3)
    assert np.array_equal(plain, (data * 255).astype(np.uint8))


def test_save_nifti_scale_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    float_data = rng.random((2, 2, 2, 3))
    expected_scaled = (float_data * 255).astype(np.uint8)

    fname_auto = str(tmp_path / "float_auto.nii.gz")
    save_nifti(fname_auto, float_data, np.eye(4), as_decfa=True)
    with pytest.warns(UserWarning, match="structured RGB"):
        loaded_auto = load_nifti_data(fname_auto)
    assert np.array_equal(loaded_auto, expected_scaled)

    fname_true = str(tmp_path / "float_scale_true.nii.gz")
    save_nifti(fname_true, float_data, np.eye(4), as_decfa=True, scale=True)
    with pytest.warns(UserWarning, match="structured RGB"):
        loaded_true = load_nifti_data(fname_true)
    assert np.array_equal(loaded_true, expected_scaled)

    fname_false = str(tmp_path / "float_scale_false.nii.gz")
    save_nifti(fname_false, float_data, np.eye(4), as_decfa=True, scale=False)
    with pytest.warns(UserWarning, match="structured RGB"):
        unscaled = load_nifti_data(fname_false)
    assert np.array_equal(unscaled, float_data.astype(np.uint8))
    assert not np.array_equal(unscaled, expected_scaled)

    uint8_data = np.array([[[[10, 200, 255]]]], dtype=np.uint8)

    fname_uint8_auto = str(tmp_path / "uint8_auto.nii.gz")
    save_nifti(fname_uint8_auto, uint8_data, np.eye(4), as_decfa=True)
    with pytest.warns(UserWarning, match="structured RGB"):
        loaded_uint8_auto = load_nifti_data(fname_uint8_auto)
    assert np.array_equal(loaded_uint8_auto, uint8_data)

    fname_uint8_forced = str(tmp_path / "uint8_scale_true.nii.gz")
    save_nifti(fname_uint8_forced, uint8_data, np.eye(4), as_decfa=True, scale=True)
    with pytest.warns(UserWarning, match="structured RGB"):
        forced = load_nifti_data(fname_uint8_forced)
    expected_wrapped = (uint8_data * 255).astype(np.uint8)
    assert np.array_equal(forced, expected_wrapped)
    assert np.array_equal(
        expected_wrapped, np.array([[[[246, 56, 1]]]], dtype=np.uint8)
    )


def test_load_nifti_data_as_ndarray_false_skips_conversion(tmp_path):
    data = np.random.default_rng(2).random((2, 2, 2, 3))
    fname = str(tmp_path / "rgb.nii.gz")
    save_nifti(fname, data, np.eye(4), as_decfa=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        raw = load_nifti_data(fname, as_ndarray=False)
    assert raw.shape == (2, 2, 2)
    assert raw.dtype == np.dtype([("R", "uint8"), ("G", "uint8"), ("B", "uint8")])


def test_load_nifti_data_plain_data_no_warning(tmp_path):
    data = np.random.default_rng(3).random((2, 2, 2))
    fname = str(tmp_path / "plain.nii.gz")
    save_nifti(fname, data, np.eye(4))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        plain = load_nifti_data(fname)
    assert plain.shape == (2, 2, 2)


@pytest.mark.parametrize("as_ndarray", [True, False])
def test_load_nifti_rgb_return_img_warns(tmp_path, as_ndarray):
    data = np.random.default_rng(4).random((2, 2, 2, 3))
    fname = str(tmp_path / "rgb.nii.gz")
    save_nifti(fname, data, np.eye(4), as_decfa=True)

    with pytest.warns(UserWarning, match="get_fdata"):
        _, _, img = load_nifti(fname, return_img=True, as_ndarray=as_ndarray)

    raw = np.asarray(img.dataobj)
    assert raw.dtype == np.dtype([("R", "uint8"), ("G", "uint8"), ("B", "uint8")])
    with pytest.raises(TypeError):
        img.get_fdata()


def test_load_nifti_rgb_without_img_no_warning(tmp_path):
    data = np.random.default_rng(5).random((2, 2, 2, 3))
    fname = str(tmp_path / "rgb.nii.gz")
    save_nifti(fname, data, np.eye(4), as_decfa=True)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded, _ = load_nifti(fname)
    assert loaded.dtype == np.dtype([("R", "uint8"), ("G", "uint8"), ("B", "uint8")])
    assert loaded.shape == (2, 2, 2)


def test_load_nifti_plain_data_return_img_no_warning(tmp_path):
    data = np.random.default_rng(6).random((2, 2, 2))
    fname = str(tmp_path / "plain.nii.gz")
    save_nifti(fname, data, np.eye(4))

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _, _, img = load_nifti(fname, return_img=True)
    assert img.get_fdata().shape == (2, 2, 2)
