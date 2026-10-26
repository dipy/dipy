"""Tests for the dipy.io.image module."""

import warnings

import nibabel as nib
import numpy as np
import pytest

from dipy.io.image import load_nifti, load_nifti_data, save_nifti
from dipy.testing.decorators import set_random_number_generator


@set_random_number_generator()
def test_save_nifti_decfa(tmp_path, rng=None):
    data = rng.random((3, 3, 3, 3))
    fname = str(tmp_path / "rgb.nii.gz")
    save_nifti(fname, data, np.eye(4), as_decfa=True)

    with pytest.warns(UserWarning, match="auto-converted"):
        out, _, loaded = load_nifti(fname, return_img=True)
    assert out.dtype == np.uint8
    assert out.shape == (3, 3, 3, 3)
    np.testing.assert_array_equal(out, (data * 255).astype(np.uint8))
    assert loaded.header.get_intent("code")[0] == 1001

    with pytest.warns(UserWarning, match="structured RGB"):
        plain = load_nifti_data(fname)
    assert plain.dtype == np.uint8
    assert plain.shape == (3, 3, 3, 3)
    assert np.array_equal(plain, (data * 255).astype(np.uint8))


@set_random_number_generator()
def test_save_nifti_scale_round_trip(tmp_path, rng=None):
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


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32])
@pytest.mark.parametrize("scale", [None, False, True])
def test_save_nifti_integer_color_preserves_channels(tmp_path, dtype, scale):
    data = np.array([[[[10, 200, 255], [0, 1, 128]]]], dtype=dtype)
    fname = tmp_path / "integer_color.nii.gz"
    save_nifti(fname, data, np.eye(4), as_decfa=True, scale=scale)

    with pytest.warns(UserWarning, match="structured RGB"):
        loaded = load_nifti_data(fname)
    assert loaded.dtype == np.uint8
    np.testing.assert_array_equal(loaded, data)


@pytest.mark.parametrize("channels", ["RGB", "RGBA"])
@pytest.mark.parametrize("as_ndarray", [True, False])
@set_random_number_generator()
def test_load_nifti_color_return_img(tmp_path, channels, as_ndarray, rng=None):
    values = rng.integers(0, 256, size=(2, 3, 4, len(channels)), dtype=np.uint8)
    color_dtype = np.dtype([(name, "uint8") for name in channels])
    raw = np.empty(values.shape[:3], dtype=color_dtype)
    for index, name in enumerate(channels):
        raw[name] = values[..., index]
    affine = np.diag([-2.0, 3.0, 4.0, 1.0])
    affine[:3, 3] = [5, 6, 7]
    source_img = nib.Nifti1Image(raw, affine)
    source_img.header.set_intent(1001)
    fname = tmp_path / "color.nii.gz"
    nib.save(source_img, fname)

    with (
        pytest.warns(UserWarning, match="auto-converted")
        if as_ndarray
        else warnings.catch_warnings(action="error")
    ):
        data, loaded_affine, img, zooms, orientation = load_nifti(
            fname,
            return_img=True,
            return_voxsize=True,
            return_coords=True,
            as_ndarray=as_ndarray,
        )
    if as_ndarray:
        assert data.shape == values.shape
        assert data.dtype == np.uint8
        np.testing.assert_array_equal(data, values)
    else:
        assert isinstance(data, nib.arrayproxy.ArrayProxy)
        assert data is img.dataobj
        assert data.shape == raw.shape
        assert data.dtype == color_dtype
        np.testing.assert_array_equal(np.asanyarray(data), raw)
    assert img.shape == raw.shape
    assert img.get_data_dtype() == color_dtype
    assert img.header.get_intent("code")[0] == 1001
    assert isinstance(img.dataobj, nib.arrayproxy.ArrayProxy)
    np.testing.assert_array_equal(np.asanyarray(img.dataobj), raw)
    np.testing.assert_array_equal(loaded_affine, affine)
    np.testing.assert_array_equal(img.affine, affine)
    assert zooms == (2, 3, 4)
    assert orientation == ("L", "A", "S")
    with pytest.raises(TypeError):
        img.get_fdata()


@pytest.mark.parametrize("channels", ["RGB", "RGBA"])
@pytest.mark.parametrize("as_ndarray", [True, False])
@pytest.mark.parametrize("loader", [load_nifti, load_nifti_data])
@set_random_number_generator()
def test_load_nifti_color_without_img(tmp_path, channels, as_ndarray, loader, rng=None):
    values = rng.integers(0, 256, size=(2, 3, 4, len(channels)), dtype=np.uint8)
    raw = np.empty(values.shape[:3], dtype=[(name, "uint8") for name in channels])
    for index, name in enumerate(channels):
        raw[name] = values[..., index]
    fname = tmp_path / "color.nii.gz"
    nib.save(nib.Nifti1Image(raw, np.eye(4)), fname)

    if as_ndarray:
        with pytest.warns(UserWarning, match="auto-converted"):
            result = loader(fname)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = loader(fname, as_ndarray=False)
    data = result[0] if loader is load_nifti else result
    if as_ndarray:
        assert data.dtype == np.uint8
        np.testing.assert_array_equal(data, values)
    else:
        assert isinstance(data, nib.arrayproxy.ArrayProxy)
        assert data.dtype == raw.dtype
        np.testing.assert_array_equal(np.asanyarray(data), raw)


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 3, 4, 5)])
@pytest.mark.parametrize("as_ndarray", [True, False])
@set_random_number_generator()
def test_load_nifti_numeric_contract(tmp_path, shape, as_ndarray, rng=None):
    data = rng.random(shape, dtype=np.float32)
    affine = np.diag([2.0, 3.0, 4.0, 1.0])
    fname = tmp_path / "plain.nii.gz"
    save_nifti(fname, data, affine)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        loaded, loaded_affine, img, zooms, orientation = load_nifti(
            fname,
            return_img=True,
            return_voxsize=True,
            return_coords=True,
            as_ndarray=as_ndarray,
        )
        only_data = load_nifti_data(fname, as_ndarray=as_ndarray)
    np.testing.assert_array_equal(np.asanyarray(loaded), data)
    np.testing.assert_array_equal(np.asanyarray(only_data), data)
    np.testing.assert_array_equal(img.get_fdata(), data)
    np.testing.assert_array_equal(loaded_affine, affine)
    assert loaded.dtype == data.dtype
    assert only_data.dtype == data.dtype
    assert zooms == (2, 3, 4)
    assert orientation == ("R", "A", "S")
