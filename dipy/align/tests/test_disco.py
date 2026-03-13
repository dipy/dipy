import numpy as np
from numpy.testing import assert_array_equal, assert_equal
import pytest

from dipy.align.disco import (
    HAVE_SYNB0,
    _save_debug_volume,
    _select_3d_volume,
    _validate_b0_index,
    _warp_image,
    synb0_syn,
)


def test_validate_b0_index_valid():
    assert _validate_b0_index(b0_index=0, n_volumes=5, image_name="dwi") == 0
    assert _validate_b0_index(b0_index=4, n_volumes=5, image_name="dwi") == 4
    # numpy integer types
    assert _validate_b0_index(b0_index=np.int64(2), n_volumes=5, image_name="dwi") == 2


def test_validate_b0_index_not_integer():
    with pytest.raises(TypeError, match="must be an integer"):
        _validate_b0_index(b0_index=1.5, n_volumes=5, image_name="dwi")


def test_validate_b0_index_out_of_range():
    with pytest.raises(ValueError, match="must be in"):
        _validate_b0_index(b0_index=5, n_volumes=5, image_name="dwi")
    with pytest.raises(ValueError, match="must be in"):
        _validate_b0_index(b0_index=-1, n_volumes=5, image_name="dwi")


def test_select_3d_volume_passthrough():
    vol = np.random.rand(10, 10, 10)
    result = _select_3d_volume(vol, image_name="test", b0_index=0)
    assert_array_equal(result, vol)


def test_select_3d_volume_from_4d():
    data = np.random.rand(10, 10, 10, 5)
    result = _select_3d_volume(data, image_name="test", b0_index=2)
    assert_array_equal(result, data[..., 2])


def test_select_3d_volume_wrong_ndim():
    with pytest.raises(ValueError, match="must be a 3D or 4D"):
        _select_3d_volume(np.zeros((5, 5)), image_name="test", b0_index=0)
    with pytest.raises(ValueError, match="must be a 3D or 4D"):
        _select_3d_volume(np.zeros((5, 5, 5, 5, 5)), image_name="test", b0_index=0)


class _FakeMapping:
    """Minimal mapping stub that applies a known transform."""

    def transform(self, image):
        return image * 2.0


def test_warp_image_3d():
    vol = np.ones((4, 4, 4))
    result = _warp_image(mapping=_FakeMapping(), image=vol)
    assert_array_equal(result, vol * 2.0)


def test_warp_image_4d():
    data = np.ones((4, 4, 4, 3))
    result = _warp_image(mapping=_FakeMapping(), image=data)
    assert_equal(result.shape, data.shape)
    assert_array_equal(result, data * 2.0)


def test_warp_image_wrong_ndim():
    with pytest.raises(ValueError, match="must be a 3D or 4D"):
        _warp_image(mapping=_FakeMapping(), image=np.zeros((4, 4)))


def test_save_debug_volume_none_dir():
    # Should be a no-op when debug_dir is None
    _save_debug_volume(
        debug_dir=None, name="test", data=np.zeros((3, 3, 3)), affine=np.eye(4)
    )


def test_save_debug_volume_writes_file(tmp_path):
    data = np.random.rand(3, 3, 3).astype(np.float32)
    _save_debug_volume(debug_dir=tmp_path, name="vol", data=data, affine=np.eye(4))
    assert (tmp_path / "vol.nii.gz").exists()


def test_save_debug_volume_bool_data(tmp_path):
    data = np.ones((3, 3, 3), dtype=bool)
    _save_debug_volume(debug_dir=tmp_path, name="mask", data=data, affine=np.eye(4))
    assert (tmp_path / "mask.nii.gz").exists()


@pytest.mark.skipif(
    not HAVE_SYNB0, reason="Only test ImportError when Synb0 is missing"
)
def test_synb0_syn_wrong_dwi_ndim():
    with pytest.raises(ValueError, match="dwi must be a 3D or 4D"):
        synb0_syn(
            dwi=np.zeros((5, 5)),
            T1=np.zeros((5, 5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
        )


@pytest.mark.skipif(
    not HAVE_SYNB0, reason="Only test ImportError when Synb0 is missing"
)
def test_synb0_syn_wrong_t1_ndim():
    with pytest.raises(ValueError, match="T1 must be a 3D or 4D"):
        synb0_syn(
            dwi=np.zeros((5, 5, 5)),
            T1=np.zeros((5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
        )


@pytest.mark.skipif(
    not HAVE_SYNB0, reason="Only test ImportError when Synb0 is missing"
)
def test_synb0_syn_bad_b0_index():
    with pytest.raises(ValueError, match="must be in"):
        synb0_syn(
            dwi=np.zeros((5, 5, 5, 3)),
            T1=np.zeros((5, 5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
            b0_index=10,
        )


@pytest.mark.skipif(
    not HAVE_SYNB0, reason="Only test ImportError when Synb0 is missing"
)
def test_synb0_syn_b0_index_type_error():
    with pytest.raises(TypeError, match="must be an integer"):
        synb0_syn(
            dwi=np.zeros((5, 5, 5, 3)),
            T1=np.zeros((5, 5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
            b0_index=0.5,
        )


@pytest.mark.skipif(
    not HAVE_SYNB0, reason="Only test ImportError when Synb0 is missing"
)
def test_synb0_syn_import_error_without_backend():
    with pytest.raises(ImportError, match="Synb0 model not available"):
        synb0_syn(
            dwi=np.zeros((5, 5, 5)),
            T1=np.zeros((5, 5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
        )


@pytest.mark.skipif(not HAVE_SYNB0, reason="Requires Synb0 (torch or tf)")
def test_synb0_syn_4d_input():
    """Run full pipeline on small 4D DWI dataset."""
    dwi = np.random.rand(32, 32, 32, 2).astype(np.float32) * 1000
    t1 = np.random.rand(32, 32, 32).astype(np.float32) * 150
    mask = np.ones((32, 32, 32), dtype=np.float32)
    affine = np.diag([2.0, 2.0, 2.0, 1.0])

    corrected, field = synb0_syn(
        dwi=dwi,
        T1=t1,
        dwi_affine=affine,
        T1_affine=affine,
        dwi_mask=mask,
        T1_mask=mask,
        return_field=True,
    )
    assert_equal(corrected.shape, dwi.shape)
    assert_equal(field.shape, dwi[..., 0].shape)
