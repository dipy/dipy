import nibabel as nib
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_array_equal, assert_equal
import pytest

from dipy.align.disco import (
    _pe_direction,
    _refine_b0_to_t1,
    _restrict_field_to_pe,
    _save_debug_volume,
    _set_forward_field,
    _warp_image,
    synb0_syn,
)
from dipy.align.imwarp import DiffeomorphicMap
from dipy.align.tests.test_bbr import (
    MOVING_AFFINE,
    STATIC_AFFINE,
    TRUTH,
    _error_mm,
    _setup,
)
from dipy.data import get_fnames
from dipy.io.image import load_nifti_data


def _translation_map(shift_mm, shape, *, inverse=False):
    """Map that warps with a constant world translation of `shift_mm`."""
    mapping = DiffeomorphicMap(dim=3, disp_shape=shape, domain_shape=shape)
    mapping.allocate()
    mapping.forward[...] = shift_mm
    mapping.backward[...] = -np.asarray(shift_mm)
    return mapping.inverse() if inverse else mapping


def test_warp_image_3d():
    vol = np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
    result = _warp_image(mapping=_translation_map([1, 0, 0], vol.shape), image=vol)
    assert_array_equal(result[:-1], vol[1:])


def test_warp_image_4d():
    data = np.arange(6 * 6 * 6 * 3, dtype=np.float32).reshape(6, 6, 6, 3)
    mapping = _translation_map([0, 1, 0], data.shape[:3])
    result = _warp_image(mapping=mapping, image=data, num_threads=1)
    assert_equal(result.shape, data.shape)
    assert_array_equal(result[:, :-1], data[:, 1:])


def test_warp_image_wrong_ndim():
    with pytest.raises(ValueError, match="must be a 3D or 4D"):
        _warp_image(mapping=_translation_map(0, (4, 4, 4)), image=np.zeros((4, 4)))


def test_set_forward_field():
    vol = np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
    field = np.zeros((6, 6, 6, 3), dtype=np.float32)
    field[..., 2] = 1
    # SymmetricDiffeomorphicRegistration returns an inverted map, whose
    # transform does not read mapping.forward.
    for inverse in (False, True):
        mapping = _translation_map([1, 0, 0], vol.shape, inverse=inverse)
        _set_forward_field(mapping, field)
        assert_array_equal(mapping.get_forward_field(), field)
        assert_array_equal(mapping.transform(vol)[..., :-1], vol[..., 1:])


def test_pe_direction():
    # Axis-aligned LAS grid: voxel axis 0 points to world -x.
    las = np.diag([-1.25, 1.25, 1.25, 1.0])
    axis, direction = _pe_direction(0, las)
    assert_equal(axis, 0)
    npt.assert_allclose(direction, [-1.0, 0.0, 0.0])
    # Permuted voxel axes: voxel axis 0 runs along world y.
    permuted = np.eye(4)[[1, 0, 2, 3]] * 2.0
    npt.assert_allclose(_pe_direction("i", permuted)[1], [0.0, 1.0, 0.0])
    npt.assert_allclose(_pe_direction("y", permuted)[1], [1.0, 0.0, 0.0])
    # BIDS PhaseEncodingDirection: the sign does not change the axis.
    npt.assert_allclose(_pe_direction("j-", permuted)[1], [1.0, 0.0, 0.0])
    npt.assert_allclose(_pe_direction("I-", permuted)[1], [0.0, 1.0, 0.0])
    # Oblique grid: the direction follows the rotation.
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    oblique = np.eye(4)
    oblique[:3, :3] = np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) * 2.0
    npt.assert_allclose(_pe_direction(1, oblique)[1], [0.0, c, s])
    for bad in (3, -1, "a", "j+", "-", True, 1.5):
        with pytest.raises(ValueError, match="pe_axis must be"):
            _pe_direction(bad, np.eye(4))


def test_restrict_field_to_pe():
    rng = np.random.default_rng(0)
    field = rng.normal(size=(6, 7, 8, 3)).astype(np.float32)
    c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
    direction = np.array([0.0, c, s])
    restricted, pe_disp = _restrict_field_to_pe(field, direction, 0)
    npt.assert_allclose(pe_disp, field @ direction, rtol=1e-5, atol=1e-6)
    npt.assert_allclose(
        restricted, pe_disp[..., None] * direction, rtol=1e-5, atol=1e-6
    )
    # Nothing is left across the direction.
    across = restricted - (restricted @ direction)[..., None] * direction
    npt.assert_allclose(across, 0, atol=1e-6)
    assert_equal(restricted.dtype, field.dtype)
    # Along a world axis this is the selection of one component.
    restricted, pe_disp = _restrict_field_to_pe(field, np.array([1.0, 0, 0]), 0)
    npt.assert_allclose(pe_disp, field[..., 0])
    npt.assert_allclose(restricted[..., 1:], 0)


def test_refine_b0_to_t1_falls_back():
    start = np.eye(4)
    start[:3, 3] = [1.0, 2.0, 3.0]
    vol = np.zeros((8, 8, 8))
    # An empty brain mask makes the boundary-based registration fail.
    affine = _refine_b0_to_t1(vol, vol, np.eye(4), np.eye(4), vol > 0, start)
    npt.assert_array_equal(affine, start)


def test_refine_b0_to_t1_refines():
    epi, t1, wm = _setup()
    affine = _refine_b0_to_t1(epi, t1, MOVING_AFFINE, STATIC_AFFINE, t1 > 0, np.eye(4))
    assert _error_mm(affine, TRUTH, wm=wm, static_affine=STATIC_AFFINE) < 0.75


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


def test_synb0_syn_wrong_dwi_ndim():
    with pytest.raises(ValueError, match="dwi must be a 3D or 4D"):
        synb0_syn(
            dwi=np.zeros((5, 5)),
            T1=np.zeros((5, 5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
        )


def test_synb0_syn_wrong_t1_ndim():
    with pytest.raises(ValueError, match="T1 must be a 3D or 4D"):
        synb0_syn(
            dwi=np.zeros((5, 5, 5)),
            T1=np.zeros((5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
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


def test_synb0_syn_b0_index_type_error():
    with pytest.raises(TypeError, match="must be an integer"):
        synb0_syn(
            dwi=np.zeros((5, 5, 5, 3)),
            T1=np.zeros((5, 5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
            b0_index=0.5,
        )


def test_synb0_syn_4d_input(tmp_path):
    """Run full pipeline on a real 4D DWI and T1 of the same subject."""
    # Random volumes do not register to the MNI template, so the synthesized
    # b0 can miss the DWI grid entirely and SyN then gets an empty image.
    fraw, _, _ = get_fnames(name="stanford_hardi")
    dwi_img = nib.load(fraw)
    dwi = np.asarray(dwi_img.dataobj[..., :2], dtype=np.float32)
    t1_img = nib.load(get_fnames(name="stanford_t1"))

    corrected, field = synb0_syn(
        dwi=dwi,
        T1=t1_img.get_fdata(dtype=np.float32),
        dwi_affine=dwi_img.affine,
        T1_affine=t1_img.affine,
        return_field=True,
        debug_dir=tmp_path,
    )
    assert_equal(corrected.shape, dwi.shape)
    assert_equal(field.shape, dwi[..., 0].shape)
    assert np.all(np.isfinite(field))
    npt.assert_(np.any(field != 0))

    # The DWI is warped with the returned field, along the phase encoding only.
    _, direction = _pe_direction(1, dwi_img.affine)
    mapping = DiffeomorphicMap(
        dim=3,
        disp_shape=field.shape,
        disp_grid2world=dwi_img.affine,
        domain_shape=field.shape,
        domain_grid2world=dwi_img.affine,
        codomain_shape=field.shape,
        codomain_grid2world=dwi_img.affine,
    )
    mapping.forward = (field[..., None] * direction).astype(np.float32)
    npt.assert_allclose(
        corrected[..., 1], mapping.transform(dwi[..., 1]), rtol=1e-4, atol=1e-3
    )

    # The correction brings the b0 closer to the synthesized undistorted b0.
    synthetic = load_nifti_data(tmp_path / "07_binf_in_dwi.nii.gz")
    inside = synthetic > 0

    def correlation(image):
        return np.corrcoef(image[inside], synthetic[inside])[0, 1]

    assert correlation(corrected[..., 0]) > correlation(dwi[..., 0])


def test_synb0_syn_bad_b0_to_t1_method():
    with pytest.raises(ValueError, match="b0_to_t1_method must be"):
        synb0_syn(
            dwi=np.zeros((5, 5, 5)),
            T1=np.zeros((5, 5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
            b0_to_t1_method="other",
        )


def test_synb0_syn_bad_pe_axis():
    with pytest.raises(ValueError, match="pe_axis must be"):
        synb0_syn(
            dwi=np.zeros((5, 5, 5)),
            T1=np.zeros((5, 5, 5)),
            dwi_affine=np.eye(4),
            T1_affine=np.eye(4),
            pe_axis=3,
        )
