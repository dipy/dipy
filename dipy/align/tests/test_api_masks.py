import nibabel as nib
import numpy as np
import numpy.testing as npt
import pytest

from dipy.align import affine_registration


def _mask_input(mask, affine, kind, tmp_path, name):
    image = nib.Nifti1Image(mask, affine)
    if kind == "nifti":
        return image
    path = tmp_path / f"{name}.nii.gz"
    nib.save(image, path)
    return str(path) if kind == "str" else path


@pytest.mark.parametrize("kind", ["nifti", "str", "path"])
@pytest.mark.parametrize("side", ["static", "moving", "both"])
@pytest.mark.parametrize("different_grid", [False, True])
def test_affine_registration_mask_inputs(kind, side, different_grid, tmp_path):
    image = np.broadcast_to(np.arange(1.0, 8.0)[:, None, None], (7, 7, 7)).copy()
    static_mask = np.zeros(image.shape, dtype=np.uint8)
    moving_mask = np.zeros(image.shape, dtype=np.uint8)
    static_mask[2:4] = 1
    moving_mask[4:6] = 1
    mask_affine = np.eye(4)
    source_static, source_moving = static_mask, moving_mask
    if different_grid:
        mask_affine = np.diag([2.0, 2.0, 2.0, 1.0])
        mask_affine[0, 3] = 0.5
        source_static = np.zeros((4, 4, 4), dtype=np.uint8)
        source_moving = np.zeros_like(source_static)
        source_static[1] = 1
        source_moving[2] = 1

    array_masks, input_masks = {}, {}
    for name, mask, source in [
        ("static_mask", static_mask, source_static),
        ("moving_mask", moving_mask, source_moving),
    ]:
        if side == "both" or name.startswith(side):
            array_masks[name] = mask
            input_masks[name] = _mask_input(source, mask_affine, kind, tmp_path, name)

    options = {
        "moving_affine": np.eye(4),
        "static_affine": np.eye(4),
        "pipeline": ["center_of_mass"],
    }
    expected_data, expected_affine = affine_registration(
        image, image, **options, **array_masks
    )
    registered, affine = affine_registration(image, image, **options, **input_masks)
    translations = {"static": 10 / 7, "moving": 6 / 11, "both": 152 / 77}
    npt.assert_allclose(affine[:3, 3], [translations[side], 0, 0], atol=1e-12)
    npt.assert_allclose(affine, expected_affine)
    npt.assert_allclose(registered, expected_data)


def test_affine_registration_mask_uses_overridden_image_affine():
    data = np.broadcast_to(np.arange(1.0, 8.0)[:, None, None], (7, 7, 7)).copy()
    image = nib.Nifti1Image(data, np.eye(4))
    image_affine = np.eye(4)
    image_affine[0, 3] = 1
    mask_affine = np.diag([2.0, 2.0, 2.0, 1.0])
    mask_affine[0, 3] = 0.5
    source_static = np.zeros((4, 4, 4), dtype=np.uint8)
    source_moving = np.zeros_like(source_static)
    source_static[1] = 1
    source_moving[2] = 1
    static_mask = np.zeros(data.shape, dtype=np.uint8)
    moving_mask = np.zeros_like(static_mask)
    static_mask[2:4] = 1
    moving_mask[3:5] = 1
    options = {
        "moving_affine": image_affine,
        "static_affine": np.eye(4),
        "pipeline": ["center_of_mass"],
    }
    expected_data, expected_affine = affine_registration(
        image, image, static_mask=static_mask, moving_mask=moving_mask, **options
    )
    registered, affine = affine_registration(
        image,
        image,
        static_mask=nib.Nifti1Image(source_static, mask_affine),
        moving_mask=nib.Nifti1Image(source_moving, mask_affine),
        **options,
    )
    npt.assert_allclose(affine[:3, 3], [125 / 63, 0, 0], atol=1e-12)
    npt.assert_allclose(affine, expected_affine)
    npt.assert_allclose(registered, expected_data)


@pytest.mark.parametrize("ndim", [2, 3])
@pytest.mark.parametrize("masked", [False, True])
def test_affine_registration_array_and_none_masks(ndim, masked):
    shape = (7,) * ndim
    image = np.broadcast_to(
        np.arange(1.0, 8.0).reshape((7,) + (1,) * (ndim - 1)), shape
    )
    static_mask = moving_mask = None
    expected_translation = 0
    if masked:
        static_mask = np.zeros(shape, dtype=bool)
        moving_mask = np.zeros(shape, dtype=bool)
        static_mask[2:4] = True
        moving_mask[4:6] = True
        expected_translation = 152 / 77
    registered, affine = affine_registration(
        image,
        image,
        moving_affine=np.eye(ndim + 1),
        static_affine=np.eye(ndim + 1),
        pipeline=["center_of_mass"],
        static_mask=static_mask,
        moving_mask=moving_mask,
    )
    expected_affine = np.eye(ndim + 1)
    expected_affine[0, -1] = expected_translation
    npt.assert_allclose(affine, expected_affine, atol=1e-12)
    if not masked:
        npt.assert_allclose(registered, image)


def test_affine_registration_image_masks_with_mutual_information():
    coordinates = np.indices((9, 9, 9), dtype=float)
    static = np.exp(-np.sum((coordinates - 4) ** 2, axis=0) / 8)
    moving = np.roll(static, 1, axis=0)
    mask = np.zeros(static.shape, dtype=np.int32)
    mask[1:-1, 1:-1, 1:-1] = 1
    options = {
        "moving_affine": np.eye(4),
        "static_affine": np.eye(4),
        "pipeline": ["translation"],
        "level_iters": [4],
        "sigmas": [0],
        "factors": [1],
        "nbins": 8,
        "ret_metric": True,
    }
    expected = affine_registration(
        moving, static, static_mask=mask, moving_mask=mask, **options
    )
    mask_image = nib.Nifti1Image(mask, np.eye(4))
    actual = affine_registration(
        moving, static, static_mask=mask_image, moving_mask=mask_image, **options
    )
    for result, reference in zip(actual, expected, strict=True):
        npt.assert_allclose(result, reference)
    assert np.isfinite(actual[-1])
