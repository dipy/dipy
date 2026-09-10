import numpy as np
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.skyline.render.roi import ROI3D, create_roi_visualization


def _binary_roi(shape=(16, 16, 16)):
    """Small solid block mask suitable for ``contour_from_roi``."""
    roi = np.zeros(shape, dtype=np.uint8)
    sl = tuple(slice(4, 12) for _ in range(len(shape)))
    roi[sl] = 1
    return roi


def test_create_roi_visualization_rejects_invalid_input():
    """``create_roi_visualization`` raises when ``input`` is not a 2- or 3-tuple."""
    with pytest.raises(ValueError, match="Input must be a tuple"):
        create_roi_visualization("not_a_tuple", 0)
    with pytest.raises(ValueError, match="Input must be a tuple"):
        create_roi_visualization((np.ones(3),), 0)


def test_create_roi_visualization_two_tuple_names_roi_by_index():
    """A two-tuple ``(roi, affine)`` uses the default ``ROI_{idx}`` filename."""
    affine = np.eye(4, dtype=np.float64)
    roi = _binary_roi()
    viz = create_roi_visualization((roi, affine), idx=7, color=(0.5, 0.25, 0.0))
    assert viz.path == "ROI_7"
    assert viz._color_picker_popup_id == "roi_color_picker_popup##ROI_7"


def test_create_roi_visualization_three_tuple_uses_filename():
    """A three-tuple carries an explicit filename as the third element."""
    affine = np.eye(4, dtype=np.float64)
    roi = _binary_roi()
    viz = create_roi_visualization((roi, affine, "left_cortex.nii"), idx=0)
    assert viz.path == "left_cortex.nii"


def test_roi3d_raises_on_none_or_non_array():
    """``ROI3D`` rejects missing ROI data and non-array inputs."""
    affine = np.eye(4, dtype=np.float64)
    with pytest.raises(ValueError, match="cannot be None"):
        ROI3D("roi", None, affine=affine)
    with pytest.raises(ValueError, match="numpy array"):
        ROI3D("roi", [1, 2, 3], affine=affine)


def test_roi3d_four_d_roi_uses_first_volume():
    """4D ROI inputs use index 0 along the last axis before meshing."""
    affine = np.eye(4, dtype=np.float64)
    roi_4d = np.zeros((8, 8, 8, 3), dtype=np.uint8)
    roi_4d[2:6, 2:6, 2:6, 0] = 1
    viz = ROI3D(
        "roi4d",
        roi_4d,
        affine=affine,
        color=(1.0, 0.0, 0.0),
        render_callback=None,
    )
    assert viz.roi.shape == (8, 8, 8)
    assert np.any(viz.roi > 0)


def test_roi3d_color_picker_initial_state():
    """Draft color and popup id match the constructor color and name."""
    affine = np.eye(4, dtype=np.float64)
    roi = _binary_roi()
    color = (0.2, 0.3, 0.4)
    viz = ROI3D("my_roi", roi, affine=affine, color=color, render_callback=None)
    assert viz._draft_color == color
    assert viz._color_picker_open is False
    assert viz._color_picker_popup_id == "roi_color_picker_popup##my_roi"


def test_roi3d_populate_info_and_actor():
    """Info text summarizes geometry; ``actor`` exposes the contour group."""
    affine = np.eye(4, dtype=np.float64)
    roi = _binary_roi(shape=(10, 10, 10))
    n_positive = int(np.sum(roi > 0))
    viz = ROI3D("n", roi, affine=affine, render_callback=None)

    info = viz._populate_info()
    assert "(10, 10, 10)" in info
    assert str(n_positive) in info
    assert viz.actor is viz._roi_surface


def test_roi3d_reports_shape_dtype_and_voxel_count():
    roi = _binary_roi()
    viz = ROI3D("mask.nii.gz", roi, affine=np.eye(4))

    info = viz._populate_info()

    assert f"ROI shape: {roi.shape}" in info
    assert "ROI dtype: uint8" in info
    assert f"Total voxels in ROI: {int(np.sum(roi > 0))}" in info
    assert "Affine:" in info


def test_roi3d_info_without_an_affine():
    viz = ROI3D("mask.nii.gz", _binary_roi(), affine=None)

    assert "Affine:" not in viz._populate_info()


def test_roi3d_default_opacity_is_opaque():
    viz = ROI3D("mask.nii.gz", _binary_roi(), affine=np.eye(4))

    assert viz.opacity == 100
    assert viz.actor is viz._roi_surface
    for actor in viz.actor.children:
        assert actor.material.alpha_mode == "blend"
        assert actor.material.depth_write is True


def test_roi3d_partial_opacity_disables_depth_writes_at_creation():
    viz = ROI3D("mask.nii.gz", _binary_roi(), affine=np.eye(4), opacity=30)

    assert viz.opacity == 30
    for actor in viz.actor.children:
        assert actor.material.depth_write is False


def test_roi3d_set_opacity_updates_every_contour_actor():
    viz = ROI3D("mask.nii.gz", _binary_roi(), affine=np.eye(4))

    viz._set_opacity(25)

    for actor in viz.actor.children:
        assert actor.material.opacity == pytest.approx(0.25)
        assert actor.material.depth_write is False

    viz._set_opacity(100)

    for actor in viz.actor.children:
        assert actor.material.opacity == pytest.approx(1.0)
        assert actor.material.depth_write is True


def test_roi3d_keeps_its_constructor_color():
    viz = ROI3D("mask.nii.gz", _binary_roi(), affine=np.eye(4), color=(0.0, 0.5, 1.0))

    assert viz.color == (0.0, 0.5, 1.0)
    assert viz._draft_color == (0.0, 0.5, 1.0)
    assert viz._color_picker_open is False
    assert viz.viz_type == "roi"
