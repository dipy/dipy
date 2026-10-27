"""End-to-end tests for the Skyline visualization workflow."""

import os

import numpy as np
import pytest

from dipy.io.image import save_nifti
from dipy.utils.optpkg import optional_package

_, has_fury_v2, _ = optional_package("fury", min_version="2.0.0")
if not has_fury_v2:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from PIL import Image

    from dipy.io.stateful_tractogram import Space, StatefulTractogram
    from dipy.io.streamline import save_tractogram
    from dipy.stats.analysis import SpectraGrid
    from dipy.tracking.streamline import Streamlines
    from dipy.workflows.viz import (
        SkylineFlow,
        SpectraVizFlow,
        _alternating_colors,
        _grid_values_to_points,
        _values_to_colors,
    )


@pytest.mark.parametrize("stealth, dipy_value", [(True, None), (False, "1")])
def test_skyline_flow_captures_offscreen(tmp_path, stealth, dipy_value):
    volume_path = tmp_path / "volume.nii.gz"
    save_nifti(
        str(volume_path),
        np.arange(64, dtype=np.float32).reshape(4, 4, 4),
        np.eye(4),
    )

    previous_dipy = os.environ.get("DIPY_OFFSCREEN")
    previous_fury = os.environ.get("FURY_OFFSCREEN")
    try:
        if dipy_value is None:
            os.environ.pop("DIPY_OFFSCREEN", None)
        else:
            os.environ["DIPY_OFFSCREEN"] = dipy_value
        os.environ["FURY_OFFSCREEN"] = "0"

        SkylineFlow().run(
            [str(volume_path)],
            stealth=stealth,
            out_dir=str(tmp_path),
            out_stealth_png="workflow.png",
        )

        assert os.environ.get("DIPY_OFFSCREEN") == dipy_value
        assert os.environ.get("FURY_OFFSCREEN") == "0"
        with Image.open(tmp_path / "workflow.png") as image:
            assert any(low != high for low, high in image.getextrema()[:3])
    finally:
        for key, value in (
            ("DIPY_OFFSCREEN", previous_dipy),
            ("FURY_OFFSCREEN", previous_fury),
        ):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def test_spectra_grid_colors():
    s_index = np.array([0, 1, 2, 2])
    r_index = np.array([0, 1, 0, 1])
    grid = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, np.nan]])
    values = _grid_values_to_points(grid, (3, 2), s_index, r_index)
    np.testing.assert_allclose(values, [0.1, 0.4, 0.5, np.nan])
    np.testing.assert_allclose(
        _grid_values_to_points(grid[:, 0], (3, 2), s_index, r_index),
        [0.1, 0.3, 0.5, 0.5],
    )
    # Values of another grid are not used
    with pytest.raises(ValueError):
        _grid_values_to_points(grid, (10, 2), s_index, r_index)
    with pytest.raises(ValueError):
        _grid_values_to_points(grid, (3, 4), s_index, r_index)

    # Invalid and missing values are gray, the range is the 5th to 95th
    # percentile by default
    valid = np.array([True, True, False, True])
    colors = _values_to_colors(values, valid, cmap="gray", vmin=0, vmax=1)
    np.testing.assert_allclose(colors[:, 0], [0.1, 0.4, 0.5, 0.5], atol=0.01)
    colors = _values_to_colors(np.arange(100.0), np.ones(100, dtype=bool), cmap="gray")
    np.testing.assert_allclose(colors[[0, 99], 0], [0, 1])

    colors = _alternating_colors(s_index, r_index, valid)
    assert colors.shape == (4, 3)
    np.testing.assert_allclose(colors[2], 128 / 255)
    assert not np.allclose(colors[0], colors[1])
    assert len(_alternating_colors(s_index, None, valid)) == 4


def test_spectra_viz_flow(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    lines = [
        np.c_[np.linspace(0, 50, 60), rng.normal(0, 3, 60), rng.normal(0, 3, 60)] + 25
        for _ in range(60)
    ]
    volume = rng.random((90, 60, 60))
    save_nifti(str(tmp_path / "fa.nii.gz"), volume, np.eye(4))
    sft = StatefulTractogram(
        Streamlines(lines), str(tmp_path / "fa.nii.gz"), Space.RASMM
    )
    save_tractogram(sft, str(tmp_path / "AF_L.trk"), bbox_valid_check=False)
    grid = SpectraGrid(lines)
    np.save(tmp_path / "values.npy", rng.random((grid.n_segments, grid.n_radial)))
    np.save(tmp_path / "tested.npy", np.ones((grid.n_segments, grid.n_radial), bool))
    bundle = str(tmp_path / "AF_L.trk")

    def is_drawn(fname):
        image = np.asarray(Image.open(fname))[..., :3]
        return (image < 250).any()

    flow = SpectraVizFlow()
    flow.run(
        [bundle],
        ref_bundle=bundle,
        bundle_color=[str(tmp_path / "values.npy")],
        color_mask=str(tmp_path / "tested.npy"),
        cam_settings=["Axial", "Sagittal_L"],
        auto_crop=True,
        out_dir=tmp_path,
        out_image="values.png",
        out_cbar="colorbar.png",
    )
    assert is_drawn(tmp_path / "values_cam01.png")
    assert is_drawn(tmp_path / "values_cam02.png")
    assert (tmp_path / "colorbar.png").exists()

    for kwargs in (
        {"bundle_color": ["alternate_2d"]},
        {"bundle_color": [str(tmp_path / "fa.nii.gz")]},
        {"bundle_color": ["red"], "no_tube": True},
        {"bundle_color": [str(tmp_path / "values.npy")], "centroid_with_bundle": True},
        {"centroid_only": True, "masks": [str(tmp_path / "fa.nii.gz")]},
    ):
        flow.run([bundle], out_dir=tmp_path, out_image="bundle.png", **kwargs)
        assert is_drawn(tmp_path / "bundle.png")

    def drawn_colors(fname):
        image = np.asarray(Image.open(fname))[..., :3].astype(int)
        return image[(image < 250).any(axis=-1)]

    # The points have the color of the bundle
    flow.run(
        [bundle],
        bundle_color=["red"],
        as_points=True,
        out_dir=tmp_path,
        out_image="points.png",
    )
    colors = drawn_colors(tmp_path / "points.png")
    assert len(colors) > 0
    assert np.all(colors[:, 0] > colors[:, 1])

    # The opacity of the lines is used, and the glass brain is visible
    brain = np.pad(np.ones((70, 40, 40)), 10)
    save_nifti(str(tmp_path / "brain.nii.gz"), brain, np.eye(4))
    flow.run(
        [bundle],
        bundle_color=["red"],
        no_tube=True,
        opacity=0.5,
        glass_brain=[str(tmp_path / "brain.nii.gz")],
        out_dir=tmp_path,
        out_image="glass.png",
    )
    colors = drawn_colors(tmp_path / "glass.png")
    is_gray = np.ptp(colors, axis=1) < 3
    assert is_gray.sum() > 1000
    assert np.median(colors[~is_gray, 1]) > 50

    with pytest.raises(ValueError):
        flow.run([bundle], cam_settings=["Top"], out_dir=tmp_path)
    with pytest.raises(ValueError):
        flow.run([bundle], centroid_only=True, centroid_with_bundle=True)
