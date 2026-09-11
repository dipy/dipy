"""Tests for the deprecated Horizon entry points.

``Horizon`` is a thin shim over Skyline; the tests build real offscreen viewers
through it rather than inspecting the forwarding in isolation.
"""

import numpy as np
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.0.0")
if not has_fury:
    pytest.skip("Requires fury>=2.0.0", allow_module_level=True)
else:
    from dipy.viz.horizon.app import Horizon, horizon

AFFINE = np.eye(4)


def _image():
    data = np.random.default_rng(0).random((6, 7, 8)).astype(np.float32)
    return (data, AFFINE, "vol.nii.gz")


def _build(**kwargs):
    kwargs.setdefault("interactive", False)
    with pytest.warns(DeprecationWarning, match="Horizon is deprecated"):
        return Horizon(**kwargs)


def test_horizon_construction_is_deprecated():
    with pytest.warns(DeprecationWarning, match="Please use Skyline instead"):
        Horizon(interactive=False)


def test_horizon_starts_without_a_show_manager():
    hz = _build(images=[_image()])

    assert hz.show_m is None
    assert hz.visualizations == []
    assert hz._tractogram_help is False


def test_horizon_maps_its_arguments_onto_skyline_options():
    images = [_image()]
    hz = _build(
        images=images,
        cluster=True,
        rgb=True,
        cluster_thr=2.0,
        length_gt=5,
        clusters_gt=3,
        bg_color=(0.2, 0.3, 0.4),
        out_png="scene.png",
    )

    assert hz._horizon_data["images"] is images
    assert hz._horizon_data["cluster"] is True
    assert hz._horizon_data["rgb"] is True
    assert hz._horizon_data["cluster_thr"] == 2.0
    assert hz._horizon_data["cluster_length_thr"] == 5
    assert hz._horizon_data["cluster_size_thr"] == 3
    assert hz._horizon_data["bg_color"] == (0.2, 0.3, 0.4)
    assert hz._horizon_data["out_stealth_png"] == "scene.png"


@pytest.mark.parametrize(
    "interactive,expected", [(False, "stealth"), (True, "standalone")]
)
def test_horizon_selects_the_visualizer_type_from_interactive(interactive, expected):
    with pytest.warns(DeprecationWarning):
        hz = Horizon(interactive=interactive)

    assert hz._horizon_data["visualizer_type"] == expected
    assert hz._horizon_data["interactive"] is interactive


def test_horizon_build_show_renders_offscreen(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    hz = _build(images=[_image()], out_png="scene.png")

    show_manager = hz.build_show()

    assert show_manager is hz.show_m
    assert (tmp_path / "scene.png").is_file()
    assert (tmp_path / "scene.png").stat().st_size > 0


def test_horizon_build_show_renders_surfaces_and_tractograms(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
        dtype=np.float32,
    )
    faces = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.uint32)
    hz = _build(surfaces=[(vertices, faces, "surf.gii")], out_png="surfaces.png")

    hz.build_show()

    assert (tmp_path / "surfaces.png").is_file()


def test_horizon_function_is_deprecated_and_returns_a_show_manager(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)

    with pytest.warns(DeprecationWarning, match="Horizon is deprecated"):
        show_manager = horizon(
            images=[_image()], interactive=False, out_png="from_function.png"
        )

    assert show_manager is not None
    assert (tmp_path / "from_function.png").is_file()


def test_horizon_function_forwards_the_background_color(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.warns(DeprecationWarning):
        show_manager = horizon(
            images=[_image()],
            interactive=False,
            bg_color=(1, 1, 1),
            out_png="white.png",
        )

    assert show_manager.screens[0].scene.background == (1, 1, 1)


def test_horizon_out_png_directory_is_dropped(tmp_path, monkeypatch):
    """Horizon has no ``out_dir``, so a directory in ``out_png`` is not kept."""
    monkeypatch.chdir(tmp_path)
    nested = tmp_path / "renders"
    nested.mkdir()

    with pytest.warns(DeprecationWarning):
        horizon(
            images=[_image()],
            interactive=False,
            out_png=str(nested / "scene.png"),
        )

    assert (tmp_path / "scene.png").is_file()
    assert not (nested / "scene.png").exists()
