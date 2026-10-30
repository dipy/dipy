from contextlib import chdir
import os

import numpy as np
import pytest

from dipy.tracking.streamline import Streamlines
from dipy.utils.optpkg import optional_package

_, has_fury_v2, _ = optional_package("fury", min_version="2.1.0")
_, has_matplotlib, _ = optional_package("matplotlib")

if has_matplotlib:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

if has_fury_v2:
    from dipy.viz import streamline

pytestmark = pytest.mark.skipif(
    not (has_fury_v2 and has_matplotlib), reason="Requires FURY >= 2.1 and Matplotlib"
)

GIZMO_WIDTH = 150
COLOR_BAR_HEIGHT = 250


def _bundle(*, along_z=False, z_shift=0.0):
    """Build five parallel 40 mm streamlines spaced 1 mm apart along x.

    Parameters
    ----------
    along_z : bool, optional
        If True the streamlines run along z, otherwise along y.
    z_shift : float, optional
        Constant z offset of streamlines running along y.

    Returns
    -------
    bundle : Streamlines
        Bundle of five streamlines with 20 points each.
    """
    t = np.linspace(0.0, 40.0, 20)
    lines = []
    for x in range(5):
        if along_z:
            points = np.stack([np.full(20, float(x)), np.zeros(20), t], axis=1)
        else:
            points = np.stack([np.full(20, float(x)), t, np.full(20, z_shift)], axis=1)
        lines.append(points)
    return Streamlines(lines)


def _read(path):
    """Read a png as an ``(H, W, 3)`` float RGB array in ``[0, 1]``."""
    return plt.imread(path)[..., :3]


def _tinted(pixels, channel, margin=0.2):
    """Count scene pixels where ``channel`` dominates the other two by ``margin``.

    The gizmo corner (first ``GIZMO_WIDTH`` columns) is excluded.
    """
    scene = pixels[:, GIZMO_WIDTH:]
    others = [c for c in range(3) if c != channel]
    strongest_other = np.maximum(scene[..., others[0]], scene[..., others[1]])
    return int(np.count_nonzero(scene[..., channel] - strongest_other > margin))


def _pink(pixels):
    """Count pink scene pixels, the low end of the displacement colormap.

    The gizmo corner (first ``GIZMO_WIDTH`` columns) is excluded.
    """
    scene = pixels[:, GIZMO_WIDTH:]
    mask = (
        (scene[..., 0] > 0.7)
        & (scene[..., 0] - scene[..., 1] > 0.2)
        & (scene[..., 2] - scene[..., 1] > 0.1)
    )
    return int(np.count_nonzero(mask))


def _non_white_rows(pixels):
    """Return the row index of every non-white scene pixel, gizmo corner excluded."""
    rows, _ = np.nonzero(np.any(pixels[:, GIZMO_WIDTH:] < 0.95, axis=-1))
    return rows


def test_show_bundles_saves_each_bundle_color(tmp_path):
    output = tmp_path / "b.png"
    streamline.show_bundles(
        [_bundle(), _bundle(z_shift=10.0)],
        colors=[(1, 0, 0), (0, 0, 1)],
        interactive=False,
        save_as=output,
    )
    pixels = _read(output)

    assert pixels.shape == (900, 900, 3)
    assert _tinted(pixels, 0) > 20
    assert _tinted(pixels, 2) > 20


def test_show_bundles_view_controls_camera_side(tmp_path):
    bundle = _bundle(along_z=True)
    spreads = {}
    for view in ("sagittal", "axial"):
        output = tmp_path / f"{view}.png"
        streamline.show_bundles(
            [bundle],
            colors=[(1, 0, 0)],
            interactive=False,
            view=view,
            save_as=output,
        )
        spreads[view] = np.ptp(_non_white_rows(_read(output)))

    assert spreads["sagittal"] > 300
    assert spreads["axial"] < 100


def test_show_bundles_rejects_unknown_view_without_writing(tmp_path):
    output = tmp_path / "b.png"
    with pytest.raises(ValueError, match="Invalid view"):
        streamline.show_bundles(
            [_bundle()],
            colors=[(1, 0, 0)],
            interactive=False,
            view="oblique",
            save_as=output,
        )

    assert not output.exists()


def test_show_bundles_deprecated_sagital_still_renders(tmp_path):
    output = tmp_path / "b.png"
    with pytest.warns(DeprecationWarning, match="deprecated"):
        streamline.show_bundles(
            [_bundle()],
            colors=[(1, 0, 0)],
            interactive=False,
            view="sagital",  # codespell:ignore sagital
            save_as=output,
        )

    assert output.is_file()
    assert _tinted(_read(output), 0) > 20


def test_show_bundles_interactive_offscreen_capture(tmp_path):
    output_directory = tmp_path / "out"
    output_directory.mkdir()
    output = output_directory / "bundles.png"

    previous_fury = os.environ.get("FURY_OFFSCREEN")
    with chdir(tmp_path):
        try:
            os.environ["FURY_OFFSCREEN"] = "1"
            streamline.show_bundles(
                [_bundle()],
                colors=[(1, 0, 0)],
                interactive=True,
                save_as=output,
            )
        finally:
            if previous_fury is None:
                os.environ.pop("FURY_OFFSCREEN", None)
            else:
                os.environ["FURY_OFFSCREEN"] = previous_fury

    start_capture = tmp_path / "bundles.png"
    assert output.is_file()
    assert start_capture.is_file()
    assert _tinted(_read(output), 0) > 20
    assert _tinted(_read(start_capture), 0) > 20


def test_viz_two_bundles_uses_both_colors(tmp_path):
    output = tmp_path / "two.png"
    streamline.viz_two_bundles(_bundle(), _bundle(z_shift=10.0), output)
    pixels = _read(output)

    assert pixels.shape == (1200, 1200, 3)
    assert _tinted(pixels, 0) > 20
    assert _tinted(pixels, 1) > 20


def test_viz_vector_field_colors_by_absolute_direction(tmp_path):
    output = tmp_path / "field.png"
    points = np.stack([np.zeros(10), np.linspace(0.0, 40.0, 10), np.zeros(10)], axis=1)
    directions = np.tile([-1.0, 0.0, 0.0], (10, 1))
    streamline.viz_vector_field(
        points,
        directions,
        directions,
        np.full(10, 2.0),
        output,
        bundle=_bundle(z_shift=10.0),
    )
    pixels = _read(output)

    assert _tinted(pixels, 0) > 20
    assert _tinted(pixels, 2) > 20


def _render_displacement_mag(output):
    """Render two 40 mm lines with 0 and 5 mm displacement and read the png."""
    t = np.linspace(0.0, 40.0, 20)
    low = np.stack([np.zeros(20), t, np.zeros(20)], axis=1)
    high = np.stack([np.zeros(20), t, np.full(20, 20.0)], axis=1)
    offsets = np.concatenate([np.zeros(20), np.full(20, 5.0)])
    streamline.viz_displacement_mag(Streamlines([low, high]), offsets, output)
    return _read(output)


def test_viz_displacement_mag_maps_extremes_to_lut_ends(tmp_path):
    pixels = _render_displacement_mag(tmp_path / "mag.png")

    assert pixels.shape == (1500, 2000, 3)
    scene = pixels[:-COLOR_BAR_HEIGHT]
    assert _tinted(scene, 1) > 20
    assert _pink(scene) > 20


def test_viz_displacement_mag_draws_color_bar_from_pink_to_green(tmp_path):
    pixels = _render_displacement_mag(tmp_path / "mag.png")

    bar = pixels[-COLOR_BAR_HEIGHT:]
    low_half, high_half = bar[:, :1600], bar[:, 1600:]
    assert _pink(low_half) > 20
    assert _pink(high_half) == 0
    assert _tinted(low_half, 1) == 0
    assert _tinted(high_half, 1) > 20
