import numpy as np
import numpy.testing as npt
import pytest

from dipy.utils.optpkg import optional_package

_, has_fury, _ = optional_package("fury", min_version="2.1.0")
if not has_fury:
    pytest.skip("Requires fury>=2.1.0", allow_module_level=True)
else:
    from fury import window
    from fury.actor import create_axes_helper

    from dipy.data import get_sphere
    from dipy.reconst.shm import sh_to_sf
    from dipy.viz import actor
    from dipy.viz.actor import _snap_camera_to_axis, _sync_gizmo, _taubin_smooth


def _isotropic_coeffs(shape=(3, 3, 3)):
    coeffs = np.zeros(shape + (6,))
    coeffs[..., 0] = 1
    return coeffs


def _offscreen_manager():
    return window.ShowManager(size=(200, 200), window_type="offscreen", imgui=False)


def _render(odf_actor):
    manager = _offscreen_manager()
    screen = manager.screens[0]
    try:
        screen.scene.add(odf_actor)
        window.update_camera(screen.camera, None, odf_actor)
        manager._draw_function()
        return manager.snapshot(fname=None)[..., :3]
    finally:
        manager.close()


def test_odf_slicer_places_slices_in_world_space():
    affine = np.diag([2.0, 2.0, 2.0, 1.0])
    affine[0, 3] = 10
    odf_actor = actor.odf_slicer(
        sh_coeffs=_isotropic_coeffs((4, 4, 4)), affine=affine, slices=(1, 2, 3)
    )
    material = odf_actor.children[0].material
    npt.assert_allclose(
        (material.active_slice_x, material.active_slice_y, material.active_slice_z),
        (12, 4, 6),
    )


@pytest.mark.parametrize(
    "visibility, drawn",
    [((True, True, True), True), ((False, False, False), False)],
)
def test_odf_slicer_visibility_controls_rendered_pixels(visibility, drawn):
    odf_actor = actor.odf_slicer(sh_coeffs=_isotropic_coeffs(), visibility=visibility)
    pixels = _render(odf_actor)
    assert np.any(pixels != pixels[0, 0]) == drawn


def test_odf_slicer_sphere_input_matches_sh_input():
    sphere = get_sphere(name="repulsion724")
    coeffs = np.random.default_rng(0).normal(scale=0.3, size=(2, 1, 1, 15))
    coeffs[..., 0] = 2
    sf = sh_to_sf(
        coeffs, sphere, sh_order_max=4, basis_type="descoteaux07", legacy=False
    )
    from_sf = _render(actor.odf_slicer(sh_func=sf, sphere=sphere, l_max=4))
    from_sh = _render(actor.odf_slicer(sh_coeffs=coeffs, l_max=4))
    assert np.abs(from_sf.astype(int) - from_sh.astype(int)).max() <= 2


@pytest.mark.parametrize("norm, same", [(True, True), (False, False)])
def test_odf_slicer_norm_controls_glyph_size(norm, same):
    coeffs = _isotropic_coeffs((1, 1, 1))
    coeffs[..., 3] = 0.3
    full = _render(actor.odf_slicer(sh_coeffs=coeffs, norm=norm))
    shrunk = _render(actor.odf_slicer(sh_coeffs=coeffs * 0.3, norm=norm))
    assert np.array_equal(full, shrunk) == same


def test_odf_slicer_opacity_blends_glyphs_with_background():
    coeffs = _isotropic_coeffs()
    hidden = _render(actor.odf_slicer(sh_coeffs=coeffs, opacity=0.0)).astype(int)
    translucent = _render(actor.odf_slicer(sh_coeffs=coeffs, opacity=0.4)).astype(int)
    opaque = _render(actor.odf_slicer(sh_coeffs=coeffs, opacity=1.0)).astype(int)
    assert 0 < np.abs(translucent - hidden).sum() < np.abs(opaque - hidden).sum()


@pytest.mark.parametrize("opacity", [-0.1, 1.5, np.nan])
def test_odf_slicer_rejects_invalid_opacity(opacity):
    with pytest.raises(ValueError, match="opacity must be between 0 and 1"):
        actor.odf_slicer(sh_coeffs=_isotropic_coeffs(), opacity=opacity)


def test_odf_slicer_rejects_sphere_size_mismatch():
    with pytest.raises(ValueError, match="vertices"):
        actor.odf_slicer(
            sh_func=np.ones((1, 1, 1, 10)), sphere=get_sphere(name="repulsion724")
        )


@pytest.mark.parametrize(
    "with_coeffs, with_func, with_sphere, match",
    [
        (False, False, False, "exactly one"),
        (True, True, False, "exactly one"),
        (False, True, False, "sphere is required"),
        (True, False, True, "only used"),
    ],
)
def test_odf_slicer_rejects_inconsistent_inputs(
    with_coeffs, with_func, with_sphere, match
):
    sphere = get_sphere(name="repulsion724")
    kwargs = {}
    if with_coeffs:
        kwargs["sh_coeffs"] = np.ones((1, 1, 1, 6))
    if with_func:
        kwargs["sh_func"] = np.ones((1, 1, 1, len(sphere.vertices)))
    if with_sphere:
        kwargs["sphere"] = sphere
    with pytest.raises(ValueError, match=match):
        actor.odf_slicer(**kwargs)


def test_odf_slicer_rejects_out_of_bounds_slices():
    with pytest.raises(ValueError, match="slices"):
        actor.odf_slicer(sh_coeffs=_isotropic_coeffs(), slices=(0, 0, 3))


def test_odf_slicer_rejects_non_4d_coefficients():
    with pytest.raises(ValueError, match="4D"):
        actor.odf_slicer(sh_coeffs=np.ones((3, 3, 3)))


def test_odf_slicer_all_zero_coefficients_build_empty_group():
    assert len(actor.odf_slicer(sh_coeffs=np.zeros((2, 2, 2, 6))).children) == 0


@pytest.mark.parametrize(
    "direction, forward, up",
    [
        ((0, 0, 1), (0, 0, -1), (0, 1, 0)),
        ((0, -1, 0), (0, 1, 0), (0, 0, 1)),
        ((-1, 0, 0), (1, 0, 0), (0, 0, 1)),
    ],
)
def test_snap_camera_to_axis_gives_anatomical_view(direction, forward, up):
    manager = _offscreen_manager()
    screen = manager.screens[0]
    try:
        screen.scene.add(actor.box(np.zeros((1, 3))))
        _snap_camera_to_axis(screen, direction)
        npt.assert_allclose(screen.camera.world.forward, forward, atol=1e-5)
        npt.assert_allclose(screen.camera.world.up, up, atol=1e-5)
        if direction == (0, 0, 1):
            npt.assert_allclose(screen.camera.world.right, (1, 0, 0), atol=1e-5)
    finally:
        manager.close()


def test_sync_gizmo_follows_camera():
    manager = _offscreen_manager()
    screen = manager.screens[0]
    try:
        parts = create_axes_helper(labels=["L", "R", "P", "A", "S", "I"])
        screen.scene.add(actor.box(np.zeros((1, 3))))
        _snap_camera_to_axis(screen, (1, 0, 0))
        _sync_gizmo(parts, screen.camera)
        assert parts["disks"][4].world.position[1] > 0
        assert parts["disks"][3].world.position[0] > 0
        assert parts["disks"][1].opacity == pytest.approx(1.0)
        assert parts["disks"][0].opacity == pytest.approx(0.1)
    finally:
        manager.close()


def test_gizmo_rejects_second_gizmo_on_same_screen():
    manager = _offscreen_manager()
    try:
        actor.gizmo(manager)
        with pytest.raises(ValueError, match="already attached"):
            actor.gizmo(manager)
    finally:
        manager.close()


@pytest.mark.parametrize(
    "label, forward, up",
    [
        ("L", (1, 0, 0), (0, 0, 1)),
        ("A", (0, -1, 0), (0, 0, 1)),
        ("S", (0, 0, -1), (0, 1, 0)),
    ],
)
def test_click_gizmo_snaps_camera_to_label_side(label, forward, up):
    manager = _offscreen_manager()
    screen = manager.screens[0]
    try:
        screen.scene.add(actor.box(np.zeros((1, 3))))
        anchor = actor.gizmo(manager)
        actor.click_gizmo(manager, anchor, label)
        npt.assert_allclose(screen.camera.world.forward, forward, atol=1e-5)
        npt.assert_allclose(screen.camera.world.up, up, atol=1e-5)
    finally:
        manager.close()


def test_click_gizmo_rejects_unknown_label():
    manager = _offscreen_manager()
    try:
        anchor = actor.gizmo(manager)
        with pytest.raises(ValueError, match="Unknown gizmo label"):
            actor.click_gizmo(manager, anchor, "X")
    finally:
        manager.close()


def _contour_vertices(contour):
    return np.vstack([child.geometry.positions.data for child in contour.children])


def _slab_roi():
    roi = np.zeros((12, 20, 12), dtype=np.uint8)
    roi[5:8, 3:17, 4:9] = 1
    return roi


def test_smooth_contour_from_roi_rounds_voxel_corners_without_shrinking():
    roi = _slab_roi()
    blocky = _contour_vertices(actor.smooth_contour_from_roi(roi, iterations=0))
    smooth = _contour_vertices(actor.smooth_contour_from_roi(roi, iterations=20))

    assert not np.allclose(smooth, blocky)
    npt.assert_array_less(smooth.min(axis=0), blocky.min(axis=0) + 1.5)
    npt.assert_array_less(blocky.max(axis=0) - 1.5, smooth.max(axis=0))
    npt.assert_allclose(smooth.mean(axis=0), blocky.mean(axis=0), atol=0.2)


def test_smooth_contour_from_roi_applies_affine():
    roi = _slab_roi()
    affine = np.eye(4)
    affine[:3, 3] = (10, -5, 2)
    plain = _contour_vertices(actor.smooth_contour_from_roi(roi))
    moved = _contour_vertices(actor.smooth_contour_from_roi(roi, affine=affine))
    npt.assert_allclose(moved, plain + (10, -5, 2), atol=1e-4)


def test_smooth_contour_from_roi_one_actor_per_object():
    roi = _slab_roi()
    roi[0:2, 0:2, 0:2] = 1
    assert len(actor.smooth_contour_from_roi(roi).children) == 2


def _bumpy_grid(size=7):
    """Flat triangulated grid in the z=0 plane with a unit bump at its center."""
    xs, ys = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    vertices = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(size * size)])
    index = np.arange(size * size).reshape(size, size)
    lower_left, lower_right = index[:-1, :-1].ravel(), index[:-1, 1:].ravel()
    upper_left, upper_right = index[1:, :-1].ravel(), index[1:, 1:].ravel()
    faces = np.vstack(
        [
            np.column_stack([lower_left, lower_right, upper_right]),
            np.column_stack([lower_left, upper_right, upper_left]),
        ]
    )
    vertices[index[size // 2, size // 2], 2] = 1.0
    return vertices, faces


def test_taubin_smooth_zero_iterations_returns_unchanged_copy():
    vertices, faces = _bumpy_grid()
    original = vertices.copy()
    smoothed = _taubin_smooth(vertices, faces, 0)
    npt.assert_array_equal(smoothed, original)
    smoothed[0] += 1
    npt.assert_array_equal(vertices, original)


def test_taubin_smooth_flattens_bump_progressively():
    vertices, faces = _bumpy_grid()
    original = vertices.copy()
    heights = [
        np.abs(_taubin_smooth(vertices, faces, iterations)[:, 2]).max()
        for iterations in (1, 5, 20)
    ]
    assert heights[0] < 1.0
    assert heights[0] > heights[1] > heights[2]
    npt.assert_array_equal(vertices, original)


def test_taubin_smooth_keeps_flat_mesh_in_plane():
    vertices, faces = _bumpy_grid()
    vertices[:, 2] = 0
    smoothed = _taubin_smooth(vertices, faces, 10)
    npt.assert_allclose(smoothed[:, 2], 0, atol=1e-12)
