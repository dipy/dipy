"""FURY actors plus DIPY-specific visualization actors."""

from functools import partial

import numpy as np
from scipy import sparse

from dipy.data import get_sphere
from dipy.reconst.shm import (
    calculate_max_order,
    real_sh_descoteaux,
    real_sh_tournier,
    sf_to_sh,
)
from dipy.utils.optpkg import optional_package
from dipy.viz.skyline.io import _validate_sh_basis
from dipy.viz.skyline.render.renderer import affine_voxel_sizes
from dipy.viz.skyline.render.sh_slicer import SHSlicer

fury, has_fury, _ = optional_package("fury", min_version="2.1.0")
if has_fury:
    from fury.actor import *
    from fury.actor import Group, create_axes_helper, image, surface, text
    from fury.colormap import create_colormap, hsv2rgb, normalize_colors
    from fury.lib import EventType, PointerEvent
    from fury.transform import apply_transformation
    from fury.utils import apply_affine, voxel_mesh_by_object
    from fury.window import update_camera
else:
    Group = create_axes_helper = surface = image = text = EventType = PointerEvent = (
        fury
    )
    apply_transformation = update_camera = normalize_colors = fury
    voxel_mesh_by_object = create_colormap = hsv2rgb = apply_affine = fury


_SCALAR_BAR_SAMPLES = 256

_SH_BASIS_FUNCTIONS = {
    "descoteaux07": (real_sh_descoteaux, False),
    "descoteaux07_legacy": (real_sh_descoteaux, True),
    "tournier07": (real_sh_tournier, True),
    "tournier19": (real_sh_tournier, False),
    "standard": (real_sh_tournier, False),
}


def _sh_abs_max(sh_coeffs, basis_type):
    """Compute the per-voxel maximum absolute amplitude of SH functions.

    Parameters
    ----------
    sh_coeffs : ndarray
        SH coefficients of shape (X, Y, Z, C).
    basis_type : str
        Resolved SH basis label.

    Returns
    -------
    peak : ndarray
        Maximum absolute amplitude per voxel, sampled on 724 directions.
    """
    full_basis = basis_type == "standard"
    sh_order_max = calculate_max_order(sh_coeffs.shape[-1], full_basis=full_basis)
    basis_function, legacy = _SH_BASIS_FUNCTIONS[basis_type]
    sphere = get_sphere(name="repulsion724")
    basis_matrix, _, _ = basis_function(
        sh_order_max,
        sphere.theta,
        sphere.phi,
        full_basis=full_basis,
        legacy=legacy,
    )
    peak = np.empty(sh_coeffs.shape[:3])
    for i, slab in enumerate(sh_coeffs):
        peak[i] = np.abs(slab @ basis_matrix.T).max(axis=-1)
    return peak


def odf_slicer(
    *,
    sh_coeffs=None,
    sh_func=None,
    sphere=None,
    affine=None,
    basis_type="descoteaux07",
    mask=None,
    slices=None,
    visibility=(True, True, True),
    scale=None,
    norm=True,
    l_max=8,
    lut_res=8,
    color_type="orientation",
    opacity=1.0,
):
    """Create a fast ODF slicer from SH coefficients or sphere samples.

    Exactly one of ``sh_coeffs`` and ``sh_func`` must be given.

    Parameters
    ----------
    sh_coeffs : ndarray, optional
        SH coefficients of shape (X, Y, Z, C).
    sh_func : ndarray, optional
        Spherical function (ODF) samples of shape (X, Y, Z, N) on ``sphere``.
        They are fit to a symmetric SH basis of order ``l_max``.
    sphere : Sphere, optional
        Sphere ``sh_func`` is sampled on. Required with ``sh_func``, not
        allowed with ``sh_coeffs``.
    affine : ndarray, optional
        4x4 voxel-to-world affine.
    basis_type : {"descoteaux07", "descoteaux07_legacy", "tournier07", \
"tournier19", "standard"}
        SH basis of ``sh_coeffs``. ``"standard"`` is FURY's full basis. Ignored
        with ``sh_func``.
    mask : ndarray, optional
        3D boolean mask of voxels to draw.
    slices : tuple of int, optional
        Voxel indices (i, j, k) of the displayed slices. Defaults to the
        volume center.
    visibility : tuple of bool
        Visibility of the x, y and z slices.
    scale : float, optional
        Glyph scale. Defaults to the mean voxel size, or 1 without an affine.
    norm : bool
        Scale each voxel's ODF by its maximum absolute amplitude.
    l_max : int
        Maximum SH order used for rendering.
    lut_res : int
        Resolution of the sphere lookup table.
    color_type : str
        Glyph coloring mode.
    opacity : float
        Glyph opacity, expected in ``[0, 1]``.

    Returns
    -------
    actor : fury.actor.Group
        The ODF slicer actor.

    Notes
    -----
    Slices are fixed at build time; build a new actor to change them.
    Asymmetric sphere functions are symmetrized by the SH fit.
    """
    if not 0 <= opacity <= 1:
        raise ValueError("opacity must be between 0 and 1")
    if (sh_coeffs is None) == (sh_func is None):
        raise ValueError("Provide exactly one of sh_coeffs or sh_func.")
    if sh_func is None:
        if sphere is not None:
            raise ValueError("sphere is only used together with sh_func.")
        name, data = "sh_coeffs", np.asarray(sh_coeffs)
    else:
        if sphere is None:
            raise ValueError("sphere is required when sh_func is given.")
        name, data = "sh_func", np.asarray(sh_func)
    if data.ndim != 4:
        raise ValueError(f"{name} must be a 4D array of shape (X, Y, Z, N).")

    shape = np.array(data.shape[:3])
    voxel = shape // 2 if slices is None else np.asarray(slices, dtype=int)
    if voxel.shape != (3,) or np.any(voxel < 0) or np.any(voxel >= shape):
        raise ValueError(f"slices must be 3 voxel indices within {tuple(shape)}.")

    if scale is None:
        scale = 1.0 if affine is None else float(np.mean(affine_voxel_sizes(affine)))

    if sh_func is not None:
        n_vertices = len(sphere.vertices)
        if data.shape[-1] != n_vertices:
            raise ValueError(
                f"sh_func has {data.shape[-1]} values per voxel but sphere has "
                f"{n_vertices} vertices."
            )
        n_coeffs = (l_max + 1) * (l_max + 2) // 2
        if n_vertices < n_coeffs:
            raise ValueError(
                f"sphere needs at least {n_coeffs} vertices to fit SH order {l_max}."
            )
        coeffs = sf_to_sh(
            data, sphere, sh_order_max=l_max, basis_type="tournier07", legacy=False
        )
        basis_type = "tournier19"
        if norm:
            peak = np.abs(data).max(axis=-1)
    else:
        coeffs = data
        if basis_type != "standard":
            basis_type = _validate_sh_basis(basis_type)
        if norm:
            peak = _sh_abs_max(coeffs, basis_type)

    if norm:
        coeffs = coeffs / np.where(peak > 0, peak, 1)[..., None]

    slicer = SHSlicer(
        coeffs,
        scale=scale,
        l_max=l_max,
        lut_res=lut_res,
        mask=mask,
        basis_type=basis_type,
        color_type=color_type,
        opacity=opacity * 100,
    )
    slicer.build()

    if affine is None:
        positions = voxel.astype(float)
    else:
        slicer.actor.transform(affine)
        positions = apply_transformation(voxel[None].astype(np.float32), affine)[0]

    for axis, position, visible in zip("xyz", positions, visibility):
        slicer.set_slice(axis, float(position))
        if not visible:
            slicer.hide_axis(axis)
    return slicer.actor


def _sync_gizmo(parts, camera):
    """Orient the gizmo and fade its axes to match the camera.

    Parameters
    ----------
    parts : dict
        Output of ``fury.actor.create_axes_helper``.
    camera : fury.lib.PerspectiveCamera
        Screen camera the gizmo follows.
    """
    x, y, z, w = camera.world.rotation
    parts["group"].local.rotation = np.array([x, y, -z, w], dtype=np.float32)

    billboard = np.array([-x, -y, z, w], dtype=np.float32)
    parts["center_disk"].local.rotation = billboard
    for disk, label in zip(parts["disks"], parts["labels"]):
        disk.local.rotation = billboard
        label.local.rotation = billboard

    depths = [float(np.dot(camera.world.forward, d)) for d in parts["axis_vectors"]]
    front = min(depths)
    span = max(max(depths) - front, 1e-6)
    for depth, disk, label, line in zip(
        depths, parts["disks"], parts["labels"], parts["lines"]
    ):
        opacity = 1.0 - (depth - front) / span * 0.9
        disk.opacity = opacity
        label.opacity = opacity
        line.opacity = opacity


def _snap_camera_to_axis(screen, direction):
    """Point the screen camera at the scene from an anatomical side.

    Parameters
    ----------
    screen : fury.window.Screen
        Screen whose camera is moved.
    direction : array_like
        World direction from the scene center toward the camera.
    """
    direction = np.asarray(direction, dtype=np.float32)
    direction /= np.linalg.norm(direction)
    camera = screen.camera
    bbox = screen.scene.main_scene.get_world_bounding_box()
    if bbox is not None and np.all(np.isfinite(bbox)):
        target = bbox.mean(axis=0)
    else:
        target = camera.world.position + camera.world.forward
    distance = max(float(np.linalg.norm(camera.world.position - target)), 1.0)
    up = (0, 1, 0) if abs(direction[2]) > 0.9 else (0, 0, 1)
    camera.world.reference_up = up
    camera.world.position = target + direction * distance
    camera.look_at(target)
    update_camera(camera, None, screen.scene)


def _on_gizmo_axis_pointer_down(event, *, screen, parts, direction):
    """Snap the camera to the clicked gizmo axis."""
    event.stop_propagation()
    _snap_camera_to_axis(screen, direction)
    _sync_gizmo(parts, screen.camera)


def gizmo(
    show_manager,
    *,
    screen=0,
    labels=("L", "R", "P", "A", "S", "I"),
    size=30,
    thickness=2,
    position=(60, 60),
):
    """Add an orientation gizmo with anatomical camera snapping.

    Parameters
    ----------
    show_manager : fury.window.ShowManager
        Window the gizmo is attached to.
    screen : int
        Index of the screen of ``show_manager`` that hosts the gizmo.
    labels : sequence of str
        Axis labels for the world directions -X, +X, -Y, +Y, +Z, -Z. The
        defaults are anatomical labels for RAS space.
    size : float
        Gizmo size in pixels.
    thickness : float
        Axis line thickness.
    position : tuple of float
        Gizmo center in pixels from the bottom-left of the screen viewport.

    Returns
    -------
    anchor : fury.actor.Group
        The gizmo actor, hosted in the screen's UI scene. Its ``endpoints``
        attribute maps each label to its clickable disk.

    Notes
    -----
    Clicking an axis label snaps the camera to that side of the scene, with
    superior up for L/R/A/P views and anterior up for S/I views.
    """
    name = f"dipy_gizmo_screen_{screen}"
    if name in show_manager.callbacks:
        raise ValueError(f"A gizmo is already attached to screen {screen}.")
    target_screen = show_manager.screens[screen]

    parts = create_axes_helper(labels=list(labels), thickness=thickness)
    anchor = Group(name="DIPY Gizmo")
    anchor.local.position = [position[0], position[1], 0.5]
    anchor.local.scale = [size, size, 0.45]
    anchor.add(parts["group"])

    for disk, label, direction in zip(
        parts["disks"], parts["labels"], parts["axis_vectors"]
    ):
        handler = partial(
            _on_gizmo_axis_pointer_down,
            screen=target_screen,
            parts=parts,
            direction=direction,
        )
        disk.add_event_handler(handler, EventType.POINTER_DOWN)
        label.add_event_handler(handler, EventType.POINTER_DOWN)

    anchor.endpoints = dict(zip(labels, parts["disks"]))
    target_screen.scene.ui_scene.add(anchor)
    _sync_gizmo(parts, target_screen.camera)
    show_manager.register_callback(
        _sync_gizmo, 0.016, True, name, parts, target_screen.camera
    )
    return anchor


def click_gizmo(show_manager, gizmo_actor, label):
    """Simulate a left click on a gizmo axis label.

    Parameters
    ----------
    show_manager : fury.window.ShowManager
        Window hosting the gizmo.
    gizmo_actor : fury.actor.Group
        Gizmo returned by :func:`gizmo`.
    label : str
        Label of the clicked axis, one of the labels given to :func:`gizmo`.

    Notes
    -----
    Useful to snap the camera in scripts and offscreen renders, where no mouse
    is available.
    """
    if label not in gizmo_actor.endpoints:
        raise ValueError(
            f"Unknown gizmo label {label!r}; expected one of "
            f"{list(gizmo_actor.endpoints)}."
        )
    show_manager.renderer.dispatch_event(
        PointerEvent(
            x=-1,
            y=-1,
            type="pointer_down",
            button=1,
            target=gizmo_actor.endpoints[label],
        )
    )


# TODO: Move `_taubin_smooth` and `smooth_contour_from_roi` to FURY and remove them
# from DIPY once https://github.com/fury-gl/fury/issues/1390 is closed
# (smooth option for `contour_from_roi` and `contour_from_label`).
def _taubin_smooth(vertices, faces, iterations, *, lam=0.5, mu=-0.53):
    """Smooth a triangle mesh with Taubin (shrink-free) Laplacian smoothing.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Mesh vertices.
    faces : ndarray, shape (M, 3)
        Triangle vertex indices.
    iterations : int
        Number of smoothing passes.
    lam : float, optional
        Positive (shrinking) step.
    mu : float, optional
        Negative (inflating) step, slightly larger in magnitude than ``lam``.

    Returns
    -------
    ndarray, shape (N, 3)
        Smoothed vertices.
    """
    n_vertices = len(vertices)
    rows = faces[:, [0, 1, 2, 1, 2, 0]].ravel()
    cols = faces[:, [1, 2, 0, 0, 1, 2]].ravel()
    adjacency = sparse.coo_matrix(
        (np.ones(len(rows)), (rows, cols)), shape=(n_vertices, n_vertices)
    ).tocsr()
    adjacency.data[:] = 1
    neighbor_mean = sparse.diags(1 / np.ravel(adjacency.sum(axis=1))) @ adjacency

    smoothed = np.array(vertices, dtype=float)
    for _ in range(iterations):
        smoothed += lam * (neighbor_mean @ smoothed - smoothed)
        smoothed += mu * (neighbor_mean @ smoothed - smoothed)
    return smoothed


def smooth_contour_from_roi(
    data, *, affine=None, color=(1, 0, 0), opacity=0.5, iterations=20
):
    """Generate a smooth surface actor from a binary ROI.

    Parameters
    ----------
    data : ndarray, shape (X, Y, Z)
        ROI volume; voxels with a value above zero are displayed.
    affine : ndarray, shape (4, 4), optional
        Affine mapping voxel coordinates to world coordinates.
    color : str, tuple, list or ndarray, optional
        Contour color (hex string, RGB in [0, 1] or in [0, 255]).
    opacity : float, optional
        Opacity from 0 (transparent) to 1 (opaque).
    iterations : int, optional
        Number of smoothing passes. Use 0 to keep the blocky voxel surface.

    Returns
    -------
    Group
        One surface actor per connected object of the ROI.

    Notes
    -----
    :func:`fury.actor.contour_from_roi` draws the voxel faces, which looks
    staircased on anatomical ROIs. This function smooths that surface with
    Taubin smoothing, which does not shrink the ROI.
    """
    color = normalize_colors(color)[0][:3]
    contour = Group()
    for mesh in voxel_mesh_by_object(data, triangulate=True).values():
        vertices = _taubin_smooth(mesh["verts"], mesh["faces"], iterations)
        if affine is not None:
            vertices = apply_affine(affine, vertices)
        contour.add(surface(vertices, mesh["faces"], colors=color, opacity=opacity))
    return contour


def _map_to_colors(values, *, clim, colors):
    """Map scalar values to the nearest of the evenly spaced ``colors``.

    Parameters
    ----------
    values : array_like
        Scalar values; those outside ``clim`` are clipped.
    clim : tuple of float
        Scalar values ``(low, high)`` mapped to the first and last colors.
    colors : ndarray of shape (M, 3)
        RGB colors ordered from low to high values.

    Returns
    -------
    ndarray of shape (..., 3)
        RGB colors of ``values``.
    """
    low, high = clim
    fraction = np.clip((np.asarray(values) - low) / (high - low), 0, 1)
    return colors[np.round(fraction * (len(colors) - 1)).astype(int)]


# TODO(fury#1392, fury#1393): FURY v2 has no scalar bar / colorbar actor, and
# ``fury.colormap.create_colormap`` only accepts colormap names. Remove
# ``scalar_bar`` and use the FURY actor, with custom colormap support, once
# https://github.com/fury-gl/fury/issues/1392 and
# https://github.com/fury-gl/fury/issues/1393 are resolved.
def scalar_bar(
    *,
    clim=(0.0, 1.0),
    cmap="plasma",
    hsv_range=None,
    orientation="horizontal",
    length=300,
    thickness=20,
    num_ticks=5,
    tick_format=".3g",
    title=None,
    font_size=14,
    color=(1.0, 1.0, 1.0),
    position=(60, 60),
):
    """Create a color bar showing how scalar values map to colors.

    Parameters
    ----------
    clim : tuple of float
        Scalar values ``(low, high)`` mapped to the first and last colors.
    cmap : str
        Name of the colormap, as accepted by ``fury.colormap.create_colormap``.
    hsv_range : tuple of three (low, high) pairs, optional
        Hue, saturation and value ramps, each in [0, 1], from the first to the
        last color. Replaces ``cmap`` when given.
    orientation : {"horizontal", "vertical"}
        Bar orientation. Low values are on the left or at the bottom.
    length : float
        Bar length in pixels.
    thickness : float
        Bar thickness in pixels.
    num_ticks : int
        Number of evenly spaced tick labels, including both ends.
    tick_format : str
        Format specification applied to each tick value.
    title : str, optional
        Text drawn above the bar.
    font_size : float
        Font size of the labels in pixels.
    color : tuple of float
        RGB color of the labels.
    position : tuple of float
        Bottom-left corner of the bar in pixels from the bottom-left of the
        screen viewport.

    Returns
    -------
    bar : fury.actor.Group
        The color bar actor, to be added to a screen's UI scene. Its
        ``map_values`` attribute maps scalar values to the RGB colors of the
        bar, clipping values outside ``clim``.

    Notes
    -----
    Add it with ``show_manager.screens[0].scene.ui_scene.add(bar)``. Temporary
    until FURY provides one with custom colormaps
    (https://github.com/fury-gl/fury/issues/1392 and
    https://github.com/fury-gl/fury/issues/1393).
    """
    if orientation not in ("horizontal", "vertical"):
        raise ValueError(
            f"orientation must be 'horizontal' or 'vertical', got {orientation!r}."
        )
    low, high = clim
    if not low < high:
        raise ValueError(f"clim must satisfy low < high, got {clim}.")
    if num_ticks < 2:
        raise ValueError(f"num_ticks must be at least 2, got {num_ticks}.")

    horizontal = orientation == "horizontal"
    if hsv_range is None:
        colors = create_colormap(
            np.linspace(0, 1, _SCALAR_BAR_SAMPLES), name=cmap, auto=False
        )
    else:
        hsv = np.stack(
            [np.linspace(*ramp, _SCALAR_BAR_SAMPLES) for ramp in hsv_range], axis=1
        )
        colors = hsv2rgb(hsv)
    num_colors = len(colors)
    gradient = colors[None] if horizontal else colors[:, None]
    texel_size = [
        length / num_colors if horizontal else thickness,
        thickness if horizontal else length / num_colors,
        1,
    ]
    gradient_actor = image(gradient, interpolation="linear")
    gradient_actor.local.scale = texel_size
    # Image texels are centered on integer coordinates; shift the first one
    # so the bar starts exactly at the group origin.
    gradient_actor.local.position = [0.5 * texel_size[0], 0.5 * texel_size[1], 0]

    bar = Group(name="DIPY Scalar Bar")
    bar.local.position = [position[0], position[1], 0]
    bar.add(gradient_actor)

    margin = 4
    values = np.linspace(low, high, num_ticks)
    offsets = np.linspace(0, length, num_ticks)
    for value, offset in zip(values, offsets):
        label_position = (
            (offset, -margin, 0) if horizontal else (thickness + margin, offset, 0)
        )
        bar.add(
            text(
                format(value, tick_format),
                colors=color,
                position=label_position,
                font_size=font_size,
                anchor="top-center" if horizontal else "middle-left",
            )
        )
    if title is not None:
        title_y = thickness + margin if horizontal else length + margin
        bar.add(
            text(
                title,
                colors=color,
                position=(0, title_y, 0),
                font_size=font_size,
                anchor="bottom-left",
            )
        )
    bar.map_values = partial(_map_to_colors, clim=clim, colors=colors)
    return bar
