"""FURY actors plus DIPY-specific visualization actors."""

from functools import partial

import numpy as np

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
    from fury.actor import Group, create_axes_helper
    from fury.lib import EventType
    from fury.transform import apply_transformation
    from fury.window import update_camera
else:
    Group = create_axes_helper = EventType = fury
    apply_transformation = update_camera = fury


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
        The gizmo actor, hosted in the screen's UI scene.

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

    target_screen.scene.ui_scene.add(anchor)
    _sync_gizmo(parts, target_screen.camera)
    show_manager.register_callback(
        _sync_gizmo, 0.016, True, name, parts, target_screen.camera
    )
    return anchor
