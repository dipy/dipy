from pathlib import Path
import warnings

import numpy as np

from dipy.utils.deprecator import warning_for_keywords
from dipy.utils.optpkg import optional_package

fury, has_fury, _ = optional_package("fury", min_version="2.1.0")
if has_fury:
    from fury import window
    from fury.colormap import line_colors

    from dipy.viz import actor

sagittal_deprecation_warning_msg = (
    "The view argument value `sagital` is deprecated and "  # codespell:ignore sagital
    "its support will be removed in a future version. Please, use `sagittal` instead."
)

_VIEW_LABELS = {"sagittal": "L", "axial": "S", "coronal": "A"}


def _show_scene(scene, *, view_label, size, fname=None, interactive=False):
    """Save a scene to a png and/or display it in an interactive window.

    Parameters
    ----------
    scene : fury.window.Scene
        Scene to be rendered.
    view_label : str
        Gizmo label clicked to orient the camera: 'L' (sagittal), 'S' (axial) or
        'A' (coronal).
    size : tuple of int
        Window size in pixels as ``(width, height)``.
    fname : str or Path, optional
        If not None, the scene is rendered offscreen into a png with that name.
    interactive : bool, optional
        If True, a 3D interactive window is opened after saving.
    """
    if fname is not None:
        manager = window.ShowManager(
            scene=scene, window_type="offscreen", size=size, pixel_ratio=1.0
        )
        gizmo = actor.gizmo(manager)
        actor.click_gizmo(manager, gizmo, view_label)
        try:
            manager.render()
            manager.window.draw()
            manager.snapshot(fname=str(fname))
        finally:
            manager.window.close()
            scene.ui_scene.remove(gizmo)

    if interactive:
        title = "DIPY" if fname is None else Path(fname).stem
        manager = window.ShowManager(scene=scene, title=title, size=size)
        actor.click_gizmo(manager, actor.gizmo(manager), view_label)
        manager.start()


@warning_for_keywords()
def show_bundles(
    bundles,
    *,
    interactive=True,
    view="sagittal",
    colors=None,
    linewidth=0.3,
    save_as=None,
):
    """Render bundles to visualize them interactively or save them into a png.

    The function allows to just render the bundles in an interactive window or
    to export them into a png file.

    Parameters
    ----------
    bundles : list
        Bundles to be rendered.
    interactive : bool, optional
        If True a 3D interactive window is opened.
    view : str, optional
        Viewing angle. Supported options: 'sagittal', 'axial' and 'coronal'.
    colors : list, optional
        Colors to be used for each bundle. If None, colors are derived from the
        orientation of each streamline segment.
    linewidth : float, optional
        Thickness of each rendered streamline.
    save_as : str or Path, optional
        If not None, the rendered scene is stored in a png file with that name.
    """

    if view == "sagital":  # codespell:ignore sagital
        warnings.warn(
            sagittal_deprecation_warning_msg,
            category=DeprecationWarning,
            stacklevel=2,
        )
        view = "sagittal"

    if view not in _VIEW_LABELS:
        raise ValueError(
            "Invalid view argument value. Use 'sagittal', 'axial' or 'coronal'."
        )

    scene = window.Scene(background=(1.0, 1.0, 1.0, 1.0))
    for i, bundle in enumerate(bundles):
        bundle_colors = line_colors(bundle) if colors is None else colors[i]
        scene.add(
            actor.streamlines(
                bundle,
                colors=bundle_colors,
                thickness=linewidth,
                outline_thickness=0,
            )
        )

    _show_scene(
        scene,
        view_label=_VIEW_LABELS[view],
        size=(900, 900),
        fname=save_as,
        interactive=interactive,
    )


@warning_for_keywords()
def viz_two_bundles(b1, b2, fname, *, c1=(1, 0, 0), c2=(0, 1, 0), interactive=False):
    """Render two bundles and save the scene into a png.

    Parameters
    ----------
    b1 : Streamlines
        Bundle one to be rendered.
    b2 : Streamlines
        Bundle two to be rendered.
    fname : str or Path
        Rendered scene is stored in a png file with that name.
    c1 : tuple, optional
        Color to be used for first bundle.
    c2 : tuple, optional
        Color to be used for second bundle.
    interactive : bool, optional
        If True a 3D interactive window is opened.
    """
    scene = window.Scene(background=(1.0, 1.0, 1.0, 1.0))
    scene.add(actor.streamlines(b1, colors=c1, thickness=0.3, outline_thickness=0))
    scene.add(actor.streamlines(b2, colors=c2, thickness=0.3, outline_thickness=0))

    _show_scene(
        scene,
        view_label="L",
        size=(1200, 1200),
        fname=fname,
        interactive=interactive,
    )


@warning_for_keywords()
def viz_vector_field(
    points_aligned,
    directions,
    colors,
    offsets,
    fname,
    *,
    bundle=None,
    interactive=False,
):
    """Render a vector field and save the scene into a png.

    Parameters
    ----------
    points_aligned : array-like
        Starting positions of the vectors.
    directions : array-like
        Unitary directions of the vectors.
    colors : array-like
        Colors for each vector.
    offsets : array-like
        Vector field magnitudes, used as the scale of each arrow.
    fname : str or Path
        Rendered scene is stored in a png file with that name.
    bundle : Streamlines, optional
        Bundle to be rendered together with the vector field.
    interactive : bool, optional
        If True a 3D interactive window is opened.

    Notes
    -----
    Arrows are colored by the absolute value of their direction components,
    since the direction colors may be negative.
    """
    scene = window.Scene(background=(1.0, 1.0, 1.0, 1.0))
    scene.add(
        actor.arrow(
            np.asarray(points_aligned),
            directions=np.asarray(directions),
            colors=np.abs(np.asarray(colors)),
            scales=np.repeat(np.asarray(offsets)[:, np.newaxis], 3, axis=1),
        )
    )

    if bundle is not None:
        scene.add(
            actor.streamlines(
                bundle, colors=(0, 0, 1), thickness=0.3, outline_thickness=0
            )
        )

    _show_scene(
        scene,
        view_label="L",
        size=(1200, 1200),
        fname=fname,
        interactive=interactive,
    )


@warning_for_keywords()
def viz_displacement_mag(bundle, offsets, fname, *, interactive=False):
    """Render the displacement magnitude over a bundle and save it into a png.

    Parameters
    ----------
    bundle : Streamlines
        Bundle to be rendered.
    offsets : array-like
        Displacement magnitudes, one value per point of ``bundle``.
    fname : str or Path
        Rendered scene is stored in a png file with that name.
    interactive : bool, optional
        If True a 3D interactive window is opened.

    Notes
    -----
    Colors run linearly from pink (smallest offset) to green (largest offset),
    as shown by the color bar at the bottom of the image. ``offsets`` must not
    be constant, since the color bar needs a non-empty range.
    """
    offsets = np.asarray(offsets)
    bar = actor.scalar_bar(
        clim=(offsets.min(), offsets.max()),
        hsv_range=((0.9, 0.3), (0.5, 1.0), (0.8, 0.8)),
        title="Displacement (mm)",
        tick_format=".1f",
        length=600,
        thickness=30,
        font_size=28,
        color=(0.0, 0.0, 0.0),
        position=(1300, 100),
    )

    scene = window.Scene(background=(1.0, 1.0, 1.0, 1.0))
    scene.add(
        actor.streamlines(
            bundle, colors=bar.map_values(offsets), thickness=0.3, outline_thickness=0
        )
    )
    scene.ui_scene.add(bar)

    _show_scene(
        scene,
        view_label="L",
        size=(2000, 1500),
        fname=fname,
        interactive=interactive,
    )
