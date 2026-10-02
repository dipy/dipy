"""
plotting functions
"""

import math
from warnings import warn

import numpy as np

from dipy.core.gradients import GradientTable
from dipy.core.sphere import Sphere
from dipy.io.utils import split_filename_extension
from dipy.utils.deprecator import warning_for_keywords
from dipy.utils.optpkg import optional_package

fury, have_fury, _ = optional_package("fury", min_version="2.0.0")
if have_fury:
    from fury import actor, ui, window

plt, have_plt, _ = optional_package("matplotlib.pyplot")


def plot_gradient_sphere(data, *, scene=None, colors=None, filename=None, show=False):
    """Render an acquisition gradient scheme or sphere in q-space with FURY.

    Parameters
    ----------
    data : GradientTable or Sphere
        Gradient scheme or unit-sphere vertices to render.
    scene : fury.window.Scene, optional
        Existing scene to add the glyphs to. A white scene is created when absent.
    colors : tuple or ndarray, optional
        One RGB color or one RGB color for every plotted point.
    filename : str or Path, optional
        Output image path.
    show : bool, optional
        Whether to display the scene interactively. Hovering a glyph updates the
        information text at the bottom of the window. If ``filename`` is given,
        its basename without the extension is used as the window title.

    Returns
    -------
    scene : fury.window.Scene
        Scene containing the rendered glyphs.

    Raises
    ------
    TypeError
        If ``data`` is neither a gradient table nor a sphere.
    ValueError
        If FURY is unavailable, the gradient table has no valid non-b0
        directions, the colors have an invalid shape, or ``scene`` is invalid.
    """
    if not have_fury:
        raise ValueError("fury package needed for visualization.")

    if isinstance(data, GradientTable):
        non_b0_mask = ~data.b0s_mask
        non_b0_bvals = data.bvals[non_b0_mask]
        if (
            not np.any(non_b0_mask)
            or not np.all(np.isfinite(non_b0_bvals))
            or np.any(non_b0_bvals <= 0)
        ):
            raise ValueError(
                "Gradient table must contain finite, positive non-b0 b-values."
            )
        points = data.gradients / np.max(non_b0_bvals)
        glyph_colors = np.zeros((len(points), 3))
        shell_span = np.ptp(non_b0_bvals)
        glyph_colors[non_b0_mask, 2] = 1.0
        if shell_span == 0:
            glyph_colors[non_b0_mask, 1] = 0.0
        else:
            glyph_colors[non_b0_mask, 1] = (
                non_b0_bvals - non_b0_bvals.min()
            ) / shell_span
    elif isinstance(data, Sphere):
        points = data.vertices
        glyph_colors = (0.0, 1.0, 0.0)
    else:
        raise TypeError("data must be a GradientTable or Sphere.")

    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape == (3,):
            glyph_colors = tuple(colors)
        elif colors.shape == (len(points), 3):
            glyph_colors = colors
        else:
            raise ValueError("colors must be one RGB tuple or an (N, 3) RGB array.")
    if scene is None:
        scene = window.Scene(background=(1.0, 1.0, 1.0, 1.0))
    elif not isinstance(scene, window.Scene):
        raise ValueError("scene must be a FURY Scene.")

    info_text = getattr(scene, "_gradient_sphere_info_text", None)
    if info_text is None:
        info_text = ui.TextBlock2D(
            text="Hover a sphere to inspect it.",
            font_size=24,
            color=(0.0, 0.0, 0.0),
            bg_color=(1.0, 1.0, 1.0),
            position=(20, 750),
            size=(760, 30),
        )
        scene.add(info_text)
        scene._gradient_sphere_info_text = info_text

    glyph_actor = actor.sphere(
        points, colors=glyph_colors, radii=0.04, impostor=False, theta=48, phi=48
    )
    faces_per_point = len(glyph_actor.geometry.indices.data) // len(points)

    def update_info(event):
        face_index = event.pick_info.get("face_index")
        if face_index is None:
            return
        point_index = face_index // faces_per_point
        point = points[point_index]
        point_text = f"q=({point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f})"
        if isinstance(data, GradientTable):
            if data.b0s_mask[point_index]:
                info_text.message = f"Gradient {point_index}: b0, {point_text}"
            else:
                info_text.message = (
                    f"Gradient {point_index}: b={data.bvals[point_index]:g} s/mm², "
                    f"{point_text}"
                )
        else:
            info_text.message = f"Sphere vertex {point_index}: {point_text}"
        info_text.update_alignment()

    def clear_info(event):
        info_text.message = "Hover a sphere to inspect it."
        info_text.update_alignment()

    glyph_actor.add_event_handler(update_info, "pointer_move")
    glyph_actor.add_event_handler(clear_info, "pointer_leave")
    scene.add(glyph_actor)
    if filename is not None or show:
        title, _ = (
            split_filename_extension(filename) if filename is not None else ("DIPY", "")
        )
        manager = window.ShowManager(
            scene=scene,
            size=(1000, 1000) if filename is not None else (800, 800),
            window_type="default" if show else "offscreen",
            title=title,
        )

        def position_info_text(size):
            available_width = max(size[0] - 40, 1)
            max_message_width = 32 * info_text.font_size
            lines = math.ceil(max_message_width / available_width)
            footer_height = min(
                max(size[1] - 40, 1),
                16 + lines * math.ceil(info_text.font_size * 1.4),
            )
            info_text.resize((available_width, footer_height))
            info_text.set_position((20, size[1] - footer_height - 20))

        manager.resize_callback(position_info_text)
        position_info_text(manager.size)
        manager.render()
        window.render_screens(manager.renderer, manager.screens, is_dirty=True)
        info_text.update_alignment()
        if filename is not None:
            manager._draw_function()
            manager.snapshot(fname=filename)
        if show:
            manager.start()

    return scene


@warning_for_keywords()
def compare_maps(
    fits,
    maps,
    *,
    transpose=None,
    fit_labels=None,
    map_labels=None,
    fit_kwargs=None,
    map_kwargs=None,
    filename=None,
):
    """Compare one or more scalar maps for different fits or models.

    Parameters
    ----------
    fits : list
        List of fits to be compared.
    maps : list
        Names of attributes to be compared.
        Default: 'rtop'.
    transpose : bool, optional
        If False, different fits are placed on different rows and different
        maps on different columns. If True, the order is transposed. If None,
        the figures are placed such that there are more columns than rows.
        Default: None.
    fit_labels : list, optional
        Labels for the different fitting routines. If None the fits are labeled
        by number.
        Default: None.
    map_labels : list, optional
        Labels for the different attributes. If None the attribute names are
        used.
        Default: None.
    fit_kwargs : list or dict, optional
        A dict or list of dicts with imshow options for each fitting routine.
        The dicts are passed to imshow as keyword-argument pairs.
        Default: {}.
    map_kwargs : list or dict, optional
        A dict or list of dicts with imshow options for each MAP-MRI scalar.
        The dicts are passed to imshow as keyword-argument pairs.
        Default: {}.
    filename : string, optional
        Filename where the image will be saved.
        Default: None.
    """
    fit_kwargs = fit_kwargs or {}
    map_kwargs = map_kwargs or {}

    if not have_plt:
        raise ValueError("matplotlib package needed for visualization.")

    fontsize = "large"
    xscale, yscale = 12, 10

    m = len(fits)
    n = len(maps)

    if transpose is None:
        transpose = m > n

    if fit_labels is None:
        fit_labels = [f"Fit {i + 1}" for i in range(m)]
    if map_labels is None:
        map_labels = maps

    if isinstance(fit_kwargs, dict):
        fit_kwargs = [fit_kwargs] * m
    if isinstance(map_kwargs, dict):
        map_kwargs = [map_kwargs] * n

    if transpose:
        fig, ax = plt.subplots(n, m, figsize=(xscale, yscale / m * n), squeeze=False)
        ax = ax.T
        for i in range(m):
            ax[i, 0].set_title(fit_labels[i], fontsize=fontsize)
        for j in range(n):
            ax[0, j].set_ylabel(map_labels[j], fontsize=fontsize)
    else:
        fig, ax = plt.subplots(m, n, figsize=(xscale, yscale / n * m), squeeze=False)
        for i in range(m):
            ax[i, 0].set_ylabel(fit_labels[i], fontsize=fontsize)
        for j in range(n):
            ax[0, j].set_title(map_labels[j], fontsize=fontsize)

    for i in range(m):
        for j in range(n):
            try:
                attr = getattr(fits[i], maps[j])
                if callable(attr):
                    attr = attr()
            except AttributeError:
                warn(f"Could not recover attribute {maps[j]}.", stacklevel=2)
                attr = np.zeros((2, 2))
            data = np.squeeze(np.array(attr, dtype=float)).T
            ax[i, j].imshow(
                data,
                interpolation="nearest",
                origin="lower",
                cmap="gray",
                **fit_kwargs[i],
                **map_kwargs[j],
            )
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            ax[i, j].spines["top"].set_visible(False)
            ax[i, j].spines["right"].set_visible(False)
            ax[i, j].spines["bottom"].set_visible(False)
            ax[i, j].spines["left"].set_visible(False)

    fig.tight_layout()

    if filename:
        plt.savefig(filename)
    else:
        plt.show()


@warning_for_keywords()
def compare_qti_maps(
    gt,
    fit1,
    fit2,
    mask,
    *,
    maps=("fa", "ufa"),
    fitname=("QTI", "QTI+"),
    xlimits=([0, 1], [0.4, 1.5]),
    disprange=([0, 1], [0, 1]),
    slice=13,
):
    """Compare one or more qti derived maps obtained with
    different fitting routines.

    Parameters
    ----------
    gt : qti fit object
        The qti fit to be considered as ground truth
    fit1 : qti fit object
        First qti fit to be compared
    fit2 : qti fit object
        Second qti fit to be compared
    mask : np.ndarray
        Boolean array indicating which voxels to retain for comparing
        the values
    maps : array-like, optional
        QTI invariants to be compared
    fitname : array-like, optional
        Names of the used QTI fitting routines
    xlimits : array-like, optional
        X-Axis limits for the histograms visualization
    disprange : array-like, optional
        Display range for maps
    slice : int, optional
        Axial brain slice to be visualized
    """
    if not have_plt:
        raise ValueError("matplotlib package needed for visualization")

    n = len(maps)
    fig, ax = plt.subplots(n, 4, figsize=(12, 9))

    background = np.zeros(gt.S0_hat.shape[0:2])
    for i in range(n):
        for j in range(3):
            ax[i, j].imshow(background, cmap="gray")
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])

    for k in range(n):
        ax[k, 0].imshow(
            np.rot90(getattr(gt, maps[k])[:, :, slice]),
            cmap="gray",
            vmin=disprange[k][0],
            vmax=disprange[k][1],
        )
        ax[k, 0].set_title("GROUND TRUTH")
        ax[k, 0].set_ylabel(maps[k], fontsize=20)

        ax[k, 1].imshow(
            np.rot90(getattr(fit1, maps[k])[:, :, slice]),
            cmap="gray",
            vmin=disprange[k][0],
            vmax=disprange[k][1],
        )
        ax[k, 1].set_title(fitname[0])

        ax[k, 2].imshow(
            np.rot90(getattr(fit2, maps[k])[:, :, slice]),
            cmap="gray",
            vmin=disprange[k][0],
            vmax=disprange[k][1],
        )
        ax[k, 2].set_title(fitname[1])

        ax[k, 3].hist(
            (getattr(fit1, maps[k])[mask, slice]).flatten(),
            density=True,
            bins=40,
            label=fitname[0],
        )
        ax[k, 3].hist(
            (getattr(fit2, maps[k])[mask, slice]).flatten(),
            density=True,
            bins=40,
            label=fitname[1],
            alpha=0.7,
        )
        ax[k, 3].hist(
            (getattr(gt, maps[k])[mask, slice]).flatten(),
            histtype="stepfilled",
            density=True,
            bins=40,
            label="GT",
            ec="k",
            alpha=1,
            linewidth=1.5,
            fc="None",
        )
        ax[k, 3].legend()
        ax[k, 3].set_title("VALUE DISTRIBUTION")
        ax[k, 3].set_xlim(xlimits[k])

    fig.tight_layout()
    plt.show()


def bundle_profile_plot(
    x, profile, ylabel, *, title="Bundle Profile", std=None, save_path=None, show=True
):
    """Plot bundle profile.

    Parameters
    ----------
    x : np.ndarray
        Integer array containing x-axis
    profile : np.ndarray
        Float array containing bundle profile
    ylabel : str
        ylabel for the plot
    title : str, optional
        Plot title
    std : np.ndarray, optional
        Float array containing standard deviations
    save_path : str, optional
        If provided, save the figure to this path (e.g., "profile.png")
    show : bool, optional
        Whether to display the plot interactively

    """
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    ax.plot(x, profile, "-", label="Mean", color="Purple", linewidth=3, markersize=12)

    if std is not None:
        std_1 = profile + std
        std_2 = profile - std
        ax.fill_between(x, std_1, std_2, alpha=0.2, label="Std", color="Purple")
        plt.ylim(0, max(std_1) + 2)

    plt.xticks(x)
    plt.ylabel(ylabel)
    plt.xlabel("Segment Number")
    plt.title(title)
    plt.legend(loc=2)

    if save_path is not None:
        fig.savefig(save_path, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def image_mosaic(
    images, *, ax_labels=None, ax_kwargs=None, figsize=None, filename=None
):
    """
    Draw a mosaic of 2D images using pyplot.imshow(). A colorbar is drawn
    beside each image.

    Parameters
    ----------
    images: list of ndarray
        Images to render.
    ax_labels: list of str, optional
        Label for each image.
    ax_kwargs: list of dictionaries, optional
        keyword arguments passed to imshow for each image. One dictionary per
        image.
    figsize: tuple of ints, optional
        Figure size.
    filename: str, optional
        When given, figure is saved to disk under this name.

    Returns
    -------
    fig: pyplot.Figure
        The figure.
    ax: pyplot.Axes or array of Axes
        The subplots for each image.
    """
    fig, ax = plt.subplots(1, len(images), figsize=figsize)

    aximages = []
    for it, (im, axe, kw) in enumerate(zip(images, ax, ax_kwargs)):
        aximages.append(axe.imshow(im, **kw))
        if ax_labels is not None:
            axe.set_title(ax_labels[it])

    for it, aximage in enumerate(aximages):
        fig.colorbar(aximage, ax=ax[it])

    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()

    return fig, ax
