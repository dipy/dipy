from dipy.utils.optpkg import optional_package
plt, have_plt, _ = optional_package("matplotlib.pyplot")
fury, has_fury, _ = optional_package('fury', min_version="0.10.0")
if has_fury:
    from dipy.viz import window, actor


def show_bundles(bundles, interactive=True, view='sagital', colors=None,
                 linewidth=0.3, save_as=None):
    """Render bundles to visualize them interactively or save them into a png.

    The function allows to just render the bundles in an interactive plot or
    to export them into a png file.

    Parameters
    ----------
    bundles : list
        Bundles to be rendered.
    interactive : boolean, optional
        If True a 3D interactive rendering is created. Default is True.
    view : str, optional
        Viewing angle. Supported options: 'sagital','axial' and 'coronal'.
        Default is 'sagital'.
    colors : list, optional
       Colors to be used for each bundle. If None default colors are used.
    linewidth : float, optional
        Width of each rendered streamline. Default is 0.3.
    save_as : str, optional
        If not None rendered scene is stored in a png file with that name.
        Default is None.

    """
    scene = window.Scene()
    scene.SetBackground(1., 1, 1)

    for i, bundle in enumerate(bundles):
        if colors is None:
            lines_actor = actor.streamtube(bundle, linewidth=linewidth)
        else:
            lines_actor = actor.streamtube(bundle, linewidth=linewidth,
                                           colors=colors[i])
        if view == 'sagital':
            lines_actor.RotateX(-90)
            lines_actor.RotateZ(90)
        elif view == 'axial':
            pass
        elif view == 'coronal':
            lines_actor.RotateX(-90)
        else:
            raise ValueError("Invalid view argument value. Use 'sagital'," +
                             "'axial' or 'coronal'.")

        scene.add(lines_actor)

    if interactive:
        window.show(scene)

    if save_as is not None:
        window.record(scene, n_frames=1, out_path=save_as, size=(900, 900))


def viz_two_bundles(b1, b2, fname, c1=(1, 0, 0), c2=(0, 1, 0),
                    interactive=False):
    """Render and plot two bundles to visualize them.

    Parameters
    ----------
    b1 : Streamlines
        Bundle one to be rendered.
    b2 : Streamlines
        Bundle two to be rendered.
    fname: str
        Rendered scene is stored in a png file with that name.
    C1 : tuple, optional
        Color to be used for first bundle. Default red.
    C2 : tuple, optional
        Color to be used for second bundle. Default green.
    interactive : boolean, optional
        If True a 3D interactive rendering is created. Default is True.

    """
    ren = window.Scene()
    ren.SetBackground(1, 1, 1)

    actor1 = actor.line(b1, colors=c1)
    actor1.GetProperty().SetEdgeVisibility(1)
    actor1.GetProperty().SetRenderLinesAsTubes(1)
    actor1.GetProperty().SetLineWidth(6)
    actor1.GetProperty().SetOpacity(1)
    actor1.RotateX(-70)
    actor1.RotateZ(90)

    ren.add(actor1)

    actor2 = actor.line(b2, colors=c2)
    actor2.GetProperty().SetEdgeVisibility(1)
    actor2.GetProperty().SetRenderLinesAsTubes(1)
    actor2.GetProperty().SetLineWidth(6)
    actor2.GetProperty().SetOpacity(1)
    actor2.RotateX(-70)
    actor2.RotateZ(90)

    ren.add(actor2)

    if interactive:
        window.show(ren)

    window.record(ren, n_frames=1, out_path=fname, size=(1200, 1200))
    if interactive:
        im = plt.imread(fname)
        plt.figure(figsize=(10, 10))
        plt.imshow(im)


def viz_vector_field(points_aligned, directions, colors, offsets, fname,
                     bundle=None, interactive=False):
    """Render and plot vector field.

    Parameters
    ----------
    points_aligned : List
        List containing starting positions of vectors.
    directions : List
        List containing unitary directions of vectors.
    colors : List
        List containing colors for each vector.
    offsets : List
        List containing vector field modules.
    fname: str
        Rendered scene is stored in a png file with that name.
    bundle : Streamlines, optional
        Bundle to be rendered with vector field (Default None).
    interactive : boolean, optional
        If True a 3D interactive rendering is created. Default is True.

    """
    scene = window.Scene()
    scene.SetBackground(1., 1, 1)
    arrows = actor.arrow(points_aligned, directions, colors, scales=offsets)
    arrows.RotateX(-70)
    arrows.RotateZ(90)
    scene.add(arrows)

    if bundle:
        actor1 = actor.line(bundle, colors=(0, 0, 1))
        actor1.RotateX(-70)
        actor1.RotateZ(90)
        scene.add(actor1)

    if interactive:
        window.show(scene)

    window.record(scene, n_frames=1, out_path=fname, size=(1200, 1200))
    if interactive:
        im = plt.imread(fname)
        plt.figure(figsize=(10, 10))
        plt.imshow(im)


def viz_displacement_mag(bundle, offsets, fname, interactive=False):
    """Render and plot displacement magnitude over the bundle.

    Parameters
    ----------
    bundle : Streamlines,
        Bundle to be rendered.
    offsets : List
        List containing displacement magnitdues per point on the bundle.
    fname: str
        Rendered scene is stored in a png file with that name.
    interactive : boolean, optional
        If True a 3D interactive rendering is created. Default is True.

    """
    scene = window.Scene()
    hue = (0.1, 0.9)
    hue = (0.9, 0.3)
    saturation = (0.5, 1)
    scene.background((1, 1, 1))
    lut_cmap = actor.colormap_lookup_table(
        scale_range=(offsets.min(), offsets.max()),
        hue_range=hue,
        saturation_range=saturation)

    stream_actor = actor.line(bundle, offsets, linewidth=7,
                              lookup_colormap=lut_cmap)

    stream_actor.RotateX(-70)
    stream_actor.RotateZ(90)

    scene.add(stream_actor)
    bar = actor.scalar_bar(lut_cmap)

    scene.add(bar)

    if interactive:
        window.show(scene)

    window.record(scene, n_frames=1, out_path=fname, size=(2000, 1500))
    if interactive:
        im = plt.imread(fname)
        plt.figure(figsize=(10, 10))
        plt.imshow(im)
