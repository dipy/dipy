from dipy.utils.optpkg import optional_package

fury, has_fury, _ = optional_package('fury')
if has_fury:
    from dipy.viz import window, actor


def show_bundles(bundles, interactive=True, view='sagital', colors=None,
                 linewidth=0.3, fname=None):
    """ Render bundles to visualize them interactively or save them into a png.

    The function allows to just render the bundles in an interactive plot or
    to export them into a png file.

    Parameters
    ----------
    bundles : list
        Bundles to be rendered.
    interactive : boolean
        If True a 3D interactive rendering is created. Default is True.
    view : string
        Viewing angle. Supported options: 'sagital','axial' and 'coronal'.
        Default is 'sagital'.
    colors : list
       Colors to be used for each bundle. If None default colors are used.
    linewidth : float
        Width of each rendered streamline. Default is 0.3.
    fname : string
        If not None rendered scene is stored in a png file with that name.
        Default is None.

    Returns
    -------
    scene : FURY Scene() object with rendered bundles.

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

    if fname is not None:
        window.record(scene, n_frames=1, out_path=fname, size=(900, 900))

    return scene
